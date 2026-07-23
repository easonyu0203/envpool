# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit test for the Phase 2 ptcg envpool (real ptcg_engine calls).

See the envpool-ptcg-integration skill's "Phased implementation checklist" --
Phase 2's bar is the real Reset()/Step() flow: 2-step deck-select handshake
into ApiBattleStart, then real ApiSelect stepping until state.isFinish().
Real games are nondeterministic in length and first-player assignment (no
seed parameter on ApiBattleStart -- see "Resets are nondeterministic" in the
skill), so unlike Phase 1's fixed 6-step fake episode, these tests assert
structural invariants across full real episodes rather than an exact
sequence table.
"""

import numpy as np
from absl.testing import absltest
from envpool.ptcg.ptcg_envpool import _PtcgEnvPool, _PtcgEnvSpec

# Registers "Ptcg-v0" as a side effect. Deliberately not routed through
# envpool/entry.py, and `make` is imported from envpool.registration rather
# than as `envpool.make` -- both avoid the top-level `envpool/__init__.py`,
# which unconditionally imports every other family's registration (and
# through it, every other family's compiled .so) via entry.py. See
# registration.py's module docstring and "Build path" in the
# envpool-ptcg-integration skill.
import envpool.ptcg.registration  # noqa: F401,E402
from envpool.registration import make  # noqa: E402

_ACTION_SLOTS = 60
# Generous safety bound on rounds/steps to reach `done`, matching the
# convention already used by ptcg_smoke_test.cc / concurrency_smoke.cpp for
# a single real game -- not a tuned expectation of how long a game runs.
_MAX_ROUNDS = 20000

# Same real, deck-legal 60-card list as ptcg_envpool_test.cc / ptcg_smoke's
# / research/envpool_smoke/concurrency_smoke.cpp, copied from
# submissions/sample_submission/deck.csv. Both seats use it.
_DECK = [
    1158, 721, 721, 722, 722, 722, 722, 723, 723, 723, 723, 1145,
    1145, 1145, 1145, 1205, 1205, 1227, 1227, 1227, 1227, 1235, 1235, 1235,
    1235, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
]


def _default_conf(**overrides: object) -> dict:
    conf = dict(
        zip(
            _PtcgEnvSpec._config_keys,
            _PtcgEnvSpec._default_config_values,
            strict=False,
        )
    )
    conf.update(overrides)
    return conf


class PtcgEnvPoolTest(absltest.TestCase):
    def test_config(self) -> None:
        # PtcgEnvFns::DefaultConfig() adds nothing beyond envpool's common
        # config (see ptcg_envpool.h) -- no family-specific knobs.
        ref_config_keys = [
            "num_envs",
            "batch_size",
            "num_threads",
            "max_num_players",
            "thread_affinity_offset",
            "base_path",
            "seed",
            "env_seed",
            "gym_reset_return_info",
            "max_episode_steps",
        ]
        self.assertEqual(
            sorted(_PtcgEnvSpec._config_keys), sorted(ref_config_keys)
        )

    def test_spec(self) -> None:
        conf = _default_conf()
        env_spec = _PtcgEnvSpec(tuple(conf.values()))
        state_spec = dict(
            zip(env_spec._state_keys, env_spec._state_spec, strict=False)
        )
        action_spec = dict(
            zip(env_spec._action_keys, env_spec._action_spec, strict=False)
        )
        # ArraySpec tuples are (shape, dtype-bounds...); index 1 is shape,
        # matching dummy_py_envpool_test.py's state_spec[...][1] usage.
        self.assertEqual(tuple(state_spec["obs:cards"][1]), (120, 5))
        self.assertEqual(tuple(state_spec["obs:pokemons"][1]), (18, 15))
        self.assertEqual(tuple(state_spec["obs:player_state"][1]), (2, 9))
        self.assertEqual(tuple(state_spec["obs:state"][1]), (8,))
        self.assertEqual(tuple(state_spec["obs:select"][1]), (16,))
        self.assertEqual(tuple(state_spec["obs:options"][1]), (63, 19))
        self.assertEqual(tuple(state_spec["info:current_player"][1]), ())
        self.assertEqual(tuple(state_spec["info:is_deck_select"][1]), ())
        self.assertEqual(
            tuple(action_spec["action"][1]), (_ACTION_SLOTS,)
        )

    def test_raw_envpool_real_engine_end_to_end(self) -> None:
        # Small num_envs == batch_size so the sync path preserves order, and
        # real games (dozens+ decisions each) force StateBuffer ring-buffer
        # slot reuse many times over -- same regression intent as Phase 1's
        # version of this test: the 6 obs tensors are zero only because
        # PtcgEnv calls Array::Zero() on every write, not because Allocate()
        # zero-inits on every call (see WriteState's comment in
        # ptcg_envpool.h).
        #
        # Also exercises AsyncEnvPool's auto-reset: Send()-ing any action
        # for an env_id that just reported done triggers Reset() instead of
        # Step() (async_envpool.h:127) -- this loop keeps driving every env,
        # including ones that already finished once, until the slowest one
        # finishes its first episode too.
        num_envs = 3
        conf = _default_conf(
            num_envs=num_envs, batch_size=num_envs, num_threads=1
        )
        env_spec = _PtcgEnvSpec(tuple(conf.values()))
        env = _PtcgEnvPool(env_spec)
        state_keys = env._state_keys

        def recv_dict() -> dict:
            return dict(zip(state_keys, env._recv(), strict=False))

        def assert_zero_obs(state: dict) -> None:
            for key in (
                "obs:cards",
                "obs:pokemons",
                "obs:player_state",
                "obs:state",
                "obs:select",
                "obs:options",
            ):
                np.testing.assert_array_equal(
                    state[key], np.zeros_like(state[key])
                )

        track = [
            {
                "is_deck_select": True,
                "done": False,
                "episode_step": 0,
                "completed_once": False,
            } for _ in range(num_envs)
        ]

        def process(state: dict) -> None:
            assert_zero_obs(state)
            np.testing.assert_array_equal(
                state["reward"], np.zeros(num_envs, dtype=np.float32)
            )
            env_ids = state["info:env_id"]
            for slot in range(num_envs):
                env_id = int(env_ids[slot])
                t = track[env_id]
                is_deck_select = bool(state["info:is_deck_select"][slot])
                current_player = int(state["info:current_player"][slot])
                done = bool(state["done"][slot])

                self.assertEqual(
                    is_deck_select,
                    t["episode_step"] < 2,
                    f"env_id={env_id} episode_step={t['episode_step']}",
                )
                if t["episode_step"] == 0:
                    self.assertEqual(current_player, 0, f"env_id={env_id}")
                elif t["episode_step"] == 1:
                    self.assertEqual(current_player, 1, f"env_id={env_id}")
                else:
                    self.assertIn(current_player, (0, 1), f"env_id={env_id}")

                t["is_deck_select"] = is_deck_select
                t["done"] = done
                t["episode_step"] += 1
                if done:
                    t["completed_once"] = True

        env._reset(np.arange(num_envs, dtype=np.int32))
        process(recv_dict())  # Reset()'s output: player-0 deck-select.

        round_count = 0
        while not all(t["completed_once"] for t in track):
            self.assertLess(
                round_count,
                _MAX_ROUNDS,
                "not every env reached done within the safety bound",
            )
            round_count += 1

            action = np.full((num_envs, _ACTION_SLOTS), -1, dtype=np.int32)
            for env_id in range(num_envs):
                t = track[env_id]
                # A `done` env's next send triggers an internal auto-reset
                # regardless of content -- see this test's docstring-comment
                # above.
                if not t["done"] and t["is_deck_select"]:
                    action[env_id] = _DECK
                if t["done"]:
                    t["episode_step"] = 0  # about to start a fresh episode

            env._send((
                np.arange(num_envs, dtype=np.int32),  # env_id
                np.arange(num_envs, dtype=np.int32),  # players.env_id
                action,
            ))
            process(recv_dict())

        for t in track:
            self.assertTrue(t["completed_once"])

    def test_make_ptcg_v0(self) -> None:
        # The literal Phase 1 bar, still true in Phase 2: "Ptcg-v0" must be
        # importable and runnable through envpool's registration/make
        # machinery -- now driving a real engine instead of Phase 1's fake
        # episode. Exhaustive structural invariants (is_deck_select
        # sequencing, current_player bounds, all-zero obs, auto-reset across
        # episodes) are already covered thoroughly by
        # test_raw_envpool_real_engine_end_to_end above -- this only needs
        # to prove the make()/Gymnasium-wrapper path itself carries a real
        # engine correctly, so it stops at the first termination rather than
        # re-proving everything per env slot (which would require handling
        # per-slot desync once episodes vary in length -- avoidable here
        # since neither env can have re-entered deck-select before either
        # one first terminates).
        num_envs = 2
        env = make("Ptcg-v0", env_type="gymnasium", num_envs=num_envs)
        obs, info = env.reset()
        self.assertEqual(obs["cards"].shape, (num_envs, 120, 5))
        self.assertEqual(obs["pokemons"].shape, (num_envs, 18, 15))
        self.assertEqual(obs["player_state"].shape, (num_envs, 2, 9))
        self.assertEqual(obs["state"].shape, (num_envs, 8))
        self.assertEqual(obs["select"].shape, (num_envs, 16))
        self.assertEqual(obs["options"].shape, (num_envs, 63, 19))
        np.testing.assert_array_equal(
            info["is_deck_select"], np.ones(num_envs, dtype=bool)
        )
        np.testing.assert_array_equal(
            info["current_player"], np.zeros(num_envs, dtype=np.int32)
        )

        deck_action = np.tile(_DECK, (num_envs, 1)).astype(np.int32)
        dummy_action = np.full((num_envs, _ACTION_SLOTS), -1, dtype=np.int32)

        # Both envs started together via the same reset() call and
        # deck-select is always exactly 2 real steps, so both are guaranteed
        # to still be deck-selecting in lockstep for exactly these 2 calls.
        obs, _reward, terminated, _truncated, info = env.step(deck_action)
        np.testing.assert_array_equal(
            info["is_deck_select"], np.ones(num_envs, dtype=bool)
        )
        np.testing.assert_array_equal(
            info["current_player"], np.ones(num_envs, dtype=np.int32)
        )

        obs, _reward, terminated, _truncated, info = env.step(deck_action)
        np.testing.assert_array_equal(
            info["is_deck_select"], np.zeros(num_envs, dtype=bool)
        )

        round_count = 0
        while not terminated.any():
            self.assertLess(
                round_count,
                _MAX_ROUNDS,
                "no env reached done within the safety bound",
            )
            round_count += 1
            obs, _reward, terminated, _truncated, info = env.step(
                dummy_action
            )
            self.assertEqual(obs["cards"].shape, (num_envs, 120, 5))

        self.assertTrue(terminated.any())


if __name__ == "__main__":
    absltest.main()
