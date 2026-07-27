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
"""Unit test for the Phases 2-4 ptcg envpool (real ptcg_engine calls, the
C++-ported observation encoder, and full action-space validation/reward).

See the envpool-ptcg-integration skill's "Phased implementation checklist":
Phase 2's real Reset()/Step() flow (2-step deck-select handshake into
ApiBattleStart, real ApiSelect stepping until state.isFinish()); Phase 2.5's
forced-full-select/prize-select auto-resolution, invisible from here except
that it means every observation Python actually sees is a genuine decision;
Phase 3's real per-field encoder output on decide() steps (ptcg_encode.h,
tested for structural self-consistency in ptcg_encode_test.cc -- this file
doesn't re-derive that, just checks the obs stops being all-zero); Phase 4's
real action-space validation (illegal-vs-under-minCount for decide(),
deck-legality for deck-select) and the resulting terminal reward.

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

    def test_registry_coverage(self) -> None:
        # Phase 6 (envpool-ptcg-integration skill): AGENTS.md's "Registry
        # Coverage" test tier -- "Ptcg-v0" is registered (through our own
        # scoped envpool.ptcg.registration import at module level above,
        # not envpool/entry.py's aggregate -- see "Build path" in the
        # skill) with the right spec/dm/gymnasium classes. No aliases:
        # this family has exactly one task id, nothing to enumerate beyond
        # it (unlike an upstream-backed family with many scenario IDs).
        from envpool.registration import registry

        self.assertIn("Ptcg-v0", registry.list_all_envs())
        import_path, spec_cls, _kwargs = registry.specs["Ptcg-v0"]
        self.assertEqual(import_path, "envpool.ptcg")
        self.assertEqual(spec_cls, "PtcgEnvSpec")
        envpool_entry = registry.envpools["Ptcg-v0"]
        self.assertEqual(envpool_entry["dm"], ("envpool.ptcg", "PtcgDMEnvPool"))
        self.assertEqual(
            envpool_entry["gymnasium"], ("envpool.ptcg", "PtcgGymnasiumEnvPool")
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
        self.assertEqual(tuple(state_spec["obs:pokemons"][1]), (18, 18))
        self.assertEqual(tuple(state_spec["obs:player_state"][1]), (2, 9))
        self.assertEqual(tuple(state_spec["obs:state"][1]), (8,))
        self.assertEqual(tuple(state_spec["obs:select"][1]), (18,))
        self.assertEqual(tuple(state_spec["obs:options"][1]), (63, 19))
        self.assertEqual(tuple(state_spec["info:current_player"][1]), ())
        self.assertEqual(tuple(state_spec["info:is_deck_select"][1]), ())
        self.assertEqual(tuple(state_spec["info:finish_reason"][1]), ())
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
        # ptcg_envpool.h). Since Phase 3, that's only true for deck-select
        # and terminal steps -- a real decide() step's obs is real encoder
        # output instead (checked for structural self-consistency in
        # ptcg_encode_test.cc, not re-derived here).
        #
        # Also exercises AsyncEnvPool's auto-reset: Send()-ing any action
        # for an env_id that just reported done triggers Reset() instead of
        # Step() (async_envpool.h:127) -- this loop keeps driving every env,
        # including ones that already finished once, until the slowest one
        # finishes its first episode too.
        #
        # Every action sent here is always legal (a real deck, or a
        # -1-filled dummy that Phase 4's under-minCount padding turns into a
        # legal pick), so these games always end via genuine win/loss/draw,
        # never the illegal-action penalty path -- see
        # test_illegal_deck_penalizes_offending_seat /
        # test_illegal_decide_action_penalizes_actor below for that.
        num_envs = 3
        conf = _default_conf(
            num_envs=num_envs, batch_size=num_envs, num_threads=1
        )
        env_spec = _PtcgEnvSpec(tuple(conf.values()))
        env = _PtcgEnvPool(env_spec)
        state_keys = env._state_keys

        def recv_dict() -> dict:
            return dict(zip(state_keys, env._recv(), strict=False))

        def assert_zero_obs(state: dict, slot: int) -> None:
            for key in (
                "obs:cards",
                "obs:pokemons",
                "obs:player_state",
                "obs:state",
                "obs:select",
                "obs:options",
            ):
                np.testing.assert_array_equal(
                    state[key][slot], np.zeros_like(state[key][slot])
                )

        def assert_not_all_zero_cards(state: dict, slot: int) -> None:
            # `cards` alone is enough to tell the encoder ran: both seats'
            # own 60 rows always carry real, non-sentinel card ids (see
            # ptcg_encode.h's WriteCards) -- a smoke signal, not the full
            # correctness check (that's ptcg_encode_test.cc's job).
            self.assertTrue(bool(np.any(state["obs:cards"][slot] != 0)))

        track = [
            {
                "is_deck_select": True,
                "done": False,
                "episode_step": 0,
                "completed_once": False,
            } for _ in range(num_envs)
        ]

        def process(state: dict) -> None:
            env_ids = state["info:env_id"]
            for slot in range(num_envs):
                env_id = int(env_ids[slot])
                t = track[env_id]
                is_deck_select = bool(state["info:is_deck_select"][slot])
                current_player = int(state["info:current_player"][slot])
                done = bool(state["done"][slot])
                reward = float(state["reward"][slot])
                finish_reason = int(state["info:finish_reason"][slot])

                if is_deck_select or done:
                    assert_zero_obs(state, slot)
                else:
                    assert_not_all_zero_cards(state, slot)
                if done:
                    self.assertIn(reward, (1.0, 0.0, -1.0), f"env_id={env_id}")
                    # Every game in this test ends via a real engine
                    # win/loss/draw, never the illegal-action penalty path
                    # (see this test's docstring) -- so finish_reason should
                    # always be a genuine non-None FinishReason (State.h:
                    # Prize0=1, Deck0=2, NoActivePokemon=3, Effect=4,
                    # Other=9), never left at its 0 default.
                    self.assertIn(
                        finish_reason, (1, 2, 3, 4, 9), f"env_id={env_id}"
                    )
                else:
                    self.assertEqual(reward, 0.0, f"env_id={env_id}")
                    self.assertEqual(finish_reason, 0, f"env_id={env_id}")

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
        self.assertEqual(obs["pokemons"].shape, (num_envs, 18, 18))
        self.assertEqual(obs["player_state"].shape, (num_envs, 2, 9))
        self.assertEqual(obs["state"].shape, (num_envs, 8))
        self.assertEqual(obs["select"].shape, (num_envs, 18))
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

        obs, reward, terminated, _truncated, info = env.step(deck_action)
        np.testing.assert_array_equal(
            info["is_deck_select"], np.zeros(num_envs, dtype=bool)
        )
        # Phase 3: a real decide() step's obs is real encoder output, not
        # all-zero (see ptcg_encode_test.cc for the actual correctness
        # check). Phase 4: reward is 0.0 on every non-terminal step.
        self.assertTrue(bool(np.any(obs["cards"] != 0)))
        np.testing.assert_array_equal(reward, np.zeros(num_envs, dtype=np.float32))

        round_count = 0
        while not terminated.any():
            self.assertLess(
                round_count,
                _MAX_ROUNDS,
                "no env reached done within the safety bound",
            )
            round_count += 1
            obs, reward, terminated, _truncated, info = env.step(
                dummy_action
            )
            self.assertEqual(obs["cards"].shape, (num_envs, 120, 5))
            for slot in range(num_envs):
                if terminated[slot]:
                    self.assertIn(float(reward[slot]), (1.0, 0.0, -1.0))
                    self.assertIn(int(info["finish_reason"][slot]), (1, 2, 3, 4, 9))
                else:
                    self.assertEqual(float(reward[slot]), 0.0)
                    self.assertEqual(int(info["finish_reason"][slot]), 0)

        self.assertTrue(terminated.any())

    def test_illegal_deck_penalizes_offending_seat(self) -> None:
        # Phase 4 test, mirroring ptcg_envpool_illegal_test.cc's C++ version
        # of the same scenario -- see that file's top comment for why
        # single-env/single-scenario tests are the right shape here (the
        # happy-path test above never exercises this path, since every
        # action it sends is legal by construction).
        conf = _default_conf(num_envs=1, batch_size=1, num_threads=1)
        env_spec = _PtcgEnvSpec(tuple(conf.values()))
        env = _PtcgEnvPool(env_spec)
        state_keys = env._state_keys

        def recv_dict() -> dict:
            return dict(zip(state_keys, env._recv(), strict=False))

        def send(slots) -> dict:
            env._send((
                np.array([0], dtype=np.int32),
                np.array([0], dtype=np.int32),
                np.array([slots], dtype=np.int32),
            ))
            return recv_dict()

        env._reset(np.array([0], dtype=np.int32))
        state = recv_dict()
        self.assertTrue(bool(state["info:is_deck_select"][0]))
        self.assertEqual(int(state["info:current_player"][0]), 0)

        state = send(_DECK)  # seat 0's real, legal deck
        self.assertTrue(bool(state["info:is_deck_select"][0]))
        self.assertEqual(int(state["info:current_player"][0]), 1)

        # Seat 1's deck: an id that doesn't exist in CardTable at all --
        # ApiBattleStart's very first per-card check (Api.h: `if
        # (!CardTable.contains(id)) return {nullptr, i, 1}`), deliberately
        # illegal, no dependency on any specific card's type/count rules.
        state = send([999999] * _ACTION_SLOTS)

        self.assertTrue(bool(state["done"][0]))
        self.assertEqual(float(state["reward"][0]), -1.0)
        self.assertEqual(
            int(state["info:current_player"][0]), 1,
            "errorPlayer should name seat 1: both decks are only validated "
            "once ApiBattleStart actually runs, which happens on seat 1's "
            "Step()",
        )
        self.assertEqual(
            int(state["info:finish_reason"][0]), 0,
            "an illegal deck never reaches the engine's own finishCheck()",
        )
        for key in (
            "obs:cards", "obs:pokemons", "obs:player_state",
            "obs:state", "obs:select", "obs:options",
        ):
            np.testing.assert_array_equal(
                state[key][0], np.zeros_like(state[key][0])
            )

    def test_illegal_decide_action_penalizes_actor(self) -> None:
        # Phase 4 test, mirroring ptcg_envpool_illegal_test.cc's C++ version.
        conf = _default_conf(num_envs=1, batch_size=1, num_threads=1)
        env_spec = _PtcgEnvSpec(tuple(conf.values()))
        env = _PtcgEnvPool(env_spec)
        state_keys = env._state_keys

        def recv_dict() -> dict:
            return dict(zip(state_keys, env._recv(), strict=False))

        def send(slots) -> dict:
            env._send((
                np.array([0], dtype=np.int32),
                np.array([0], dtype=np.int32),
                np.array([slots], dtype=np.int32),
            ))
            return recv_dict()

        env._reset(np.array([0], dtype=np.int32))
        recv_dict()  # Reset()'s own response -- must be drained before the
                     # send()/recv() pairs below, or every round after it
                     # reads one round stale (see the identical gotcha this
                     # hit in ptcg_envpool_illegal_test.cc).

        send(_DECK)              # seat 0's deck
        state = send(_DECK)      # seat 1's deck -> ApiBattleStart
        # Phase 2.5's bypass loop guarantees whatever's pending now is a
        # genuine, non-bypassed decision -- safe to answer illegally on
        # purpose.
        self.assertFalse(bool(state["info:is_deck_select"][0]))
        actor_before = int(state["info:current_player"][0])

        bad_action = [-1] * _ACTION_SLOTS
        # Safely within ActionSpec's declared bound ({-1, N_CARD_IDS-1}) but
        # essentially certain to exceed the real options count for any
        # early-game decision -- an out-of-range pick, per "Action space" in
        # the skill.
        bad_action[0] = 500
        state = send(bad_action)

        self.assertTrue(bool(state["done"][0]))
        self.assertEqual(float(state["reward"][0]), -1.0)
        self.assertEqual(
            int(state["info:current_player"][0]), actor_before,
            "no SyncFromEngine happens on the illegal path -- "
            "current_player_ stays exactly what it was announced as",
        )
        self.assertEqual(
            int(state["info:finish_reason"][0]), 0,
            "ApiSelect returned before advancing state, so the engine's "
            "own finishCheck() never ran on this action",
        )
        for key in (
            "obs:cards", "obs:pokemons", "obs:player_state",
            "obs:state", "obs:select", "obs:options",
        ):
            np.testing.assert_array_equal(
                state[key][0], np.zeros_like(state[key][0])
            )


if __name__ == "__main__":
    absltest.main()
