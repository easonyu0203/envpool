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
C++-ported observation encoder, and full action-space validation/reward),
plus the later config-deck redesign.

See the envpool-ptcg-integration skill's "Phased implementation checklist":
Phase 2's real Reset()/Step() flow into ApiBattleStart, real ApiSelect
stepping until state.isFinish(); Phase 2.5's forced-full-select/prize-select
auto-resolution, invisible from here except that it means every observation
Python actually sees is a genuine decision; Phase 3's real per-field encoder
output on decide() steps (ptcg_encode.h, tested for structural
self-consistency in ptcg_encode_test.cc -- this file doesn't re-derive that,
just checks the obs stops being all-zero); Phase 4's real action-space
validation and the resulting terminal reward. The config-deck redesign moved
deck selection out of the trajectory entirely: `deck0s`/`deck1s` (one deck
pair per env slot) are fixed EnvPool-lifetime config (see ptcg_envpool.h's
DefaultConfig) -- Reset()'s own response is already a genuine decide()-type
observation, and the action space is a single scalar index per env, not a
60-wide vector.

Real games are nondeterministic in length and first-player assignment (no
seed parameter on ApiBattleStart -- see "Resets are nondeterministic" in the
skill), so unlike Phase 1's fixed 6-step fake episode, these tests assert
structural invariants across full real episodes rather than an exact
sequence table.

`deck0s`/`deck1s` (see ptcg_envpool.h's DefaultConfig) are one 60-card deck
per env slot, flattened -- length num_envs*60 each. Every test here gives
every slot the same pair (`_DECK * num_envs`), since none of them are
testing per-slot pairing itself (that's the training pipeline's concern, not
this env's) -- just that the config plumbs through to the right env_id.
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


_OPT_IS_VALID_COL = 0  # OPTIONS_COLUMNS[0], see schema.py
_OPT_ALREADY_SELECTED_COL = 19  # OPTIONS_COLUMNS[-1], see schema.py


def _pick_legal_action(options_row: np.ndarray) -> int:
    """options_row: (64, 20) -- one env's raw obs:options. Picks the first
    legal (is_valid && !already_selected) row; real options sort before the
    STOP row (index 63), so this greedily keeps picking real options up to
    selectMax before ever reaching STOP. is_valid alone isn't enough: a real
    row stays is_valid=1 forever once picked (structural, unaffected by
    already_selected -- see schema.py's OPTIONS_COLUMNS comment), so without
    the already_selected check this would pick the same already-chosen row
    again on a decision's second+ sub-pick, which the env's LegalActions()
    correctly rejects as illegal. Every genuine decide()-step observation is
    guaranteed to offer at least one real legal row (Python only ever sees
    >=2-way decisions -- see ptcg_envpool.h's
    AutoResolveForcedSubPicksThenRespond -- and STOP alone can never
    account for more than one of those two-plus legal slots), so this
    always finds a real pick, never STOP, keeping every game's action
    sequence trivially legal by construction."""
    legal = np.flatnonzero(
        (options_row[:, _OPT_IS_VALID_COL] == 1) & (options_row[:, _OPT_ALREADY_SELECTED_COL] == 0)
    )
    assert legal.size > 0, "no legal action found -- envpool invariant violated"
    return int(legal[0])


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
        # PtcgEnvFns::DefaultConfig() adds deck0s/deck1s (one deck per env
        # slot, the config-deck redesign, see ptcg_envpool.h) on top of
        # envpool's common config.
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
            "deck0s",
            "deck1s",
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
        self.assertEqual(tuple(state_spec["obs:options"][1]), (64, 20))
        self.assertEqual(tuple(state_spec["info:current_player"][1]), ())
        self.assertEqual(tuple(state_spec["info:finish_reason"][1]), ())
        # Scalar now -- decks are config, not part of the action anymore
        # (see ptcg_envpool.h's ActionSpec).
        self.assertEqual(tuple(action_spec["action"][1]), ())

    def test_raw_envpool_real_engine_end_to_end(self) -> None:
        # Small num_envs == batch_size so the sync path preserves order, and
        # real games (dozens+ decisions each) force StateBuffer ring-buffer
        # slot reuse many times over -- same regression intent as Phase 1's
        # version of this test: the 6 obs tensors are zero only because
        # PtcgEnv calls Array::Zero() on every write, not because Allocate()
        # zero-inits on every call (see WriteState's comment in
        # ptcg_envpool.h). Since Phase 3, that's only true for the terminal
        # step -- a real decide() step's obs is real encoder output instead
        # (checked for structural self-consistency in ptcg_encode_test.cc,
        # not re-derived here).
        #
        # Also exercises AsyncEnvPool's auto-reset: Send()-ing any action
        # for an env_id that just reported done triggers Reset() instead of
        # Step() (async_envpool.h:127) -- this loop keeps driving every env,
        # including ones that already finished once, until the slowest one
        # finishes its first episode too.
        #
        # Every action sent here is always legal (a real option row read
        # off the just-observed obs:options and picked by
        # _pick_legal_action(), see its docstring), so these games always
        # end via genuine win/loss/draw, never the illegal-action penalty
        # path -- see test_illegal_decide_action_penalizes_actor below for
        # that.
        num_envs = 3
        conf = _default_conf(
            num_envs=num_envs, batch_size=num_envs, num_threads=1,
            deck0s=_DECK * num_envs, deck1s=_DECK * num_envs,
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
                "done": False,
                "completed_once": False,
                "last_options": None,
            } for _ in range(num_envs)
        ]

        def process(state: dict) -> None:
            env_ids = state["info:env_id"]
            for slot in range(num_envs):
                env_id = int(env_ids[slot])
                t = track[env_id]
                current_player = int(state["info:current_player"][slot])
                done = bool(state["done"][slot])
                reward = float(state["reward"][slot])
                finish_reason = int(state["info:finish_reason"][slot])

                if done:
                    assert_zero_obs(state, slot)
                else:
                    assert_not_all_zero_cards(state, slot)
                    t["last_options"] = np.array(state["obs:options"][slot])
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

                self.assertIn(current_player, (0, 1), f"env_id={env_id}")

                t["done"] = done
                if done:
                    t["completed_once"] = True

        env._reset(np.arange(num_envs, dtype=np.int32))
        process(recv_dict())  # Reset()'s output: every env's first decide().

        round_count = 0
        while not all(t["completed_once"] for t in track):
            self.assertLess(
                round_count,
                _MAX_ROUNDS,
                "not every env reached done within the safety bound",
            )
            round_count += 1

            action = np.zeros((num_envs,), dtype=np.int32)
            for env_id in range(num_envs):
                t = track[env_id]
                # A `done` env's next send triggers an internal auto-reset
                # regardless of content -- see this test's docstring-comment
                # above.
                if not t["done"]:
                    action[env_id] = _pick_legal_action(t["last_options"])

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
        # episode. Exhaustive structural invariants (current_player bounds,
        # all-zero terminal obs, auto-reset across episodes) are already
        # covered thoroughly by test_raw_envpool_real_engine_end_to_end
        # above -- this only needs to prove the make()/Gymnasium-wrapper
        # path itself carries a real engine correctly, so it stops at the
        # first termination rather than re-proving everything per env slot.
        num_envs = 2
        env = make(
            "Ptcg-v0", env_type="gymnasium", num_envs=num_envs,
            deck0s=_DECK * num_envs, deck1s=_DECK * num_envs,
        )
        obs, info = env.reset()
        self.assertEqual(obs["cards"].shape, (num_envs, 120, 5))
        self.assertEqual(obs["pokemons"].shape, (num_envs, 18, 18))
        self.assertEqual(obs["player_state"].shape, (num_envs, 2, 9))
        self.assertEqual(obs["state"].shape, (num_envs, 8))
        self.assertEqual(obs["select"].shape, (num_envs, 18))
        self.assertEqual(obs["options"].shape, (num_envs, 64, 20))
        # Reset()'s own response is already a genuine decide()-type
        # observation now (decks are config, see ptcg_envpool.h's
        # DefaultConfig) -- current_player is whatever ApiBattleStart/the
        # bypass loop settled on, not a fixed seat-0 deck-select prompt.
        self.assertTrue(bool(np.any(obs["cards"] != 0)))

        def legal_action(obs: dict) -> np.ndarray:
            action = np.zeros((num_envs,), dtype=np.int32)
            for slot in range(num_envs):
                action[slot] = _pick_legal_action(obs["options"][slot])
            return action

        round_count = 0
        terminated = np.zeros(num_envs, dtype=bool)
        while not terminated.any():
            self.assertLess(
                round_count,
                _MAX_ROUNDS,
                "no env reached done within the safety bound",
            )
            round_count += 1
            obs, reward, terminated, _truncated, info = env.step(
                legal_action(obs)
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

    def test_illegal_decide_action_penalizes_actor(self) -> None:
        # Phase 4 test, mirroring ptcg_envpool_illegal_test.cc's C++ version.
        # The old sibling "illegal deck" test is gone: decks are fixed
        # EnvPool-lifetime config now (see ptcg_envpool.h's DefaultConfig),
        # not a per-episode action a policy submits, so that scenario is
        # structurally unreachable (an invalid configured deck now
        # CHECK-fails Reset(), a construction/config bug).
        conf = _default_conf(
            num_envs=1, batch_size=1, num_threads=1, deck0s=_DECK, deck1s=_DECK,
        )
        env_spec = _PtcgEnvSpec(tuple(conf.values()))
        env = _PtcgEnvPool(env_spec)
        state_keys = env._state_keys

        def recv_dict() -> dict:
            return dict(zip(state_keys, env._recv(), strict=False))

        def send(action_value: int) -> dict:
            env._send((
                np.array([0], dtype=np.int32),
                np.array([0], dtype=np.int32),
                np.array([action_value], dtype=np.int32),
            ))
            return recv_dict()

        env._reset(np.array([0], dtype=np.int32))
        state = recv_dict()
        # Reset()'s own response is already a genuine, non-bypassed
        # decide() prompt -- safe to answer illegally on purpose right away.
        actor_before = int(state["info:current_player"][0])

        # Within ActionSpec's declared bound ({0, kMaxOptions}) -- a real
        # option-row index, just not one that exists yet -- but essentially
        # certain to exceed the real options count for any early-game
        # decision, per "Action space" in the skill.
        state = send(62)  # kMaxOptions - 1

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
