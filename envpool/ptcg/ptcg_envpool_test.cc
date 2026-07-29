// Copyright 2021 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "envpool/ptcg/ptcg_envpool.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <vector>

using PtcgAction = typename ptcg::PtcgEnv::Action;
using PtcgState = typename ptcg::PtcgEnv::State;

namespace {

void ExpectAllZero(const Array& arr) {
  const auto* data = reinterpret_cast<const int*>(arr.Data());
  for (std::size_t i = 0; i < arr.size; ++i) {
    ASSERT_EQ(data[i], 0);
  }
}

// Phase 3: a real decide() step's obs should never be all-zero -- `cards`
// alone is enough to tell, since both seats' own 60 rows always carry real,
// non-sentinel card ids (see ptcg_encode.h's WriteCards). Not a full
// correctness check (that's ptcg_encode_test.cc's job) -- just a smoke
// signal that the encoder actually ran instead of silently falling through
// to the Zero()'d default.
void ExpectNotAllZero(const Array& arr) {
  const auto* data = reinterpret_cast<const int*>(arr.Data());
  for (std::size_t i = 0; i < arr.size; ++i) {
    if (data[i] != 0) {
      return;
    }
  }
  FAIL() << "expected at least one non-zero cell";
}

// Same real, deck-legal 60-card list as ptcg_smoke_test.cc and
// research/envpool_smoke/concurrency_smoke.cpp, copied from
// submissions/sample_submission/deck.csv. Both seats use it.
constexpr std::array<int, 60> kDeck = {
    1158, 721,  721,  722,  722,  722,  722,  723,  723,  723,  723,  1145,
    1145, 1145, 1145, 1205, 1205, 1227, 1227, 1227, 1227, 1235, 1235, 1235,
    1235, 3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
};

struct EnvTrack {
  bool is_deck_select = true;
  bool done = false;
  int episode_step = 0;  // resets to 0 whenever a fresh episode begins
  bool completed_once = false;
  std::vector<int> last_options;  // valid only when !is_deck_select && !done
};

// Copies an obs:options view into stable storage -- the Array it wraps is a
// slice of AsyncEnvPool's ring buffer, liable to be overwritten by the time
// the next round's Send() is built.
std::vector<int> CopyOptions(const TArray<int>& options) {
  std::vector<int> out(ptcg::kOptionRows * ptcg::kOptionsCols);
  for (int row = 0; row < ptcg::kOptionRows; ++row) {
    for (int col = 0; col < ptcg::kOptionsCols; ++col) {
      out[row * ptcg::kOptionsCols + col] = options[row][col];
    }
  }
  return out;
}

// Picks the first legal (is_valid && !already_selected) row -- real options
// sort before ptcg::kStopSlot, so this greedily keeps picking real options
// up to selectMax before ever reaching STOP. is_valid alone isn't enough: a
// real row stays is_valid=1 forever once picked (structural, unaffected by
// already_selected -- see schema.py's OPTIONS_COLUMNS comment), so without
// the already_selected check this would pick the same already-chosen row
// again on a decision's second+ sub-pick, which LegalActions() correctly
// rejects as illegal. Every genuine decide()-step observation is guaranteed
// to offer at least one real legal row (Python only ever sees >=2-way
// decisions -- see ptcg_envpool.h's AutoResolveForcedSubPicksThenRespond --
// and STOP alone can never account for more than one of those two-plus
// legal slots), so this always finds a real pick, never STOP, keeping every
// game's action sequence trivially legal by construction -- exactly the
// same role the old -1-filled dummy action played back when under-minCount
// padding made any action legal.
int PickLegalAction(const std::vector<int>& options_flat) {
  constexpr int kIsValidCol = 0;          // OPTIONS_COLUMNS[0], see schema.py
  constexpr int kAlreadySelectedCol = 19;  // OPTIONS_COLUMNS[-1]
  for (int row = 0; row < ptcg::kOptionRows; ++row) {
    if (options_flat[row * ptcg::kOptionsCols + kIsValidCol] == 1 &&
        options_flat[row * ptcg::kOptionsCols + kAlreadySelectedCol] == 0) {
      return row;
    }
  }
  ADD_FAILURE() << "no legal action found in observed options -- envpool invariant violated";
  return ptcg::kStopSlot;
}

}  // namespace

// Phases 2-4 test (envpool-ptcg-integration skill): drives real
// ApiBattleStart/ApiSelect calls, not Phase 1's fake fixed-length episode.
// Real games are nondeterministic in length and first-player assignment (see
// "Resets are nondeterministic" in the skill -- ApiBattleStart has no seed
// parameter), so unlike Phase 1's exact-sequence table, this asserts
// structural invariants across full real episodes: deck-select is always
// exactly the first 2 steps of an episode with current_player 0 then 1;
// decide() steps keep current_player in {0,1}; obs tensors are all-zero on
// deck-select/terminal steps and non-trivial (real encoder output, Phase 3)
// on decide() steps; reward is 0.0 except on the step done becomes true,
// where it's the real apiResult()-derived terminal reward (Phase 4) --
// every action sent here is always legal (a real deck, or a real option
// row read off the just-observed obs:options and picked by
// PickLegalAction(), see its comment), so this test's games always end via
// genuine win/loss/draw, never the illegal-action penalty path (see
// ptcg_envpool_illegal_test.cc for that); every env eventually reaches
// done. Same "diff structurally, not bit-exact" adjustment Phase 0 already
// made for the same underlying reason.
//
// num_envs == batch_size == 3 (sync mode): real games run dozens+ of
// decide() calls each, which alone cycles every StateBuffer ring-buffer slot
// many times over within a single episode -- still a real regression check
// on Array::Zero(), same intent as Phase 1's version of this test.
//
// This also exercises AsyncEnvPool's auto-reset: `reset = force_reset ||
// envs_[env_id]->IsDone()` in the worker loop (async_envpool.h:127) means
// Send()-ing *any* action for an env_id that just reported done triggers
// Reset() instead of Step(), regardless of what was sent -- confirmed here
// by continuing to drive every env (including ones that already completed
// once) until the slowest env finishes its first episode too.
TEST(PtcgEnvPoolTest, RealEngineEndToEnd) {
  auto config = ptcg::PtcgEnvSpec::kDefaultConfig;
  const int num_envs = 3;
  config["num_envs"_] = num_envs;
  config["batch_size"_] = num_envs;
  config["num_threads"_] = 1;
  ptcg::PtcgEnvSpec spec(config);
  ptcg::PtcgEnvPool envpool(spec);

  Array all_env_ids(Spec<int>({num_envs}));
  for (int i = 0; i < num_envs; ++i) {
    all_env_ids[i] = i;
  }
  envpool.Reset(all_env_ids);

  std::vector<EnvTrack> track(num_envs);

  auto recv_and_check = [&]() {
    PtcgState state(envpool.Recv());
    for (int i = 0; i < num_envs; ++i) {
      int env_id = static_cast<int>(state["info:env_id"_][i]);
      EnvTrack& t = track[env_id];

      bool is_deck_select =
          static_cast<bool>(state["info:is_deck_select"_][i]);
      int current_player = static_cast<int>(state["info:current_player"_][i]);
      bool done = static_cast<bool>(state["done"_][i]);
      float reward = static_cast<float>(state["reward"_][i]);
      int finish_reason = static_cast<int>(state["info:finish_reason"_][i]);

      if (is_deck_select || done) {
        ExpectAllZero(state["obs:cards"_][i]);
        ExpectAllZero(state["obs:pokemons"_][i]);
        ExpectAllZero(state["obs:player_state"_][i]);
        ExpectAllZero(state["obs:state"_][i]);
        ExpectAllZero(state["obs:select"_][i]);
        ExpectAllZero(state["obs:options"_][i]);
      } else {
        ExpectNotAllZero(state["obs:cards"_][i]);
        t.last_options = CopyOptions(TArray<int>(state["obs:options"_][i]));
      }
      if (done) {
        EXPECT_TRUE(reward == 1.0F || reward == 0.0F || reward == -1.0F)
            << "env_id=" << env_id << " reward=" << reward;
        // Every game in this test ends via a real engine win/loss/draw,
        // never the illegal-action penalty path (see the doc comment
        // above) -- so finish_reason should always be a genuine non-None
        // FinishReason (State.h: Prize0=1, Deck0=2, NoActivePokemon=3,
        // Effect=4, Other=9), never left at its 0 default.
        EXPECT_TRUE(finish_reason == 1 || finish_reason == 2 ||
                    finish_reason == 3 || finish_reason == 4 ||
                    finish_reason == 9)
            << "env_id=" << env_id << " finish_reason=" << finish_reason;
      } else {
        EXPECT_FLOAT_EQ(reward, 0.0F) << "env_id=" << env_id;
        EXPECT_EQ(finish_reason, 0) << "env_id=" << env_id;
      }

      EXPECT_EQ(is_deck_select, t.episode_step < 2)
          << "env_id=" << env_id << " episode_step=" << t.episode_step;
      if (t.episode_step == 0) {
        EXPECT_EQ(current_player, 0) << "env_id=" << env_id;
      } else if (t.episode_step == 1) {
        EXPECT_EQ(current_player, 1) << "env_id=" << env_id;
      } else {
        EXPECT_GE(current_player, 0) << "env_id=" << env_id;
        EXPECT_LE(current_player, 1) << "env_id=" << env_id;
      }

      t.is_deck_select = is_deck_select;
      t.done = done;
      ++t.episode_step;
      if (done) {
        t.completed_once = true;
      }
    }
  };

  recv_and_check();  // Reset()'s output: player-0 deck-select, every env.

  const int kMaxRounds = 20000;
  int round = 0;
  auto all_completed = [&] {
    return std::all_of(track.begin(), track.end(),
                        [](const EnvTrack& t) { return t.completed_once; });
  };
  while (!all_completed()) {
    ASSERT_LT(round, kMaxRounds)
        << "not every env reached done within the safety bound";
    ++round;

    std::vector<Array> raw_action(
        {Array(Spec<int>({num_envs})), Array(Spec<int>({num_envs})),
         Array(Spec<int>({num_envs, ptcg::kActionSlots}))});
    PtcgAction action(raw_action);
    for (int i = 0; i < num_envs; ++i) {
      action["env_id"_][i] = i;
      action["players.env_id"_][i] = i;
      EnvTrack& t = track[i];
      // A `done` env's next Send() triggers AsyncEnvPool's internal
      // auto-reset (see this test's top comment) -- content is irrelevant
      // for that slot this round, it starts a fresh episode's deck-select
      // regardless of what's sent.
      bool send_real_deck = !t.done && t.is_deck_select;
      int first_slot = -1;
      if (send_real_deck) {
        first_slot = kDeck[0];
      } else if (!t.done) {
        first_slot = PickLegalAction(t.last_options);
      }
      action["action"_][i][0] = first_slot;
      for (int j = 1; j < ptcg::kActionSlots; ++j) {
        action["action"_][i][j] = send_real_deck ? kDeck[j] : -1;
      }
      if (t.done) {
        t.episode_step = 0;  // about to start a fresh episode
      }
    }
    envpool.Send(action);
    recv_and_check();
  }

  for (const EnvTrack& t : track) {
    EXPECT_TRUE(t.completed_once);
  }
}
