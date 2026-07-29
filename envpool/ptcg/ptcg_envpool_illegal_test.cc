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

#include <array>
#include <vector>

using PtcgAction = typename ptcg::PtcgEnv::Action;
using PtcgState = typename ptcg::PtcgEnv::State;

// Phase 4 test (envpool-ptcg-integration skill): the two illegal-action
// penalty paths ("Action space" / "Reward" in the skill) that
// ptcg_envpool_test.cc's happy-path test never exercises, since every
// action it sends there is legal by construction (a real deck, or a real
// option row read off the just-observed obs:options). Both single-env,
// single-scripted-scenario tests (num_envs=1) -- sequencing/ring-buffer
// coverage across many envs and real games is ptcg_envpool_test.cc's job,
// this file only needs to prove the penalty wiring itself.
namespace {

// Same real, deck-legal 60-card list as ptcg_envpool_test.cc /
// ptcg_encode_test.cc / ptcg_smoke_test.cc, copied from
// submissions/sample_submission/deck.csv.
constexpr std::array<int, 60> kDeck = {
    1158, 721,  721,  722,  722,  722,  722,  723,  723,  723,  723,  1145,
    1145, 1145, 1145, 1205, 1205, 1227, 1227, 1227, 1227, 1235, 1235, 1235,
    1235, 3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
};

void ExpectAllZero(const Array& arr) {
  const auto* data = reinterpret_cast<const int*>(arr.Data());
  for (std::size_t i = 0; i < arr.size; ++i) {
    EXPECT_EQ(data[i], 0);
  }
}

// Single-env send/recv helper shared by both tests below.
PtcgState Send(ptcg::PtcgEnvPool& envpool,  // NOLINT
                const std::array<int, ptcg::kActionSlots>& slots) {
  std::vector<Array> raw_action(
      {Array(Spec<int>({1})), Array(Spec<int>({1})),
       Array(Spec<int>({1, ptcg::kActionSlots}))});
  PtcgAction action(raw_action);
  action["env_id"_][0] = 0;
  action["players.env_id"_][0] = 0;
  for (int j = 0; j < ptcg::kActionSlots; ++j) {
    action["action"_][0][j] = slots[j];
  }
  envpool.Send(action);
  return PtcgState(envpool.Recv());
}

}  // namespace

TEST(PtcgEnvPoolIllegalTest, IllegalDeckPenalizesOffendingSeat) {
  auto config = ptcg::PtcgEnvSpec::kDefaultConfig;
  config["num_envs"_] = 1;
  config["batch_size"_] = 1;
  config["num_threads"_] = 1;
  ptcg::PtcgEnvSpec spec(config);
  ptcg::PtcgEnvPool envpool(spec);

  Array env_ids(Spec<int>({1}));
  env_ids[0] = 0;
  envpool.Reset(env_ids);
  PtcgState state(envpool.Recv());  // seat-0 deck-select prompt
  EXPECT_TRUE(static_cast<bool>(state["info:is_deck_select"_][0]));
  EXPECT_EQ(static_cast<int>(state["info:current_player"_][0]), 0);

  state = Send(envpool, kDeck);  // seat 0's real, legal deck
  EXPECT_TRUE(static_cast<bool>(state["info:is_deck_select"_][0]));
  EXPECT_EQ(static_cast<int>(state["info:current_player"_][0]), 1);

  // Seat 1's deck: an id that doesn't exist in CardTable at all --
  // ApiBattleStart's very first per-card check (Api.h: `if
  // (!CardTable.contains(id)) return {nullptr, i, 1}`), deliberately
  // illegal, no dependency on any specific card's type/count rules.
  std::array<int, ptcg::kActionSlots> bad_deck{};
  bad_deck.fill(999999);
  state = Send(envpool, bad_deck);

  EXPECT_TRUE(static_cast<bool>(state["done"_][0]));
  EXPECT_FLOAT_EQ(static_cast<float>(state["reward"_][0]), -1.0F);
  EXPECT_EQ(static_cast<int>(state["info:current_player"_][0]), 1)
      << "errorPlayer should name seat 1: both decks are only validated "
         "once ApiBattleStart actually runs, which happens on seat 1's "
         "Step() (see StepDeckSelect's comment on this)";
  EXPECT_EQ(static_cast<int>(state["info:finish_reason"_][0]), 0)
      << "an illegal deck never reaches the engine's own finishCheck()";
  ExpectAllZero(state["obs:cards"_][0]);
  ExpectAllZero(state["obs:pokemons"_][0]);
  ExpectAllZero(state["obs:player_state"_][0]);
  ExpectAllZero(state["obs:state"_][0]);
  ExpectAllZero(state["obs:select"_][0]);
  ExpectAllZero(state["obs:options"_][0]);
}

TEST(PtcgEnvPoolIllegalTest, IllegalDecideActionPenalizesActor) {
  auto config = ptcg::PtcgEnvSpec::kDefaultConfig;
  config["num_envs"_] = 1;
  config["batch_size"_] = 1;
  config["num_threads"_] = 1;
  ptcg::PtcgEnvSpec spec(config);
  ptcg::PtcgEnvPool envpool(spec);

  Array env_ids(Spec<int>({1}));
  env_ids[0] = 0;
  envpool.Reset(env_ids);
  envpool.Recv();  // Reset()'s own response (seat-0 deck-select prompt) --
                    // must be drained before the Send()/Recv() pairs below,
                    // or every round after it reads one round stale.

  Send(envpool, kDeck);                    // seat 0's deck
  PtcgState state = Send(envpool, kDeck);  // seat 1's deck -> ApiBattleStart
  // Phase 2.5's bypass loop guarantees whatever's pending now is a genuine,
  // non-bypassed decision -- safe to answer illegally on purpose.
  EXPECT_FALSE(static_cast<bool>(state["info:is_deck_select"_][0]));
  int actor_before = static_cast<int>(state["info:current_player"_][0]);

  std::array<int, ptcg::kActionSlots> bad_action{};
  bad_action.fill(-1);
  // Safely within ActionSpec's declared bound ({-1, N_CARD_IDS-1}) but
  // essentially certain to exceed the real state.options.size() for any
  // early-game decision -- an out-of-range pick, per "Action space" in the
  // skill: "any non--1 entry ... pointing at an options row not marked
  // valid".
  bad_action[0] = 500;
  state = Send(envpool, bad_action);

  EXPECT_TRUE(static_cast<bool>(state["done"_][0]));
  EXPECT_FLOAT_EQ(static_cast<float>(state["reward"_][0]), -1.0F);
  EXPECT_EQ(static_cast<int>(state["info:current_player"_][0]), actor_before)
      << "no SyncFromEngine happens on the illegal path -- ApiSelect "
         "returns before calling data->next() on any error (Api.h), so "
         "current_player_ stays exactly what it was announced as";
  EXPECT_EQ(static_cast<int>(state["info:finish_reason"_][0]), 0)
      << "ApiSelect returned before advancing state, so the engine's own "
         "finishCheck() never ran on this action";
  ExpectAllZero(state["obs:cards"_][0]);
  ExpectAllZero(state["obs:pokemons"_][0]);
  ExpectAllZero(state["obs:player_state"_][0]);
  ExpectAllZero(state["obs:state"_][0]);
  ExpectAllZero(state["obs:select"_][0]);
  ExpectAllZero(state["obs:options"_][0]);
}
