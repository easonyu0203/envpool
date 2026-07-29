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

// Phase 4 test (envpool-ptcg-integration skill): the illegal-decide-action
// penalty path ("Action space" / "Reward" in the skill) that
// ptcg_envpool_test.cc's happy-path test never exercises, since every
// action it sends there is legal by construction (a real option row read
// off the just-observed obs:options). Single-env, single-scripted-scenario
// test (num_envs=1) -- sequencing/ring-buffer coverage across many envs and
// real games is ptcg_envpool_test.cc's job, this file only needs to prove
// the penalty wiring itself.
//
// The old sibling "illegal deck" test (an adversarial per-episode deck-
// select action naming a nonexistent card id) is gone: decks are fixed
// EnvPool-lifetime config now (see ptcg_envpool.h's DefaultConfig), not an
// action a policy submits per episode, so that scenario is structurally
// unreachable -- an invalid configured deck now CHECK-fails Reset()
// (a construction/config bug, not a per-episode reward-penalty condition).
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

// Single-env send/recv helper.
PtcgState Send(ptcg::PtcgEnvPool& envpool, int action_value) {  // NOLINT
  std::vector<Array> raw_action({Array(Spec<int>({1})), Array(Spec<int>({1})),
                                  Array(Spec<int>({1}))});
  PtcgAction action(raw_action);
  action["env_id"_][0] = 0;
  action["players.env_id"_][0] = 0;
  action["action"_][0] = action_value;
  envpool.Send(action);
  return PtcgState(envpool.Recv());
}

}  // namespace

TEST(PtcgEnvPoolIllegalTest, IllegalDecideActionPenalizesActor) {
  auto config = ptcg::PtcgEnvSpec::kDefaultConfig;
  config["num_envs"_] = 1;
  config["batch_size"_] = 1;
  config["num_threads"_] = 1;
  config["deck0"_] = std::vector<int>(kDeck.begin(), kDeck.end());
  config["deck1"_] = std::vector<int>(kDeck.begin(), kDeck.end());
  ptcg::PtcgEnvSpec spec(config);
  ptcg::PtcgEnvPool envpool(spec);

  Array env_ids(Spec<int>({1}));
  env_ids[0] = 0;
  envpool.Reset(env_ids);
  // Reset()'s own response is already a genuine, non-bypassed decide()
  // prompt (decks are config now -- ApiBattleStart + Phase 2.5's bypass
  // loop both run inside Reset() itself) -- safe to answer illegally on
  // purpose right away.
  PtcgState state(envpool.Recv());
  int actor_before = static_cast<int>(state["info:current_player"_][0]);

  // Within ActionSpec's declared bound ({0, kMaxOptions}) -- a real
  // option-row index, just not one that exists yet -- but essentially
  // certain to exceed the real state.options.size() for any early-game
  // decision, per "Action space" in the skill.
  state = Send(envpool, ptcg::kMaxOptions - 1);

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
