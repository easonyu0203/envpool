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

#include "envpool/ptcg/ptcg_encode.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <vector>

// Phase 6 test (envpool-ptcg-integration skill): AGENTS.md's "deterministic
// test" tier, adapted for the accepted nondeterministic-Reset() tradeoff
// (ApiBattleStart has no seed parameter -- see "Resets are nondeterministic"
// in the skill). Can't do the literal "same seed -> same Reset() rollout"
// AGENTS.md describes, so instead: clone a real mid-episode State via the
// engine's own binary serialize()/deserialize() round-trip (not through
// ApiGetBattleData/erasePlayerData -- this test isn't about masking, just
// raw engine determinism, see ptcg_cross_validation_test.cc for the masked
// version), sync each clone's RNG once at the clone point (AGENTS.md
// explicitly allows this: "synchronize internal state at most once
// immediately after reset if needed to eliminate RNG/bootstrap differences
// ... after that, drive both sides only with the same external actions"),
// then replay the identical action sequence on both and assert bit-exact
// agreement at every step, not just the end state.
namespace {

constexpr std::array<int, 60> kDeck = {
    1158, 721,  721,  722,  722,  722,  722,  723,  723,  723,  723,  1145,
    1145, 1145, 1145, 1205, 1205, 1227, 1227, 1227, 1227, 1235, 1235, 1235,
    1235, 3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
};

constexpr unsigned kCloneRngSeed = 424242U;

// Deliberately RNG-free: always select [0, selectMin) -- the same
// trivially-legal policy ptcg_smoke_test.cc / concurrency_smoke.cpp use.
// The two clones still exercise the engine's own internal RNG (coin flips,
// shuffle effects, ...) identically, since both start from byte-identical
// state with byte-identically-seeded game.rng and receive byte-identical
// actions -- this policy only needs to be legal, not random, to prove that.
std::vector<int> TriviallyLegalSelection(const State& state) {
  std::vector<int> sel(state.selectMin);
  std::iota(sel.begin(), sel.end(), 0);
  return sel;
}

struct Encoded {
  Array cards{Spec<int>({ptcg::kMaxCards, ptcg::kCardsCols})};
  Array pokemons{Spec<int>({ptcg::kMaxPokemon, ptcg::kPokemonsCols})};
  Array player_state{Spec<int>({ptcg::kPlayerStateRows, ptcg::kPlayerStateCols})};
  Array state{Spec<int>({ptcg::kStateCols})};
  Array select{Spec<int>({ptcg::kSelectCols})};
  Array options{Spec<int>({ptcg::kOptionRows, ptcg::kOptionsCols})};

  // Deliberately always the very start of a decision (already_selected
  // empty) -- this test is about raw engine/encoder determinism given
  // identical inputs, not about exercising PtcgEnv's own per-index
  // accumulation (that's ptcg_envpool_test.cc's job).
  static Encoded From(const State& state, const std::array<int, ptcg::kDeckSize>& deck) {
    Encoded e;
    ptcg::EncodeObservation(state, deck, {}, e.cards, e.pokemons, e.player_state,
                            e.state, e.select, e.options);
    return e;
  }
};

void ExpectArrayEqual(const Array& a, const Array& b, const char* name, int step) {
  ASSERT_EQ(a.size, b.size) << name << " size mismatch, step " << step;
  const auto* da = reinterpret_cast<const int*>(a.Data());
  const auto* db = reinterpret_cast<const int*>(b.Data());
  for (std::size_t i = 0; i < a.size; ++i) {
    EXPECT_EQ(da[i], db[i]) << name << "[" << i << "] mismatch, step " << step;
  }
}

void ExpectEncodedEqual(const Encoded& a, const Encoded& b, int step) {
  ExpectArrayEqual(a.cards, b.cards, "cards", step);
  ExpectArrayEqual(a.pokemons, b.pokemons, "pokemons", step);
  ExpectArrayEqual(a.player_state, b.player_state, "player_state", step);
  ExpectArrayEqual(a.state, b.state, "state", step);
  ExpectArrayEqual(a.select, b.select, "select", step);
  ExpectArrayEqual(a.options, b.options, "options", step);
}

std::unique_ptr<ApiData, void (*)(ApiData*)> CloneFrom(const BinaryWriter& writer) {
  auto* clone = new ApiData();
  clone->state.game = &clone->game;
  BinaryReader reader;
  reader.buf = writer.buf;  // raw bytes, no base64 needed within one process
  clone->state.deserialize(reader);
  // Throughput fix, same as PtcgEnv's own (see "ptcg_engine C++ surface to
  // call directly" in the skill) -- without it both clones would draw from
  // real OS entropy (std::random_device) instead of the seeded mt19937
  // below, and diverge from each other immediately.
  clone->game.config.deviceRand = false;
  // The "reset-time RNG sync" AGENTS.md's alignment-test standard allows --
  // done once, here, not re-synced mid-rollout.
  clone->game.rng = std::mt19937(kCloneRngSeed);
  return {clone, [](ApiData* d) { ApiBattleFinish(d); }};
}

}  // namespace

TEST(PtcgDeterministicTest, ClonedStateReplaysIdentically) {
  static std::once_flag init_flag;
  std::call_once(init_flag, InitializeAll);

  std::array<int, 2 * 60> cards{};
  std::copy(kDeck.begin(), kDeck.end(), cards.begin());
  std::copy(kDeck.begin(), kDeck.end(), cards.begin() + 60);
  StartData start = ApiBattleStart(cards.data());
  ASSERT_NE(start.battlePtr, nullptr)
      << "errorPlayer=" << start.errorPlayer << " errorType=" << start.errorType;
  ApiData* origin = start.battlePtr;
  origin->game.config.deviceRand = false;

  constexpr int kWarmupSteps = 10;
  for (int i = 0; i < kWarmupSteps && !origin->state.isFinish(); ++i) {
    std::vector<int> sel = TriviallyLegalSelection(origin->state);
    ASSERT_EQ(ApiSelect(origin, sel.data(), static_cast<int>(sel.size())), 0)
        << "warmup step " << i;
  }
  ASSERT_FALSE(origin->state.isFinish())
      << "warmup finished the game before reaching a real mid-episode "
         "clone point -- reduce kWarmupSteps or accept this run's length";

  BinaryWriter writer;
  origin->state.serialize(writer);
  ApiBattleFinish(origin);

  auto clone_a = CloneFrom(writer);
  auto clone_b = CloneFrom(writer);

  constexpr int kRolloutSteps = 30;
  int steps_compared = 0;
  for (int step = 0; step < kRolloutSteps; ++step) {
    bool a_done = clone_a->state.isFinish();
    bool b_done = clone_b->state.isFinish();
    ASSERT_EQ(a_done, b_done) << "done mismatch, step " << step;
    if (a_done) {
      break;
    }

    ASSERT_EQ(clone_a->state.selectMin, clone_b->state.selectMin) << "step " << step;
    ASSERT_EQ(clone_a->state.selectMax, clone_b->state.selectMax) << "step " << step;
    ASSERT_EQ(clone_a->state.options.size(), clone_b->state.options.size())
        << "step " << step;
    ASSERT_EQ(clone_a->state.selectPlayer, clone_b->state.selectPlayer)
        << "step " << step;

    Encoded enc_a = Encoded::From(clone_a->state, kDeck);
    Encoded enc_b = Encoded::From(clone_b->state, kDeck);
    ExpectEncodedEqual(enc_a, enc_b, step);
    steps_compared++;

    std::vector<int> sel = TriviallyLegalSelection(clone_a->state);
    int err_a = ApiSelect(clone_a.get(), sel.data(), static_cast<int>(sel.size()));
    int err_b = ApiSelect(clone_b.get(), sel.data(), static_cast<int>(sel.size()));
    ASSERT_EQ(err_a, 0) << "step " << step;
    ASSERT_EQ(err_b, 0) << "step " << step;
  }

  EXPECT_EQ(clone_a->state.isFinish(), clone_b->state.isFinish());
  if (clone_a->state.isFinish() && clone_b->state.isFinish()) {
    EXPECT_EQ(clone_a->state.apiResult(), clone_b->state.apiResult());
  }
  // Sanity: this test is only meaningful if it actually compared real
  // decide()-step encodings, not just the done/isFinish bookkeeping.
  EXPECT_GT(steps_compared, 0);
}
