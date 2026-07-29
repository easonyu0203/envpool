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
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "tools/cpp/runfiles/runfiles.h"

using bazel::tools::cpp::runfiles::Runfiles;

// Phase 5 test (envpool-ptcg-integration skill): bit-exact cross-validation
// against the (already-trusted, already-tested) Python `encode_observation()`
// -- also doubles as Phase 6's AGENTS.md-style "alignment test" (oracle =
// our own Python encoder), since for this env family they're the same
// comparison. Fixtures are generated once by
// tests/encoding/generate_ptcg_encode_fixtures.py (main repo, run manually,
// not part of this test) and checked in at testdata/ -- see that script's
// docstring for exactly how each case's `search_begin_input` (a raw
// state.serialize() blob, masked once via state.erasePlayerData()) and
// matching EncodedObservation were captured from the *same* instant of one
// real ctypes-driven game, per "The mechanism matters" in the skill: driving
// two independently-seeded games and diffing decision-by-decision would
// desync silently the moment either side's RNG-call order diverges. Each
// case also carries an `already_selected` set (empty for most cases, a
// random non-empty subset for maxCount>1 decisions) exercising the
// options.already_selected column and the engine-appended STOP row
// (schema.py's STOP_SLOT) under a genuine mid-multi-pick state, not just a
// decision's first call.
namespace {

struct FixtureCase {
  std::string base64;
  std::array<int, ptcg::kActionSlots> my_deck{};
  std::vector<int> already_selected;  // indices into select.option, see generate_ptcg_encode_fixtures.py
  std::vector<int> cards, pokemons, player_state, state, select, options;
};

void ReadInts(std::ifstream& ifs, std::vector<int>* out) {
  for (int& v : *out) {
    ifs >> v;
  }
}

std::vector<FixtureCase> LoadFixtures(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs.good()) {
    ADD_FAILURE() << "failed to open fixture file: " << path;
    return {};
  }
  int n = 0;
  ifs >> n;
  std::vector<FixtureCase> cases(n);
  for (auto& c : cases) {
    ifs >> c.base64;
    for (int& v : c.my_deck) {
      ifs >> v;
    }
    int k = 0;
    ifs >> k;
    c.already_selected.resize(k);
    ReadInts(ifs, &c.already_selected);
    c.cards.resize(ptcg::kMaxCards * ptcg::kCardsCols);
    ReadInts(ifs, &c.cards);
    c.pokemons.resize(ptcg::kMaxPokemon * ptcg::kPokemonsCols);
    ReadInts(ifs, &c.pokemons);
    c.player_state.resize(ptcg::kPlayerStateRows * ptcg::kPlayerStateCols);
    ReadInts(ifs, &c.player_state);
    c.state.resize(ptcg::kStateCols);
    ReadInts(ifs, &c.state);
    c.select.resize(ptcg::kSelectCols);
    ReadInts(ifs, &c.select);
    c.options.resize(ptcg::kOptionRows * ptcg::kOptionsCols);
    ReadInts(ifs, &c.options);
  }
  if (ifs.fail() && !ifs.eof()) {
    ADD_FAILURE() << "fixture file parse error (not enough tokens?): " << path;
  }
  return cases;
}

void ExpectFlatEqual(const Array& arr, const std::vector<int>& expected,
                     const char* name, int case_index) {
  const auto* data = reinterpret_cast<const int*>(arr.Data());
  ASSERT_EQ(arr.size, expected.size())
      << name << " size mismatch, case " << case_index;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(data[i], expected[i])
        << name << "[" << i << "] mismatch, case " << case_index
        << " (fixture regenerated? or a real encoder divergence)";
  }
}

}  // namespace

TEST(PtcgCrossValidationTest, BitExactAgainstPythonEncoder) {
  std::string error;
  std::unique_ptr<Runfiles> runfiles(Runfiles::CreateForTest(&error));
  ASSERT_NE(runfiles, nullptr) << "Runfiles::CreateForTest failed: " << error;
  std::string fixture_path = runfiles->Rlocation(
      "envpool/envpool/ptcg/testdata/cross_validation_fixtures.txt");
  ASSERT_FALSE(fixture_path.empty()) << "could not resolve fixture rlocation";

  std::vector<FixtureCase> cases = LoadFixtures(fixture_path);
  ASSERT_GT(cases.size(), 0u)
      << "no fixture cases loaded -- did generate_ptcg_encode_fixtures.py "
         "run and write a non-empty file?";

  // Same InitializeAll()-once-per-process requirement as PtcgEnv's own
  // constructor (see ptcg_envpool.h) -- CardTable must be populated before
  // any getMaster() call the encoder makes.
  static std::once_flag init_flag;
  std::call_once(init_flag, InitializeAll);

  for (int i = 0; i < static_cast<int>(cases.size()); ++i) {
    const FixtureCase& c = cases[i];

    ApiData* data = new ApiData();
    // Not created via ApiBattleStart, so this wiring (normally done inside
    // BattleData::setState()) has to happen explicitly -- same pattern as
    // Api.h's own ApiAgentStart().
    data->state.game = &data->game;
    SetBattleData(data, c.base64.data(), static_cast<int>(c.base64.size()));

    Array cards_arr(Spec<int>({ptcg::kMaxCards, ptcg::kCardsCols}));
    Array pokemons_arr(Spec<int>({ptcg::kMaxPokemon, ptcg::kPokemonsCols}));
    Array player_state_arr(
        Spec<int>({ptcg::kPlayerStateRows, ptcg::kPlayerStateCols}));
    Array state_arr(Spec<int>({ptcg::kStateCols}));
    Array select_arr(Spec<int>({ptcg::kSelectCols}));
    Array options_arr(Spec<int>({ptcg::kOptionRows, ptcg::kOptionsCols}));
    // No explicit Zero() needed here, unlike ptcg_envpool.h's WriteState --
    // this is Array's *owning* constructor (a fresh std::vector<char>(size),
    // which value-inits to zero), not Allocate()'s reused ring-buffer slice.

    ptcg::EncodeObservation(data->state, c.my_deck, c.already_selected,
                            cards_arr, pokemons_arr, player_state_arr,
                            state_arr, select_arr, options_arr);

    ExpectFlatEqual(cards_arr, c.cards, "cards", i);
    ExpectFlatEqual(pokemons_arr, c.pokemons, "pokemons", i);
    ExpectFlatEqual(player_state_arr, c.player_state, "player_state", i);
    ExpectFlatEqual(state_arr, c.state, "state", i);
    ExpectFlatEqual(select_arr, c.select, "select", i);
    ExpectFlatEqual(options_arr, c.options, "options", i);

    delete data;
  }
}
