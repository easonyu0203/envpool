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
#include <string>

using PtcgAction = typename ptcg::PtcgEnv::Action;
using PtcgState = typename ptcg::PtcgEnv::State;

// Phase 3 test (envpool-ptcg-integration skill): structural/self-consistency
// invariants on real decide()-step observations from the C++ encoder
// (ptcg_encode.h), mirroring tests/support/invariants.py's
// assert_valid_encoding -- shapes (already guaranteed by StateSpec, not
// re-checked here), card id ranges, the 60/60 mine-vs-opponent split,
// options padding-zero beyond is_valid, and every card/pokemon pointer
// resolving to a real row of the exact claimed id. This is the
// self-consistency bar Phase 3 can clear standing alone; it does NOT
// attempt bit-exact comparison against the Python ctypes-backed
// encode_observation() -- that dual-encode comparator, plus the Bazel
// plumbing to reach across into the main repo's tests/support from this
// workspace, is explicitly Phase 5's job ("cross-validation suite"), not
// this one's.
namespace {

// Same real, deck-legal 60-card list as ptcg_envpool_test.cc /
// ptcg_smoke_test.cc / research/envpool_smoke/concurrency_smoke.cpp, copied
// from submissions/sample_submission/deck.csv. Both seats use it.
constexpr std::array<int, 60> kDeck = {
    1158, 721,  721,  722,  722,  722,  722,  723,  723,  723,  723,  1145,
    1145, 1145, 1145, 1205, 1205, 1227, 1227, 1227, 1227, 1235, 1235, 1235,
    1235, 3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
};

// Column indices, mirrored from schema.py's *_COLUMNS tuples (same
// source-of-truth duplication story as ptcg_schema.h/ptcg_encode.h -- no
// shared codegen between Python and C++).
namespace cards_col {
constexpr int kId = 0;
constexpr int kIsMe = 1;
constexpr int kPokemonPos = 2;
constexpr int kArea = 3;
constexpr int kPosInArea = 4;
}  // namespace cards_col

namespace pokemons_col {
constexpr int kIsValid = 0;
constexpr int kIsMe = 1;
constexpr int kPokemonPos = 2;
}  // namespace pokemons_col

namespace options_col {
constexpr int kIsValid = 0;
constexpr int kType = 1;
constexpr int kCardValid = 10;
constexpr int kCardIsMe = 11;
constexpr int kCardPokemonPos = 12;
constexpr int kCardArea = 13;
constexpr int kCardPosInArea = 14;
constexpr int kCardId = 15;
constexpr int kPokemonValid = 16;
constexpr int kPokemonIsMe = 17;
constexpr int kPokemonPos = 18;
constexpr int kAlreadySelected = 19;
}  // namespace options_col

namespace select_col {
constexpr int kMinCount = 2;
constexpr int kMaxCount = 3;
constexpr int kContextCardValid = 6;
constexpr int kContextCardIsMe = 7;
constexpr int kContextCardPokemonPos = 8;
constexpr int kContextCardArea = 9;
constexpr int kContextCardPosInArea = 10;
constexpr int kContextCardId = 11;
constexpr int kEffectCardValid = 12;
constexpr int kEffectCardIsMe = 13;
constexpr int kEffectCardPokemonPos = 14;
constexpr int kEffectCardArea = 15;
constexpr int kEffectCardPosInArea = 16;
constexpr int kEffectCardId = 17;
}  // namespace select_col

int Cell(const TArray<int>& arr, int row, int col) { return arr[row][col]; }
int Cell1D(const TArray<int>& arr, int col) { return arr[col]; }

// Copies an obs:options view into stable storage -- the Array it wraps is a
// slice of AsyncEnvPool's ring buffer, liable to be overwritten by the time
// the next round's Send() is built (see ptcg_envpool.h's WriteState comment
// on why an unconditional Zero() is needed at all), so it can't be read
// lazily from the original TArray across a Send()/Recv() boundary.
std::vector<int> CopyOptions(const TArray<int>& options) {
  std::vector<int> out(ptcg::kOptionRows * ptcg::kOptionsCols);
  for (int row = 0; row < ptcg::kOptionRows; ++row) {
    for (int col = 0; col < ptcg::kOptionsCols; ++col) {
      out[row * ptcg::kOptionsCols + col] = Cell(options, row, col);
    }
  }
  return out;
}

// Picks the first legal (is_valid && !already_selected) row -- real options
// sort before ptcg::kStopSlot, so this greedily keeps picking real options
// up to selectMax before ever reaching STOP. is_valid alone isn't enough: a
// real row stays is_valid=1 forever once picked (structural, unaffected by
// already_selected), so without the already_selected check this would pick
// the same already-chosen row again on a decision's second+ sub-pick, which
// LegalActions() correctly rejects as illegal. Every genuine decide()-step
// observation is guaranteed to offer at least one real legal row (Python
// only ever sees >=2-way decisions -- see ptcg_envpool.h's
// AutoResolveForcedSubPicksThenRespond -- and STOP alone can never account
// for more than one of those two-plus legal slots), so this always finds
// a real pick, never STOP, keeping every game's action sequence trivially
// legal by construction.
int PickLegalAction(const std::vector<int>& options_flat) {
  for (int row = 0; row < ptcg::kOptionRows; ++row) {
    if (options_flat[row * ptcg::kOptionsCols + options_col::kIsValid] == 1 &&
        options_flat[row * ptcg::kOptionsCols + options_col::kAlreadySelected] == 0) {
      return row;
    }
  }
  ADD_FAILURE() << "no legal action found in observed options -- envpool invariant violated";
  return ptcg::kStopSlot;
}

// Mirrors tests/support/invariants.py's _find_card_row/_check_pointer.
bool FindCardRow(const TArray<int>& cards, int is_me, int pokemon_pos, int area,
                  int pos_in_area, int* out_id) {
  for (int row = 0; row < ptcg::kMaxCards; ++row) {
    if (Cell(cards, row, cards_col::kIsMe) == is_me &&
        Cell(cards, row, cards_col::kPokemonPos) == pokemon_pos &&
        Cell(cards, row, cards_col::kArea) == area &&
        Cell(cards, row, cards_col::kPosInArea) == pos_in_area) {
      *out_id = Cell(cards, row, cards_col::kId);
      return true;
    }
  }
  return false;
}

void CheckCardPointer(const TArray<int>& cards, const std::string& source, int valid, int is_me,
                      int pokemon_pos, int area, int pos_in_area, int claimed_id) {
  if (valid != 1) {
    return;
  }
  int found_id = 0;
  ASSERT_TRUE(FindCardRow(cards, is_me, pokemon_pos, area, pos_in_area, &found_id))
      << source << ": pointer (is_me=" << is_me << ", pokemon_pos=" << pokemon_pos
      << ", area=" << area << ", pos_in_area=" << pos_in_area
      << ") matches no row in cards tensor";
  ASSERT_EQ(found_id, claimed_id)
      << source << ": claimed id=" << claimed_id << " but cards tensor row has id=" << found_id;
}

void CheckPokemonPointer(const TArray<int>& pokemons, const std::string& source, int valid,
                         int is_me, int pos) {
  if (valid != 1) {
    return;
  }
  for (int row = 0; row < ptcg::kMaxPokemon; ++row) {
    if (Cell(pokemons, row, pokemons_col::kIsMe) == is_me &&
        Cell(pokemons, row, pokemons_col::kPokemonPos) == pos &&
        Cell(pokemons, row, pokemons_col::kIsValid) == 1) {
      return;
    }
  }
  FAIL() << source << ": pokemon pointer (is_me=" << is_me << ", pos=" << pos
         << ") matches no valid row in pokemons tensor";
}

void CheckStructuralInvariants(const PtcgState& state, int slot) {
  TArray<int> cards(state["obs:cards"_][slot]);
  TArray<int> pokemons(state["obs:pokemons"_][slot]);
  TArray<int> options(state["obs:options"_][slot]);
  TArray<int> select(state["obs:select"_][slot]);

  for (int row = 0; row < ptcg::kMaxCards; ++row) {
    int id = Cell(cards, row, cards_col::kId);
    EXPECT_GE(id, 0) << "row=" << row;
    EXPECT_LE(id, ptcg::kNCardIds) << "row=" << row;
  }

  int n_mine = 0;
  int n_opp = 0;
  for (int row = 0; row < ptcg::kMaxCards; ++row) {
    if (Cell(cards, row, cards_col::kIsMe) == 1) {
      n_mine++;
    } else {
      n_opp++;
    }
  }
  EXPECT_EQ(n_mine, 60);
  EXPECT_EQ(n_opp, 60);

  for (int row = 0; row < ptcg::kMaxPokemon; ++row) {
    int v = Cell(pokemons, row, pokemons_col::kIsValid);
    EXPECT_TRUE(v == 0 || v == 1) << "row=" << row << " is_valid=" << v;
  }

  int n_valid_opts = 0;
  for (int row = 0; row < ptcg::kMaxOptions; ++row) {
    n_valid_opts += Cell(options, row, options_col::kIsValid);
  }
  for (int row = n_valid_opts; row < ptcg::kMaxOptions; ++row) {
    for (int col = 0; col < ptcg::kOptionsCols; ++col) {
      EXPECT_EQ(Cell(options, row, col), 0)
          << "non-zero padding at options[" << row << "][" << col << "]";
    }
  }

  // STOP row (ptcg::kStopSlot, one past the real rows checked above): its
  // is_valid directly encodes whether stopping is currently legal
  // (picks-so-far, read off the real rows' already_selected column, vs
  // select.min_count) -- no separate padding-zero expectation the way real
  // rows past n_valid_opts have, since this row is always meaningfully
  // populated on a genuine decide()-step observation.
  int n_already_selected = 0;
  for (int row = 0; row < n_valid_opts; ++row) {
    n_already_selected += Cell(options, row, options_col::kAlreadySelected);
  }
  bool expected_can_stop = n_already_selected >= Cell1D(select, select_col::kMinCount);
  EXPECT_EQ(Cell(options, ptcg::kStopSlot, options_col::kIsValid), expected_can_stop ? 1 : 0)
      << "n_already_selected=" << n_already_selected
      << " min_count=" << Cell1D(select, select_col::kMinCount);
  EXPECT_EQ(Cell(options, ptcg::kStopSlot, options_col::kType), ptcg::kStopOptionType);
  EXPECT_EQ(Cell(options, ptcg::kStopSlot, options_col::kAlreadySelected), 0)
      << "STOP row must never itself be already_selected";
  for (int col = 0; col < ptcg::kOptionsCols; ++col) {
    if (col == options_col::kIsValid || col == options_col::kType) {
      continue;
    }
    EXPECT_EQ(Cell(options, ptcg::kStopSlot, col), 0)
        << "STOP row has non-zero data outside is_valid/type at col=" << col;
  }

  for (int row = 0; row < n_valid_opts; ++row) {
    CheckCardPointer(cards, "options.card", Cell(options, row, options_col::kCardValid),
                     Cell(options, row, options_col::kCardIsMe),
                     Cell(options, row, options_col::kCardPokemonPos),
                     Cell(options, row, options_col::kCardArea),
                     Cell(options, row, options_col::kCardPosInArea),
                     Cell(options, row, options_col::kCardId));
    CheckPokemonPointer(pokemons, "options.pokemon", Cell(options, row, options_col::kPokemonValid),
                        Cell(options, row, options_col::kPokemonIsMe),
                        Cell(options, row, options_col::kPokemonPos));
  }

  CheckCardPointer(cards, "select.context_card", Cell1D(select, select_col::kContextCardValid),
                   Cell1D(select, select_col::kContextCardIsMe),
                   Cell1D(select, select_col::kContextCardPokemonPos),
                   Cell1D(select, select_col::kContextCardArea),
                   Cell1D(select, select_col::kContextCardPosInArea),
                   Cell1D(select, select_col::kContextCardId));
  CheckCardPointer(cards, "select.effect_card", Cell1D(select, select_col::kEffectCardValid),
                   Cell1D(select, select_col::kEffectCardIsMe),
                   Cell1D(select, select_col::kEffectCardPokemonPos),
                   Cell1D(select, select_col::kEffectCardArea),
                   Cell1D(select, select_col::kEffectCardPosInArea),
                   Cell1D(select, select_col::kEffectCardId));
}

}  // namespace

// Drives real games (num_envs == batch_size, sync mode) and runs the
// structural checks above on every decide()-step observation encountered --
// deck-select steps are skipped (all-zero by design, nothing to check) via
// the same episode_step/is_deck_select bookkeeping ptcg_envpool_test.cc
// uses. Stops once every env has completed at least one full episode.
TEST(PtcgEncodeTest, StructuralInvariantsOnRealDecideSteps) {
  auto config = ptcg::PtcgEnvSpec::kDefaultConfig;
  const int num_envs = 2;
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

  struct EnvTrack {
    bool is_deck_select = true;
    bool done = false;
    bool completed_once = false;
    int checked_decide_steps = 0;
    std::vector<int> last_options;  // valid only when !is_deck_select && !done
  };
  std::vector<EnvTrack> track(num_envs);

  auto recv_and_check = [&]() {
    PtcgState state(envpool.Recv());
    for (int i = 0; i < num_envs; ++i) {
      int env_id = static_cast<int>(state["info:env_id"_][i]);
      EnvTrack& t = track[env_id];
      bool is_deck_select = static_cast<bool>(state["info:is_deck_select"_][i]);
      bool done = static_cast<bool>(state["done"_][i]);
      if (!is_deck_select && !done) {
        CheckStructuralInvariants(state, i);
        t.checked_decide_steps++;
        t.last_options = CopyOptions(TArray<int>(state["obs:options"_][i]));
      }
      t.is_deck_select = is_deck_select;
      t.done = done;
      if (done) {
        t.completed_once = true;
      }
    }
  };

  recv_and_check();

  const int kMaxRounds = 20000;
  int round = 0;
  auto all_completed = [&] {
    return std::all_of(track.begin(), track.end(),
                        [](const EnvTrack& t) { return t.completed_once; });
  };
  while (!all_completed()) {
    ASSERT_LT(round, kMaxRounds);
    ++round;
    std::vector<Array> raw_action(
        {Array(Spec<int>({num_envs})), Array(Spec<int>({num_envs})),
         Array(Spec<int>({num_envs, ptcg::kActionSlots}))});
    PtcgAction action(raw_action);
    for (int i = 0; i < num_envs; ++i) {
      action["env_id"_][i] = i;
      action["players.env_id"_][i] = i;
      EnvTrack& t = track[i];
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
    }
    envpool.Send(action);
    recv_and_check();
  }

  int total_checked = 0;
  for (const EnvTrack& t : track) {
    EXPECT_TRUE(t.completed_once);
    total_checked += t.checked_decide_steps;
  }
  // Sanity: this test is only meaningful if it actually exercised the
  // encoder on real decide() steps, not just deck-select/terminal ones.
  EXPECT_GT(total_checked, 0);
}
