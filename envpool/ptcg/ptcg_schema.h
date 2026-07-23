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

#ifndef ENVPOOL_PTCG_PTCG_SCHEMA_H_
#define ENVPOOL_PTCG_PTCG_SCHEMA_H_

namespace ptcg {

// EncodedObservation shape constants, mirrored from
// submissions/template_submission/schema.py:61-67. Keep in sync by hand --
// there is no shared codegen between the Python and C++ sides (see the
// envpool-ptcg-integration skill's "EncodedObservation schema" section).
// Split into its own file (mirroring schema.py/encode.py's own split) so
// both ptcg_envpool.h and ptcg_encode.h depend on one definition instead of
// two copies drifting apart.
constexpr int kMaxCards = 120;
constexpr int kMaxPokemon = 18;  // ROW count: (1 active + 8 bench) x 2 players.
constexpr int kMaxOptions = 63;
constexpr int kNCardIds = 1267;

constexpr int kCardsCols = 5;
// COLUMN count: 18, coincidentally the same number as kMaxPokemon (a row
// count) above -- unrelated constants, verified against schema.py's
// POKEMONS_COLUMNS directly (`len(schema.POKEMONS_COLUMNS) == 18`), not
// derived from kMaxPokemon. An earlier version of this file had this at 15
// -- wrong, caught in Phase 3 by cross-checking against the real schema.py
// instead of trusting the value carried over from Phase 1. It went
// undetected through Phase 1/2 because those phases' tests only checked
// shapes against ptcg_envpool.h's own constants (circular), never against
// schema.py.
constexpr int kPokemonsCols = 18;
constexpr int kPlayerStateRows = 2;
constexpr int kPlayerStateCols = 9;
constexpr int kStateCols = 8;
// Same Phase-1-wrong/Phase-3-fixed story as kPokemonsCols: was 16, schema.py
// says `len(SELECT_COLUMNS) == 18` (6 scalar fields + two 6-field card
// pointers -- see ptcg_encode.h's WriteSelect).
constexpr int kSelectCols = 18;
constexpr int kOptionsCols = 19;

// Deck-select and decide() share one action shape (see "Action space" in the
// envpool-ptcg-integration skill): 60 slots, all real values (card ids) for
// deck-select, up to maxCount real option-row indices + -1 padding for
// decide().
constexpr int kActionSlots = 60;

}  // namespace ptcg

#endif  // ENVPOOL_PTCG_PTCG_SCHEMA_H_
