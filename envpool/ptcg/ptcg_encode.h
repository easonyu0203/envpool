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

#ifndef ENVPOOL_PTCG_PTCG_ENCODE_H_
#define ENVPOOL_PTCG_PTCG_ENCODE_H_

#include <algorithm>
#include <array>
#include <map>
#include <unordered_map>
#include <vector>

#include "All.h"

#include "envpool/core/array.h"
#include "envpool/ptcg/ptcg_schema.h"

// Phase 3 (envpool-ptcg-integration skill): a direct C++ port of
// submissions/template_submission/encode.py's `encode_observation()`, fused
// with the hidden-info-masking/computed-getter logic that in the Python path
// lives one layer further down, in ptcg_engine/ToJson.h (specifically
// `PokemonJson`/`PlayerJson`/`Current`/`SelectJson`/`SelectOptionJson`) --
// bypassing JSON entirely means both layers collapse into this one file (see
// "port risk areas" in the skill). This file is organized to mirror
// encode.py's own section order and helper-function names as closely as
// possible (register/register_pokemon/resolve_direct/resolve_attached/
// owning_player_of_own_action/resolve_inplay/resolve_by_serial/
// card_ptr_or_none/encode_option -> RegisterAll/RegisterPokemon/
// ResolveDirect/ResolveAttached/OwningPlayerOfOwnAction/ResolveInPlay/
// ResolveBySerial/EncodeOption), so a side-by-side read against encode.py is
// the intended way to audit this file, not just reading it standalone.
//
// Every non-obvious mapping between the engine's raw State and the JSON
// shape encode.py actually consumes is cited inline against ToJson.h/
// ApiJson.h -- see the envpool-ptcg-integration skill's "EncodedObservation
// schema and port risk areas" section for the full risk ranking this file
// was written against.
namespace ptcg {

// Mirrors schema.py's CardArea IntEnum exactly (value-for-value).
namespace card_area {
constexpr int kHand = 0;
constexpr int kDiscard = 1;
constexpr int kActive = 2;
constexpr int kBench = 3;
constexpr int kStadium = 4;
constexpr int kEnergy = 5;
constexpr int kTool = 6;
constexpr int kPreEvolution = 7;
constexpr int kLooking = 8;
constexpr int kDeck = 9;
constexpr int kPrizeDeck = 10;
constexpr int kPrizeDeckHand = 11;
constexpr int kResolving = 12;
}  // namespace card_area

// Mirrors schema.py's PokemonPos IntEnum exactly (value-for-value).
namespace pokemon_pos {
constexpr int kNone = 0;
constexpr int kActive = 1;
constexpr int kBench0 = 2;
constexpr int Bench(int bench_index) { return kBench0 + bench_index; }
}  // namespace pokemon_pos

namespace detail {

// (card_id, is_me, pokemon_pos, area, pos_in_area) -- what encode.py's
// `loc_by_serial`/`mine_rows`/`opp_rows` store per registered card.
struct CardLoc {
  int card_id = 0;
  int is_me = 0;
  int pokemon_pos = 0;
  int area = 0;
  int pos_in_area = 0;
};

// (valid, is_me, pokemon_pos, area, pos_in_area, id) -- encode.py's
// `_CARD_PTR` tuple shape, used for select/options card pointers. Distinct
// field *order* from CardLoc (valid first, id last) is why this is a
// separate type rather than reusing CardLoc with a bolted-on `valid` flag.
struct CardPtr {
  int valid = 0;
  int is_me = 0;
  int pokemon_pos = 0;
  int area = 0;
  int pos_in_area = 0;
  int id = 0;
};

// (valid, is_me, pos) -- encode.py's `_POKEMON_PTR`: a pointer to an
// in-play Pokemon *slot*, not a specific card (ATTACH/EVOLVE's inPlay
// target).
struct PokemonPtr {
  int valid = 0;
  int is_me = 0;
  int pos = 0;
};

// encode.py's `_NO_CARD` / `_NO_POKEMON` sentinels, replicated field-for-
// field (including the arbitrary-but-fixed area=HAND placeholder on an
// invalid card pointer) in case anything downstream ever inspects the
// sentinel's bare values despite valid=0.
constexpr CardPtr kNoCardPtr{0, 0, pokemon_pos::kNone, card_area::kHand, 0, 0};
constexpr PokemonPtr kNoPokemonPtr{0, 0, pokemon_pos::kNone};

inline CardPtr FromLoc(const CardLoc& loc) {
  return {1, loc.is_me, loc.pokemon_pos, loc.area, loc.pos_in_area, loc.card_id};
}

// Writes consecutive int32 cells starting at base[0] -- base is a raw
// pointer to the first cell of the target row, e.g.
// static_cast<int*>(cards_arr.Data()) + row * kCardsCols, or
// static_cast<int*>(select_arr.Data()) for a genuinely 1-D tensor. Mirrors
// encode.py's positional tuple assignment (`cards[row] = (a, b, c, ...)`);
// every call site lists column names in a comment in the same order as the
// corresponding schema.py *_COLUMNS tuple, since there's no shared codegen
// to enforce it.
//
// Deliberately raw-pointer, not Array::operator[]: Array::operator()
// heap-allocates a fresh std::vector<size_t> (its shape_) on *every* call
// (envpool/core/array.h), sized for occasional coarse-grained indexing (one
// slice per env per step in envpool's typical usage), not this encoder's
// per-scalar-cell access pattern. Going through it here cost ~18x throughput
// (WriteCards's ~720 calls/decision dominated) -- see
// research/env_speed/README.md's "Encoder bug" section (main repo) for the
// measured before/after.
inline void WriteRow(int* base, std::initializer_list<int> values) {
  int i = 0;
  for (int v : values) {
    base[i] = v;
    ++i;
  }
}

/**
 * One-shot, single-use encoder for one decide()-step observation. Mirrors
 * encode.py's encode_observation()'s local closures as member functions
 * instead, since C++ doesn't have Python's capture-by-closure-over-locals as
 * conveniently -- the *_rows_/loc_by_serial_/etc. members here are exactly
 * encode_observation()'s local variables of the same name.
 */
class ObservationEncoder {
 public:
  ObservationEncoder(const State& state, const std::array<int, kActionSlots>& my_deck)
      : state_(state),
        your_index_(state.selectPlayer),
        opp_index_(1 - state.selectPlayer),
        my_deck_(my_deck) {}

  void Encode(const Array& cards_arr, const Array& pokemons_arr,
              const Array& player_state_arr, const Array& state_arr,
              const Array& select_arr, const Array& options_arr) {
    RegisterAll();
    WriteCards(cards_arr);
    WritePokemons(pokemons_arr);
    WritePlayerState(player_state_arr);
    WriteStateVec(state_arr);
    WriteSelect(select_arr);
    WriteOptions(options_arr);
  }

 private:
  const State& state_;
  int your_index_;
  int opp_index_;
  const std::array<int, kActionSlots>& my_deck_;

  std::unordered_map<int, CardLoc> loc_by_serial_;
  std::vector<CardLoc> mine_rows_;
  std::vector<CardLoc> opp_rows_;
  std::vector<int> mine_revealed_ids_;
  std::unordered_map<int, std::vector<int>> pending_prize_deck_pointer_;
  std::map<int, int> mine_hidden_ids_;  // card_id -> count, sorted by id (std::map)
  int opp_hidden_count_ = 0;

  // --- registration walk (encode.py's `register`/`register_pokemon` +
  // main loop + "Playing-limbo fallback") ---

  void Register(CardRef ref, int is_me, int pos, int area, int pos_in_area) {
    int serial = ref.cardIndex;
    int card_id = state_.getCard(ref).getMaster().cardId;
    CardLoc loc{card_id, is_me, pos, area, pos_in_area};
    loc_by_serial_[serial] = loc;
    (is_me != 0 ? mine_rows_ : opp_rows_).push_back(loc);
    if (is_me != 0) {
      mine_revealed_ids_.push_back(card_id);
    }
  }

  void RegisterPokemon(CardRef ref, int is_me, int pos) {
    int base_area = (pos == pokemon_pos::kActive) ? card_area::kActive : card_area::kBench;
    Register(ref, is_me, pos, base_area, 0);
    const Card& card = state_.getCard(ref);
    // Own local scratch vector, not state_.game->cardList -- that's an
    // engine-internal "temporary use" scratch buffer; using our own avoids
    // any risk of interfering with it if something else during this
    // decision point assumes a particular scratch-buffer state. (The
    // *energies* list -- as opposed to energyCards here -- is a separate
    // concept used only by WritePokemons, not this registration walk.)
    std::vector<CardRef> energy_cards;
    state_.getEnergyCards(ref, energy_cards);
    for (int i = 0; i < static_cast<int>(energy_cards.size()); ++i) {
      Register(energy_cards[i], is_me, pos, card_area::kEnergy, i);
    }
    auto tools = state_.getAttachedToolRef(card);
    for (int i = 0; i < tools.size(); ++i) {
      Register(tools[i], is_me, pos, card_area::kTool, i);
    }
    auto pre_evolutions = state_.getPreEvolutions(card);
    for (int i = 0; i < pre_evolutions.size(); ++i) {
      Register(pre_evolutions[i], is_me, pos, card_area::kPreEvolution, i);
    }
  }

  // Whether the (possibly empty) state_.looking is visible to your_index_ --
  // shared by the registration walk and ResolveDirect's LOOKING case. See
  // ToJson.h's Current(): looking is all-or-nothing per viewer (full reveal
  // or nothing), never per-card masked, except the masked-count-only case
  // (lookingPlayer 3/4) which this deliberately treats identically to "not
  // visible" -- both result in zero registrations, matching encode.py's own
  // `if c is not None: register(...)` (masked entries are never non-None).
  [[nodiscard]] bool LookingVisible() const {
    return !state_.looking.empty() &&
           (state_.lookingPlayer == your_index_ || state_.lookingPlayer == 2);
  }

  // A reversed (face-down) active/bench Pokemon is normally represented as
  // a valid CardRef whose Card has reverse=true -- that's the only
  // representation PtcgEnv's own live pipeline ever produces (it reads
  // battle_->state directly, never through State::erasePlayerData()). But
  // erasePlayerData() (ApiGetBattleData's masking -- used by the
  // cross-validation fixture harness, Phase 5) goes one step further for
  // the *opponent's* reversed active Pokemon specifically: eraseCard()
  // takes the CardRef by reference and zeroes the ref itself, not just the
  // Card it points to (`ref = {};`, ptcg_engine/State.h's eraseCard). A
  // plain `!getCard(ref).reverse` check would call getCard() on that
  // now-null ref and hit its `assert(cardIndex > 0)` -- confirmed the hard
  // way, see the skill's Phase 5 checklist entry. isNull() must be checked
  // first; both representations mean the same thing to this encoder.
  [[nodiscard]] bool IsReversedOrErased(CardRef ref) const {
    return ref.isNull() || state_.getCard(ref).reverse;
  }

  void RegisterAll() {
    std::array<int, 2> masked_active_bench{0, 0};
    for (int player_index = 0; player_index < 2; ++player_index) {
      int is_me = (player_index == your_index_) ? 1 : 0;
      const auto& ps = state_.players[player_index];
      if (!ps.active.empty()) {
        CardRef ref = ps.active[0];
        if (!IsReversedOrErased(ref)) {
          RegisterPokemon(ref, is_me, pokemon_pos::kActive);
        } else {
          masked_active_bench[player_index]++;
        }
      }
      for (int bidx = 0; bidx < ps.bench.size(); ++bidx) {
        CardRef ref = ps.bench[bidx];
        if (!IsReversedOrErased(ref)) {
          RegisterPokemon(ref, is_me, pokemon_pos::Bench(bidx));
        } else {
          masked_active_bench[player_index]++;
        }
      }
      if (is_me != 0) {  // hand populated only for me
        for (int i = 0; i < ps.hand.size(); ++i) {
          Register(ps.hand[i], is_me, pokemon_pos::kNone, card_area::kHand, i);
        }
      }
      for (int i = 0; i < ps.trash.size(); ++i) {
        Register(ps.trash[i], is_me, pokemon_pos::kNone, card_area::kDiscard, i);
      }
      // ps.prize entries are always face-down, even to their own owner, in
      // real play -- count only, folded into the hidden-pool bookkeeping
      // below (see encode.py's own comment on this).
    }

    for (int i = 0; i < state_.stadium.size(); ++i) {
      CardRef ref = state_.stadium[i];
      int is_me = (state_.getCard(ref).playerIndex == your_index_) ? 1 : 0;
      Register(ref, is_me, pokemon_pos::kNone, card_area::kStadium, i);
    }

    if (LookingVisible()) {
      for (int i = 0; i < state_.looking.size(); ++i) {
        CardRef ref = state_.looking[i];
        int is_me = (state_.getCard(ref).playerIndex == your_index_) ? 1 : 0;
        Register(ref, is_me, pokemon_pos::kNone, card_area::kLooking, i);
      }
    }

    if (state_.selectDeck) {
      // Always my own deck -- only the currently-selecting player ever gets
      // select.deck (see SelectJson in ToJson.h).
      const auto& deck_list = state_.players[your_index_].deck;
      for (int i = 0; i < deck_list.size(); ++i) {
        Register(deck_list[i], 1, pokemon_pos::kNone, card_area::kDeck, i);
      }
    }

    // --- hidden pools ---
    const auto& my_ps = state_.players[your_index_];
    const auto& opp_ps = state_.players[opp_index_];
    int mine_hidden_count = static_cast<int>(my_ps.prize.size()) +
                            (state_.selectDeck ? 0 : static_cast<int>(my_ps.deck.size())) +
                            masked_active_bench[your_index_];
    opp_hidden_count_ = static_cast<int>(opp_ps.hand.size()) +
                        static_cast<int>(opp_ps.deck.size()) +
                        static_cast<int>(opp_ps.prize.size()) +
                        masked_active_bench[opp_index_];

    // --- Playing-limbo fallback (see encode.py's comment of the same name)
    // ---
    int resolving_counter = 0;
    int mine_room = 60 - mine_hidden_count - static_cast<int>(mine_rows_.size());
    int opp_room = 60 - opp_hidden_count_ - static_cast<int>(opp_rows_.size());
    auto maybe_register_fallback = [&](CardRef ref) {
      if (ref.isNull()) {
        return;
      }
      int serial = ref.cardIndex;
      if (loc_by_serial_.count(serial) != 0) {
        return;
      }
      const Card& c = state_.getCard(ref);
      int is_me = (c.playerIndex == your_index_) ? 1 : 0;
      int& room = (is_me != 0) ? mine_room : opp_room;
      if (room <= 0) {
        if (is_me != 0) {
          pending_prize_deck_pointer_[c.getMaster().cardId].push_back(serial);
        }
        // else: opponent has no identity-tracked bucket to point into;
        // leave this pointer unresolved rather than break the row count.
        return;
      }
      Register(ref, is_me, pokemon_pos::kNone, card_area::kResolving, resolving_counter);
      resolving_counter++;
      room--;
    };
    // Order matters: mine_room/opp_room are shared mutable state across both
    // calls, so which of the two (if both are simultaneously unregistered
    // and room is scarce) gets the RESOLVING row vs the PRIZE_DECK fallback
    // pointer depends on call order. Must match encode.py's `for c in
    // (select.effect, select.contextCard)` (effect first) exactly.
    if (state_.onEffect()) {
      maybe_register_fallback(state_.getEffectCard().card);
    }
    maybe_register_fallback(state_.contextCard);

    // mine_hidden_ids = +(Counter(my_deck) - Counter(mine_revealed_ids)) --
    // mine_revealed_ids_ already includes any RESOLVING rows registered just
    // above, since Register() appends to it unconditionally.
    for (int id : my_deck_) {
      mine_hidden_ids_[id]++;
    }
    for (int id : mine_revealed_ids_) {
      auto it = mine_hidden_ids_.find(id);
      if (it == mine_hidden_ids_.end()) {
        continue;
      }
      it->second--;
      if (it->second <= 0) {
        mine_hidden_ids_.erase(it);
      }
    }
  }

  // --- cards tensor ---

  void WriteCards(const Array& cards_arr) {
    auto* base = static_cast<int*>(cards_arr.Data());
    int row = 0;
    for (const CardLoc& loc : mine_rows_) {
      // cards_id, is_me, pokemon_pos, area, pos_in_area
      WriteRow(base + row * kCardsCols, {loc.card_id, 1, loc.pokemon_pos, loc.area, loc.pos_in_area});
      row++;
    }
    int pos = 0;
    for (const auto& [card_id, count] : mine_hidden_ids_) {  // std::map: sorted by card_id
      for (int c = 0; c < count; ++c) {
        WriteRow(base + row * kCardsCols,
                 {card_id, 1, pokemon_pos::kNone, card_area::kPrizeDeck, pos});
        auto it = pending_prize_deck_pointer_.find(card_id);
        if (it != pending_prize_deck_pointer_.end() && !it->second.empty()) {
          int serial = it->second.back();
          it->second.pop_back();
          loc_by_serial_[serial] = CardLoc{card_id, 1, pokemon_pos::kNone, card_area::kPrizeDeck, pos};
        }
        row++;
        pos++;
      }
    }
    row = 60;
    for (const CardLoc& loc : opp_rows_) {
      WriteRow(base + row * kCardsCols, {loc.card_id, 0, loc.pokemon_pos, loc.area, loc.pos_in_area});
      row++;
    }
    for (int p = 0; p < opp_hidden_count_; ++p) {
      WriteRow(base + row * kCardsCols, {0, 0, pokemon_pos::kNone, card_area::kPrizeDeckHand, p});
      row++;
    }
    // If registration accounting is off, some rows past here stay at
    // whatever the caller Zero()'d them to -- see ptcg_envpool.h's
    // WriteState, which always Zero()s before calling this encoder.
  }

  // --- pokemons tensor ---

  void FillPokemonRow(const Array& pokemons_arr, int row_idx, CardRef ref, int is_me,
                      int pos, int player_index) {
    const Card& card = state_.getCard(ref);
    std::vector<EnergyType> energies;
    state_.getEnergies(player_index, ref, energies);
    std::array<int, 12> energy_counts{};
    for (EnergyType e : energies) {
      energy_counts[EnergyTypeIndex(e)]++;
    }
    auto* row = static_cast<int*>(pokemons_arr.Data()) + row_idx * kPokemonsCols;
    // is_valid, is_me, pokemon_pos, hp, max_hp, appear_this_turn, energy_count_0..11
    row[0] = 1;
    row[1] = is_me;
    row[2] = pos;
    row[3] = state_.getHp(card);
    row[4] = state_.getMaxHp(card);
    row[5] = static_cast<int>(card.appear);
    for (int i = 0; i < 12; ++i) {
      row[6 + i] = energy_counts[i];
    }
  }

  void WritePokemons(const Array& pokemons_arr) {
    for (int player_index = 0; player_index < 2; ++player_index) {
      int is_me = (player_index == your_index_) ? 1 : 0;
      int base = (is_me != 0) ? 0 : 9;
      const auto& ps = state_.players[player_index];
      // A face-down (reversed) active/bench Pokemon leaves its row at the
      // Zero()'d default, same as an altogether-absent slot -- both are
      // "no information" from this viewer's position (mirrors encode.py's
      // `if pk is None: return` inside fill_pokemon_row, since a reversed
      // Pokemon comes through as a masked None entry at the JSON layer --
      // see ToJson.h's PokemonJson).
      if (!ps.active.empty()) {
        CardRef ref = ps.active[0];
        if (!IsReversedOrErased(ref)) {
          FillPokemonRow(pokemons_arr, base, ref, is_me, pokemon_pos::kActive, player_index);
        }
      }
      for (int bidx = 0; bidx < ps.bench.size(); ++bidx) {
        CardRef ref = ps.bench[bidx];
        if (!IsReversedOrErased(ref)) {
          FillPokemonRow(pokemons_arr, base + 1 + bidx, ref, is_me, pokemon_pos::Bench(bidx),
                         player_index);
        }
      }
    }
  }

  // --- player_state tensor ---

  void WritePlayerState(const Array& player_state_arr) {
    auto* base = static_cast<int*>(player_state_arr.Data());
    std::array<int, 2> order = {your_index_, opp_index_};
    for (int row_idx = 0; row_idx < 2; ++row_idx) {
      int player_index = order[row_idx];
      const auto& ps = state_.players[player_index];
      // bench_max, deck_count, prize_count, hand_count, poisoned, burned, asleep, paralyzed, confused
      WriteRow(base + row_idx * kPlayerStateCols, {
        state_.benchCapacity(player_index),
        static_cast<int>(ps.deck.size()),
        static_cast<int>(ps.prize.size()),
        static_cast<int>(ps.hand.size()),
        static_cast<int>(ps.isPoisoned()),
        static_cast<int>(ps.burned),
        static_cast<int>(ps.badStatus == BadStatusType::Asleep),
        static_cast<int>(ps.badStatus == BadStatusType::Paralyzed),
        static_cast<int>(ps.badStatus == BadStatusType::Confused),
      });
    }
  }

  // --- state vector ---

  void WriteStateVec(const Array& state_arr) {
    bool first_player_known = state_.firstPlayer != -1;
    bool is_first = first_player_known && (state_.firstPlayer == your_index_);
    // turn, turn_action_count, is_first, first_player_known, supporter_played,
    // stadium_played, energy_attached, retreated
    // NOTE: "energy_attached" reads state_.energyPlayed, not a field literally
    // named energyAttached -- that's the JSON *key* name (ToJson.h's
    // Current()), the underlying State field is energyPlayed.
    WriteRow(static_cast<int*>(state_arr.Data()), {
      state_.turn,
      state_.turnActionCount,
      static_cast<int>(is_first),
      static_cast<int>(first_player_known),
      static_cast<int>(state_.supporterPlayed),
      static_cast<int>(state_.stadiumPlayed),
      static_cast<int>(state_.energyPlayed),
      static_cast<int>(state_.retreated),
    });
  }

  // --- card-pointer resolvers, shared by select and options blocks ---

  CardPtr ResolveBySerial(int serial, int card_id) {
    // Skill option's cardId==0 means "this is a special-condition
    // resolution, not a real card" (see ApiType.h's OptionType::SKILL doc
    // comment) -- the only caller where card_id can legitimately be 0.
    if (card_id == 0) {
      return kNoCardPtr;
    }
    auto it = loc_by_serial_.find(serial);
    if (it == loc_by_serial_.end()) {
      return kNoCardPtr;
    }
    return FromLoc(it->second);
  }

  CardPtr ResolveDirect(int area, int index, int player_index) {
    int is_me = (player_index == your_index_) ? 1 : 0;
    const auto& ps = state_.players[player_index];
    auto id_of = [&](CardRef ref) { return state_.getCard(ref).getMaster().cardId; };
    if (area == static_cast<int>(AreaType::Hand)) {
      CardRef ref = ps.hand[index];
      return {1, is_me, pokemon_pos::kNone, card_area::kHand, index, id_of(ref)};
    }
    if (area == static_cast<int>(AreaType::Trash)) {  // "Discard" in schema.py's naming
      CardRef ref = ps.trash[index];
      return {1, is_me, pokemon_pos::kNone, card_area::kDiscard, index, id_of(ref)};
    }
    if (area == static_cast<int>(AreaType::Active)) {
      CardRef ref = ps.active[0];
      return {1, is_me, pokemon_pos::kActive, card_area::kActive, 0, id_of(ref)};
    }
    if (area == static_cast<int>(AreaType::Bench)) {
      CardRef ref = ps.bench[index];
      return {1, is_me, pokemon_pos::Bench(index), card_area::kBench, 0, id_of(ref)};
    }
    if (area == static_cast<int>(AreaType::Stadium)) {
      CardRef ref = state_.stadium[index];
      return {1, is_me, pokemon_pos::kNone, card_area::kStadium, index, id_of(ref)};
    }
    if (area == static_cast<int>(AreaType::Looking)) {
      // Can genuinely be masked -- Redeemable Ticket's forced "select all as
      // new Prizes" is the only select that does this, and Phase 2.5's
      // bypass logic auto-resolves every forced-full-select before this
      // encoder ever runs (see encode.py's own docstring on this). Kept as
      // a graceful fallback rather than an assumption, same as encode.py.
      if (!LookingVisible()) {
        return kNoCardPtr;
      }
      CardRef ref = state_.looking[index];
      return {1, is_me, pokemon_pos::kNone, card_area::kLooking, index, id_of(ref)};
    }
    if (area == static_cast<int>(AreaType::Deck)) {
      // select.deck is always the *selecting* player's own deck (see
      // SelectJson) -- your_index_ here, not player_index (which for this
      // option type is unused/redundant with the fact select.deck has no
      // separate playerIndex concept).
      CardRef ref = state_.players[your_index_].deck[index];
      return {1, is_me, pokemon_pos::kNone, card_area::kDeck, index, id_of(ref)};
    }
    // Unresolvable area for a direct card pointer -- matches encode.py's
    // warn-and-fall-back-to-_NO_CARD path (engine effect defs changed in a
    // way this port didn't anticipate, if this is ever actually hit).
    return kNoCardPtr;
  }

  CardPtr ResolveAttached(int area, int index, int player_index, int sub_index, int kind) {
    int is_me = (player_index == your_index_) ? 1 : 0;
    const auto& ps = state_.players[player_index];
    CardRef pk_ref = (area == static_cast<int>(AreaType::Active)) ? ps.active[0] : ps.bench[index];
    const Card& pk_card = state_.getCard(pk_ref);
    int pos = (area == static_cast<int>(AreaType::Active)) ? pokemon_pos::kActive
                                                            : pokemon_pos::Bench(index);
    CardRef attached_ref;
    if (kind == card_area::kTool) {
      auto tools = state_.getAttachedToolRef(pk_card);
      attached_ref = tools[sub_index];
    } else {
      std::vector<CardRef> energy_cards;
      state_.getEnergyCards(pk_ref, energy_cards);
      attached_ref = energy_cards[sub_index];
    }
    int card_id = state_.getCard(attached_ref).getMaster().cardId;
    return {1, is_me, pos, kind, sub_index, card_id};
  }

  int OwningPlayerOfOwnAction(int area, int index) {
    // ABILITY/DISCARD's (area, index) is always your own card, except
    // Stadium: either player can use/discard it regardless of who played
    // it, so its owner comes from the card itself, not the selecting
    // player (see encode.py's comment + regression test 4921468).
    if (area == static_cast<int>(AreaType::Stadium)) {
      return state_.getCard(state_.stadium[index]).playerIndex;
    }
    return your_index_;
  }

  PokemonPtr ResolveInPlay(int area, int index) {
    int pos = (area == static_cast<int>(AreaType::Active)) ? pokemon_pos::kActive
                                                            : pokemon_pos::Bench(index);
    return {1, 1, pos};
  }

  // --- select vector ---

  void WriteSelect(const Array& select_arr) {
    CardPtr ctx_ptr = kNoCardPtr;
    if (!state_.contextCard.isNull()) {
      CardRef ref = state_.contextCard;
      ctx_ptr = ResolveBySerial(ref.cardIndex, state_.getCard(ref).getMaster().cardId);
    }
    CardPtr eff_ptr = kNoCardPtr;
    if (state_.onEffect()) {
      CardRef ref = state_.getEffectCard().card;
      eff_ptr = ResolveBySerial(ref.cardIndex, state_.getCard(ref).getMaster().cardId);
    }
    // type, context, min_count, max_count, remain_damage_counter, remain_energy_cost,
    // context_card_{valid,is_me,pokemon_pos,area,pos_in_area,id},
    // effect_card_{valid,is_me,pokemon_pos,area,pos_in_area,id}
    //
    // SelectType/SelectContext are -1 from their C++ enum values to land on
    // schema.py's 0-based SelectType/SelectContext IntEnums (C++'s None=0
    // has no Python equivalent -- confirmed value-for-value against
    // ApiType.h and cg/api.py; see ToJson.h's SelectJson doing the same
    // `max(0, (int)x - 1)` for the (non-web) API path).
    WriteRow(static_cast<int*>(select_arr.Data()), {
      std::max(0, static_cast<int>(state_.selectType) - 1),
      std::max(0, static_cast<int>(state_.selectContext) - 1),
      state_.selectMin,
      state_.selectMax,
      state_.remainDamageCounter,
      state_.remainEnergyCost,
      ctx_ptr.valid, ctx_ptr.is_me, ctx_ptr.pokemon_pos, ctx_ptr.area, ctx_ptr.pos_in_area, ctx_ptr.id,
      eff_ptr.valid, eff_ptr.is_me, eff_ptr.pokemon_pos, eff_ptr.area, eff_ptr.pos_in_area, eff_ptr.id,
    });
  }

  // --- options tensor ---

  void EncodeOption(const SelectOption& opt, int* row_arr) {
    int number = 0;
    int number_valid = 0;
    int count = 0;
    int count_valid = 0;
    int sct = 0;
    int sct_valid = 0;
    int attack_id = 0;
    int attack_id_valid = 0;
    CardPtr card = kNoCardPtr;
    PokemonPtr pokemon = kNoPokemonPtr;

    SelectOptionType ot = opt.type;
    // number/count/special_condition_type are only ever populated by the
    // engine for their one matching SelectOptionType (see
    // ApiJson.h's SelectOptionJson) -- generic top-level extraction here
    // mirrors encode.py's own generic (opt.number is not None)-style checks,
    // just keyed on `ot` directly since there's no "None" sentinel on a raw
    // `short` param the way Python's Optional[int] has one.
    if (ot == SelectOptionType::Number) {
      number = opt.param0;
      number_valid = 1;
    } else if (ot == SelectOptionType::SpecialCondition) {
      sct = opt.param0;
      sct_valid = 1;
    } else if (ot == SelectOptionType::Energy) {
      count = opt.param4;
      count_valid = 1;
    }

    switch (ot) {
      case SelectOptionType::Number:
      case SelectOptionType::Yes:
      case SelectOptionType::No:
      case SelectOptionType::Retreat:
      case SelectOptionType::End:
      case SelectOptionType::SpecialCondition:
        break;
      case SelectOptionType::Card:
        card = ResolveDirect(opt.param0, opt.param1, opt.param2);
        break;
      case SelectOptionType::ToolCard:
        card = ResolveAttached(opt.param0, opt.param1, opt.param2, opt.param3, card_area::kTool);
        break;
      case SelectOptionType::EnergyCard:
      case SelectOptionType::Energy:
        card = ResolveAttached(opt.param0, opt.param1, opt.param2, opt.param3, card_area::kEnergy);
        break;
      case SelectOptionType::Play: {
        CardRef ref = state_.players[your_index_].hand[opt.param0];
        card = {1, 1, pokemon_pos::kNone, card_area::kHand, opt.param0,
                state_.getCard(ref).getMaster().cardId};
        break;
      }
      case SelectOptionType::Attach:
      case SelectOptionType::Evolve:
        // the card being attached/evolved, and the in-play Pokemon it targets.
        card = ResolveDirect(opt.param0, opt.param1, your_index_);
        pokemon = ResolveInPlay(opt.param2, opt.param3);
        break;
      case SelectOptionType::Ability:
      case SelectOptionType::Discard:
        card = ResolveDirect(opt.param0, opt.param1, OwningPlayerOfOwnAction(opt.param0, opt.param1));
        break;
      case SelectOptionType::Attack:
        attack_id = opt.param0;
        attack_id_valid = 1;
        break;
      case SelectOptionType::Skill:
        card = ResolveBySerial(opt.param1, opt.param0);  // (serial, cardId)
        break;
      default:
        // Unrecognized SelectOptionType -- matches encode.py's
        // warn-and-leave-bare fallback (is_valid/type still get written
        // below; nothing else does).
        break;
    }

    // is_valid, type, number, number_valid, count, count_valid,
    // special_condition_type, special_condition_type_valid, attack_id, attack_id_valid,
    // card_{valid,is_me,pokemon_pos,area,pos_in_area,id},
    // pokemon_{valid,is_me,pos}
    WriteRow(row_arr, {
      1, static_cast<int>(ot),
      number, number_valid,
      count, count_valid,
      sct, sct_valid,
      attack_id, attack_id_valid,
      card.valid, card.is_me, card.pokemon_pos, card.area, card.pos_in_area, card.id,
      pokemon.valid, pokemon.is_me, pokemon.pos,
    });
  }

  void WriteOptions(const Array& options_arr) {
    auto* base = static_cast<int*>(options_arr.Data());
    int n = std::min(static_cast<int>(state_.options.size()), kMaxOptions);
    for (int i = 0; i < n; ++i) {
      EncodeOption(state_.options[i], base + i * kOptionsCols);
    }
    // Rows [n, kMaxOptions) stay at the Zero()'d default -- see WriteCards's
    // closing comment; same caller contract.
  }
};

}  // namespace detail

// Public entry point. `state` must be a genuine, non-terminal, non-bypassed
// decision point (state.selectType != SelectType::None, state.options
// non-empty) -- exactly encode_observation()'s own documented precondition.
// `my_deck` is the persistent per-seat deck cache (PtcgEnv's deck0_/deck1_),
// standing in for production's `get_deck()`/`@lru_cache` (see "Persistent
// per-seat deck cache" in the skill).
inline void EncodeObservation(const State& state, const std::array<int, kActionSlots>& my_deck,
                               const Array& cards_arr, const Array& pokemons_arr,
                               const Array& player_state_arr, const Array& state_arr,
                               const Array& select_arr, const Array& options_arr) {
  detail::ObservationEncoder(state, my_deck)
      .Encode(cards_arr, pokemons_arr, player_state_arr, state_arr, select_arr, options_arr);
}

}  // namespace ptcg

#endif  // ENVPOOL_PTCG_PTCG_ENCODE_H_
