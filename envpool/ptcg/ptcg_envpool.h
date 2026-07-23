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

#ifndef ENVPOOL_PTCG_PTCG_ENVPOOL_H_
#define ENVPOOL_PTCG_PTCG_ENVPOOL_H_

#include <algorithm>
#include <array>
#include <mutex>
#include <numeric>
#include <random>
#include <vector>

#include "absl/log/check.h"

#include "All.h"

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

namespace ptcg {

// EncodedObservation shape constants, mirrored from
// submissions/template_submission/schema.py:61-67. Keep in sync by hand --
// there is no shared codegen between the Python and C++ sides (see the
// envpool-ptcg-integration skill's "EncodedObservation schema" section).
constexpr int kMaxCards = 120;
constexpr int kMaxPokemon = 18;
constexpr int kMaxOptions = 63;
constexpr int kNCardIds = 1267;

constexpr int kCardsCols = 5;
constexpr int kPokemonsCols = 15;
constexpr int kPlayerStateRows = 2;
constexpr int kPlayerStateCols = 9;
constexpr int kStateCols = 8;
constexpr int kSelectCols = 16;
constexpr int kOptionsCols = 19;

// Deck-select and decide() share one action shape (see "Action space" in the
// envpool-ptcg-integration skill): 60 slots, all real values (card ids) for
// deck-select, up to maxCount real option-row indices + -1 padding for
// decide().
constexpr int kActionSlots = 60;

static_assert(kActionSlots == DECK_SIZE,
              "action space's 60 deck-select slots must equal the engine's "
              "own deck size (ptcg_engine/Core.h's DECK_SIZE) -- deck-select "
              "hands the action's slots to ApiBattleStart verbatim, see "
              "'Action space' in the envpool-ptcg-integration skill");

class PtcgEnvFns {
 public:
  static decltype(auto) DefaultConfig() { return MakeDict(); }

  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict(
        "obs:cards"_.Bind(Spec<int>({kMaxCards, kCardsCols})),
        "obs:pokemons"_.Bind(Spec<int>({kMaxPokemon, kPokemonsCols})),
        "obs:player_state"_.Bind(
            Spec<int>({kPlayerStateRows, kPlayerStateCols})),
        "obs:state"_.Bind(Spec<int>({kStateCols})),
        "obs:select"_.Bind(Spec<int>({kSelectCols})),
        "obs:options"_.Bind(Spec<int>({kMaxOptions, kOptionsCols})),
        // Routing metadata, not part of the 6 encoded tensors above -- see
        // "Observation space" in the envpool-ptcg-integration skill.
        "info:current_player"_.Bind(Spec<int>({}, {0, 1})),
        "info:is_deck_select"_.Bind(Spec<bool>({})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict(
        "action"_.Bind(Spec<int>({kActionSlots}, {-1, kNCardIds - 1})));
  }
};

using PtcgEnvSpec = EnvSpec<PtcgEnvFns>;

/**
 * Phase 2 (envpool-ptcg-integration skill): real ptcg_engine calls.
 * Reset()/Step() drive the actual 2-step deck-select handshake into
 * ApiBattleStart, then real ApiSelect stepping until state.isFinish() --
 * see "Deck-select sequencing" and "ptcg_engine C++ surface to call
 * directly" in the skill.
 *
 * Two things are deliberately still stubbed, each named as its own later
 * phase in the skill, not oversights:
 *  - decide() steps don't interpret `action`'s content yet (the
 *    legal-vs-illegal / under-minCount-padding tiering is Phase 4). Step()
 *    self-drives with a uniformly random legal-sized, legal-index pick from
 *    the live engine state instead, using this env's own gen_ (Env<Spec>'s
 *    per-env RNG). That proves the real ApiSelect/IsDone sequencing end to
 *    end without depending on Phase 3's observation encoding to exist first
 *    (there'd otherwise be no way for an external caller to know what
 *    `options` even means well enough to choose from it).
 *  - the 6 obs tensors stay all-zero for every step, decide() included, and
 *    reward stays 0.0 always including the terminal step -- Phase 3 ports
 *    the real per-field encoder, Phase 4 adds the apiResult()-derived
 *    +1/-1/0 terminal reward rule.
 *
 * Deck-select *is* fully real already: the 60-slot action is read verbatim
 * as that seat's deck and handed to ApiBattleStart, since (per "Action
 * space" in the skill) that has no legal/illegal-vs-padding tiering to defer
 * -- legality there is just whatever ApiBattleStart itself enforces.
 */
class PtcgEnv : public Env<PtcgEnvSpec> {
 protected:
  bool done_{true};
  bool is_deck_select_{true};
  int current_player_{0};
  ApiData* battle_{nullptr};
  std::array<int, kActionSlots> deck0_{};
  std::array<int, kActionSlots> deck1_{};

 public:
  PtcgEnv(const Spec& spec, int env_id) : Env<PtcgEnvSpec>(spec, env_id) {
    // InitializeAll() asserts CardTable.size() == 0 on entry (All.h:15).
    // AsyncEnvPool builds one PtcgEnv per env slot from a ThreadPool (see
    // async_envpool.h's constructor: `init_pool.enqueue([i, this] {
    // envs_[i].reset(new Env(...)); })` for every i) -- these constructor
    // calls genuinely race across real worker threads, not just
    // hypothetically, so this must run exactly once *total* via a
    // synchronizing guard, not once per env. A function-local static
    // once_flag is shared across every call to this constructor and is
    // thread-safe to initialize per the C++11 guarantee on function-local
    // statics -- see "Concurrency safety" in the skill.
    static std::once_flag init_flag;
    std::call_once(init_flag, InitializeAll);
  }

  ~PtcgEnv() override {
    if (battle_ != nullptr) {
      ApiBattleFinish(battle_);
    }
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    if (battle_ != nullptr) {
      // AsyncEnvPool reuses this same PtcgEnv instance across episodes in
      // this env slot -- the worker loop calls Reset() instead of Step()
      // whenever IsDone() is true (async_envpool.h:127), it never
      // reconstructs the object. Without this, every episode after the
      // first in this slot leaks the previous battle_.
      ApiBattleFinish(battle_);
      battle_ = nullptr;
    }
    done_ = false;
    is_deck_select_ = true;
    current_player_ = 0;
    WriteState(0.0F);
  }

  void Step(const Action& action) override {
    if (is_deck_select_) {
      StepDeckSelect(action);
    } else {
      StepDecide();
    }
  }

 private:
  void StepDeckSelect(const Action& action) {
    // "Deck-select sequencing" in the skill: all 60 slots are real card ids,
    // no -1 padding -- legality is whatever ApiBattleStart enforces, so this
    // is a verbatim copy, not a filtered/validated read. current_player_
    // still holds "whose deck this is" (set by the previous WriteState),
    // read here before this call updates it.
    std::array<int, kActionSlots>& deck =
        current_player_ == 0 ? deck0_ : deck1_;
    for (int i = 0; i < kActionSlots; ++i) {
      deck[i] = action["action"_][i];
    }

    if (current_player_ == 0) {
      current_player_ = 1;
      WriteState(0.0F);
      return;
    }

    std::array<int, 2 * kActionSlots> cards{};
    std::copy(deck0_.begin(), deck0_.end(), cards.begin());
    std::copy(deck1_.begin(), deck1_.end(), cards.begin() + kActionSlots);
    StartData start = ApiBattleStart(cards.data());
    // Phase 2's own test methodology always supplies two real legal decks
    // (the skill's Phase 2 checklist "random valid deck" means the *caller*
    // picks a deck that's valid, not that this env tolerates an invalid
    // one) -- hitting this means a test bug, not a real scenario to handle
    // gracefully yet. Phase 4 ("deck-legality check for deck-select")
    // replaces this CHECK with the real illegal-action penalty, attributed
    // via errorPlayer.
    CHECK(start.battlePtr != nullptr)
        << "ApiBattleStart rejected the Phase 2 test decks: errorPlayer="
        << start.errorPlayer << " errorType=" << start.errorType;
    battle_ = start.battlePtr;
    // Throughput fix -- see "Architecture at a glance" / "ptcg_engine C++
    // surface to call directly" in the skill. Must happen before any
    // ApiSelect call; ApiBattleStart hardcodes deviceRand=true internally
    // with no override parameter, but GameConfig is read live at every use
    // site, so mutating it here still takes effect for the rest of the
    // battle.
    battle_->game.config.deviceRand = false;

    is_deck_select_ = false;
    SyncFromEngine();
    WriteState(0.0F);
  }

  void StepDecide() {
    auto& state = battle_->state;
    // state.selectMin <= state.selectMax <= state.options.size() is already
    // guaranteed by the engine itself (State::step(), State.h) by the time
    // ApiSelect/ApiBattleStart return -- no extra clamping needed here.
    //
    // Real interpretation of `action` (legal-vs-illegal, under-minCount
    // padding) is Phase 4 -- see the class comment. This instead picks
    // uniformly among legal-sized, legal-index choices using gen_
    // (Env<Spec>'s own per-env RNG, seeded from config at construction --
    // distinct from the engine's own battle_->game.rng, which Phase 2.5's
    // prize-select bypass will use instead for production parity; no such
    // parity concern applies to this Phase 2 stand-in).
    std::uniform_int_distribution<int> count_dist(state.selectMin,
                                                   state.selectMax);
    int count = count_dist(gen_);
    std::vector<int> indices(state.options.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen_);
    indices.resize(count);

    int error = ApiSelect(battle_, indices.data(),
                           static_cast<int>(indices.size()));
    CHECK_EQ(error, 0)
        << "legal-but-dumb pick rejected by ApiSelect -- bug in the Phase 2 "
           "stand-in's selectMin/selectMax/options.size() bookkeeping, not "
           "a real scenario";

    SyncFromEngine();
    WriteState(0.0F);
  }

  void SyncFromEngine() {
    const auto& state = battle_->state;
    current_player_ = state.selectPlayer;
    done_ = state.isFinish();
  }

  void WriteState(float reward) {
    auto state = Allocate();
    // Both deck-select and (still-stubbed) decide steps leave these 6
    // tensors at all-zeros through Phase 2 -- Phase 3 is what starts writing
    // real per-field encoder output for decide() steps. NOT a free
    // Allocate() default either way: StateBuffer's backing arrays are a
    // ring buffer allocated once per AsyncEnvPool and zero-initialized only
    // at that one construction -- Allocate() hands back a *reused* slice on
    // every later call, still holding whatever a previous env/step wrote
    // there. Explicit Zero() is required on every write that doesn't fill a
    // field itself, or an observation can leak a stale, possibly-hidden
    // previous game's encoded board.
    state["obs:cards"_].Zero();
    state["obs:pokemons"_].Zero();
    state["obs:player_state"_].Zero();
    state["obs:state"_].Zero();
    state["obs:select"_].Zero();
    state["obs:options"_].Zero();
    state["info:current_player"_] = current_player_;
    state["info:is_deck_select"_] = is_deck_select_;
    state["reward"_] = reward;
  }
};

using PtcgEnvPool = AsyncEnvPool<PtcgEnv>;

}  // namespace ptcg

#endif  // ENVPOOL_PTCG_PTCG_ENVPOOL_H_
