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
#include <vector>

#include "absl/log/check.h"

#include "All.h"

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/ptcg/ptcg_encode.h"
#include "envpool/ptcg/ptcg_schema.h"

namespace ptcg {

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

// Phase 2.5 (envpool-ptcg-integration skill): two decision types the engine
// can present that must never reach Python -- see "Trajectory & prompt
// routing". Free functions (only need `state`, no PtcgEnv instance state),
// mirroring tests/support/engine_harness.py's
// is_forced_select/is_prize_select/is_bypassed_select -- an independent
// mirror of the same checks in the same order, used as a cross-check while
// writing this.
inline bool IsForcedFullSelect(const State& state) {
  return state.selectMin == state.selectMax &&
         state.selectMax == static_cast<int>(state.options.size());
}

inline bool IsPrizeSelect(const State& state) {
  return !state.options.empty() &&
         state.options[0].getCardPosition().area == AreaType::Prize;
}

inline bool IsBypassedSelect(const State& state) {
  return IsForcedFullSelect(state) || IsPrizeSelect(state);
}

/**
 * Phases 2-4 (envpool-ptcg-integration skill) are all wired in:
 *  - Phase 2: real Reset()/Step() flow -- 2-step deck-select handshake into
 *    ApiBattleStart, deviceRand=false throughput fix, real ApiSelect
 *    stepping, IsDone() via state.isFinish().
 *  - Phase 2.5: forced-full-select and prize-select are auto-resolved
 *    inside ResolveBypassedSelectsThenRespond()'s loop -- see "Trajectory &
 *    prompt routing" -- so they never reach Python as their own decision
 *    point, matching production main.py:agent()'s exact behavior.
 *  - Phase 3: decide()-step observations are real encoder output
 *    (ptcg_encode.h's EncodeObservation, a C++ port of encode.py), not
 *    Zero()'d placeholders. Deck-select steps (and the terminal step -- see
 *    WriteState) still Zero(), since there's no board/decision to encode.
 *  - Phase 4: decide()'s `action` is genuinely interpreted now (illegal-vs-
 *    under-minCount tiering, "Action space" in the skill) instead of Phase
 *    2's self-driven random-legal-pick stand-in; deck-select's illegal-deck
 *    case and decide()'s illegal-action case both apply the same instant-
 *    loss-for-the-offender reward pattern board_games::IllegalRewards uses
 *    (envpool/pgx/board_games.h), adapted to this env's single-reward-per-
 *    step shape (see "Reward" in the skill) instead of that pattern's dual-
 *    player broadcast, which doesn't fit here.
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
      StepDecide(action);
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
    if (start.battlePtr == nullptr) {
      // Illegal deck -> instant loss for whichever seat ApiBattleStart
      // named, mirroring board_games::IllegalRewards's pattern (see the
      // class comment) -- "Action space" in the skill: "An invalid deck
      // gets the illegal-action penalty, attributed via errorPlayer." Note
      // errorPlayer isn't necessarily current_player_ (==1 here): a bad
      // seat-0 deck submitted on the *previous* Step() call only surfaces
      // once ApiBattleStart actually runs, on this one.
      current_player_ = start.errorPlayer;
      done_ = true;
      WriteState(-1.0F);
      return;
    }
    battle_ = start.battlePtr;
    // Throughput fix -- see "Architecture at a glance" / "ptcg_engine C++
    // surface to call directly" in the skill. Must happen before any
    // ApiSelect call; ApiBattleStart hardcodes deviceRand=true internally
    // with no override parameter, but GameConfig is read live at every use
    // site, so mutating it here still takes effect for the rest of the
    // battle.
    battle_->game.config.deviceRand = false;

    is_deck_select_ = false;
    ResolveBypassedSelectsThenRespond();
  }

  void StepDecide(const Action& action) {
    auto& state = battle_->state;
    std::vector<int> chosen;
    for (int i = 0; i < kActionSlots; ++i) {
      int v = action["action"_][i];
      if (v != -1) {
        chosen.push_back(v);
      }
    }
    if (static_cast<int>(chosen.size()) < state.selectMin) {
      // Under minCount -> pad, don't penalize: production explicitly
      // tolerates and pads short answers with random remaining valid
      // indices ("Action space" in the skill, mirroring
      // main.py:agent()'s own tolerant behavior). gen_ (Env<Spec>'s
      // per-env RNG) is fine here -- unlike Phase 2.5's prize-select bypass,
      // there's no production-RNG-parity concern: the policy already made
      // its intentional picks, this only fills in blanks it declined to
      // specify.
      std::vector<int> remaining;
      for (int i = 0; i < static_cast<int>(state.options.size()); ++i) {
        if (std::find(chosen.begin(), chosen.end(), i) == chosen.end()) {
          remaining.push_back(i);
        }
      }
      std::shuffle(remaining.begin(), remaining.end(), gen_);
      int need = state.selectMin - static_cast<int>(chosen.size());
      for (int i = 0; i < need && i < static_cast<int>(remaining.size());
           ++i) {
        chosen.push_back(remaining[i]);
      }
    }

    // Everything else -- out-of-range, exceeds maxCount, duplicate -- is
    // left for the engine's own state.checkPlayerSelect() (State.h), run at
    // the top of ApiSelect, rather than re-derived by hand here: reusing
    // the authoritative check mirrors deck-select's "legality is whatever
    // ApiBattleStart itself enforces" philosophy ("Action space" in the
    // skill), and checkPlayerSelect()'s error codes are exactly the
    // illegal-vs-not distinction that section describes.
    int error =
        ApiSelect(battle_, chosen.data(), static_cast<int>(chosen.size()));
    if (error != 0) {
      // Illegal action -> instant loss for the offender (state.selectPlayer,
      // == current_player_, unambiguous here unlike deck-select's
      // errorPlayer indirection). ApiSelect returns before calling
      // data->next() on any error (Api.h), so battle_->state is completely
      // unchanged -- no SyncFromEngine() call, and WriteState correctly
      // Zero()s the obs rather than re-encoding a decision that never
      // advanced.
      done_ = true;
      WriteState(-1.0F);
      return;
    }
    ResolveBypassedSelectsThenRespond();
  }

  // Runs ApiSelect's own bypass loop forward through any forced-full-select/
  // prize-select decision points (Phase 2.5, "Trajectory & prompt routing"),
  // then computes the terminal-only reward ("Reward" in the skill) and
  // writes the response. Shared by both StepDeckSelect's post-ApiBattleStart
  // path and StepDecide's post-ApiSelect path -- deliberately, since a
  // just-started battle's first decision point could in principle itself be
  // bypassed (astronomically unlikely, but free to handle correctly by
  // reusing this rather than special-casing it away).
  void ResolveBypassedSelectsThenRespond() {
    while (!battle_->state.isFinish() && IsBypassedSelect(battle_->state)) {
      std::vector<int> bypass = ComputeBypassSelection();
      int error = ApiSelect(battle_, bypass.data(),
                            static_cast<int>(bypass.size()));
      CHECK_EQ(error, 0)
          << "bypass-computed selection was rejected by ApiSelect -- bug in "
             "the Phase 2.5 bypass logic, not a real scenario";
    }
    float reward = 0.0F;
    if (battle_->state.isFinish()) {
      // "Reward" in the skill: state.apiResult() (-1/0/1/2), not a raw
      // GameResult cast. For *that step's player* (state.selectPlayer):
      // +1.0 if apiResult()==selectPlayer, -1.0 if it's the other player's
      // index, 0.0 on draw.
      int result = battle_->state.apiResult();
      int actor = battle_->state.selectPlayer;
      if (result != 2) {
        reward = (result == actor) ? 1.0F : -1.0F;
      }
    }
    SyncFromEngine();
    WriteState(reward);
  }

  std::vector<int> ComputeBypassSelection() {
    auto& state = battle_->state;
    if (IsForcedFullSelect(state)) {
      std::vector<int> all(state.options.size());
      std::iota(all.begin(), all.end(), 0);
      return all;
    }
    // Prize-select: uniform random sample of maxCount indices, using the
    // battle's own per-instance RNG (battle_->game.rng) rather than gen_
    // (Env<Spec>'s per-env RNG) -- "Trajectory & prompt routing" in the
    // skill: this needs to be consistent with the rest of *that battle's*
    // randomness for production parity, unlike StepDecide's under-minCount
    // padding above, which has no such parity concern.
    std::vector<int> indices(state.options.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), battle_->game.rng);
    indices.resize(state.selectMax);
    return indices;
  }

  void SyncFromEngine() {
    const auto& state = battle_->state;
    current_player_ = state.selectPlayer;
    done_ = state.isFinish();
  }

  void WriteState(float reward) {
    auto state = Allocate();
    // Not a free Allocate() default -- StateBuffer's backing arrays are a
    // ring buffer allocated once per AsyncEnvPool and zero-initialized only
    // at that one construction; every later Allocate() call hands back a
    // *reused* slice, still holding whatever a previous env/step wrote
    // there. Explicit Zero() is required unconditionally here, even though
    // EncodeObservation below overwrites most of it in the common case --
    // it never writes 100% of every tensor (pokemons rows with no Pokemon
    // there, options rows past the real option count, ...), and skipping
    // Zero() would leak whatever a prior ring-buffer occupant last wrote
    // into exactly those gaps.
    state["obs:cards"_].Zero();
    state["obs:pokemons"_].Zero();
    state["obs:player_state"_].Zero();
    state["obs:state"_].Zero();
    state["obs:select"_].Zero();
    state["obs:options"_].Zero();
    if (!is_deck_select_ && !done_) {
      // Real encoder output only for a genuine, non-terminal, non-bypassed
      // decide() step -- exactly encode_observation()'s own documented
      // precondition. Deck-select steps have no board to encode; the
      // terminal step is intentionally left Zero()'d too (production never
      // calls encode_observation() in an already-finished game either, so
      // there's no ground truth to match, and state.selectType may already
      // be cleared by then -- see ptcg_encode.h's EncodeObservation doc
      // comment).
      const auto& deck = current_player_ == 0 ? deck0_ : deck1_;
      EncodeObservation(battle_->state, deck, state["obs:cards"_],
                        state["obs:pokemons"_], state["obs:player_state"_],
                        state["obs:state"_], state["obs:select"_],
                        state["obs:options"_]);
    }
    state["info:current_player"_] = current_player_;
    state["info:is_deck_select"_] = is_deck_select_;
    state["reward"_] = reward;
  }
};

using PtcgEnvPool = AsyncEnvPool<PtcgEnv>;

}  // namespace ptcg

#endif  // ENVPOOL_PTCG_PTCG_ENVPOOL_H_
