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
        "obs:options"_.Bind(Spec<int>({kOptionRows, kOptionsCols})),
        // Routing metadata, not part of the 6 encoded tensors above -- see
        // "Observation space" in the envpool-ptcg-integration skill.
        "info:current_player"_.Bind(Spec<int>({}, {0, 1})),
        "info:is_deck_select"_.Bind(Spec<bool>({})),
        // Raw State.h FinishReason cast to int: None=0, Prize0=1, Deck0=2,
        // NoActivePokemon=3, Effect=4, Other=9. Only meaningful on a row
        // where `terminated` came from the engine's own state.isFinish()
        // (see ResolveBypassedSelectsThenRespond) -- stays 0 on every other
        // row, including the envpool-side illegal-action instant-loss
        // terminations, which never run the engine's real finishCheck().
        "info:finish_reason"_.Bind(Spec<int>({}, {0, 9})));
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
 * Phases 2-4 (envpool-ptcg-integration skill) are all wired in, plus a
 * later single-index-action redesign (see CLAUDE.md's "Current focus" /
 * schema.py's STOP_SLOT/N_OPTION_SLOTS comment):
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
 *  - decide()'s `action` is a single index per Step() call (only slot 0 of
 *    the 60-slot action vector is read; the rest are ignored -- the vector
 *    stays 60-wide only because deck-select needs it), not the up-to-
 *    maxCount padded list Phase 4 originally used: 0..kMaxOptions-1 picks
 *    a real state.options row, kStopSlot submits the accumulated selection
 *    (chosen_) as final. A decision needing multiple picks is therefore
 *    multiple genuine Step() calls, not one call carrying multiple indices
 *    -- see LegalActions()/AutoResolveForcedSubPicksThenRespond()/
 *    FinalizeSelection() below. Any sub-pick where LegalActions() collapses
 *    to exactly one option (e.g. only one valid card remains, or selectMax
 *    was just reached so only STOP is legal) is auto-resolved internally,
 *    generalizing Phase 2.5's whole-decision bypass to intra-decision
 *    sub-picks -- Python only ever sees a genuine multi-way choice.
 *    Illegal actions (anything LegalActions() doesn't contain) apply the
 *    same instant-loss-for-the-offender reward pattern
 *    board_games::IllegalRewards uses (envpool/pgx/board_games.h), adapted
 *    to this env's single-reward-per-step shape (see "Reward" in the
 *    skill) instead of that pattern's dual-player broadcast, which doesn't
 *    fit here; deck-select's illegal-deck case uses the identical pattern.
 */
class PtcgEnv : public Env<PtcgEnvSpec> {
 protected:
  bool done_{true};
  bool is_deck_select_{true};
  int current_player_{0};
  int finish_reason_{0};
  ApiData* battle_{nullptr};
  std::array<int, kActionSlots> deck0_{};
  std::array<int, kActionSlots> deck1_{};
  // Real state.options-row indices already picked earlier in the current
  // decide()-type decision -- reset to empty every time a fresh, non-
  // bypassed decide() prompt begins (see ResolveBypassedSelectsThenRespond),
  // accumulated across however many Step() calls that decision takes, and
  // submitted to the real ApiSelect exactly once, at FinalizeSelection().
  std::vector<int> chosen_;

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
    finish_reason_ = 0;
    chosen_.clear();
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
      finish_reason_ = 0;
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
      finish_reason_ = 0;
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
    int a = action["action"_][0];
    std::vector<int> legal = LegalActions();
    if (std::find(legal.begin(), legal.end(), a) == legal.end()) {
      // Illegal action -> instant loss for the offender (state.selectPlayer,
      // == current_player_, unambiguous here unlike deck-select's
      // errorPlayer indirection). ApiSelect is never even called on this
      // path (validated against LegalActions() first), so battle_->state
      // is completely unchanged -- no SyncFromEngine() call, and WriteState
      // correctly Zero()s the obs rather than re-encoding a decision that
      // never advanced.
      done_ = true;
      finish_reason_ = 0;
      WriteState(-1.0F);
      return;
    }
    if (a == kStopSlot) {
      FinalizeSelection();
      return;
    }
    chosen_.push_back(a);
    AutoResolveForcedSubPicksThenRespond();
  }

  // Single source of truth for "what actions are legal right now", given
  // battle_->state and chosen_ -- used both to validate an incoming Python
  // action (StepDecide) and to drive the auto-resolve loop below. A real
  // option row i is legal iff it's one of the actual state.options
  // (i < n_real), not already picked, and selectMax hasn't been reached
  // yet; kStopSlot is legal iff selectMin has been satisfied. Mirrors
  // encode.py's/ptcg_encode.h's STOP-row is_valid computation exactly (see
  // schema.py's STOP_SLOT comment) -- same rule, read here for control flow
  // instead of written into an observation.
  std::vector<int> LegalActions() const {
    const auto& state = battle_->state;
    int n_real = std::min(static_cast<int>(state.options.size()), kMaxOptions);
    std::vector<int> legal;
    if (static_cast<int>(chosen_.size()) < state.selectMax) {
      for (int i = 0; i < n_real; ++i) {
        if (std::find(chosen_.begin(), chosen_.end(), i) == chosen_.end()) {
          legal.push_back(i);
        }
      }
    }
    if (static_cast<int>(chosen_.size()) >= state.selectMin) {
      legal.push_back(kStopSlot);
    }
    return legal;
  }

  // Runs from the current chosen_ state (either just-appended-by-Python in
  // StepDecide, or freshly reset for a new decide()-type prompt in
  // ResolveBypassedSelectsThenRespond): auto-advances through any round
  // where LegalActions() collapses to exactly one option -- generalizes
  // the forced-full-select/prize-select whole-decision bypass
  // (IsBypassedSelect) to intra-decision sub-picks (e.g. "only one valid
  // card remains" or "selectMax was just reached, only STOP is legal"), so
  // Python only ever sees a genuine multi-way decide() step, never a
  // deterministic one. Finalizes (real ApiSelect + recurse into
  // ResolveBypassedSelectsThenRespond) the instant that happens; otherwise
  // presents the resulting decision with WriteState(0.0F).
  void AutoResolveForcedSubPicksThenRespond() {
    while (true) {
      std::vector<int> legal = LegalActions();
      CHECK(!legal.empty())
          << "no legal action at a decide() sub-pick -- engine invariant "
             "violated (selectMin exceeds available options?)";
      if (legal.size() != 1) {
        break;
      }
      if (legal[0] == kStopSlot) {
        FinalizeSelection();
        return;
      }
      chosen_.push_back(legal[0]);
    }
    SyncFromEngine();
    WriteState(0.0F);
  }

  // Submits chosen_ as the final answer to this decide()-type selection.
  // Every constituent pick was already validated against LegalActions()
  // before being added to chosen_ (either by StepDecide, for a genuine
  // Python-provided pick, or by AutoResolveForcedSubPicksThenRespond, for
  // an auto-resolved one), so ApiSelect itself should never reject it --
  // CHECK_EQ (not a penalized runtime path) mirrors
  // ResolveBypassedSelectsThenRespond's own bypass-selection assertion:
  // a failure here means LegalActions() has drifted from the engine's real
  // state.checkPlayerSelect() rule, a bug to fix, not a real game outcome.
  void FinalizeSelection() {
    int error =
        ApiSelect(battle_, chosen_.data(), static_cast<int>(chosen_.size()));
    CHECK_EQ(error, 0)
        << "PtcgEnv's own tracked selection was rejected by ApiSelect -- "
           "LegalActions() has drifted from the engine's real legality rule";
    ResolveBypassedSelectsThenRespond();
  }

  // Runs ApiSelect's own bypass loop forward through any forced-full-select/
  // prize-select decision points (Phase 2.5, "Trajectory & prompt routing"),
  // then either hands off to a fresh decide()-type prompt's own
  // auto-resolve loop, or -- if the game ended instead -- computes the
  // terminal-only reward ("Reward" in the skill) and writes the response.
  // Shared by StepDeckSelect's post-ApiBattleStart path, StepDecide's
  // post-illegal-check path (via AutoResolveForcedSubPicksThenRespond /
  // FinalizeSelection), and FinalizeSelection itself -- deliberately, since
  // a just-started battle's first decision point could in principle itself
  // be bypassed (astronomically unlikely, but free to handle correctly by
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
    if (!battle_->state.isFinish()) {
      // A fresh, non-bypassed decide()-type prompt: reset the per-decision
      // pick accumulator and let AutoResolveForcedSubPicksThenRespond
      // either settle immediately (e.g. exactly one valid option with
      // selectMin==selectMax==1 -- this whole decision has only one legal
      // path start-to-finish) or present a genuine choice.
      finish_reason_ = 0;
      chosen_.clear();
      AutoResolveForcedSubPicksThenRespond();
      return;
    }
    // "Reward" in the skill: state.apiResult() (-1/0/1/2), not a raw
    // GameResult cast. For *that step's player* (state.selectPlayer):
    // +1.0 if apiResult()==selectPlayer, -1.0 if it's the other player's
    // index, 0.0 on draw.
    float reward = 0.0F;
    int result = battle_->state.apiResult();
    int actor = battle_->state.selectPlayer;
    if (result != 2) {
      reward = (result == actor) ? 1.0F : -1.0F;
    }
    // Real engine-side finish -- state.finishCheck() has already run
    // (called internally by ApiSelect/data->next()) and set finishReason,
    // so this is the one place a non-zero value is ever recorded.
    finish_reason_ = static_cast<int>(battle_->state.finishReason);
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
    // randomness for production parity, unlike LegalActions()-driven
    // auto-resolution above, which has no such parity concern (it only
    // ever narrows to a single legal choice, nothing to randomize).
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
      EncodeObservation(battle_->state, deck, chosen_, state["obs:cards"_],
                        state["obs:pokemons"_], state["obs:player_state"_],
                        state["obs:state"_], state["obs:select"_],
                        state["obs:options"_]);
    }
    state["info:current_player"_] = current_player_;
    state["info:is_deck_select"_] = is_deck_select_;
    state["info:finish_reason"_] = finish_reason_;
    state["reward"_] = reward;
  }
};

using PtcgEnvPool = AsyncEnvPool<PtcgEnv>;

}  // namespace ptcg

#endif  // ENVPOOL_PTCG_PTCG_ENVPOOL_H_
