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

// Non-engine finish_reason sentinel for this env's own round-cap timeout
// close-out (AutoResolveForcedSubPicksThenRespond) -- never produced by the
// engine itself (State.h's FinishReason enum is 0-9). Mirrors
// training/finish_reason.py's FINISH_REASON_TIMEOUT, which reserves this
// exact value outside the engine's own range for exactly this.
constexpr int kFinishReasonTimeout = -1;

class PtcgEnvFns {
 public:
  // `deck0s`/`deck1s`: one fixed 60-card-id deck pair *per env slot*,
  // flattened to length `num_envs * kDeckSize` each -- slot `env_id`'s pair
  // is `deck0s[env_id*kDeckSize : (env_id+1)*kDeckSize]` (and likewise for
  // deck1s), read once at that PtcgEnv's construction and fixed for the
  // whole pool's lifetime (every episode in a given slot replays the same
  // pair; different slots may hold entirely different pairs). Mirrors
  // envpool/core/env.h's per-env `env_seed` config (`ResolveSeed`'s
  // `env_seed.at(env_id)` slicing), generalized from one int per env to one
  // 60-int deck per env. This is what lets one `make("Ptcg-v0", num_envs=N,
  // ...)` call run N independently-paired matchups at once, rather than one
  // shared pair broadcast to every env -- see the training pipeline's deck
  // pairing scheme. Changing any pairing means re-`make()`-ing the pool
  // (cheap -- measured ~9-27ms even at num_envs=1024), not a runtime setter.
  //
  // `max_rounds`: per-episode cap on genuine (Python-visible) decisions --
  // see PtcgEnv::round_count_/max_rounds_ and
  // AutoResolveForcedSubPicksThenRespond. Default matches
  // training/config.py's RolloutConfig.max_rounds fallback; real training
  // runs always pass their own value explicitly.
  static decltype(auto) DefaultConfig() {
    return MakeDict("deck0s"_.Bind(std::vector<int>{}),
                     "deck1s"_.Bind(std::vector<int>{}),
                     "max_rounds"_.Bind(128));
  }

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
        // Who this step's `reward` belongs to: the player that SUBMITTED
        // the action being resolved, snapshotted before any mutation
        // (Step()'s `acting_player`, threaded through the whole resolve
        // chain). Distinct from info:current_player, which names whoever
        // decides the *returned* observation (who's next, not who just
        // acted) -- the two coincide only on a terminal row, where there's
        // no "next" to advance to. -1 only on the very first observation
        // of a fresh episode (Reset()'s own response, before any action
        // has ever been submitted); always 0/1 on every other row. See
        // "Reward" in the envpool-ptcg-integration skill.
        "info:reward_player"_.Bind(Spec<int>({}, {-1, 1})),
        // Raw State.h FinishReason cast to int: None=0, Prize0=1, Deck0=2,
        // NoActivePokemon=3, Effect=4, Other=9. Only meaningful on a row
        // where `terminated` came from the engine's own state.isFinish()
        // (see ResolveBypassedSelectsThenRespond) -- stays 0 on every other
        // row, including the envpool-side illegal-action instant-loss
        // terminations, which never run the engine's real finishCheck().
        // kFinishReasonTimeout (-1) is the one non-engine value: this
        // env's own round-cap close-out
        // (AutoResolveForcedSubPicksThenRespond), never produced by the
        // engine itself.
        "info:finish_reason"_.Bind(Spec<int>({}, {kFinishReasonTimeout, 9})),
        // True prize-cards-remaining count for player 0/1 (absolute index,
        // not ego-relative like obs:player_state's rows) -- valid on EVERY
        // row, including the terminal one, unlike the 6 encoded obs:
        // tensors above (Zero()'d when done_, see WriteState). Just
        // prev_prize_remaining_ plumbed out: ConsumeDensePrizeReward already
        // refreshes it from live engine state on every call that reaches
        // it, and the one path that doesn't call it (illegal-action instant
        // loss) never mutates battle_->state either, so the previous
        // snapshot is still correct there too.
        "info:prize_p0"_.Bind(Spec<int>({}, {0, PRIZE_SIZE})),
        "info:prize_p1"_.Bind(Spec<int>({}, {0, PRIZE_SIZE})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    // Decks are configured, not part of the trajectory anymore (see
    // DefaultConfig) -- every Step() is a decide()-type step, a single
    // option-row index in [0, kMaxOptions-1], or kStopSlot (== kMaxOptions)
    // to submit the accumulated selection as final.
    return MakeDict("action"_.Bind(Spec<int>({}, {0, kMaxOptions})));
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

// GitHub #4: SelectOption::getCardPosition() casts param0 to AreaType with
// no type gate, but param0 only actually holds an area for the option types
// listed below (mirroring ApiJson.h's own list of types that emit an
// "area" key) -- for every other type param0 holds something else entirely
// (e.g. Play's hand index, Attack's attackId), and since AreaType::Prize
// == 6, an unrelated Play(6) or Attack(attackId % 256 == 6) option would
// otherwise misfire as a prize select and get silently resolved by
// envpool's own random bypass instead of reaching the policy. Gate on
// option.type first, exactly like every other engine call site that reads
// getCardPosition() already does.
inline bool IsPrizeSelect(const State& state) {
  if (state.options.empty()) {
    return false;
  }
  const SelectOption& o = state.options[0];
  switch (o.type) {
    case SelectOptionType::Card:
    case SelectOptionType::ToolCard:
    case SelectOptionType::EnergyCard:
    case SelectOptionType::Energy:
    case SelectOptionType::Attach:
    case SelectOptionType::Evolve:
    case SelectOptionType::Ability:
    case SelectOptionType::Discard:
      return static_cast<AreaType>(o.param0) == AreaType::Prize;
    default:
      return false;
  }
}

inline bool IsBypassedSelect(const State& state) {
  return IsForcedFullSelect(state) || IsPrizeSelect(state);
}

/**
 * Phases 2-4 (envpool-ptcg-integration skill) are all wired in, plus a
 * single-index-action redesign (see CLAUDE.md's "Current focus" /
 * schema.py's STOP_SLOT/N_OPTION_SLOTS comment), plus a later config-deck
 * redesign, plus a dense-reward/round-cap-timeout redesign:
 *  - Phase 2: real Reset()/Step() flow into ApiBattleStart, deviceRand=false
 *    throughput fix, real ApiSelect stepping, IsDone() via
 *    state.isFinish().
 *  - Phase 2.5: forced-full-select and prize-select are auto-resolved
 *    inside ResolveBypassedSelectsThenRespond()'s loop -- see "Trajectory &
 *    prompt routing" -- so they never reach Python as their own decision
 *    point, matching production main.py:agent()'s exact behavior.
 *  - Phase 3: decide()-step observations are real encoder output
 *    (ptcg_encode.h's EncodeObservation, a C++ port of encode.py), not
 *    Zero()'d placeholders. The terminal step (see WriteState) still
 *    Zero()s, since there's no decision to encode.
 *  - decide()'s `action` is a single index per Step() call: 0..kMaxOptions-1
 *    picks a real state.options row, kStopSlot submits the accumulated
 *    selection (chosen_) as final. A decision needing multiple picks is
 *    therefore multiple genuine Step() calls, not one call carrying
 *    multiple indices -- see LegalActions()/
 *    AutoResolveForcedSubPicksThenRespond()/FinalizeSelection() below. Any
 *    sub-pick where LegalActions() collapses to exactly one option (e.g.
 *    only one valid card remains, or selectMax was just reached so only
 *    STOP is legal) is auto-resolved internally, generalizing Phase 2.5's
 *    whole-decision bypass to intra-decision sub-picks -- Python only ever
 *    sees a genuine multi-way choice. Illegal actions (anything
 *    LegalActions() doesn't contain) apply the same instant-loss-for-the-
 *    offender reward pattern board_games::IllegalRewards uses
 *    (envpool/pgx/board_games.h), adapted to this env's
 *    single-reward-per-step shape (see "Reward" in the skill) instead of
 *    that pattern's dual-player broadcast, which doesn't fit here.
 *  - Config-deck redesign: deck selection is no longer part of the
 *    trajectory. `deck0s`/`deck1s` are fixed EnvPool-lifetime config, one
 *    pair per env slot (see DefaultConfig), each env's own pair read once
 *    at construction; Reset() drives ApiBattleStart with it directly and
 *    its own response is already a real decide()-type observation. There
 *    is exactly one Step() shape now.
 *  - Dense-reward/round-cap-timeout redesign: reward is no longer
 *    terminal-only. Every Step()/Reset() response carries a dense
 *    per-prize component (ConsumeDensePrizeReward -- diffs each player's
 *    remaining prize count against the last call, attributed to
 *    `acting_player`: who submitted the action being resolved, threaded
 *    through the whole resolve chain and distinct from
 *    current_player_/info:current_player, which names whoever decides the
 *    *returned* obs), plus, on a genuine engine-driven finish, a lump sum
 *    crediting the winner their own still-unclaimed prizes
 *    (NoActivePokemon/Deck0/Effect/Prize0 all reduce to this one rule).
 *    Separately, max_rounds_/round_count_ enforce a per-episode cap on
 *    genuine decisions inside AutoResolveForcedSubPicksThenRespond --
 *    training's own collection loop no longer truncates itself, trusting
 *    every episode to close (normally or via this env-side timeout) within
 *    max_rounds. `timeout_penalty` itself is intentionally NOT env config:
 *    applying it in Python (training/buffer.py) instead of folding it into
 *    one flat scalar here keeps a dense delta that happens to land on the
 *    exact round-cap step correctly mirrored rather than accidentally
 *    unmirrored. See "Reward" in the envpool-ptcg-integration skill.
 */
class PtcgEnv : public Env<PtcgEnvSpec> {
 protected:
  bool done_{true};
  int current_player_{0};
  int finish_reason_{0};
  ApiData* battle_{nullptr};
  // Populated once, at construction, from this env_id's own slice of the
  // config's deck0s/deck1s -- fixed for this PtcgEnv's whole lifetime
  // (every episode in this slot plays the same pair; a different slot in
  // the same pool may hold an entirely different pair). Never mutated by
  // Step()/Reset() the way an earlier per-episode deck-select handshake
  // used to.
  std::array<int, kDeckSize> deck0_{};
  std::array<int, kDeckSize> deck1_{};
  // Per-episode cap on genuine (Python-visible) decisions -- config, fixed
  // for this PtcgEnv's whole lifetime, like deck0_/deck1_ above. See
  // round_count_ below and AutoResolveForcedSubPicksThenRespond.
  int max_rounds_{0};
  // Real state.options-row indices already picked earlier in the current
  // decide()-type decision -- reset to empty every time a fresh, non-
  // bypassed decide() prompt begins (see ResolveBypassedSelectsThenRespond),
  // accumulated across however many Step() calls that decision takes, and
  // submitted to the real ApiSelect exactly once, at FinalizeSelection().
  std::vector<int> chosen_;
  // How many genuine decisions the CURRENT episode has already presented
  // -- reset at Reset(), gates the round-cap timeout below. Counts every
  // Step() round-trip, including sub-picks of a multi-pick decision,
  // matching what training/rollout.py's own round counter already counts.
  int round_count_{0};
  // Each player's ps.prize.size() ("remaining prizes") as of the last
  // WriteState-producing resolution -- ConsumeDensePrizeReward's diff
  // baseline, refreshed on every call. The {PRIZE_SIZE, PRIZE_SIZE}
  // member-initializer (also reset at Reset()) is only ever a placeholder,
  // not a claim that prizes are already dealt at that point -- they
  // aren't (see ConsumeDensePrizeReward's own doc comment); it's
  // overwritten with the real live value on the very first call regardless.
  std::array<int, 2> prev_prize_remaining_{PRIZE_SIZE, PRIZE_SIZE};

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

    // deck0s/deck1s config -- see DefaultConfig's comment. Each is expected
    // to be exactly `num_envs * kDeckSize` long (one deck per env slot,
    // flattened); a size mismatch is a config bug (wrong-length list passed
    // to make()), not a runtime condition to recover from -- fail loudly at
    // construction rather than silently reading garbage/out-of-bounds
    // later. This env_id's own slice is `[env_id*kDeckSize,
    // (env_id+1)*kDeckSize)`, mirroring envpool/core/env.h's ResolveSeed
    // per-env `env_seed.at(env_id)` pattern.
    int num_envs = spec.config["num_envs"_];
    const auto& deck0s_cfg = spec.config["deck0s"_];
    const auto& deck1s_cfg = spec.config["deck1s"_];
    CHECK_EQ(deck0s_cfg.size(), static_cast<std::size_t>(num_envs * kDeckSize))
        << "`deck0s` config must contain exactly num_envs * " << kDeckSize
        << " card ids (one " << kDeckSize << "-card deck per env slot)";
    CHECK_EQ(deck1s_cfg.size(), static_cast<std::size_t>(num_envs * kDeckSize))
        << "`deck1s` config must contain exactly num_envs * " << kDeckSize
        << " card ids (one " << kDeckSize << "-card deck per env slot)";
    auto deck0_begin = deck0s_cfg.begin() + env_id * kDeckSize;
    auto deck1_begin = deck1s_cfg.begin() + env_id * kDeckSize;
    std::copy(deck0_begin, deck0_begin + kDeckSize, deck0_.begin());
    std::copy(deck1_begin, deck1_begin + kDeckSize, deck1_.begin());

    max_rounds_ = spec.config["max_rounds"_];
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
    current_player_ = 0;
    finish_reason_ = 0;
    chosen_.clear();
    round_count_ = 0;
    prev_prize_remaining_ = {PRIZE_SIZE, PRIZE_SIZE};

    std::array<int, 2 * kDeckSize> cards{};
    std::copy(deck0_.begin(), deck0_.end(), cards.begin());
    std::copy(deck1_.begin(), deck1_.end(), cards.begin() + kDeckSize);
    StartData start = ApiBattleStart(cards.data());
    // Decks are fixed config now, not a possibly-adversarial per-episode
    // action -- an invalid pair is a setup bug (wrong ids, illegal
    // deck-building constraints, ...), not something to attribute a
    // per-episode illegal-action penalty to. Fail loudly instead of the
    // old errorPlayer/instant-loss-reward path.
    CHECK(start.battlePtr != nullptr)
        << "configured deck0/deck1 pair is illegal per ApiBattleStart -- "
           "fix the make() config, this is not a per-episode condition";
    battle_ = start.battlePtr;
    // Throughput fix -- see "Architecture at a glance" / "ptcg_engine C++
    // surface to call directly" in the skill. Must happen before any
    // ApiSelect call; ApiBattleStart hardcodes deviceRand=true internally
    // with no override parameter, but GameConfig is read live at every use
    // site, so mutating it here still takes effect for the rest of the
    // battle.
    battle_->game.config.deviceRand = false;

    // No real prior actor for the very first decision of a fresh episode
    // -- reward is trivially 0 there regardless (nothing has happened
    // yet), see ConsumeDensePrizeReward's actor<0 no-op guard.
    ResolveBypassedSelectsThenRespond(-1);
  }

  void Step(const Action& action) override {
    // Snapshot before any mutation: current_player_ still names whoever
    // this action was asked of (see SyncFromEngine, called only after this
    // point) -- the reward this call produces always belongs to them,
    // distinct from info:current_player on the *returned* row (which names
    // whoever decides next). See "Reward" in the skill.
    const int acting_player = current_player_;
    int a = action["action"_];
    std::vector<int> legal = LegalActions();
    if (std::find(legal.begin(), legal.end(), a) == legal.end()) {
      // Illegal action -> instant loss for the offender (acting_player).
      // ApiSelect is never even called on this path (validated against
      // LegalActions() first), so battle_->state is completely unchanged
      // -- no SyncFromEngine() call, no dense component (nothing advanced
      // to diff), and WriteState correctly Zero()s the obs rather than
      // re-encoding a decision that never advanced.
      done_ = true;
      finish_reason_ = 0;
      WriteState(-1.0F, acting_player);
      return;
    }
    if (a == kStopSlot) {
      FinalizeSelection(acting_player);
      return;
    }
    chosen_.push_back(a);
    AutoResolveForcedSubPicksThenRespond(acting_player);
  }

 private:
  // Single source of truth for "what actions are legal right now", given
  // battle_->state and chosen_ -- used both to validate an incoming Python
  // action (Step) and to drive the auto-resolve loop below. A real
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
  // Step, or freshly reset for a new decide()-type prompt in
  // ResolveBypassedSelectsThenRespond): auto-advances through any round
  // where LegalActions() collapses to exactly one option (e.g. "only one
  // valid card remains" or "selectMax was just reached, only STOP is
  // legal"), so Python only ever sees a genuine multi-way decide() step,
  // never a deterministic one. Finalizes (real ApiSelect + recurse into
  // ResolveBypassedSelectsThenRespond) the instant that happens; otherwise
  // presents the resulting decision -- unless this episode has already
  // used up its max_rounds_ budget of genuine decisions, in which case it
  // force-closes with a timeout instead of presenting another one (see
  // "Dense-reward/round-cap-timeout redesign" above).
  void AutoResolveForcedSubPicksThenRespond(int acting_player) {
    while (true) {
      std::vector<int> legal = LegalActions();
      CHECK(!legal.empty())
          << "no legal action at a decide() sub-pick -- engine invariant "
             "violated (selectMin exceeds available options?)";
      if (legal.size() != 1) {
        break;
      }
      if (legal[0] == kStopSlot) {
        FinalizeSelection(acting_player);
        return;
      }
      chosen_.push_back(legal[0]);
    }
    // Always credit whatever the just-finalized action changed, whether or
    // not this row turns out to be a timeout close-out below -- a dense
    // delta that happens to land on the exact round-cap boundary must
    // still be credited, not dropped.
    float dense = ConsumeDensePrizeReward(acting_player);
    if (round_count_ >= max_rounds_) {
      // Round-cap timeout: this episode already used up its budget of
      // genuine decisions. Force-close instead of presenting another one.
      // Reward is *only* the dense component above -- training applies
      // timeout_penalty itself (buffer.py), not this env (see the class
      // doc comment's "Dense-reward/round-cap-timeout redesign"). No
      // SyncFromEngine(): current_player_ keeps its last real value, not
      // meaningful on a terminal row regardless (mirrors the
      // illegal-action path in Step()).
      done_ = true;
      finish_reason_ = kFinishReasonTimeout;
      WriteState(dense, acting_player);
      return;
    }
    round_count_++;
    SyncFromEngine();
    WriteState(dense, acting_player);
  }

  // Submits chosen_ as the final answer to this decide()-type selection.
  // Every constituent pick was already validated against LegalActions()
  // before being added to chosen_ (either by Step, for a genuine
  // Python-provided pick, or by AutoResolveForcedSubPicksThenRespond, for
  // an auto-resolved one), so ApiSelect itself should never reject it --
  // CHECK_EQ (not a penalized runtime path) mirrors
  // ResolveBypassedSelectsThenRespond's own bypass-selection assertion:
  // a failure here means LegalActions() has drifted from the engine's real
  // state.checkPlayerSelect() rule, a bug to fix, not a real game outcome.
  void FinalizeSelection(int acting_player) {
    int error =
        ApiSelect(battle_, chosen_.data(), static_cast<int>(chosen_.size()));
    CHECK_EQ(error, 0)
        << "PtcgEnv's own tracked selection was rejected by ApiSelect -- "
           "LegalActions() has drifted from the engine's real legality rule";
    ResolveBypassedSelectsThenRespond(acting_player);
  }

  // Runs ApiSelect's own bypass loop forward through any forced-full-select/
  // prize-select decision points (Phase 2.5, "Trajectory & prompt routing"),
  // then either hands off to a fresh decide()-type prompt's own
  // auto-resolve loop, or -- if the game ended instead -- computes this
  // step's reward ("Reward" in the skill) and writes the response.
  // Shared by Reset()'s post-ApiBattleStart path, Step's post-illegal-check
  // path (via AutoResolveForcedSubPicksThenRespond / FinalizeSelection), and
  // FinalizeSelection itself -- deliberately, since a just-started battle's
  // first decision point could in principle itself be bypassed
  // (astronomically unlikely, but free to handle correctly by reusing this
  // rather than special-casing it away).
  void ResolveBypassedSelectsThenRespond(int acting_player) {
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
      AutoResolveForcedSubPicksThenRespond(acting_player);
      return;
    }
    // Reward ("Reward" in the skill): dense component (prizes taken
    // resolving this action, for either player -- see
    // ConsumeDensePrizeReward) plus, only on a real apiResult() winner (not
    // a draw), a lump sum crediting the winner their own still-unclaimed
    // prizes. NoActivePokemon/Deck0/Effect/Prize0 all reduce to this one
    // rule: on a normal Prize0 win the winner already has 0 prizes left,
    // so the lump sum is a no-op and the dense stream already accounts for
    // the whole game; on the other three, the winner banks whatever they
    // hadn't gotten around to taking yet. apiResult()==2 (draw, including
    // the engine's own turn>=10000/actionCount>=3000 safety valves) adds
    // no lump sum -- there's no winner to credit.
    float reward = ConsumeDensePrizeReward(acting_player);
    int result = battle_->state.apiResult();
    if (result == 0 || result == 1) {
      float remaining_winner =
          static_cast<float>(battle_->state.players[result].prize.size()) /
          PRIZE_SIZE;
      reward += (acting_player == result) ? remaining_winner : -remaining_winner;
    }
    // Real engine-side finish -- state.finishCheck() has already run
    // (called internally by ApiSelect/data->next()) and set finishReason,
    // so this is the one place a non-zero value is ever recorded.
    finish_reason_ = static_cast<int>(battle_->state.finishReason);
    SyncFromEngine();
    WriteState(reward, acting_player);
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

  // Dense per-prize reward ("Reward" in the skill): diffs each player's
  // remaining prize count (ps.prize.size()) against its value as of the
  // last call, attributes the net change to `actor` (positive if actor
  // gained ground, negative if the other player did -- this covers both
  // "actor's own attack KO'd something" and "actor's own effect handed the
  // other player a free prize", without needing to know which happened),
  // and unconditionally advances the baseline. Two cases are a safe no-op
  // (baseline refreshed, reward stays 0): `actor` outside {0,1} (only the
  // Reset()-time first call, via ResolveBypassedSelectsThenRespond(-1) in
  // Reset()), and `state.phase == GamePhase::Setup`.
  //
  // The Setup guard is required, not defensive: SetupProc.h's SetupGame ->
  // SelectedIsFirst presents the very first Python-visible decide() (a
  // Yes/No "who goes first" prompt) *before* SetupPrize ever runs for
  // either player -- prev_prize_remaining_'s {PRIZE_SIZE, PRIZE_SIZE}
  // member-initializer is just a placeholder value, not yet true at that
  // point. Worse, SetupProc.h's AfterSetupActivePokemon deals a mulligan-
  // free player's prizes with SetupPrize(state, i) *alone*, one whole
  // resolution before the other (mulliganing) player's own SetupPrize call
  // -- an ordinary (no phase guard) delta computation on that resolution
  // would see one side's remaining jump 0->PRIZE_SIZE with the other still
  // at 0, i.e. exactly what a real multi-prize KO looks like, producing a
  // full spurious +-1.0 "reward" for getting your own starting prizes
  // dealt. state.phase stays GamePhase::Setup for the whole setup
  // sequence (mulligan/active/bench selection, SetupPrize itself) and only
  // flips to Main inside TurnStart, right as the first real turn begins
  // (GameProc.h) -- a reliable "has real gameplay actually started yet"
  // signal, unlike trying to infer it from prize counts themselves (a
  // legitimate mid-game effect, EffectType::DeckToPrize, can also
  // *increase* a prize pile, so "prizes went up" alone doesn't imply
  // setup).
  float ConsumeDensePrizeReward(int actor) {
    const auto& state = battle_->state;
    std::array<int, 2> cur = {
        static_cast<int>(state.players[0].prize.size()),
        static_cast<int>(state.players[1].prize.size()),
    };
    float result = 0.0F;
    if ((actor == 0 || actor == 1) && state.phase != GamePhase::Setup) {
      int other = 1 - actor;
      int delta_actor = prev_prize_remaining_[actor] - cur[actor];
      int delta_other = prev_prize_remaining_[other] - cur[other];
      result = static_cast<float>(delta_actor - delta_other) / PRIZE_SIZE;
    }
    prev_prize_remaining_ = cur;
    return result;
  }

  void SyncFromEngine() {
    const auto& state = battle_->state;
    current_player_ = state.selectPlayer;
    done_ = state.isFinish();
  }

  void WriteState(float reward, int reward_player) {
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
    if (!done_) {
      // Real encoder output only for a genuine, non-terminal, non-bypassed
      // decide() step -- exactly encode_observation()'s own documented
      // precondition. The terminal step is intentionally left Zero()'d
      // (production never calls encode_observation() in an already-finished
      // game either, so there's no ground truth to match, and
      // state.selectType may already be cleared by then -- see
      // ptcg_encode.h's EncodeObservation doc comment).
      const auto& deck = current_player_ == 0 ? deck0_ : deck1_;
      EncodeObservation(battle_->state, deck, chosen_, state["obs:cards"_],
                        state["obs:pokemons"_], state["obs:player_state"_],
                        state["obs:state"_], state["obs:select"_],
                        state["obs:options"_]);
    }
    state["info:current_player"_] = current_player_;
    state["info:reward_player"_] = reward_player;
    state["info:finish_reason"_] = finish_reason_;
    state["info:prize_p0"_] = prev_prize_remaining_[0];
    state["info:prize_p1"_] = prev_prize_remaining_[1];
    state["reward"_] = reward;
  }
};

using PtcgEnvPool = AsyncEnvPool<PtcgEnv>;

}  // namespace ptcg

#endif  // ENVPOOL_PTCG_PTCG_ENVPOOL_H_
