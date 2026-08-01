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
#include <cmath>
#include <vector>

// Dense-reward/round-cap-timeout redesign tests (envpool-ptcg-integration
// skill's "Reward" section / ptcg_envpool.h's class doc comment). Separate
// file from ptcg_envpool_test.cc (structural coverage) and
// ptcg_envpool_illegal_test.cc (the illegal-action penalty path) since both
// scenarios here are genuinely new mechanisms, each worth a focused,
// self-contained test -- same reasoning ptcg_envpool_illegal_test.cc's own
// doc comment gives for its own split. Duplicates the small kDeck/TileDeck/
// PickLegalAction/CopyOptions/Send helpers rather than sharing them with the
// other two files -- an established pattern here already (each file's
// anonymous-namespace helpers have internal linkage, so there's nothing to
// share without a new shared-test-util target, which isn't worth it for a
// handful of small functions).

using PtcgAction = typename ptcg::PtcgEnv::Action;
using PtcgState = typename ptcg::PtcgEnv::State;

namespace {

// Same real, deck-legal 60-card list as ptcg_envpool_test.cc /
// ptcg_envpool_illegal_test.cc / ptcg_encode_test.cc / ptcg_smoke_test.cc,
// copied from submissions/sample_submission/deck.csv.
constexpr std::array<int, 60> kDeck = {
    1158, 721,  721,  722,  722,  722,  722,  723,  723,  723,  723,  1145,
    1145, 1145, 1145, 1205, 1205, 1227, 1227, 1227, 1227, 1235, 1235, 1235,
    1235, 3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
};

std::vector<int> TileDeck(const std::array<int, 60>& deck, int num_envs) {
  std::vector<int> out;
  out.reserve(deck.size() * num_envs);
  for (int i = 0; i < num_envs; ++i) {
    out.insert(out.end(), deck.begin(), deck.end());
  }
  return out;
}

void ExpectAllZero(const Array& arr) {
  const auto* data = reinterpret_cast<const int*>(arr.Data());
  for (std::size_t i = 0; i < arr.size; ++i) {
    EXPECT_EQ(data[i], 0);
  }
}

std::vector<int> CopyOptions(const TArray<int>& options) {
  std::vector<int> out(ptcg::kOptionRows * ptcg::kOptionsCols);
  for (int row = 0; row < ptcg::kOptionRows; ++row) {
    for (int col = 0; col < ptcg::kOptionsCols; ++col) {
      out[row * ptcg::kOptionsCols + col] = options[row][col];
    }
  }
  return out;
}

// See ptcg_envpool_test.cc's own copy of this helper for the full rationale
// (greedily picks a real, not-yet-picked option over STOP -- every genuine
// decide() observation offers at least one, so this is always legal and
// never accidentally ends a decision early).
int PickLegalAction(const std::vector<int>& options_flat) {
  constexpr int kIsValidCol = 0;
  constexpr int kAlreadySelectedCol = 19;
  for (int row = 0; row < ptcg::kOptionRows; ++row) {
    if (options_flat[row * ptcg::kOptionsCols + kIsValidCol] == 1 &&
        options_flat[row * ptcg::kOptionsCols + kAlreadySelectedCol] == 0) {
      return row;
    }
  }
  ADD_FAILURE() << "no legal action found in observed options -- envpool invariant violated";
  return ptcg::kStopSlot;
}

// Same legality scan as PickLegalAction, but works through a fixed
// priority order of OptionTypes (cg.api.OptionType / ApiType.h's
// SelectOptionType -- 0-indexed, matches cg/api.py's OptionType exactly,
// see the envpool-ptcg-integration skill's "Enum alignment" note) instead
// of just the lowest-index legal row: ATTACK first, then ATTACH (energy)
// and PLAY/EVOLVE (deploy more Pokemon) ahead of RETREAT/END. A plain
// lowest-index-first (or even attack-only-preferring) policy can complete
// a whole game without ever attacking for lethal -- e.g. never attaching
// enough energy to make an attack legal in the first place, if some
// always-legal lower-priority option (END) keeps winning ties -- which
// measurably happened often enough with just ATTACK preference alone to
// make DenseRewardMatchesObservedPrizeDeltas flaky. This biases play
// toward actually landing a KO without controlling the game's outcome or
// legality in any other way.
int PickAggressiveAction(const std::vector<int>& options_flat) {
  constexpr int kIsValidCol = 0;
  constexpr int kOptionTypeCol = 1;
  constexpr int kAlreadySelectedCol = 19;
  constexpr int kPriority[] = {13, 8, 7, 9};  // ATTACK, ATTACH, PLAY, EVOLVE

  auto is_legal = [&](int row) {
    return options_flat[row * ptcg::kOptionsCols + kIsValidCol] == 1 &&
           options_flat[row * ptcg::kOptionsCols + kAlreadySelectedCol] == 0;
  };

  for (int want_type : kPriority) {
    for (int row = 0; row < ptcg::kOptionRows; ++row) {
      if (is_legal(row) && options_flat[row * ptcg::kOptionsCols + kOptionTypeCol] == want_type) {
        return row;
      }
    }
  }
  for (int row = 0; row < ptcg::kOptionRows; ++row) {
    if (is_legal(row)) {
      return row;
    }
  }
  ADD_FAILURE() << "no legal action found in observed options -- envpool invariant violated";
  return ptcg::kStopSlot;
}

// Single-env send/recv helper (same shape as ptcg_envpool_illegal_test.cc's).
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

// Cross-checks every non-terminal step's dense reward against ground truth
// decoded straight from that same step's own encoded observation --
// obs:player_state's prize_count column (ego-centric: row 0 is
// info:current_player's own row, row 1 the opponent's -- see
// ptcg_encode.h's WritePlayerState) rather than trusting
// ConsumeDensePrizeReward's own bookkeeping, which is exactly the thing
// under test. The terminal row itself is Zero()'d (no ground truth
// observable there), so this only bounds it loosely -- the terminal lump
// sum's exact composition is exercised precisely with synthetic,
// hand-computed data in tests/training/test_buffer.py instead, where an
// exact scenario can be constructed without needing to steer the whole
// engine into one.
TEST(PtcgRewardTest, DenseRewardMatchesObservedPrizeDeltas) {
  // Retried across up to kMaxAttempts independent games, not just one: even
  // with PickAggressiveAction, an attack doesn't always connect for lethal
  // (energy requirements, damage vs. HP, ...) -- a single random game not
  // infrequently never lands a single KO. The per-row correctness
  // check (EXPECT_NEAR below) still runs, and must pass, on *every*
  // attempt, real or not -- only the "did we ever actually exercise a
  // nonzero delta" coverage bar is allowed to need more than one try.
  constexpr int kMaxAttempts = 20;
  bool any_attempt_showed_dense = false;
  for (int attempt = 0; attempt < kMaxAttempts && !any_attempt_showed_dense; ++attempt) {
    auto config = ptcg::PtcgEnvSpec::kDefaultConfig;
    config["num_envs"_] = 1;
    config["batch_size"_] = 1;
    config["num_threads"_] = 1;
    config["deck0s"_] = TileDeck(kDeck, 1);
    config["deck1s"_] = TileDeck(kDeck, 1);
    // Generous -- this test wants the round-cap timeout to never interfere
    // (that's RoundCapForcesTimeoutAtExactlyMaxRounds's job, below).
    config["max_rounds"_] = 20000;
    ptcg::PtcgEnvSpec spec(config);
    ptcg::PtcgEnvPool envpool(spec);

    Array env_ids(Spec<int>({1}));
    env_ids[0] = 0;
    envpool.Reset(env_ids);
    PtcgState state(envpool.Recv());

    // Not {PRIZE_SIZE, PRIZE_SIZE} -- SetupProc.h's SetupGame presents the
    // very first Python-visible decide() (a Yes/No "who goes first" prompt)
    // *before* either player's prizes are dealt, and an asymmetric
    // mulligan can deal one side's prizes a whole resolution before the
    // other's (see ConsumeDensePrizeReward's doc comment in
    // ptcg_envpool.h for the full story). The engine skips dense
    // computation entirely while GamePhase::Setup, so this test doesn't
    // try to predict the exact round that settles -- it just requires
    // `reward == 0` on every row until it has independently observed both
    // piles simultaneously at PRIZE_SIZE, and only trusts the exact delta
    // formula from that point on.
    std::array<int, 2> last_remaining = {0, 0};
    bool baseline_trusted = false;
    bool checked_any_nonzero_dense = false;

    const int kMaxRounds = 20000;
    int round = 0;
    for (; round < kMaxRounds; ++round) {
      bool done = static_cast<bool>(state["done"_][0]);
      float reward = static_cast<float>(state["reward"_][0]);
      int reward_player = static_cast<int>(state["info:reward_player"_][0]);
      ASSERT_GE(reward_player, -1);
      ASSERT_LE(reward_player, 1);

      if (done) {
        EXPECT_TRUE(std::isfinite(reward));
        // Loose bound, not exact: dense (bounded by ~2, both players could
        // in principle move in one resolution) plus lump sum (bounded by
        // 1). Catches gross errors (wrong scale, forgetting to divide by
        // PRIZE_SIZE) without being fragile to which exact finish reason a
        // real, randomly-run game happens to end on.
        EXPECT_LE(std::abs(reward), 3.0F) << "reward=" << reward;
        // info:prize_p0/p1 are readable ground truth on this exact row
        // (unlike obs:player_state, which WriteState Zero()'s here) -- a
        // real Prize0 finish means the winner's own pile hit exactly 0.
        int prize_p0 = static_cast<int>(state["info:prize_p0"_][0]);
        int prize_p1 = static_cast<int>(state["info:prize_p1"_][0]);
        EXPECT_GE(prize_p0, 0);
        EXPECT_LE(prize_p0, PRIZE_SIZE);
        EXPECT_GE(prize_p1, 0);
        EXPECT_LE(prize_p1, PRIZE_SIZE);
        int finish_reason = static_cast<int>(state["info:finish_reason"_][0]);
        if (finish_reason == static_cast<int>(FinishReason::Prize0)) {
          EXPECT_TRUE(prize_p0 == 0 || prize_p1 == 0)
              << "Prize0 finish but neither player's info:prize_p* hit 0 -- "
                 "prize_p0="
              << prize_p0 << " prize_p1=" << prize_p1;
        }
        break;
      }

      int current_player = static_cast<int>(state["info:current_player"_][0]);
      TArray<int> player_state(state["obs:player_state"_][0]);
      constexpr int kPrizeCountCol = 2;  // PLAYER_STATE_COLUMNS[2], see schema.py
      std::array<int, 2> remaining{};
      remaining[current_player] = player_state[0][kPrizeCountCol];
      remaining[1 - current_player] = player_state[1][kPrizeCountCol];

      if (!baseline_trusted) {
        EXPECT_FLOAT_EQ(reward, 0.0F)
            << "attempt=" << attempt << " round=" << round
            << ": still (possibly) mid-setup -- the engine skips dense "
               "computation until GamePhase leaves Setup";
        if (remaining[0] == PRIZE_SIZE && remaining[1] == PRIZE_SIZE) {
          baseline_trusted = true;
        }
      } else if (reward_player == 0 || reward_player == 1) {
        int actor = reward_player;
        int other = 1 - actor;
        int delta_actor = last_remaining[actor] - remaining[actor];
        int delta_other = last_remaining[other] - remaining[other];
        float expected = static_cast<float>(delta_actor - delta_other) / static_cast<float>(PRIZE_SIZE);
        EXPECT_NEAR(reward, expected, 1e-4F)
            << "attempt=" << attempt << " round=" << round << " actor=" << actor
            << " last_remaining={" << last_remaining[0] << "," << last_remaining[1]
            << "} remaining={" << remaining[0] << "," << remaining[1] << "}";
        if (delta_actor != 0 || delta_other != 0) {
          checked_any_nonzero_dense = true;
        }
      }
      last_remaining = remaining;

      std::vector<int> options_flat = CopyOptions(TArray<int>(state["obs:options"_][0]));
      int action_value = PickAggressiveAction(options_flat);
      state = Send(envpool, action_value);
    }

    ASSERT_LT(round, kMaxRounds) << "attempt=" << attempt << ": game never finished within the safety bound";
    ASSERT_TRUE(baseline_trusted)
        << "attempt=" << attempt << ": never observed both piles "
           "simultaneously at PRIZE_SIZE -- setup never completed, or the "
           "prize_count column mapping is wrong";
    any_attempt_showed_dense = checked_any_nonzero_dense;
  }

  EXPECT_TRUE(any_attempt_showed_dense)
      << "none of " << kMaxAttempts << " independent games ever showed a "
         "single prize-taking event -- either PickAggressiveAction never "
         "gets a lethal attack in with this deck, or something's wrong; "
         "this test needs at least one real dense event to be meaningful";
}

// max_rounds is a genuine per-episode cap now (not the batch-wide
// round_idx < max_rounds truncation training/rollout.py used to do) --
// this drives a real game with a tiny cap and checks the env force-closes
// at exactly the right round, with the right finish_reason, rather than
// either running past the cap or off-by-one stopping short of it.
TEST(PtcgRewardTest, RoundCapForcesTimeoutAtExactlyMaxRounds) {
  auto config = ptcg::PtcgEnvSpec::kDefaultConfig;
  config["num_envs"_] = 1;
  config["batch_size"_] = 1;
  config["num_threads"_] = 1;
  config["deck0s"_] = TileDeck(kDeck, 1);
  config["deck1s"_] = TileDeck(kDeck, 1);
  constexpr int kCap = 5;  // tiny -- a real game never finishes this fast
  config["max_rounds"_] = kCap;
  ptcg::PtcgEnvSpec spec(config);
  ptcg::PtcgEnvPool envpool(spec);

  Array env_ids(Spec<int>({1}));
  env_ids[0] = 0;
  envpool.Reset(env_ids);
  PtcgState state(envpool.Recv());

  int genuine_rounds = 0;
  bool saw_timeout = false;
  for (int i = 0; i < kCap + 1; ++i) {
    ASSERT_FALSE(static_cast<bool>(state["done"_][0]))
        << "game finished naturally before the round cap -- adjust kCap, "
           "this scenario needs the cap to be what stops it";
    std::vector<int> options_flat = CopyOptions(TArray<int>(state["obs:options"_][0]));
    int action_value = PickLegalAction(options_flat);
    state = Send(envpool, action_value);
    genuine_rounds++;
    if (static_cast<bool>(state["done"_][0])) {
      saw_timeout = true;
      break;
    }
  }

  ASSERT_TRUE(saw_timeout) << "round cap never fired within kCap+1 sends";
  // Exactly kCap genuine decisions allowed; the (kCap+1)-th attempt is what
  // force-closes -- the off-by-one this test exists to catch.
  EXPECT_EQ(genuine_rounds, kCap);
  EXPECT_EQ(static_cast<int>(state["info:finish_reason"_][0]), ptcg::kFinishReasonTimeout);
  EXPECT_TRUE(std::isfinite(static_cast<float>(state["reward"_][0])));
  ExpectAllZero(state["obs:cards"_][0]);
  ExpectAllZero(state["obs:pokemons"_][0]);
  ExpectAllZero(state["obs:player_state"_][0]);
  ExpectAllZero(state["obs:state"_][0]);
  ExpectAllZero(state["obs:select"_][0]);
  ExpectAllZero(state["obs:options"_][0]);
  // Unlike obs:player_state above, info:prize_p0/p1 are NOT Zero()'d on this
  // terminal row -- they're the whole point of exposing them separately.
  // This cap is tiny enough that a real game can't have finished by round 5
  // (kCap+1's ASSERT_FALSE above already established that), so this can't
  // pin exact values without over-fitting to engine setup-phase timing --
  // just confirms the fields are wired and in range, not left uninitialized
  // or leaking a stale ring-buffer slot.
  EXPECT_GE(static_cast<int>(state["info:prize_p0"_][0]), 0);
  EXPECT_LE(static_cast<int>(state["info:prize_p0"_][0]), PRIZE_SIZE);
  EXPECT_GE(static_cast<int>(state["info:prize_p1"_][0]), 0);
  EXPECT_LE(static_cast<int>(state["info:prize_p1"_][0]), PRIZE_SIZE);
}

// A dense delta landing exactly on the cap-triggering step still being
// credited (not silently dropped) is NOT separately tested here: the
// terminal row's observation is always Zero()'d (see WriteState), so
// there's no ground truth to decode and cross-check the coincident
// resolution's prize counts against -- the whole point of the scenario is
// unobservable from outside by construction. The guarantee instead rests
// on ConsumeDensePrizeReward being called exactly once, unconditionally,
// before AutoResolveForcedSubPicksThenRespond's round-cap branch (see that
// function's doc comment) -- a single code path feeding both the "present
// normally" and "force timeout" branches the same `dense` value, plain
// enough to verify by inspection. The precise composition (dense +
// mirroring + timeout_penalty layered on top) is exercised exactly, with
// hand-constructed synthetic data, in tests/training/test_buffer.py
// instead, where the scenario can be built directly rather than hoped for
// from a real game.
