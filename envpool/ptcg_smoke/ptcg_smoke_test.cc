// Phase 0 smoke test (envpool-ptcg-integration skill): does the @ptcg_engine
// new_local_repository vendoring actually compile and link through Bazel, end to end -
// not just "headers parse", but symbols resolve and a real battle runs to completion?
//
// This deliberately does NOT re-litigate the concurrency/determinism question - that
// was already answered thoroughly (TSan/ASan clean across thousands of battles) by
// research/envpool_smoke/concurrency_smoke.cpp, compiled directly with g++ for fast
// iteration outside of Bazel. This target's only job is to de-risk the Bazel
// vendoring plumbing itself.

#include "All.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

namespace {

// Same real, deck-legal 60-card list as concurrency_smoke.cpp, copied from
// submissions/sample_submission/deck.csv.
constexpr std::array<int, 60> kDeck = {
    1158, 721,  721,  722,  722,  722,  722,  723,  723,  723,  723,  1145,
    1145, 1145, 1145, 1205, 1205, 1227, 1227, 1227, 1227, 1235, 1235, 1235,
    1235, 3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
};

// InitializeAll() asserts CardTable.size() == 0 on entry (ptcg_engine/All.h:15), so it
// can only run once per process. SetUpTestSuite() (not per-test SetUp(), and not a
// second TEST() relying on source order) is what guarantees that regardless of
// --gtest_filter/--gtest_shuffle.
class PtcgEngineBazelVendoring : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { InitializeAll(); }
};

}  // namespace

TEST_F(PtcgEngineBazelVendoring, InitializeAllPopulatedGlobalTables) {
  EXPECT_GT(CardTable.size(), 0u);
  EXPECT_GT(FunctionTable.size(), 0u);
  EXPECT_GT(AttackTable.size(), 0u);
}

TEST_F(PtcgEngineBazelVendoring, OneBattleRunsToCompletionThroughBazelLinkedSymbols) {
  std::array<int, 120> cards;
  std::copy(kDeck.begin(), kDeck.end(), cards.begin());
  std::copy(kDeck.begin(), kDeck.end(), cards.begin() + 60);

  StartData start = ApiBattleStart(cards.data());
  ASSERT_NE(start.battlePtr, nullptr) << "errorPlayer=" << start.errorPlayer
                                       << " errorType=" << start.errorType;
  ApiData* data = start.battlePtr;

  // Trivially-legal policy (see concurrency_smoke.cpp for why this is always
  // accepted): always select indices [0, selectMin).
  int steps = 0;
  while (!data->state.isFinish() && steps < 20000) {
    int select_min = data->state.selectMin;
    std::vector<int> selected(select_min);
    std::iota(selected.begin(), selected.end(), 0);
    ASSERT_EQ(ApiSelect(data, selected.data(), (int)selected.size()), 0);
    steps++;
  }

  EXPECT_TRUE(data->state.isFinish());
  EXPECT_GT(steps, 0);
  EXPECT_LE((int)data->state.gameResult, 3);
  EXPECT_GE((int)data->state.gameResult, 1);
  ApiBattleFinish(data);
}
