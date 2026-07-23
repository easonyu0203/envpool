# Phase 0 throwaway stub (envpool-ptcg-integration skill): proves the @ptcg_engine
# new_local_repository vendoring (see //WORKSPACE, //third_party/ptcg_engine) compiles
# and links through Bazel end to end. Superseded once Phase 1 stands up the real
# //envpool/ptcg package - safe to delete this directory at that point, or keep it as
# a cheap standalone regression check on the vendoring itself.
load("@rules_cc//cc:defs.bzl", "cc_test")

cc_test(
    name = "ptcg_smoke_test",
    size = "small",
    srcs = ["ptcg_smoke_test.cc"],
    # copts do NOT propagate from a dependency's cc_library to this target - every
    # target that (transitively, via All.h) touches ptcg_engine's std::u8string/
    # char8_t usage needs its own -std=c++20, not just @ptcg_engine's cc_library.
    # Confirmed empirically: omitting this failed with -std=c++17 from
    # envpool/.bazelrc's global `build:linux --cxxopt=-std=c++17`.
    copts = ["-std=c++20"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@ptcg_engine//:ptcg_engine",
    ],
)
