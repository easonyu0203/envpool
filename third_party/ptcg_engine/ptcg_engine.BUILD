# BUILD file injected into the @ptcg_engine external repo by new_local_repository
# (see //WORKSPACE). ptcg_engine (repo root's ptcg_engine/, the PTCG AI Battle
# Challenge's C++ battle engine - the "cabt Engine") has no build system of its own:
# header-only, all functions `inline` in Api.h/reachable via All.h, no CMake/Makefile,
# only a Windows-only .sln. See the envpool-ptcg-integration skill for design context.
#
# Export.cpp (the ctypes/ FFI extern "C" wrapper, unused by the direct-call envpool
# env) is intentionally excluded by the hdrs glob only matching *.h.
load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "ptcg_engine",
    hdrs = glob(["*.h"]),
    includes = ["."],
    copts = ["-std=c++20"],
)
