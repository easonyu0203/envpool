# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Phase 2 (envpool-ptcg-integration skill): links @ptcg_engine for real
# ApiBattleStart/ApiSelect calls -- every target that transitively #includes
# All.h needs its own copts = ["-std=c++20"], since copts don't propagate
# across cc_library boundaries (see "Bazel vendoring" in the skill; confirmed
# the hard way in Phase 0 via envpool_smoke/ptcg_smoke). Deliberately not
# wired into //envpool:entry or //envpool:envpool (see "Build path" in
# the skill and registration.py's docstring) -- those targets pull in every
# other envpool family (ALE ROMs, MuJoCo, VizDoom, ...), which this
# training-only, never-published project has no reason to build.
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//envpool:requirements.bzl", "requirement")

py_library(
    name = "ptcg",
    srcs = ["__init__.py"],
    data = [":ptcg_envpool"],
    imports = ["../.."],
    deps = ["//envpool/python:api"],
)

py_library(
    name = "ptcg_registration",
    srcs = ["registration.py"],
    imports = ["../.."],
    deps = ["//envpool:registration"],
)

cc_library(
    name = "ptcg_envpool_h",
    hdrs = ["ptcg_envpool.h"],
    copts = ["-std=c++20"],
    deps = [
        "//envpool/core:async_envpool",
        "//envpool/core:env",
        "//envpool/core:logging",
        "@ptcg_engine//:ptcg_engine",
    ],
)

cc_test(
    name = "ptcg_envpool_test",
    size = "small",
    srcs = ["ptcg_envpool_test.cc"],
    copts = ["-std=c++20"],
    deps = [
        ":ptcg_envpool_h",
        "//envpool/core:logging",
        "@com_google_googletest//:gtest_main",
    ],
)

py_test(
    name = "ptcg_py_envpool_test",
    size = "small",
    srcs = ["ptcg_py_envpool_test.py"],
    data = [":ptcg_envpool"],
    imports = ["../.."],
    deps = [
        ":ptcg",
        ":ptcg_registration",
        "//envpool:registration",
        "//envpool/python:api",
        requirement("numpy"),
        requirement("absl-py"),
    ],
)

pybind_extension(
    name = "ptcg_envpool",
    srcs = [
        "ptcg_envpool.cc",
    ],
    copts = ["-std=c++20"],
    linkopts = select({
        "//:windows": [],
        "//conditions:default": [
            "-ldl",
        ],
    }),
    deps = [
        ":ptcg_envpool_h",
        "//envpool/core:py_envpool",
    ],
)
