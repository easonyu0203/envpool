# Copyright 2021 Garena Online Private Limited
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
"""Phase 7 (envpool-ptcg-integration skill): parallel-envpool throughput,
extending research/env_speed/'s convention (meta block, timestamped results
JSON) with a num_envs scaling curve -- the ctypes baseline there
(research/env_speed/bench.py) is inherently single-process/single-core
(the ctypes bindings hold one global battle pointer per process), so it has
no such axis to sweep in the first place.

Must run through Bazel (`bazel run //envpool/ptcg:bench_envpool`), not a bare
`python`/`uv run` invocation -- `envpool.ptcg`'s compiled extension only
resolves correctly via Bazel's runfiles, same reasoning as every ptcg_py_*
test in this package. See "Build path" in the skill for the
`envpool.ptcg.registration` + `envpool.registration.make` access pattern
this reuses.

Self-play policy: every action sent is legal by construction -- a real deck
during deck-select, or a -1-filled dummy during decide() that Phase 4's
under-minCount padding always turns into a legal random pick. Same policy
shape research/env_speed/bench.py's baseline uses (uniformly random among
valid options), so the two numbers are a fair throughput comparison even
though neither is a trained policy.

Usage (num_envs defaults to a sweep from 1 to os.cpu_count()):
    bazel run //envpool/ptcg:bench_envpool -- \\
        --out /abs/path/to/research/env_speed/results_envpool_TIMESTAMP.json
    bazel run //envpool/ptcg:bench_envpool -- --num-envs 1,2,4,8 --duration-sec 5
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

import envpool.ptcg.registration  # noqa: F401 -- side effect: registers "Ptcg-v0"
from envpool.registration import make  # NOT `envpool.make` -- see "Build path" in the skill

_ACTION_SLOTS = 60
# Same real, deck-legal 60-card list used throughout this package's tests
# (submissions/sample_submission/deck.csv).
_DECK = [
    1158, 721, 721, 722, 722, 722, 722, 723, 723, 723, 723, 1145,
    1145, 1145, 1145, 1205, 1205, 1227, 1227, 1227, 1227, 1235, 1235, 1235,
    1235, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
]


def _git_commit() -> str | None:
    # Best-effort only -- under `bazel run`, __file__ resolves to a runfiles
    # path, not the source tree, so there's no reliable relative-path way
    # to locate either repo's root from here. Whatever the process's actual
    # cwd happens to be (BUILD_WORKING_DIRECTORY under `bazel run`, or
    # wherever this was invoked from) is the best available guess; None on
    # any failure is an acceptable outcome for a metadata-only field.
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _gymnasium_stepper(num_envs: int, num_threads: int):
    """The normal, documented path: make() + the Gymnasium wrapper."""
    env = make(
        "Ptcg-v0", env_type="gymnasium", num_envs=num_envs, batch_size=num_envs,
        num_threads=num_threads,
    )
    dummy_action = np.full((num_envs, _ACTION_SLOTS), -1, dtype=np.int32)
    deck_action = np.tile(_DECK, (num_envs, 1)).astype(np.int32)

    _obs, info = env.reset()
    is_deck_select = info["is_deck_select"]

    def step_once() -> int:
        nonlocal is_deck_select
        action = np.where(is_deck_select[:, None], deck_action, dummy_action)
        _obs, _reward, terminated, _truncated, info = env.step(action)
        is_deck_select = info["is_deck_select"]
        return int(terminated.sum())

    return step_once


def _raw_api_stepper(num_envs: int, num_threads: int):
    """Diagnostic path: raw `_PtcgEnvPool`/`_send`/`_recv`, bypassing both
    make()'s config-kwargs layer and the Gymnasium wrapper's obs/info dict
    reshaping -- same raw pattern ptcg_py_envpool_test.py's
    test_raw_envpool_real_engine_end_to_end uses. Isolates how much of the
    num_envs=1 gap against the ctypes baselines (see the
    envpool-ptcg-integration skill's "Cross-validation"/Part-1 discussion in
    research/env_speed/README.md) is the wrapper layers specifically, versus
    AsyncEnvPool's own queue/pybind cost underneath both paths.
    """
    from envpool.ptcg.ptcg_envpool import _PtcgEnvPool, _PtcgEnvSpec

    conf = dict(
        zip(_PtcgEnvSpec._config_keys, _PtcgEnvSpec._default_config_values, strict=False)
    )
    conf.update(num_envs=num_envs, batch_size=num_envs, num_threads=num_threads)
    env_spec = _PtcgEnvSpec(tuple(conf.values()))
    env = _PtcgEnvPool(env_spec)
    state_keys = env._state_keys
    env_id = np.arange(num_envs, dtype=np.int32)

    def recv_dict() -> dict:
        return dict(zip(state_keys, env._recv(), strict=False))

    dummy_action = np.full((num_envs, _ACTION_SLOTS), -1, dtype=np.int32)
    deck_action = np.tile(_DECK, (num_envs, 1)).astype(np.int32)

    env._reset(env_id)
    state = recv_dict()
    is_deck_select = state["info:is_deck_select"]

    def step_once() -> int:
        nonlocal is_deck_select
        action = np.where(is_deck_select[:, None], deck_action, dummy_action)
        env._send((env_id, env_id, action))
        state = recv_dict()
        is_deck_select = state["info:is_deck_select"]
        return int(state["done"].sum())

    return step_once


def run_one_config(
    num_envs: int, num_threads: int, duration_sec: float, warmup_sec: float, raw_api: bool
) -> dict:
    step_once = (_raw_api_stepper if raw_api else _gymnasium_stepper)(num_envs, num_threads)

    t_end = time.perf_counter() + warmup_sec
    while time.perf_counter() < t_end:
        step_once()

    completed_games = 0
    rounds = 0
    t_start = time.perf_counter()
    t_end = t_start + duration_sec
    while time.perf_counter() < t_end:
        completed_games += step_once()
        rounds += 1
    wall_s = time.perf_counter() - t_start
    total_decisions = rounds * num_envs

    return {
        "num_envs": num_envs,
        "num_threads": num_threads,
        "raw_api": raw_api,
        "rounds": rounds,
        "total_decisions": total_decisions,
        "completed_games": completed_games,
        "wall_s": wall_s,
        "decisions_per_sec": total_decisions / wall_s,
        "games_per_sec": completed_games / wall_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--num-envs", default=None,
        help="comma-separated num_envs values to sweep (default: 1,2,4,...,os.cpu_count())",
    )
    parser.add_argument("--num-threads", type=int, default=0, help="0 = envpool auto (min(batch, cpu_count))")
    parser.add_argument("--duration-sec", type=float, default=10.0, help="timed duration per config")
    parser.add_argument("--warmup-sec", type=float, default=2.0, help="untimed warmup duration per config")
    parser.add_argument(
        "--raw-api", action="store_true",
        help="bypass make()/Gymnasium, use raw _PtcgEnvPool/_send/_recv directly (diagnostic)",
    )
    parser.add_argument("--out", default=None, help="results JSON path (required)")
    args = parser.parse_args()

    if args.num_envs:
        num_envs_list = [int(x) for x in args.num_envs.split(",")]
    else:
        cpu_count = os.cpu_count() or 4
        num_envs_list = []
        n = 1
        while n < cpu_count:
            num_envs_list.append(n)
            n *= 2
        num_envs_list.append(cpu_count)

    meta = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "num_threads_requested": args.num_threads,
        "duration_sec_per_config": args.duration_sec,
        "warmup_sec_per_config": args.warmup_sec,
        "raw_api": args.raw_api,
    }

    runs = []
    for num_envs in num_envs_list:
        print(f"running num_envs={num_envs} num_threads={args.num_threads} raw_api={args.raw_api} ...", file=sys.stderr)
        result = run_one_config(
            num_envs, args.num_threads, args.duration_sec, args.warmup_sec, args.raw_api
        )
        runs.append(result)
        print(
            f"  num_envs={num_envs}: games/sec={result['games_per_sec']:.2f} "
            f"decisions/sec={result['decisions_per_sec']:.1f} "
            f"completed_games={result['completed_games']}",
            file=sys.stderr,
        )

    baseline = runs[0]["games_per_sec"] if runs else 1.0
    for r in runs:
        r["scaling_vs_num_envs_1"] = r["games_per_sec"] / baseline if baseline > 0 else None

    out = {"meta": meta, "runs": runs}

    if not args.out:
        raise SystemExit("--out is required (pass an absolute path -- bazel run's cwd is not the repo root)")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
