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

--batch-size controls async overlap: batch_size == num_envs (the default,
one entry per --num-envs value) is fully synchronous -- every round waits
for all num_envs results before sending the next batch. batch_size <
num_envs runs AsyncEnvPool's actual async mode via async_reset()+recv()/
send(): each recv() returns whichever `batch_size` envs finished first,
so a slow env (e.g. one mid-reset, paying the one-time per-episode
ApiBattleStart shuffle cost -- see the envpool-ptcg-integration skill)
doesn't stall the ones that are ready. --batch-size accepts either one
value (broadcast to every --num-envs entry, for sweeping the num_envs:
batch_size ratio at a fixed batch_size) or a comma list the same length
as --num-envs (per-entry override).

Usage (num_envs defaults to a sweep from 1 to os.cpu_count(), batch_size
defaults to sync i.e. batch_size == num_envs):
    bazel run //envpool/ptcg:bench_envpool -- \\
        --out /abs/path/to/research/env_speed/results_envpool_TIMESTAMP.json
    bazel run //envpool/ptcg:bench_envpool -- --num-envs 1,2,4,8 --duration-sec 5
    # async ratio sweep at a fixed batch_size == num_threads == 8:
    bazel run //envpool/ptcg:bench_envpool -- \\
        --num-envs 8,10,12,16,24,32 --batch-size 8 --num-threads 8
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


def _gymnasium_stepper(num_envs: int, batch_size: int, num_threads: int):
    """The normal, documented path: make() + the Gymnasium wrapper's async
    send/recv API (async_reset() once, then recv()/send() in a loop). This
    is envpool's actual async mechanism -- each recv() drains exactly
    `batch_size` env results, whichever finish first, and send() re-arms
    only those env_ids -- and it subsumes the synchronous case for free:
    when batch_size == num_envs, the first recv() drains the whole round
    just like env.step() would, so callers don't need a separate sync path.
    """
    env = make(
        "Ptcg-v0", env_type="gymnasium", num_envs=num_envs, batch_size=batch_size,
        num_threads=num_threads,
    )
    dummy_action = np.full((batch_size, _ACTION_SLOTS), -1, dtype=np.int32)
    deck_action = np.tile(_DECK, (batch_size, 1)).astype(np.int32)

    env.async_reset()

    def step_once() -> int:
        _obs, _reward, terminated, _truncated, info = env.recv()
        env_id = info["env_id"]
        is_deck_select = info["is_deck_select"]
        action = np.where(is_deck_select[:, None], deck_action, dummy_action)
        env.send(action, env_id)
        return int(terminated.sum())

    return step_once


def _raw_api_stepper(num_envs: int, batch_size: int, num_threads: int):
    """Diagnostic path: raw `_PtcgEnvPool`/`_send`/`_recv`, bypassing both
    make()'s config-kwargs layer and the Gymnasium wrapper's obs/info dict
    reshaping -- same raw pattern ptcg_py_envpool_test.py's
    test_raw_envpool_real_engine_end_to_end uses. Isolates how much of the
    num_envs=1 gap against the ctypes baselines (see the
    envpool-ptcg-integration skill's "Cross-validation"/Part-1 discussion in
    research/env_speed/README.md) is the wrapper layers specifically, versus
    AsyncEnvPool's own queue/pybind cost underneath both paths.

    Sync-only (batch_size must equal num_envs): this path's env_id handling
    always sends/expects the full arange(num_envs) every round, unlike
    _gymnasium_stepper's async recv()-driven env_id. Not worth generalizing
    -- it exists to isolate wrapper overhead, not to run the async sweep.
    """
    if batch_size != num_envs:
        raise ValueError("--raw-api only supports batch_size == num_envs (sync)")

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
    num_envs: int, batch_size: int, num_threads: int, duration_sec: float, warmup_sec: float,
    raw_api: bool
) -> dict:
    step_once = (_raw_api_stepper if raw_api else _gymnasium_stepper)(
        num_envs, batch_size, num_threads
    )

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
    # Each round's step_once() advances exactly one recv()-sized batch of
    # envs (== batch_size, whether or not that equals num_envs -- see
    # _gymnasium_stepper), not the full num_envs.
    total_decisions = rounds * batch_size

    return {
        "num_envs": num_envs,
        "batch_size": batch_size,
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
    parser.add_argument(
        "--batch-size", default=None,
        help="batch_size per config: one value (broadcast to every --num-envs entry, for an "
        "async num_envs:batch_size ratio sweep) or a comma list the same length as --num-envs "
        "(per-entry override). Default: batch_size == num_envs for every entry (fully sync).",
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

    if args.batch_size is None:
        batch_size_list = list(num_envs_list)
    else:
        raw = [int(x) for x in args.batch_size.split(",")]
        batch_size_list = raw if len(raw) > 1 else raw * len(num_envs_list)
        if len(batch_size_list) != len(num_envs_list):
            raise SystemExit(
                f"--batch-size has {len(batch_size_list)} values, expected 1 or "
                f"{len(num_envs_list)} (matching --num-envs)"
            )
    for ne, bs in zip(num_envs_list, batch_size_list, strict=True):
        if bs > ne:
            raise SystemExit(f"batch_size ({bs}) must be <= num_envs ({ne})")

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
    for num_envs, batch_size in zip(num_envs_list, batch_size_list, strict=True):
        print(
            f"running num_envs={num_envs} batch_size={batch_size} "
            f"num_threads={args.num_threads} raw_api={args.raw_api} ...",
            file=sys.stderr,
        )
        result = run_one_config(
            num_envs, batch_size, args.num_threads, args.duration_sec, args.warmup_sec,
            args.raw_api
        )
        runs.append(result)
        print(
            f"  num_envs={num_envs} batch_size={batch_size}: games/sec={result['games_per_sec']:.2f} "
            f"decisions/sec={result['decisions_per_sec']:.1f} "
            f"completed_games={result['completed_games']}",
            file=sys.stderr,
        )

    baseline = runs[0]["games_per_sec"] if runs else 1.0
    for r in runs:
        r["scaling_vs_first_run"] = r["games_per_sec"] / baseline if baseline > 0 else None

    out = {"meta": meta, "runs": runs}

    if not args.out:
        raise SystemExit("--out is required (pass an absolute path -- bazel run's cwd is not the repo root)")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
