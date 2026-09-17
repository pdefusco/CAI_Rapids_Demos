#!/usr/bin/env python3
"""Compare the most recent CPU vs GPU Spark ETL runs in this project.

Auto-discovers event logs under spark-event-logs-dir (the directory every
ETL script here points spark.eventLog.dir at), profiles the two most
recent ones with ../spark-log-profiler, and reports the speedup as a text
summary -- optionally also saving a matplotlib bar chart. Run it after
02_etl_cpu.py and its GPU counterpart 04_etl_gpu.py have both run.

Usage:
  python3 05_compare_cpu_gpu.py                latest two runs, text summary
  python3 05_compare_cpu_gpu.py --chart         also save a PNG bar chart
  python3 05_compare_cpu_gpu.py --single        analyze only the single latest run
  python3 05_compare_cpu_gpu.py --list          list discovered event logs, newest first
  python3 05_compare_cpu_gpu.py --dir <path>    look in a different directory

No required arguments -- works the same whether launched from a terminal
or run directly in a Cloudera AI Workbench session.

Author: Brandon Antone
"""
import argparse
import importlib.util
import os
import sys

DEFAULT_EVENT_LOG_DIR = "/home/cdsw/spark-rapids-qualification-tool/spark-event-logs-dir"

PROFILER_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "spark-log-profiler", "profile.py"
)

# Prefixes Spark uses for event log files/directories: v2 rolling
# (eventlog_v2_spark-<appid>), legacy single-file (local-<ts>,
# spark-application-<ts>, app-<ts>).
EVENT_LOG_PREFIXES = ("eventlog_v2_", "local-", "spark-application-", "app-")


def load_profiler():
    spec = importlib.util.spec_from_file_location(
        "_spark_log_profiler", PROFILER_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_event_logs(event_log_dir):
    if not os.path.isdir(event_log_dir):
        return []

    candidates = [
        os.path.join(event_log_dir, name)
        for name in os.listdir(event_log_dir)
        if name.startswith(EVENT_LOG_PREFIXES)
    ]
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates


def save_chart(run_a, run_b, label_a, label_b, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ta, tb = run_a["timing"], run_b["timing"]
    naive = (
        ta["total_ms"] / tb["total_ms"]
        if ta["total_ms"] and tb["total_ms"] else None
    )
    true_speedup = (
        ta["execution_ms"] / tb["execution_ms"]
        if ta["execution_ms"] and tb["execution_ms"] else None
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(
        [label_a, label_b],
        [ta["total_ms"] / 1000, tb["total_ms"] / 1000],
        color=["tab:blue", "tab:green"],
    )
    axes[0].set_ylabel("Total time (s)")
    axes[0].set_title("Total runtime")

    speedup_pairs = [("naive", naive), ("true", true_speedup)]
    speedup_labels = [n for n, v in speedup_pairs if v is not None]
    speedups = [v for _, v in speedup_pairs if v is not None]

    axes[1].bar(speedup_labels, speedups, color="tab:orange")
    axes[1].set_ylabel("Speedup (x)")
    axes[1].set_title(f"{label_a} / {label_b} speedup")
    for i, v in enumerate(speedups):
        axes[1].text(i, v, f"{v:.2f}x", ha="center", va="bottom")

    fig.suptitle("CPU vs GPU comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nChart saved to {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--dir",
        default=DEFAULT_EVENT_LOG_DIR,
        help="event log directory to search (default: %(default)s)",
    )
    ap.add_argument(
        "--single",
        action="store_true",
        help="analyze only the single most recent run instead of comparing two",
    )
    ap.add_argument(
        "--chart",
        action="store_true",
        help="also save a matplotlib PNG bar chart of the comparison",
    )
    ap.add_argument(
        "--chart-out",
        default="cpu_vs_gpu_comparison.png",
        help="path to save the chart PNG (default: %(default)s)",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="list discovered event logs (newest first) and exit",
    )
    args = ap.parse_args()

    logs = find_event_logs(args.dir)
    if not logs:
        sys.exit(f"No event logs found under {args.dir}")

    if args.list:
        for path in logs:
            print(path)
        return

    profiler = load_profiler()

    if args.single:
        run = profiler.profile_run(logs[0])
        print(profiler.summarize_run(run))
        return

    if len(logs) < 2:
        sys.exit(
            f"Need two event logs to compare, only found one under {args.dir} "
            "(use --single to analyze it alone)"
        )

    # Oldest of the two first so labels/ordering read naturally
    # (e.g. CPU run before GPU run, if CPU ran first).
    path_a, path_b = logs[1], logs[0]
    run_a = profiler.profile_run(path_a)
    run_b = profiler.profile_run(path_b)
    label_a, label_b = profiler.label_run(run_a), profiler.label_run(run_b)

    print(profiler.summarize_run(run_a, label=label_a))
    print()
    print(profiler.summarize_run(run_b, label=label_b))
    print(profiler.summarize_comparison(run_a, run_b, label_a, label_b))

    if args.chart:
        try:
            save_chart(run_a, run_b, label_a, label_b, args.chart_out)
        except ImportError:
            print(
                "\nmatplotlib not installed -- skipping chart. "
                "Install with: pip install matplotlib"
            )


if __name__ == "__main__":
    main()
