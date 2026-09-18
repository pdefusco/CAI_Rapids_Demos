# Spark RAPIDS Qualification Tool Demo

This repository demonstrates how to use the **NVIDIA Spark RAPIDS Qualification Tool** to evaluate Apache Spark workloads for GPU acceleration and compare CPU execution against GPU-accelerated execution using the Spark RAPIDS Accelerator.

The project walks through a complete end-to-end workflow:

1. Generate a synthetic data warehouse
2. Execute representative Spark ETL workloads
3. Analyze the Spark Event Logs with the Spark RAPIDS Qualification Tool
4. Re-run the same workloads with Spark RAPIDS enabled to measure performance improvements

---

# What is Spark RAPIDS?

The **Spark RAPIDS Accelerator** is an open-source plugin for Apache Spark that enables Spark SQL and DataFrame operations to execute on NVIDIA GPUs instead of CPUs.

Rather than requiring applications to be rewritten, Spark RAPIDS replaces many Spark physical operators with GPU-accelerated implementations while preserving the existing Spark APIs. This allows many Spark applications to achieve significant reductions in execution time with minimal code changes.

Before enabling GPU acceleration, it is useful to determine whether an application is actually a good candidate for acceleration. This is the purpose of the **Spark RAPIDS Qualification Tool**.

The Qualification Tool analyzes Spark Event Logs and estimates:

- Whether an application is a good candidate for GPU acceleration
- Which Spark operators are GPU compatible
- Estimated execution time improvements
- Potential GPU utilization
- Recommended GPU cluster sizing

---

# Repository Workflow

The repository is organized into four stages.

## Step 1 — Create the Demo Tables

Run the scripts whose filenames begin with **`1_`**.

These scripts generate the synthetic datasets used throughout the demo, including fact tables and dimension tables that resemble a small analytical data warehouse.

Typical datasets include:

- Customers
- Accounts
- Transactions
- Branches
- Products
- Additional supporting dimension tables

These datasets provide enough scale to exercise Spark joins, aggregations, filters, and shuffle operations.

---

## Step 2 — Execute the ETL Workloads

Run the scripts whose filenames begin with **`2_`**.

These scripts execute increasingly complex Spark ETL pipelines against the generated data.

The workloads are designed to exercise operations that are commonly accelerated by Spark RAPIDS, including:

- Large joins
- Wide aggregations
- GroupBy operations
- Sorting
- Window functions
- Shuffle-intensive transformations

Running these jobs also generates the Spark Event Logs that will later be analyzed by the Qualification Tool.

---

## Step 3 — Run the Spark RAPIDS Qualification Tool

Run the script whose filename begins with **`3_`**.

This step analyzes the Spark Event Logs produced during the ETL runs.

The Qualification Tool generates reports describing:

- GPU compatibility
- Estimated acceleration
- SQL operator analysis
- Recommended GPU configuration
- Estimated runtime improvements

The generated report helps determine whether the workloads are good candidates for GPU acceleration before any infrastructure changes are made.

---

## Step 4 — Re-run the ETL Jobs with Spark RAPIDS

Finally, enable the Spark RAPIDS Accelerator and execute the same ETL applications again.

The goal of this step is to compare:

- CPU execution time
- GPU execution time
- Overall speedup
- Resource utilization

Because the application code remains unchanged, the comparison highlights the performance gains obtained simply by enabling Spark RAPIDS.

---

# Prerequisites

Before running the Qualification Tool, ensure that Spark Event Logging is enabled and configured to write logs into the project directory.

In your **Cloudera AI Workbench** project settings, configure the Spark Event Log directory as:

```text
/home/cdsw/spark-rapids-qualification-tool
```

This allows the Qualification Tool to locate and analyze the generated Spark Event Logs.

---

# Repository Structure

```text
1_*    Generate demo datasets

2_*    Execute Spark ETL workloads

3_*    Run the Spark RAPIDS Qualification Tool

4_*    Execute the same workloads with Spark RAPIDS enabled
```

---

# Expected Outcome

After completing this workflow, you will have:

- Generated a realistic Spark analytics workload
- Produced Spark Event Logs
- Evaluated the workload using the Spark RAPIDS Qualification Tool
- Identified GPU acceleration opportunities
- Measured the performance improvements achieved by enabling Spark RAPIDS

This repository provides a practical introduction to evaluating and benchmarking Spark GPU acceleration using NVIDIA Spark RAPIDS.

---

# GPU Tuning Log (250M transactions, T4)

Baseline (before any tuning): **CPU ≈ 300s, GPU ≈ 290s** — GPU was barely winning, indicating shuffle-bound behavior rather than a compute bottleneck.

Each attempt below changes **only** SparkSession config. ETL logic, data, and cluster shape are unchanged so wall-clocks stay comparable.

**Version-per-attempt files.** Each row is applied cumulatively to a distinct `04_etl_gpu_Vx.py` file, so nothing gets overwritten. `V1` = baseline + row 1's change, `V2` = `V1` + row 2's change, and so on. To test a row: open its `Vx` file in a CAI session, run it, and copy the printed `GPU wall-clock (ETL)` value into the table.

| # | File | Config change | From → To | Why | GPU wall-clock (ETL) | Δ vs baseline | Commit |
|---|---|---|---|---|---|---|---|
| 0 | *pre-tuning* | *baseline* | — | Config as-shipped in `t4 use case` commit | ~290s | — | `ceb336a` |
| 1 | `04_etl_gpu_V1.py` | `spark.sql.autoBroadcastJoinThreshold` | `-1` → `512m` | Broadcasts customers/merchants/branches/calendar; eliminates 4 of 5 join shuffles. account_dim (~1 GB) stays as shuffle-hash join. | **151.8s** | **138.2s** | `68d6c37` |
| 2 | `04_etl_gpu_V2.py` | `spark.sql.files.maxPartitionBytes` | `4g` → `1g` | 4 GB per input partition under-parallelizes the initial fact scan. 1 GB gives ~30–60 read partitions across 8 executors × 8 cores. | **128.2s** | **161.8s** |  |
| 3 | `04_etl_gpu_V3.py` | `spark.rapids.sql.concurrentGpuTasks` | `2` → `3` | Standard T4 sweet spot at 12g executor + 4g pinned. Improves GPU utilization on the withColumn-heavy stage. |  |  |  |
| 4 | `04_etl_gpu_V4.py` | `spark.rapids.memory.pinnedPool.size` | `2g` → `4g` | Faster host↔device transfers. Comes out of the 8g overhead, so no container change. |  |  |  |
| 5 | `04_etl_gpu_V5.py` | `spark.task.resource.gpu.amount` | `0.125` → `0.25` | 8 task slots per executor competing for a GPU that only runs 3 concurrent tasks creates scheduler churn. 4 slots aligns better. |  |  |  |
| 6 | `04_etl_gpu_V6.py` | *(new)* `spark.rapids.sql.reader.multithreaded.combine.sizeBytes` | (unset) → `32m` | Combines small parquet row groups on the fact read. Small but free win. |  |  |  |

**How to run:** open the `Vx` file for the row you want to test in a CAI session and execute it. Paste the terminal output back in this thread and I'll parse the `GPU wall-clock (ETL)` line, fill in the row, and prepare the next `Vx`. `Δ vs baseline` = baseline − current (positive = faster).

## Additional tunings to try after the six above

Kept separate because each has a caveat — apply only if the first six leave headroom on the table.

| Config change | From → To | Caveat |
|---|---|---|
| `spark.rapids.sql.hasNans` | (default `true`) → `false` | Speeds up float aggregations. Safe only if the data has no NaN — our synthetic generators don't produce any, so this is safe here. |
| `spark.executor.memoryOverhead` + `spark.executor.memory` rebalance | `8g` + `12g` → `6g` + `14g` | Moves 2 GB back into the JVM heap for shuffle spill headroom. Only worth it if we see disk spill in the Spark UI. |
| `spark.rapids.sql.batchSizeBytes` | `1g` → `512m` | If GPU OOMs or spills appear at concurrentGpuTasks=3, halve the batch size to trade throughput for headroom. |
| `spark.rapids.filecache.enabled` | `false` → `true` | Only helps if the fact table is scanned more than once. Currently it isn't — leave off unless the ETL evolves. |
| Drop `spark.sql.adaptive.coalescePartitions.initialPartitionNum` and `minPartitionSize` | remove both | Both are no-ops today because the ETL uses explicit `.repartition(SHUFFLE_PARTITIONS, keys)`, which locks partition count past AQE. Cosmetic cleanup once broadcast joins remove the AQE-visible join shuffles. |
| `spark.rapids.sql.explain` | (unset) → `NOT_ON_GPU` | Diagnostic, not a tuning. Set once, read executor logs, confirm zero CPU fallbacks, then unset. |

---
