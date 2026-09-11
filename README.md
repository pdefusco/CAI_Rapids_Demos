# Rapids in Cloudera AI

Collection of articles and quickstarts focused on using Nvidia RAPIDS in Cloudera AI.

Each folder is a self-contained quickstart — click through for the walkthrough.

## Demos

### [cuDF](./cudf)
Single-script demo of a local `dask_cuda.LocalCUDACluster` generating and operating on two 125M-row cuDF DataFrames in parallel from a Cloudera AI session. Entry point: [`cudf/001_local_cuda_cluster.py`](./cudf/001_local_cuda_cluster.py).

### [cuGraph](./cugraph)
GPU-accelerated graph analytics with RAPIDS cuGraph: PageRank benchmarks against NetworkX on public SuiteSparse datasets (DIMACS10, LAW dblp-2010, etc.), personalized PageRank, and multi-GPU PageRank on a `LocalCUDACluster`. Work through the notebooks in order:
- [`00_install_requirements.ipynb`](./cugraph/00_install_requirements.ipynb)
- [`01_pagerank_benchmark.ipynb`](./cugraph/01_pagerank_benchmark.ipynb)
- [`02_personalized_pagerank.ipynb`](./cugraph/02_personalized_pagerank.ipynb)
- [`03_dask_multi_gpu_cugraph.py`](./cugraph/03_dask_multi_gpu_cugraph.py)

### [Dask-CUDA](./dask-cuda)
Multi-node, multi-GPU Dask CUDA clusters on Cloudera AI via the `cmlextensions` `DaskCudaCluster` wrapper — no manual Kubernetes or Docker setup. Includes `dask.array` stress tests, distributed cuDF joins on S3-staged data, and a CDE data-generation pipeline that produces a 50B-row / 10K-column synthetic dataset. See the folder's [README](./dask-cuda/README.md) for setup.

### [Spark-RAPIDS ETL](./spark-rapids-etl)
End-to-end tutorial on running a GPU-accelerated Spark ETL job in Cloudera AI using the Spark 3.3.0 DEX CDE runtime add-on and the `com.nvidia:rapids-4-spark_2.12` plugin: synthetic datagen to S3, an ETL pipeline configured with all the required Spark RAPIDS options (pinned pool, GPU discovery script, shim provider), and `.explain()` output confirming the physical plan runs on the GPU. Full step-by-step in the folder's [README](./spark-rapids-etl/README.md).

### [Spark-RAPIDS ML](./spark-rapids-ml)
Trains a `spark_rapids_ml.classification.RandomForestClassifier` (the GPU-accelerated drop-in for `pyspark.ml`), logs it with MLflow, then deploys it to Cloudera AI Inference (CAII) via the `deployEndpoint` API for real-time inference. Numbered scripts drive the flow:
- [`00_install_requirements.py`](./spark-rapids-ml/00_install_requirements.py)
- [`01_datagen.py`](./spark-rapids-ml/01_datagen.py)
- [`02_model_training.py`](./spark-rapids-ml/02_model_training.py)
- [`03_deploy_model_ai_inf.ipynb`](./spark-rapids-ml/03_deploy_model_ai_inf.ipynb)

### [Spark-RAPIDS Qualification Tool](./spark-rapids-qualification-tool)
Four-stage workflow around NVIDIA's Spark RAPIDS Qualification Tool:
1. Generate a synthetic star-schema warehouse (`01_generate_*.py` — customers, accounts, transactions, branches, merchants, calendar, with skewed variants).
2. Run progressively more complex CPU Spark ETL pipelines that emit event logs ([`02_etl_v1.py`](./spark-rapids-qualification-tool/02_etl_v1.py) … [`02_etl_v11.py`](./spark-rapids-qualification-tool/02_etl_v11.py)).
3. Score those event logs for GPU-acceleration candidacy with [`03_qualification_tool.py`](./spark-rapids-qualification-tool/03_qualification_tool.py).
4. Re-run the ETL with Spark RAPIDS enabled and compare CPU vs GPU ([`04_spark_rapids_etl.py`](./spark-rapids-qualification-tool/04_spark_rapids_etl.py), [`05_compare_cpu_gpu.py`](./spark-rapids-qualification-tool/05_compare_cpu_gpu.py)).

See the folder's [README](./spark-rapids-qualification-tool/README.md) for the full walkthrough.

### [Spark Log Profiler](./spark-log-profiler)
A pure-stdlib Python tool that parses Spark event logs (rolling v2 directories or legacy single-file, gzipped or not) into structured job / stage / task / executor / shuffle / spill / GC data plus SQL-execution timing and RAPIDS GPU accumulator metrics. Its distinguishing feature is a "true speedup" comparison for two runs (e.g. CPU vs GPU) that excludes startup time so warm-vs-cold-start effects don't skew the reported ratio. Entry point: [`profile.py`](./spark-log-profiler/profile.py). Result-interpretation and report-style guides live under [`spark-log-profiler/reference/`](./spark-log-profiler/reference).

## Repo layout

- **[`img/`](./img)** — Shared screenshots and small helpers used across the demos.
- **[`src/`](./src)** — Shared source utilities.
- **[`install_requirements.py`](./install_requirements.py)** — Repo-wide dependency helper.
- **`spark-rapids-qualification-tool/qual_2026*/`** — Timestamped output directories produced by the Spark RAPIDS Qualification Tool. These are run artifacts, not demos.
