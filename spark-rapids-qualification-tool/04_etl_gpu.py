#****************************************************************************
# (C) Cloudera, Inc. 2020-2026
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
# #  Author(s): Paul de Fusco
#***************************************************************************/
#
# ============================================================
# GPU ETL: RAPIDS version of 02_etl_cpu.py
# ============================================================
#
# The transformation logic below is BYTE-FOR-BYTE identical to
# 02_etl_cpu.py. Only the SparkSession builder differs: it applies
# the qualification-tool recommendations plus the CAI-specific GPU
# plumbing needed for the RAPIDS Accelerator plugin to actually
# load in this workspace.
#
# See 02_etl_cpu.py header for the rationale on what was kept and
# cut vs the full v9 pipeline (kept under archive/02_etl_v9.py).
#
# Compare wall-clock against 02_etl_cpu.py -- NOT against
# archive/02_etl_v9.py -- because the fact filter and aggregation
# shape only match between 02_etl_cpu.py and 04_etl_gpu.py.
# ============================================================

import os
import time
import warnings
import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Module-level timer: includes SparkSession startup so we can see
# session-launch overhead separately from ETL work.
t0 = time.time()


# ============================================================
# Configuration
# ============================================================

DATABASE = "DEMO_pauldefusco"

TRANSACTION_TABLE = f"{DATABASE}.TRS_v26" # 1 B transactions, filtered below
CUSTOMER_TABLE = f"{DATABASE}.CUSTOMERS_skewed"
ACCOUNT_TABLE = f"{DATABASE}.ACCOUNTS_skewed"
MERCHANT_TABLE = f"{DATABASE}.MERCHANTS_skewed"
BRANCH_TABLE = f"{DATABASE}.BRANCHES_skewed"
CALENDAR_TABLE = f"{DATABASE}.CALENDAR"

# Output tables suffixed with _GPU so CPU and GPU runs can coexist
# and be compared without either clobbering the other.
OUTPUT_TABLE_CUSTOMER = f"{DATABASE}.ETL_V9_SIMPLE_CUSTOMER_MONTH_GPU"
OUTPUT_TABLE_MERCHANT = f"{DATABASE}.ETL_V9_SIMPLE_MERCHANT_QUARTER_GPU"

# MUST match 02_etl_cpu.py exactly, or the wall-clock
# comparison is not apples-to-apples.
FACT_ROW_ID_CEILING = 500_000_000

SHUFFLE_PARTITIONS = 1000

RAPIDS_JAR_PATH = "/home/cdsw/.ivy2/jars/com.nvidia_rapids-4-spark_2.12-26.02.0.jar"

# Fresh event log for a follow-up profiling pass.
EVENT_LOG_DIR_LOCAL = "/home/cdsw/spark-rapids-qualification-tool/spark-event-logs-dir"
os.makedirs(EVENT_LOG_DIR_LOCAL, exist_ok=True)


# ============================================================
# Spark Session (GPU / RAPIDS)
#
# Session config carried from archive/04_spark_rapids_etl_v9.2.py
# (the most recent known-good T4 loader): 8 static executors, network
# resilience configs raised, RAPIDS spark354 shim, RapidsShuffleManager,
# discovery script. Simpler pipeline should keep wall-clock well
# under the ~5-min pod-deletion window, so the network resilience
# knobs are belt-and-suspenders.
# ============================================================

spark = (
    SparkSession.builder

    .appName("Spark-Rapids-ETL-GPU")

    # ------------------------------------------------------------
    # Resources (constrained to CAI T4 GPU capacity)
    # ------------------------------------------------------------
    .config("spark.dynamicAllocation.enabled", "false")
    .config("spark.executor.instances", "8")           # matches available GPUs
    .config("spark.executor.cores", "8")
    .config("spark.executor.memory", "12g")
    .config("spark.executor.memoryOverhead", "8g")     # RAPIDS pinned + spill
    .config("spark.driver.cores", 4)
    .config("spark.driver.memory", "10g")

    # ------------------------------------------------------------
    # Network resilience for shuffle fetch under peer churn
    # ------------------------------------------------------------
    # Kept from v9.2 as a safety net. On this simple pipeline the
    # wall-clock should be short enough that pod-deletion at the
    # ~5-min mark never triggers, but if it does, these give the
    # RPC layer more room to recover before task retry ceilings hit.
    .config("spark.network.timeout", "600s")
    .config("spark.rpc.askTimeout", "600s")
    .config("spark.executor.heartbeatInterval", "60s")
    .config("spark.shuffle.io.maxRetries", "10")
    .config("spark.shuffle.io.retryWait", "15s")
    .config("spark.shuffle.io.connectionTimeout", "600s")
    .config("spark.rpc.io.connectionTimeout", "600s")
    .config("spark.rpc.lookupTimeout", "600s")
    .config("spark.core.connection.ack.wait.timeout", "600s")
    .config("spark.stage.maxConsecutiveAttempts", "8")

    # ------------------------------------------------------------
    # GPU scheduling
    # ------------------------------------------------------------
    .config("spark.executor.resource.gpu.amount", "1")
    .config("spark.task.resource.gpu.amount", 0.125)
    .config("spark.executor.resource.gpu.vendor", "nvidia.com")
    .config(
        "spark.executor.resource.gpu.discoveryScript",
        "/home/cdsw/getGpusResources.sh",
    )

    # ------------------------------------------------------------
    # RAPIDS plugin + jar classpath
    # ------------------------------------------------------------
    .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
    .config("spark.jars.packages", "com.nvidia:rapids-4-spark_2.12:26.02.0")
    .config("spark.driver.extraClassPath",   RAPIDS_JAR_PATH)
    .config("spark.executor.extraClassPath", RAPIDS_JAR_PATH)
    # CAI runtime is Spark 3.5.4 -> use the matching spark354 shim.
    .config(
        "spark.rapids.shims-provider-override",
        "com.nvidia.spark.rapids.shims.spark354.SparkShimServiceProvider",
    )

    # ------------------------------------------------------------
    # RAPIDS shuffle + kryo
    # ------------------------------------------------------------
    .config(
        "spark.shuffle.manager",
        "com.nvidia.spark.rapids.spark354.RapidsShuffleManager",
    )
    .config(
        "spark.kryo.registrator",
        "com.nvidia.spark.rapids.GpuKryoRegistrator",
    )

    # ------------------------------------------------------------
    # RAPIDS SQL knobs
    # ------------------------------------------------------------
    .config("spark.rapids.sql.enabled", "true")
    .config("spark.rapids.filecache.enabled", "false")
    .config("spark.rapids.memory.pinnedPool.size", "2g")
    .config("spark.rapids.sql.batchSizeBytes", "1g")
    .config("spark.rapids.sql.concurrentGpuTasks", "2")
    .config("spark.rapids.sql.multiThreadedRead.numThreads", "32")
    .config("spark.rapids.shuffle.multiThreaded.reader.threads", "24")
    .config("spark.rapids.shuffle.multiThreaded.writer.threads", "24")

    # ------------------------------------------------------------
    # Spark SQL / AQE
    # ------------------------------------------------------------
    .config("spark.sql.shuffle.partitions", SHUFFLE_PARTITIONS)
    .config("spark.sql.files.maxPartitionBytes", "4g")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.initialPartitionNum", "1000")
    .config("spark.sql.adaptive.coalescePartitions.minPartitionSize", "4m")
    .config("spark.sql.adaptive.coalescePartitions.parallelismFirst", "false")
    .config("spark.locality.wait", "0")

    # ------------------------------------------------------------
    # Event log (comparable to CPU-simple)
    # ------------------------------------------------------------
    .config("spark.eventLog.enabled", "true")
    .config("spark.eventLog.dir", f"file://{EVENT_LOG_DIR_LOCAL}")

    # ------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------
    .config(
        "spark.kerberos.access.hadoopFileSystems",
        "s3a://goes-se-sandbox/data",
    )

    .getOrCreate()
)

# Skewed dims make auto-broadcast heuristics unreliable here.
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# Spark UI
print("SPARK UI:\n")
print("https://spark-"+os.environ["CDSW_ENGINE_ID"]+"."+os.environ["CDSW_DOMAIN"])

# Post-session timer: measures just the ETL work, excluding session startup.
_v9_simple_gpu_start = time.time()


# ============================================================
# Helper
# ============================================================

def section(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)
    print()


# ============================================================
# Load Tables
# ============================================================

section("Loading source tables (GPU ETL)")

transactions = (
    spark.table(TRANSACTION_TABLE)
    .filter(
        F.col("transaction_id") < FACT_ROW_ID_CEILING
    )
)

customers = spark.table(CUSTOMER_TABLE)
accounts = spark.table(ACCOUNT_TABLE)
merchants = spark.table(MERCHANT_TABLE)
branches = spark.table(BRANCH_TABLE)
calendar = spark.table(CALENDAR_TABLE)


# ============================================================
# Dimension Projections
# ============================================================

customer_dim = (
    customers
    .select(
        "customer_id",
        "age",
        "credit_score",
        "state",
        "city",
        "income_band",
        "estimated_income",
        "customer_segment",
        "tenure_years",
        "risk_rating",
    )
)


account_dim = (
    accounts
    .select(
        "account_id",
        "customer_id",
        "account_type",
        "currency",
        "account_status",
        "opened_year",
        "branch_id",
        "current_balance",
        "credit_limit",
        "interest_rate",
    )
)


merchant_dim = (
    merchants
    .select(
        "merchant_id",
        "state",
        "region",
        "merchant_category",
        "merchant_name",
        "risk_level",
        "annual_revenue",
        "merchant_size",
        "opened_year",
        "active",
    )
)


branch_dim = (
    branches
    .select(
        "branch_id",
        "branch_name",
        "state",
        "region",
        "branch_type",
        "employee_count",
        "assets_under_management",
        "annual_operating_cost",
        "opened_year",
        "manager_id",
        "status",
    )
)


calendar_dim = (
    calendar
    .select(
        "calendar_date",
        "date_key",
        "year",
        "quarter",
        "month",
        "month_name",
        "week_of_year",
        "day_of_month",
        "day_of_week",
        "day_name",
        "is_weekend",
        "is_month_end",
    )
)


# ============================================================
# Transaction Enrichment
# ============================================================

section("Transaction enrichment")


enriched = (
    transactions.alias("t")

    .join(
        customer_dim.alias("c"),
        F.col("t.customer_id") == F.col("c.customer_id"),
        "left",
    )

    .join(
        account_dim.alias("a"),
        F.col("t.account_id") == F.col("a.account_id"),
        "left",
    )

    .join(
        merchant_dim.alias("m"),
        F.col("t.merchant_id") == F.col("m.merchant_id"),
        "left",
    )

    .join(
        branch_dim.alias("b"),
        F.col("t.branch_id") == F.col("b.branch_id"),
        "left",
    )

    .join(
        calendar_dim.alias("cal"),
        F.col("t.transaction_date")
        == F.col("cal.calendar_date"),
        "left",
    )

    .select(

        # ----- Transaction identifiers -----
        F.col("t.transaction_id"),
        F.col("t.customer_id"),
        F.col("t.account_id"),
        F.col("t.merchant_id"),
        F.col("t.branch_id"),

        F.col("t.transaction_date"),
        F.col("t.transaction_timestamp"),

        # ----- Transaction attributes -----
        F.col("t.merchant_category").alias("txn_merchant_category"),
        F.col("t.transaction_amount"),
        F.col("t.payment_channel"),
        F.col("t.payment_type"),
        F.col("t.device_type"),
        F.col("t.fraud_flag"),
        F.col("t.latitude"),
        F.col("t.longitude"),

        # ----- Customer attributes -----
        F.col("c.age"),
        F.col("c.credit_score"),
        F.col("c.state").alias("customer_state"),
        F.col("c.city"),
        F.col("c.income_band"),
        F.col("c.estimated_income"),
        F.col("c.customer_segment"),
        F.col("c.tenure_years"),
        F.col("c.risk_rating"),

        # ----- Account attributes -----
        F.col("a.account_type"),
        F.col("a.currency"),
        F.col("a.account_status"),
        F.col("a.opened_year").alias("account_opened_year"),
        F.col("a.current_balance"),
        F.col("a.credit_limit"),
        F.col("a.interest_rate"),

        # ----- Merchant attributes -----
        F.col("m.state").alias("merchant_state"),
        F.col("m.region").alias("merchant_region"),
        F.col("m.merchant_category").alias("dimension_merchant_category"),
        F.col("m.risk_level").alias("merchant_risk_level"),
        F.col("m.annual_revenue"),
        F.col("m.merchant_size"),
        F.col("m.active").alias("merchant_active"),

        # ----- Branch attributes -----
        F.col("b.state").alias("branch_state"),
        F.col("b.region").alias("branch_region"),
        F.col("b.branch_type"),
        F.col("b.employee_count"),
        F.col("b.assets_under_management"),
        F.col("b.annual_operating_cost"),
        F.col("b.status").alias("branch_status"),

        # ----- Calendar attributes -----
        F.col("cal.year"),
        F.col("cal.quarter"),
        F.col("cal.month"),
        F.col("cal.month_name"),
        F.col("cal.week_of_year"),
        F.col("cal.day_of_month"),
        F.col("cal.day_of_week"),
        F.col("cal.day_name"),
        F.col("cal.is_weekend"),
        F.col("cal.is_month_end"),
    )
)


# ============================================================
# Transaction-Level Financial / Risk Analytics
# ============================================================

section("Transaction-level financial and risk analytics")


analytical_transactions = (
    enriched

    .withColumn(
        "transaction_size_band",
        F.when(F.col("transaction_amount") < 50, "MICRO")
        .when(F.col("transaction_amount") < 250, "SMALL")
        .when(F.col("transaction_amount") < 1000, "MEDIUM")
        .when(F.col("transaction_amount") < 5000, "LARGE")
        .otherwise("VERY_LARGE")
    )

    .withColumn(
        "credit_risk_factor",
        F.when(F.col("credit_score").isNull(), F.lit(1.0))
        .when(F.col("credit_score") < 550, F.lit(2.00))
        .when(F.col("credit_score") < 650, F.lit(1.50))
        .when(F.col("credit_score") < 700, F.lit(1.20))
        .when(F.col("credit_score") < 760, F.lit(1.00))
        .otherwise(F.lit(0.75))
    )

    .withColumn(
        "merchant_risk_factor",
        F.when(F.col("merchant_risk_level") == "HIGH", F.lit(2.00))
        .when(F.col("merchant_risk_level") == "MEDIUM", F.lit(1.35))
        .when(F.col("merchant_risk_level") == "LOW", F.lit(1.00))
        .otherwise(F.lit(1.00))
    )

    .withColumn(
        "credit_utilization",
        F.when(
            F.col("credit_limit") > 0,
            F.col("current_balance") / F.col("credit_limit")
        )
        .otherwise(F.lit(0.0))
    )

    .withColumn(
        "balance_exposure",
        F.when(F.col("current_balance").isNotNull(), F.col("current_balance"))
        .otherwise(F.lit(0.0))
        *
        F.when(F.col("credit_risk_factor") > 1.5, F.lit(1.25))
        .otherwise(F.lit(1.0))
    )

    .withColumn(
        "interest_exposure",
        F.when(
            F.col("current_balance").isNotNull()
            & F.col("interest_rate").isNotNull(),
            F.col("current_balance") * F.col("interest_rate") / F.lit(100.0)
        )
        .otherwise(F.lit(0.0))
    )

    .withColumn(
        "fraud_exposure",
        F.when(F.col("fraud_flag") == 1, F.col("transaction_amount"))
        .otherwise(F.lit(0.0))
    )

    .withColumn(
        "risk_adjusted_amount",
        F.col("transaction_amount")
        * F.col("credit_risk_factor")
        * F.col("merchant_risk_factor")
    )

    .withColumn(
        "risk_exposure",
        F.col("risk_adjusted_amount")
        + F.col("fraud_exposure")
        + F.col("interest_exposure")
    )

    .withColumn(
        "transaction_to_income_ratio",
        F.when(
            F.col("estimated_income") > 0,
            F.col("transaction_amount") / F.col("estimated_income")
        )
        .otherwise(F.lit(0.0))
    )

    .withColumn(
        "weekend_risk",
        F.when(
            F.col("is_weekend") == True,
            F.col("transaction_amount") * F.lit(1.15)
        )
        .otherwise(F.col("transaction_amount"))
    )

    .withColumn(
        "high_risk_transaction",
        F.when(
            (
                (F.col("fraud_flag") == 1)
                | (F.col("merchant_risk_level") == "HIGH")
                | (F.col("credit_score") < 600)
                | (F.col("transaction_amount") > 5000)
            ),
            1
        )
        .otherwise(0)
    )

    .withColumn(
        "analytical_score",
        F.col("risk_exposure")
        + F.col("weekend_risk")
        + (F.col("high_risk_transaction") * F.lit(1000.0))
    )
)


# ============================================================
# Branch 1: Customer / Month
# ============================================================

section("Customer monthly analytics")


customer_month = (
    analytical_transactions

    .repartition(
        SHUFFLE_PARTITIONS,
        "customer_id",
        "year",
        "month",
        "customer_segment",
    )

    .sortWithinPartitions(
        "customer_id",
        "year",
        "month",
    )

    .groupBy(
        "customer_id",
        "customer_segment",
        "income_band",
        "risk_rating",
        "year",
        "month",
    )

    .agg(
        F.count("*").alias("customer_txns"),
        F.sum("transaction_amount").alias("customer_volume"),
        F.avg("transaction_amount").alias("customer_avg_amount"),
        F.max("transaction_amount").alias("customer_max_amount"),
        F.sum("fraud_flag").alias("customer_fraud_events"),
        F.sum("fraud_exposure").alias("customer_fraud_exposure"),
        F.sum("risk_adjusted_amount").alias("customer_risk_volume"),
        F.sum("balance_exposure").alias("customer_balance_exposure"),
        F.sum("interest_exposure").alias("customer_interest_exposure"),
        F.sum("analytical_score").alias("customer_score"),
    )
)


# ============================================================
# Branch 2: Merchant / Quarter
# ============================================================

section("Merchant quarterly analytics")


merchant_quarter = (
    analytical_transactions

    .repartition(
        SHUFFLE_PARTITIONS,
        "merchant_id",
        "year",
        "quarter",
    )

    .sortWithinPartitions(
        "merchant_id",
        "year",
        "quarter",
    )

    .groupBy(
        "merchant_id",
        "txn_merchant_category",
        "merchant_region",
        "merchant_risk_level",
        "merchant_size",
        "year",
        "quarter",
    )

    .agg(
        F.count("*").alias("merchant_txns"),
        F.sum("transaction_amount").alias("merchant_volume"),
        F.avg("transaction_amount").alias("merchant_avg_amount"),
        F.max("transaction_amount").alias("merchant_max_amount"),
        F.sum("fraud_flag").alias("merchant_fraud_events"),
        F.sum("fraud_exposure").alias("merchant_fraud_exposure"),
        F.sum("risk_adjusted_amount").alias("merchant_risk_volume"),
        F.sum("analytical_score").alias("merchant_score"),
        F.avg("annual_revenue").alias("merchant_revenue"),
    )
)


# ============================================================
# Terminal actions
# ============================================================

section("Writing customer_month result")

(
    customer_month
    .write
    .mode("overwrite")
    .saveAsTable(OUTPUT_TABLE_CUSTOMER)
)


section("Writing merchant_quarter result")

(
    merchant_quarter
    .write
    .mode("overwrite")
    .saveAsTable(OUTPUT_TABLE_MERCHANT)
)


_v9_simple_gpu_elapsed = time.time() - _v9_simple_gpu_start
_v9_simple_gpu_total   = time.time() - t0

print()
print("=" * 90)
print("SPARK RAPIDS ETL (GPU) COMPLETE")
print("=" * 90)
print(f"Source transactions   : {TRANSACTION_TABLE}")
print(f"Fact filter           : transaction_id < {FACT_ROW_ID_CEILING:,}")
print(f"Output tables         : {OUTPUT_TABLE_CUSTOMER}")
print(f"                        {OUTPUT_TABLE_MERCHANT}")
print(f"Shuffle partitions    : {SHUFFLE_PARTITIONS}")
print(f"GPU wall-clock (ETL)  : {_v9_simple_gpu_elapsed:.1f}s")
print(f"GPU wall-clock (total): {_v9_simple_gpu_total:.1f}s  (includes SparkSession startup)")
print()
print("Two terminal writes completed. Compare against 02_etl_cpu.py.")
print("=" * 90)


spark.stop()
