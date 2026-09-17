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
# CPU ETL: legitimate subset of the full v9 pipeline
# ============================================================
#
# Purpose:
#   Produce a version of the v9 ETL that is small and shuffle-light
#   enough to run to completion on the current CAI T4 GPU environment,
#   while still exercising the join + column-math + aggregation shape
#   that Spark-RAPIDS actually accelerates. This is the CPU baseline
#   for that simplified pipeline; 04_etl_gpu.py runs the SAME
#   transformation on GPU. Both scripts must be compared against each
#   other (NOT against the full v9, which is kept under archive/), so
#   the wall-clock comparison stays apples-to-apples.
#
# What is kept from the full v9 (archive/02_etl_v9.py):
#   - Same 5 source tables (fact + 4 skewed dims + calendar)
#   - Same 5 LEFT joins to build the enriched fact
#   - All ~10 analytical withColumn computations (risk factors,
#     exposures, weekend risk, high-risk indicator, analytical_score)
#   - Two aggregation branches: customer/month and merchant/quarter
#   - Two saveAsTable terminal actions
#
# What is cut vs the full v9 (with reason):
#   - Fact filter narrowed to transaction_id < 500_000_000 (~4% of the
#     original filtered scope) so wall-clock stays under the ~5-min
#     pod-deletion window observed on this CAI environment.
#   - 4 of the 6 v9 aggregation branches (account_month, regional_month,
#     payment_month, fraud_month) -- each was one extra shuffle stage.
#   - All 3 spine-and-rejoin blocks (customer_account, merchant_analysis,
#     analytical_spine) -- these were the biggest shuffle contributors
#     because they re-join branch aggregates back to the 40M-row fact.
#   - regional_fraud spine, final_joined, and the final mega-groupBy on
#     17 grouping columns with 50+ aggregates.
#   - Global orderBy at the very end.
#
# Net vs full v9: ~24 shuffle boundaries -> ~7. Data volume ~1B rows -> ~40M rows.
# This is still a real ETL: fact-to-dim joins, heavy per-row analytics,
# and two grouped aggregations that produce genuine business tables.
# ============================================================

import os
import time
import warnings
import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


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

# Two output tables: one per surviving aggregation branch.
OUTPUT_TABLE_CUSTOMER = f"{DATABASE}.ETL_V9_SIMPLE_CUSTOMER_MONTH"
OUTPUT_TABLE_MERCHANT = f"{DATABASE}.ETL_V9_SIMPLE_MERCHANT_QUARTER"

# Narrower fact filter than the full v9 (archive/02_etl_v9.py uses
# < 12_500_000_000). This must match the GPU filter in 04_etl_gpu.py
# exactly for the comparison to be fair.
FACT_ROW_ID_CEILING = 500_000_000

EVENT_LOG_DIR = (
    "file:///home/cdsw/"
    "spark-rapids-qualification-tool/"
    "spark-event-logs-dir"
)

# Same as v9. Simple pipeline still has real shuffles, so keeping
# 1000 partitions gives AQE room to coalesce down where appropriate.
SHUFFLE_PARTITIONS = 1000


# ============================================================
# Spark Session (CPU)
#
# Identical shape to archive/02_etl_v9.py -- do not change without also
# changing the GPU builder in 04_etl_gpu.py. Only the RAPIDS-related
# configs differ between CPU and GPU; the resource shape stays comparable.
# ============================================================

spark = (
    SparkSession.builder

    .appName(
        "Spark-ETL-CPU"
    )

    .config(
        "spark.driver.cores",
        4
    )

    .config(
        "spark.driver.memory",
        "4g"
    )

    .config(
        "spark.dynamicAllocation.enabled",
        "true"
    )

    .config(
        "spark.executor.cores",
        4
    )

    .config(
        "spark.executor.memory",
        "16g"
    )

    .config(
        "spark.sql.shuffle.partitions",
        SHUFFLE_PARTITIONS
    )

    .config(
        "spark.kerberos.access.hadoopFileSystems",
        "s3a://goes-se-sandbox/data"
    )

    .config(
        "spark.eventLog.dir",
        "file:///home/cdsw/spark-rapids-qualification-tool/spark-event-logs-dir"
    )

    .getOrCreate()
)

# Spark UI
print("SPARK UI:\n")
print("https://spark-"+os.environ["CDSW_ENGINE_ID"]+"."+os.environ["CDSW_DOMAIN"])

# Wall-clock timer (compare against GPU-simple).
_v9_simple_start = time.time()


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

section("Loading source tables (CPU ETL)")

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
# (unchanged from v9)
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
#
# 5 LEFT joins -- identical to v9. LEFT because the generators
# make FK ranges wider than the dimension row counts.
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
#
# Heavy per-row column math. This section is where RAPIDS
# actually earns its keep -- pure column-wise operations,
# no shuffles.
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
# (identical to v9 branch of the same name)
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
# (identical to v9 branch of the same name)
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
#
# Two saveAsTable writes -- one per branch. No count/show/collect
# between them (same discipline as v9), so the event log stays
# clean for a follow-up qualification pass.
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


_v9_simple_elapsed = time.time() - _v9_simple_start

print()
print("=" * 90)
print("SPARK ETL (CPU) COMPLETE")
print("=" * 90)
print(f"Source transactions   : {TRANSACTION_TABLE}")
print(f"Fact filter           : transaction_id < {FACT_ROW_ID_CEILING:,}")
print(f"Output tables         : {OUTPUT_TABLE_CUSTOMER}")
print(f"                        {OUTPUT_TABLE_MERCHANT}")
print(f"Shuffle partitions    : {SHUFFLE_PARTITIONS}")
print(f"CPU wall-clock        : {_v9_simple_elapsed:.1f}s")
print()
print("Two terminal writes completed.")
print("=" * 90)


spark.stop()
