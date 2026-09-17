#****************************************************************************
# (C) Cloudera, Inc. 2020-2026
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

#****************************************************************************
#
# Spark RAPIDS Benchmark - ETL v11
#
# Purpose:
#   v11 keeps the exact same skewed source data as v10, but increases the
#   amount of legitimate GPU-friendly SQL work in the analytical pipeline.
#
#   Targeted operators:
#       Exchange / Sort / SortMergeJoin / HashAggregate / Project / Filter
#
#   Changes from v10:
#     - additional transaction-level financial/risk expressions
#     - additional predicates
#     - additional group-by dimensions
#     - multiple richer aggregation stages
#     - second aggregation over the first-stage result
#     - additional repartition + sort operations
#     - additional aggregate-to-aggregate SortMergeJoin
#     - final multidimensional aggregation and global sort
#
#   Source tables are intentionally unchanged from v10:
#       TRS_v14, CUSTOMERS_v2, ACCOUNTS_skewed, MERCHANTS_v2,
#       BRANCHES_v2, CALENDAR
#****************************************************************************/

import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# Configuration
# ============================================================

DATABASE = "DEMO_pauldefusco"

TRANSACTION_TABLE = f"{DATABASE}.TRS_v25"
CUSTOMER_TABLE = f"{DATABASE}.CUSTOMERS_skewed"
ACCOUNT_TABLE = f"{DATABASE}.ACCOUNTS_skewed"
MERCHANT_TABLE = f"{DATABASE}.MERCHANTS_skewed"
BRANCH_TABLE = f"{DATABASE}.BRANCHES_skewed"
CALENDAR_TABLE = f"{DATABASE}.CALENDAR"

OUTPUT_TABLE = f"{DATABASE}.ETL_V11_RESULT"

EVENT_LOG_DIR = (
    "file:///home/cdsw/"
    "spark-rapids-qualification-tool/"
    "spark-event-logs-dir"
)

SHUFFLE_PARTITIONS = 1000


# ============================================================
# Spark Session
# ============================================================

os.makedirs(
    "/home/cdsw/spark-rapids-qualification-tool/spark-rapids-logs",
    exist_ok=True,
)

spark = (
    SparkSession.builder
    .appName("Spark-ETL-v11")
    .config("spark.driver.cores", 4)
    .config("spark.driver.memory", "4g")
    .config("spark.dynamicAllocation.enabled", "true")
    .config("spark.executor.cores", 4)
    .config("spark.executor.memory", "16g")
    .config("spark.sql.shuffle.partitions", SHUFFLE_PARTITIONS)
    .config("spark.kerberos.access.hadoopFileSystems", "s3a://goes-se-sandbox/data")
    .config("spark.eventLog.dir", EVENT_LOG_DIR)
    .getOrCreate()
)

# Force analytical joins to remain shuffle joins. This is intentional for
# this qualification workload: we want SortMergeJoin + Exchange operators.
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
spark.conf.set("spark.sql.shuffle.partitions", SHUFFLE_PARTITIONS)


def section(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)
    print()


# ============================================================
# Load the SAME skewed source data as v10
# ============================================================

section("Loading skewed source tables")

transactions = (
    spark.table(TRANSACTION_TABLE)
    .filter(F.col("transaction_id") < 12500000000)
)

customers = spark.table(CUSTOMER_TABLE)
accounts = spark.table(ACCOUNT_TABLE)
merchants = spark.table(MERCHANT_TABLE)
branches = spark.table(BRANCH_TABLE)
calendar = spark.table(CALENDAR_TABLE)


# ============================================================
# Dimension projections
# ============================================================

customer_dim = customers.select(
    "customer_id", "age", "credit_score", "state", "city",
    "income_band", "estimated_income", "customer_segment",
    "tenure_years", "risk_rating",
)

account_dim = accounts.select(
    "account_id", "customer_id", "account_type", "currency",
    "account_status", "opened_year", "branch_id", "current_balance",
    "credit_limit", "interest_rate",
)

merchant_dim = merchants.select(
    "merchant_id", "state", "region", "merchant_category",
    "merchant_name", "risk_level", "annual_revenue", "merchant_size",
    "opened_year", "active",
)

branch_dim = branches.select(
    "branch_id", "branch_name", "state", "region", "branch_type",
    "employee_count", "assets_under_management", "annual_operating_cost",
    "opened_year", "manager_id", "status",
)

calendar_dim = calendar.select(
    "calendar_date", "date_key", "year", "quarter", "month",
    "month_name", "week_of_year", "day_of_month", "day_of_week",
    "day_name", "is_weekend", "is_month_end",
)


# ============================================================
# Fact + dimension enrichment
# ============================================================

section("Transaction enrichment")

enriched = (
    transactions.alias("t")
    .join(customer_dim.alias("c"), F.col("t.customer_id") == F.col("c.customer_id"), "left")
    .join(account_dim.alias("a"), F.col("t.account_id") == F.col("a.account_id"), "left")
    .join(merchant_dim.alias("m"), F.col("t.merchant_id") == F.col("m.merchant_id"), "left")
    .join(branch_dim.alias("b"), F.col("t.branch_id") == F.col("b.branch_id"), "left")
    .join(calendar_dim.alias("cal"), F.col("t.transaction_date") == F.col("cal.calendar_date"), "left")
    .select(
        "t.transaction_id",
        "t.customer_id",
        "t.account_id",
        "t.merchant_id",
        "t.branch_id",
        "t.transaction_date",
        "t.transaction_timestamp",
        F.col("t.merchant_category").alias("txn_merchant_category"),
        "t.transaction_amount",
        "t.payment_channel",
        "t.payment_type",
        "t.device_type",
        "t.fraud_flag",
        "t.latitude",
        "t.longitude",
        "c.age",
        "c.credit_score",
        F.col("c.state").alias("customer_state"),
        "c.city",
        "c.income_band",
        "c.estimated_income",
        "c.customer_segment",
        "c.tenure_years",
        "c.risk_rating",
        F.col("a.customer_id").alias("account_customer_id"),
        "a.account_type",
        "a.currency",
        "a.account_status",
        F.col("a.opened_year").alias("account_opened_year"),
        F.col("a.branch_id").alias("account_branch_id"),
        "a.current_balance",
        "a.credit_limit",
        "a.interest_rate",
        F.col("m.state").alias("merchant_state"),
        F.col("m.region").alias("merchant_region"),
        F.col("m.merchant_category").alias("dimension_merchant_category"),
        F.col("m.risk_level").alias("merchant_risk_level"),
        "m.annual_revenue",
        "m.merchant_size",
        F.col("m.active").alias("merchant_active"),
        F.col("b.state").alias("branch_state"),
        F.col("b.region").alias("branch_region"),
        "b.branch_type",
        "b.employee_count",
        "b.assets_under_management",
        "b.annual_operating_cost",
        F.col("b.status").alias("branch_status"),
        "cal.year",
        "cal.quarter",
        "cal.month",
        "cal.month_name",
        "cal.week_of_year",
        "cal.day_of_month",
        "cal.day_of_week",
        "cal.day_name",
        "cal.is_weekend",
        "cal.is_month_end",
    )
)


# ============================================================
# Expanded transaction-level analytics
# ============================================================

section("Expanded financial and risk analytics")

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
        F.when(F.col("credit_score").isNull(), 1.0)
         .when(F.col("credit_score") < 550, 2.00)
         .when(F.col("credit_score") < 650, 1.50)
         .when(F.col("credit_score") < 700, 1.20)
         .when(F.col("credit_score") < 760, 1.00)
         .otherwise(0.75)
    )
    .withColumn(
        "merchant_risk_factor",
        F.when(F.col("merchant_risk_level") == "HIGH", 2.00)
         .when(F.col("merchant_risk_level") == "MEDIUM", 1.35)
         .otherwise(1.00)
    )
    .withColumn(
        "credit_utilization",
        F.when(F.col("credit_limit") > 0,
               F.col("current_balance") / F.col("credit_limit"))
         .otherwise(0.0)
    )
    .withColumn(
        "balance_exposure",
        F.coalesce(F.col("current_balance"), F.lit(0.0)) *
        F.when(F.col("credit_risk_factor") > 1.5, 1.25).otherwise(1.0)
    )
    .withColumn(
        "interest_exposure",
        F.when(
            F.col("current_balance").isNotNull() & F.col("interest_rate").isNotNull(),
            F.col("current_balance") * F.col("interest_rate") / 100.0,
        ).otherwise(0.0)
    )
    .withColumn(
        "fraud_exposure",
        F.when(F.col("fraud_flag") == 1, F.col("transaction_amount")).otherwise(0.0)
    )
    .withColumn(
        "risk_adjusted_amount",
        F.col("transaction_amount") *
        F.col("credit_risk_factor") *
        F.col("merchant_risk_factor")
    )
    .withColumn(
        "risk_exposure",
        F.col("risk_adjusted_amount") +
        F.col("fraud_exposure") +
        F.col("interest_exposure")
    )
    .withColumn(
        "transaction_to_income_ratio",
        F.when(F.col("estimated_income") > 0,
               F.col("transaction_amount") / F.col("estimated_income"))
         .otherwise(0.0)
    )
    .withColumn(
        "weekend_risk",
        F.when(F.col("is_weekend") == True,
               F.col("transaction_amount") * 1.15)
         .otherwise(F.col("transaction_amount"))
    )
    .withColumn(
        "high_risk_transaction",
        F.when(
            (F.col("fraud_flag") == 1) |
            (F.col("merchant_risk_level") == "HIGH") |
            (F.col("credit_score") < 600) |
            (F.col("transaction_amount") > 5000) |
            (F.col("credit_utilization") > 0.80),
            1,
        ).otherwise(0)
    )
    # NEW v11 financial measures
    .withColumn(
        "available_credit",
        F.greatest(
            F.coalesce(F.col("credit_limit"), F.lit(0.0)) -
            F.coalesce(F.col("current_balance"), F.lit(0.0)),
            F.lit(0.0),
        )
    )
    .withColumn(
        "net_financial_exposure",
        F.col("risk_exposure") + F.col("balance_exposure") + F.col("available_credit")
    )
    .withColumn(
        "merchant_margin_proxy",
        F.col("transaction_amount") *
        F.when(F.col("merchant_size") == "LARGE", 0.030)
         .when(F.col("merchant_size") == "MEDIUM", 0.045)
         .otherwise(0.060)
    )
    .withColumn(
        "customer_capacity_score",
        F.when(F.col("estimated_income") > 0,
               F.col("estimated_income") / (F.abs(F.col("current_balance")) + 1.0))
         .otherwise(0.0)
    )
    .withColumn(
        "transaction_intensity",
        F.col("transaction_amount") *
        (F.col("credit_utilization") + F.lit(1.0)) *
        (F.col("tenure_years") + F.lit(1.0))
    )
    .withColumn(
        "analytical_score",
        F.col("risk_exposure") +
        F.col("weekend_risk") +
        (F.col("high_risk_transaction") * 1000.0) +
        F.col("merchant_margin_proxy")
    )
)


# ============================================================
# Additional predicates
# ============================================================

section("Additional predicate filtering")

filtered = (
    analytical_transactions
    .filter(F.col("transaction_amount") > 10)
    .filter(F.col("transaction_amount") < 100000)
    .filter(F.col("transaction_id").isNotNull())
    .filter(
        (F.col("account_status").isNull()) |
        (F.col("account_status") != "CLOSED")
    )
    .filter(
        (F.col("merchant_active").isNull()) |
        (F.col("merchant_active") == True)
    )
    .filter(
        (F.col("fraud_flag") == 1) |
        (F.col("credit_utilization") > 0.10) |
        (F.col("transaction_amount") > 250)
    )
)


# ============================================================
# Stage 1: high-cardinality customer/account aggregation
# ============================================================

section("Stage 1 customer-account aggregation")

customer_account_month = (
    filtered
    .repartition(
        SHUFFLE_PARTITIONS,
        "customer_id", "account_id", "year", "month",
        "customer_segment", "account_type",
    )
    .sortWithinPartitions(
        "customer_id", "account_id", "year", "month",
    )
    .groupBy(
        "customer_id",
        "account_id",
        "customer_segment",
        "income_band",
        "risk_rating",
        "account_type",
        "account_status",
        "branch_region",
        "year",
        "month",
    )
    .agg(
        F.count("*").alias("txn_count"),
        F.sum("transaction_amount").alias("gross_volume"),
        F.avg("transaction_amount").alias("avg_amount"),
        F.min("transaction_amount").alias("min_amount"),
        F.max("transaction_amount").alias("max_amount"),
        F.sum("fraud_flag").alias("fraud_events"),
        F.sum("fraud_exposure").alias("fraud_volume"),
        F.sum("risk_adjusted_amount").alias("risk_volume"),
        F.sum("interest_exposure").alias("interest_volume"),
        F.sum("balance_exposure").alias("balance_exposure"),
        F.avg("credit_utilization").alias("avg_utilization"),
        F.max("credit_utilization").alias("max_utilization"),
        F.sum("net_financial_exposure").alias("net_exposure"),
        F.sum("transaction_intensity").alias("transaction_intensity"),
        F.sum("analytical_score").alias("analytical_score"),
    )
)


# ============================================================
# Stage 1b: merchant/category aggregation
# ============================================================

section("Stage 1 merchant aggregation")

merchant_month = (
    filtered
    .filter(F.col("merchant_id").isNotNull())
    .filter(F.col("merchant_region").isNotNull())
    .repartition(
        SHUFFLE_PARTITIONS,
        "merchant_id", "merchant_region", "year", "month",
        "txn_merchant_category", "merchant_size",
    )
    .sortWithinPartitions(
        "merchant_id", "year", "month", "txn_merchant_category",
    )
    .groupBy(
        "merchant_id",
        "txn_merchant_category",
        "merchant_region",
        "merchant_risk_level",
        "merchant_size",
        "year",
        "month",
    )
    .agg(
        F.count("*").alias("merchant_txns"),
        F.sum("transaction_amount").alias("merchant_volume"),
        F.avg("transaction_amount").alias("merchant_avg_amount"),
        F.max("transaction_amount").alias("merchant_max_amount"),
        F.sum("fraud_flag").alias("merchant_fraud_events"),
        F.sum("fraud_exposure").alias("merchant_fraud_volume"),
        F.sum("risk_adjusted_amount").alias("merchant_risk_volume"),
        F.sum("analytical_score").alias("merchant_score"),
    )
)


# ============================================================
# Stage 2: aggregate the aggregate
# ============================================================

section("Stage 2 regional rollup")

regional_rollup = (
    customer_account_month
    .filter(F.col("gross_volume") > 0)
    .repartition(
        SHUFFLE_PARTITIONS,
        "branch_region", "customer_segment", "account_type", "year", "month",
    )
    .sortWithinPartitions(
        "branch_region", "customer_segment", "account_type", "year", "month",
    )
    .groupBy(
        "branch_region",
        "customer_segment",
        "income_band",
        "risk_rating",
        "account_type",
        "year",
        "month",
    )
    .agg(
        F.sum("txn_count").alias("regional_txns"),
        F.sum("gross_volume").alias("regional_volume"),
        F.avg("avg_amount").alias("regional_avg_amount"),
        F.max("max_amount").alias("regional_max_amount"),
        F.sum("fraud_events").alias("regional_fraud_events"),
        F.sum("fraud_volume").alias("regional_fraud_volume"),
        F.sum("risk_volume").alias("regional_risk_volume"),
        F.sum("interest_volume").alias("regional_interest_volume"),
        F.sum("balance_exposure").alias("regional_balance_exposure"),
        F.avg("avg_utilization").alias("regional_avg_utilization"),
        F.max("max_utilization").alias("regional_max_utilization"),
        F.sum("net_exposure").alias("regional_net_exposure"),
        F.sum("analytical_score").alias("regional_score"),
    )
)


# ============================================================
# Stage 2b: merchant rollup
# ============================================================

merchant_rollup = (
    merchant_month
    .filter(F.col("merchant_volume") > 0)
    .repartition(
        SHUFFLE_PARTITIONS,
        "merchant_region", "txn_merchant_category", "year", "month",
    )
    .sortWithinPartitions(
        "merchant_region", "txn_merchant_category", "year", "month",
    )
    .groupBy(
        "merchant_region",
        "txn_merchant_category",
        "merchant_risk_level",
        "merchant_size",
        "year",
        "month",
    )
    .agg(
        F.sum("merchant_txns").alias("rollup_txns"),
        F.sum("merchant_volume").alias("rollup_volume"),
        F.avg("merchant_avg_amount").alias("rollup_avg_amount"),
        F.max("merchant_max_amount").alias("rollup_max_amount"),
        F.sum("merchant_fraud_events").alias("rollup_fraud_events"),
        F.sum("merchant_fraud_volume").alias("rollup_fraud_volume"),
        F.sum("merchant_risk_volume").alias("rollup_risk_volume"),
        F.sum("merchant_score").alias("rollup_score"),
    )
)


# ============================================================
# Additional aggregate-to-aggregate SortMergeJoin
# ============================================================

section("Aggregate-to-aggregate analytical join")

joined_rollups = (
    regional_rollup.alias("r")
    .join(
        merchant_rollup.alias("m"),
        (
            (F.col("r.branch_region") == F.col("m.merchant_region")) &
            (F.col("r.year") == F.col("m.year")) &
            (F.col("r.month") == F.col("m.month"))
        ),
        "left",
    )
    .select(
        F.col("r.branch_region"),
        F.col("r.customer_segment"),
        F.col("r.income_band"),
        F.col("r.risk_rating"),
        F.col("r.account_type"),
        F.col("r.year"),
        F.col("r.month"),
        F.col("r.regional_txns"),
        F.col("r.regional_volume"),
        F.col("r.regional_avg_amount"),
        F.col("r.regional_max_amount"),
        F.col("r.regional_fraud_events"),
        F.col("r.regional_fraud_volume"),
        F.col("r.regional_risk_volume"),
        F.col("r.regional_interest_volume"),
        F.col("r.regional_balance_exposure"),
        F.col("r.regional_avg_utilization"),
        F.col("r.regional_max_utilization"),
        F.col("r.regional_net_exposure"),
        F.col("r.regional_score"),
        F.coalesce(F.col("m.rollup_txns"), F.lit(0)).alias("merchant_txns"),
        F.coalesce(F.col("m.rollup_volume"), F.lit(0.0)).alias("merchant_volume"),
        F.coalesce(F.col("m.rollup_avg_amount"), F.lit(0.0)).alias("merchant_avg_amount"),
        F.coalesce(F.col("m.rollup_fraud_events"), F.lit(0)).alias("merchant_fraud_events"),
        F.coalesce(F.col("m.rollup_fraud_volume"), F.lit(0.0)).alias("merchant_fraud_volume"),
        F.coalesce(F.col("m.rollup_risk_volume"), F.lit(0.0)).alias("merchant_risk_volume"),
        F.coalesce(F.col("m.rollup_score"), F.lit(0.0)).alias("merchant_score"),
    )
)


# ============================================================
# Second aggregation after the join
# ============================================================

section("Post-join aggregation")

post_join_agg = (
    joined_rollups
    .filter(F.col("regional_volume") > 100)
    .repartition(
        SHUFFLE_PARTITIONS,
        "branch_region", "customer_segment", "year", "month",
    )
    .sortWithinPartitions(
        "branch_region", "customer_segment", "year", "month",
    )
    .groupBy(
        "branch_region",
        "customer_segment",
        "income_band",
        "risk_rating",
        "year",
        "month",
    )
    .agg(
        F.sum("regional_txns").alias("total_txns"),
        F.sum("regional_volume").alias("total_volume"),
        F.avg("regional_avg_amount").alias("avg_transaction_amount"),
        F.max("regional_max_amount").alias("max_transaction_amount"),
        F.sum("regional_fraud_events").alias("total_fraud_events"),
        F.sum("regional_fraud_volume").alias("total_fraud_volume"),
        F.sum("regional_risk_volume").alias("total_risk_volume"),
        F.sum("regional_interest_volume").alias("total_interest_volume"),
        F.sum("regional_balance_exposure").alias("total_balance_exposure"),
        F.avg("regional_avg_utilization").alias("avg_utilization"),
        F.max("regional_max_utilization").alias("max_utilization"),
        F.sum("regional_net_exposure").alias("total_net_exposure"),
        F.sum("merchant_txns").alias("merchant_txns"),
        F.sum("merchant_volume").alias("merchant_volume"),
        F.avg("merchant_avg_amount").alias("merchant_avg_amount"),
        F.sum("merchant_fraud_events").alias("merchant_fraud_events"),
        F.sum("merchant_fraud_volume").alias("merchant_fraud_volume"),
        F.sum("merchant_risk_volume").alias("merchant_risk_volume"),
        F.sum("merchant_score").alias("merchant_score"),
        F.sum("regional_score").alias("regional_score"),
    )
)


# ============================================================
# Final sort + output
# ============================================================

section("Final sort")

result = (
    post_join_agg
    .repartition(
        SHUFFLE_PARTITIONS,
        "year", "month", "branch_region", "customer_segment",
    )
    .sortWithinPartitions(
        "year", "month", "branch_region", "customer_segment",
        F.desc("total_volume"),
    )
    .orderBy(
        F.desc("total_volume"),
        F.desc("total_fraud_events"),
        F.desc("total_risk_volume"),
        F.asc("year"),
        F.asc("month"),
    )
)


section("Writing ETL v11 result")

result.write.mode("overwrite").saveAsTable(OUTPUT_TABLE)

print(f"ETL v11 complete: {OUTPUT_TABLE}")
print(f"Shuffle partitions: {SHUFFLE_PARTITIONS}")
print(f"Elapsed time: {time.time():.2f}")

spark.stop()
