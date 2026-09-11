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

import os
import warnings
import logging

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

OUTPUT_TABLE = f"{DATABASE}.ETL_V9_RESULT"

EVENT_LOG_DIR = (
    "file:///home/cdsw/"
    "spark-rapids-qualification-tool/"
    "spark-event-logs-dir"
)

# Deliberately somewhat higher than the previous 800.
# This is a tuning variable, not a fixed requirement.
SHUFFLE_PARTITIONS = 1000


# ============================================================
# Spark Session
# ============================================================

spark = (
    SparkSession.builder

    .appName(
        "Spark-ETL-v9"
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

section("Loading source tables")

transactions = (
    spark.table(TRANSACTION_TABLE)
    .filter(
        F.col("transaction_id") < 12500000000
    )
)

customers = spark.table(CUSTOMER_TABLE)
accounts = spark.table(ACCOUNT_TABLE)
merchants = spark.table(MERCHANT_TABLE)
branches = spark.table(BRANCH_TABLE)
calendar = spark.table(CALENDAR_TABLE)


# ============================================================
# Dimension Projections
#
# These columns come directly from the 01_generate_* scripts.
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
# IMPORTANT:
# Use LEFT joins.
#
# The generators deliberately create FK ranges larger than
# the corresponding dimensions. An INNER JOIN would discard
# much of the 2.5M transaction population.
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

        # ----------------------------------------------------
        # Transaction identifiers
        # ----------------------------------------------------

        F.col("t.transaction_id"),
        F.col("t.customer_id"),
        F.col("t.account_id"),
        F.col("t.merchant_id"),
        F.col("t.branch_id"),

        F.col("t.transaction_date"),
        F.col("t.transaction_timestamp"),

        # ----------------------------------------------------
        # Transaction attributes
        # ----------------------------------------------------

        F.col("t.merchant_category").alias(
            "txn_merchant_category"
        ),

        F.col("t.transaction_amount"),

        F.col("t.payment_channel"),
        F.col("t.payment_type"),
        F.col("t.device_type"),

        F.col("t.fraud_flag"),

        F.col("t.latitude"),
        F.col("t.longitude"),

        # ----------------------------------------------------
        # Customer attributes
        # ----------------------------------------------------

        F.col("c.age"),
        F.col("c.credit_score"),
        F.col("c.state").alias(
            "customer_state"
        ),
        F.col("c.city"),
        F.col("c.income_band"),
        F.col("c.estimated_income"),
        F.col("c.customer_segment"),
        F.col("c.tenure_years"),
        F.col("c.risk_rating"),

        # ----------------------------------------------------
        # Account attributes
        # ----------------------------------------------------

        F.col("a.customer_id").alias(
            "account_customer_id"
        ),

        F.col("a.account_type"),
        F.col("a.currency"),
        F.col("a.account_status"),
        F.col("a.opened_year").alias(
            "account_opened_year"
        ),
        F.col("a.branch_id").alias(
            "account_branch_id"
        ),
        F.col("a.current_balance"),
        F.col("a.credit_limit"),
        F.col("a.interest_rate"),

        # ----------------------------------------------------
        # Merchant attributes
        # ----------------------------------------------------

        F.col("m.state").alias(
            "merchant_state"
        ),

        F.col("m.region").alias(
            "merchant_region"
        ),

        F.col("m.merchant_category").alias(
            "dimension_merchant_category"
        ),

        F.col("m.risk_level").alias(
            "merchant_risk_level"
        ),

        F.col("m.annual_revenue"),
        F.col("m.merchant_size"),
        F.col("m.active").alias(
            "merchant_active"
        ),

        # ----------------------------------------------------
        # Branch attributes
        # ----------------------------------------------------

        F.col("b.state").alias(
            "branch_state"
        ),

        F.col("b.region").alias(
            "branch_region"
        ),

        F.col("b.branch_type"),
        F.col("b.employee_count"),
        F.col("b.assets_under_management"),
        F.col("b.annual_operating_cost"),
        F.col("b.status").alias(
            "branch_status"
        ),

        # ----------------------------------------------------
        # Calendar attributes
        # ----------------------------------------------------

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
# These calculations intentionally use the fields generated
# by the data generators.
# ============================================================

section("Transaction-level financial and risk analytics")


analytical_transactions = (
    enriched

    # --------------------------------------------------------
    # Basic transaction classification
    # --------------------------------------------------------

    .withColumn(
        "transaction_size_band",
        F.when(
            F.col("transaction_amount") < 50,
            "MICRO"
        )
        .when(
            F.col("transaction_amount") < 250,
            "SMALL"
        )
        .when(
            F.col("transaction_amount") < 1000,
            "MEDIUM"
        )
        .when(
            F.col("transaction_amount") < 5000,
            "LARGE"
        )
        .otherwise(
            "VERY_LARGE"
        )
    )

    # --------------------------------------------------------
    # Customer credit risk
    # --------------------------------------------------------

    .withColumn(
        "credit_risk_factor",
        F.when(
            F.col("credit_score").isNull(),
            F.lit(1.0)
        )
        .when(
            F.col("credit_score") < 550,
            F.lit(2.00)
        )
        .when(
            F.col("credit_score") < 650,
            F.lit(1.50)
        )
        .when(
            F.col("credit_score") < 700,
            F.lit(1.20)
        )
        .when(
            F.col("credit_score") < 760,
            F.lit(1.00)
        )
        .otherwise(
            F.lit(0.75)
        )
    )

    # --------------------------------------------------------
    # Merchant risk factor
    # --------------------------------------------------------

    .withColumn(
        "merchant_risk_factor",
        F.when(
            F.col("merchant_risk_level") == "HIGH",
            F.lit(2.00)
        )
        .when(
            F.col("merchant_risk_level") == "MEDIUM",
            F.lit(1.35)
        )
        .when(
            F.col("merchant_risk_level") == "LOW",
            F.lit(1.00)
        )
        .otherwise(
            F.lit(1.00)
        )
    )

    # --------------------------------------------------------
    # Account utilization
    # --------------------------------------------------------

    .withColumn(
        "credit_utilization",
        F.when(
            F.col("credit_limit") > 0,
            F.col("current_balance")
            / F.col("credit_limit")
        )
        .otherwise(
            F.lit(0.0)
        )
    )

    # --------------------------------------------------------
    # Account balance exposure
    # --------------------------------------------------------

    .withColumn(
        "balance_exposure",
        F.when(
            F.col("current_balance").isNotNull(),
            F.col("current_balance")
        )
        .otherwise(
            F.lit(0.0)
        )
        *
        F.when(
            F.col("credit_risk_factor") > 1.5,
            F.lit(1.25)
        )
        .otherwise(
            F.lit(1.0)
        )
    )

    # --------------------------------------------------------
    # Interest exposure
    # --------------------------------------------------------

    .withColumn(
        "interest_exposure",
        F.when(
            F.col("current_balance").isNotNull()
            & F.col("interest_rate").isNotNull(),
            F.col("current_balance")
            * F.col("interest_rate")
            / F.lit(100.0)
        )
        .otherwise(
            F.lit(0.0)
        )
    )

    # --------------------------------------------------------
    # Fraud exposure
    # --------------------------------------------------------

    .withColumn(
        "fraud_exposure",
        F.when(
            F.col("fraud_flag") == 1,
            F.col("transaction_amount")
        )
        .otherwise(
            F.lit(0.0)
        )
    )

    # --------------------------------------------------------
    # Risk-adjusted transaction value
    # --------------------------------------------------------

    .withColumn(
        "risk_adjusted_amount",
        F.col("transaction_amount")
        * F.col("credit_risk_factor")
        * F.col("merchant_risk_factor")
    )

    # --------------------------------------------------------
    # Fraud + risk exposure
    # --------------------------------------------------------

    .withColumn(
        "risk_exposure",
        F.col("risk_adjusted_amount")
        +
        F.col("fraud_exposure")
        +
        F.col("interest_exposure")
    )

    # --------------------------------------------------------
    # Customer financial capacity ratio
    # --------------------------------------------------------

    .withColumn(
        "transaction_to_income_ratio",
        F.when(
            F.col("estimated_income") > 0,
            F.col("transaction_amount")
            / F.col("estimated_income")
        )
        .otherwise(
            F.lit(0.0)
        )
    )

    # --------------------------------------------------------
    # Weekend transaction risk
    # --------------------------------------------------------

    .withColumn(
        "weekend_risk",
        F.when(
            F.col("is_weekend") == True,
            F.col("transaction_amount")
            * F.lit(1.15)
        )
        .otherwise(
            F.col("transaction_amount")
        )
    )

    # --------------------------------------------------------
    # High-risk transaction indicator
    # --------------------------------------------------------

    .withColumn(
        "high_risk_transaction",
        F.when(
            (
                (F.col("fraud_flag") == 1)
                |
                (
                    F.col("merchant_risk_level")
                    == "HIGH"
                )
                |
                (
                    F.col("credit_score") < 600
                )
                |
                (
                    F.col("transaction_amount") > 5000
                )
            ),
            1
        )
        .otherwise(0)
    )

    # --------------------------------------------------------
    # Analytical score
    # --------------------------------------------------------

    .withColumn(
        "analytical_score",
        F.col("risk_exposure")
        +
        F.col("weekend_risk")
        +
        (
            F.col("high_risk_transaction")
            * F.lit(1000.0)
        )
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
        F.count("*").alias(
            "customer_txns"
        ),

        F.sum("transaction_amount").alias(
            "customer_volume"
        ),

        F.avg("transaction_amount").alias(
            "customer_avg_amount"
        ),

        F.max("transaction_amount").alias(
            "customer_max_amount"
        ),

        F.sum("fraud_flag").alias(
            "customer_fraud_events"
        ),

        F.sum("fraud_exposure").alias(
            "customer_fraud_exposure"
        ),

        F.sum("risk_adjusted_amount").alias(
            "customer_risk_volume"
        ),

        F.sum("balance_exposure").alias(
            "customer_balance_exposure"
        ),

        F.sum("interest_exposure").alias(
            "customer_interest_exposure"
        ),

        F.sum("analytical_score").alias(
            "customer_score"
        ),
    )
)


# ============================================================
# Branch 2: Account / Month
# ============================================================

section("Account monthly analytics")


account_month = (
    analytical_transactions

    .filter(
        F.col("account_id").isNotNull()
    )

    .repartition(
        SHUFFLE_PARTITIONS,
        "account_id",
        "year",
        "month",
    )

    .sortWithinPartitions(
        "account_id",
        "year",
        "month",
    )

    .groupBy(
        "account_id",
        "account_type",
        "account_status",
        "year",
        "month",
    )

    .agg(
        F.count("*").alias(
            "account_txns"
        ),

        F.sum("transaction_amount").alias(
            "account_volume"
        ),

        F.avg("transaction_amount").alias(
            "account_avg_amount"
        ),

        F.sum("fraud_flag").alias(
            "account_fraud_events"
        ),

        F.sum("fraud_exposure").alias(
            "account_fraud_exposure"
        ),

        F.avg("credit_utilization").alias(
            "account_avg_utilization"
        ),

        F.max("credit_utilization").alias(
            "account_max_utilization"
        ),

        F.sum("balance_exposure").alias(
            "account_balance_exposure"
        ),

        F.sum("interest_exposure").alias(
            "account_interest_exposure"
        ),

        F.sum("risk_adjusted_amount").alias(
            "account_risk_volume"
        ),

        F.sum("analytical_score").alias(
            "account_score"
        ),
    )
)


# ============================================================
# Branch 3: Merchant / Quarter
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
        F.count("*").alias(
            "merchant_txns"
        ),

        F.sum("transaction_amount").alias(
            "merchant_volume"
        ),

        F.avg("transaction_amount").alias(
            "merchant_avg_amount"
        ),

        F.max("transaction_amount").alias(
            "merchant_max_amount"
        ),

        F.sum("fraud_flag").alias(
            "merchant_fraud_events"
        ),

        F.sum("fraud_exposure").alias(
            "merchant_fraud_exposure"
        ),

        F.sum("risk_adjusted_amount").alias(
            "merchant_risk_volume"
        ),

        F.sum("analytical_score").alias(
            "merchant_score"
        ),

        F.avg("annual_revenue").alias(
            "merchant_revenue"
        ),
    )
)


# ============================================================
# Branch 4: Region / Segment / Month
# ============================================================

section("Regional monthly analytics")


regional_month = (
    analytical_transactions

    .repartition(
        SHUFFLE_PARTITIONS,
        "branch_region",
        "customer_segment",
        "year",
        "month",
    )

    .sortWithinPartitions(
        "branch_region",
        "customer_segment",
        "year",
        "month",
    )

    .groupBy(
        "branch_region",
        "branch_type",
        "customer_segment",
        "income_band",
        "year",
        "month",
    )

    .agg(
        F.count("*").alias(
            "regional_txns"
        ),

        F.sum("transaction_amount").alias(
            "regional_volume"
        ),

        F.avg("transaction_amount").alias(
            "regional_avg_amount"
        ),

        F.sum("fraud_flag").alias(
            "regional_fraud_events"
        ),

        F.sum("fraud_exposure").alias(
            "regional_fraud_exposure"
        ),

        F.sum("risk_adjusted_amount").alias(
            "regional_risk_volume"
        ),

        F.sum("assets_under_management").alias(
            "regional_aum"
        ),

        F.sum("annual_operating_cost").alias(
            "regional_operating_cost"
        ),

        F.sum("analytical_score").alias(
            "regional_score"
        ),
    )
)


# ============================================================
# Branch 5: Payment / Device / Month
#
# Independent analytical branch using transaction-native
# dimensions. This gives us another substantial aggregation
# path without depending on dimension join cardinality.
# ============================================================

section("Payment-channel analytics")


payment_month = (
    analytical_transactions

    .repartition(
        SHUFFLE_PARTITIONS,
        "payment_channel",
        "payment_type",
        "device_type",
        "year",
        "month",
    )

    .sortWithinPartitions(
        "payment_channel",
        "payment_type",
        "device_type",
        "year",
        "month",
    )

    .groupBy(
        "payment_channel",
        "payment_type",
        "device_type",
        "year",
        "month",
    )

    .agg(
        F.count("*").alias(
            "payment_txns"
        ),

        F.sum("transaction_amount").alias(
            "payment_volume"
        ),

        F.avg("transaction_amount").alias(
            "payment_avg_amount"
        ),

        F.sum("fraud_flag").alias(
            "payment_fraud_events"
        ),

        F.sum("fraud_exposure").alias(
            "payment_fraud_exposure"
        ),

        F.sum("risk_adjusted_amount").alias(
            "payment_risk_volume"
        ),

        F.sum("analytical_score").alias(
            "payment_score"
        ),
    )
)


# ============================================================
# Branch 6: Fraud / Category / Month
# ============================================================

section("Fraud category analytics")


fraud_month = (
    analytical_transactions

    .filter(
        F.col("fraud_flag") == 1
    )

    .repartition(
        SHUFFLE_PARTITIONS,
        "txn_merchant_category",
        "merchant_region",
        "year",
        "month",
    )

    .sortWithinPartitions(
        "txn_merchant_category",
        "merchant_region",
        "year",
        "month",
    )

    .groupBy(
        "txn_merchant_category",
        "merchant_region",
        "year",
        "month",
    )

    .agg(
        F.count("*").alias(
            "fraud_txns"
        ),

        F.sum("transaction_amount").alias(
            "fraud_volume"
        ),

        F.sum("fraud_exposure").alias(
            "fraud_exposure"
        ),

        F.sum("risk_adjusted_amount").alias(
            "fraud_risk_volume"
        ),

        F.sum("analytical_score").alias(
            "fraud_score"
        ),
    )
)


# ============================================================
# Customer + Account Relationship Spine
# ============================================================

section("Customer/account analytical join")


customer_account_spine = (
    analytical_transactions

    .select(
        "customer_id",
        "account_id",
        "account_type",
        "account_status",
        "customer_segment",
        "income_band",
        "risk_rating",
        "year",
        "month",
    )

    .dropDuplicates()

    .repartition(
        SHUFFLE_PARTITIONS,
        "customer_id",
        "account_id",
        "year",
        "month",
    )
)


customer_account = (
    customer_account_spine.alias("s")

    .join(
        customer_month.alias("c"),
        (
            (F.col("s.customer_id") == F.col("c.customer_id"))
            &
            (
                F.col("s.customer_segment")
                == F.col("c.customer_segment")
            )
            &
            (
                F.col("s.income_band")
                == F.col("c.income_band")
            )
            &
            (
                F.col("s.risk_rating")
                == F.col("c.risk_rating")
            )
            &
            (F.col("s.year") == F.col("c.year"))
            &
            (F.col("s.month") == F.col("c.month"))
        ),
        "left",
    )

    .join(
        account_month.alias("a"),
        (
            (F.col("s.account_id") == F.col("a.account_id"))
            &
            (
                F.col("s.account_type")
                == F.col("a.account_type")
            )
            &
            (
                F.col("s.account_status")
                == F.col("a.account_status")
            )
            &
            (F.col("s.year") == F.col("a.year"))
            &
            (F.col("s.month") == F.col("a.month"))
        ),
        "left",
    )

    .select(
        F.col("s.customer_id"),
        F.col("s.account_id"),
        F.col("s.account_type"),
        F.col("s.customer_segment"),
        F.col("s.income_band"),
        F.col("s.risk_rating"),
        F.col("s.year"),
        F.col("s.month"),

        F.col("c.customer_txns"),
        F.col("c.customer_volume"),
        F.col("c.customer_avg_amount"),
        F.col("c.customer_max_amount"),
        F.col("c.customer_fraud_events"),
        F.col("c.customer_fraud_exposure"),
        F.col("c.customer_risk_volume"),
        F.col("c.customer_balance_exposure"),
        F.col("c.customer_interest_exposure"),
        F.col("c.customer_score"),

        F.col("a.account_txns"),
        F.col("a.account_volume"),
        F.col("a.account_avg_amount"),
        F.col("a.account_fraud_events"),
        F.col("a.account_fraud_exposure"),
        F.col("a.account_avg_utilization"),
        F.col("a.account_max_utilization"),
        F.col("a.account_balance_exposure"),
        F.col("a.account_interest_exposure"),
        F.col("a.account_risk_volume"),
        F.col("a.account_score"),
    )
)


# ============================================================
# Merchant Analytical Join
# ============================================================

section("Merchant analytical join")


merchant_spine = (
    analytical_transactions

    .select(
        "customer_id",
        "merchant_id",
        "txn_merchant_category",
        "merchant_region",
        "merchant_risk_level",
        "merchant_size",
        "year",
        "quarter",
    )

    .dropDuplicates()

    .repartition(
        SHUFFLE_PARTITIONS,
        "customer_id",
        "merchant_id",
        "year",
        "quarter",
    )
)


merchant_analysis = (
    merchant_spine.alias("s")

    .join(
        merchant_quarter.alias("m"),
        (
            (F.col("s.merchant_id") == F.col("m.merchant_id"))
            &
            (
                F.col("s.txn_merchant_category")
                == F.col("m.txn_merchant_category")
            )
            &
            (
                F.col("s.merchant_region")
                == F.col("m.merchant_region")
            )
            &
            (
                F.col("s.merchant_risk_level")
                == F.col("m.merchant_risk_level")
            )
            &
            (
                F.col("s.merchant_size")
                == F.col("m.merchant_size")
            )
            &
            (F.col("s.year") == F.col("m.year"))
            &
            (F.col("s.quarter") == F.col("m.quarter"))
        ),
        "left",
    )

    .select(
        F.col("s.customer_id"),
        F.col("s.merchant_id"),
        F.col("s.txn_merchant_category"),
        F.col("s.merchant_region"),
        F.col("s.year"),
        F.col("s.quarter"),

        F.col("m.merchant_txns"),
        F.col("m.merchant_volume"),
        F.col("m.merchant_avg_amount"),
        F.col("m.merchant_max_amount"),
        F.col("m.merchant_fraud_events"),
        F.col("m.merchant_fraud_exposure"),
        F.col("m.merchant_risk_volume"),
        F.col("m.merchant_score"),
        F.col("m.merchant_revenue"),
    )
)


# ============================================================
# Combined Analytical Spine
# ============================================================

section("Building combined analytical spine")


analytical_spine = (
    analytical_transactions.alias("t")

    .join(
        customer_account.alias("ca"),
        (
            (F.col("t.customer_id") == F.col("ca.customer_id"))
            &
            (F.col("t.account_id") == F.col("ca.account_id"))
            &
            (
                F.col("t.account_type")
                == F.col("ca.account_type")
            )
            &
            (
                F.col("t.customer_segment")
                == F.col("ca.customer_segment")
            )
            &
            (
                F.col("t.income_band")
                == F.col("ca.income_band")
            )
            &
            (
                F.col("t.risk_rating")
                == F.col("ca.risk_rating")
            )
            &
            (F.col("t.year") == F.col("ca.year"))
            &
            (F.col("t.month") == F.col("ca.month"))
        ),
        "left",
    )

    .join(
        merchant_analysis.alias("ma"),
        (
            (F.col("t.customer_id") == F.col("ma.customer_id"))
            &
            (F.col("t.merchant_id") == F.col("ma.merchant_id"))
            &
            (
                F.col("t.txn_merchant_category")
                == F.col("ma.txn_merchant_category")
            )
            &
            (
                F.col("t.merchant_region")
                == F.col("ma.merchant_region")
            )
            &
            (F.col("t.year") == F.col("ma.year"))
            &
            (F.col("t.quarter") == F.col("ma.quarter"))
        ),
        "left",
    )

    .join(
        payment_month.alias("p"),
        (
            (F.col("t.payment_channel") == F.col("p.payment_channel"))
            &
            (F.col("t.payment_type") == F.col("p.payment_type"))
            &
            (F.col("t.device_type") == F.col("p.device_type"))
            &
            (F.col("t.year") == F.col("p.year"))
            &
            (F.col("t.month") == F.col("p.month"))
        ),
        "left",
    )

    .select(
        # ----------------------------------------------------
        # Keys
        # ----------------------------------------------------

        F.col("t.customer_id"),
        F.col("t.account_id"),
        F.col("t.merchant_id"),
        F.col("t.branch_id"),

        F.col("t.customer_segment"),
        F.col("t.income_band"),
        F.col("t.risk_rating"),

        F.col("t.account_type"),
        F.col("t.account_status"),

        F.col("t.txn_merchant_category"),
        F.col("t.merchant_region"),
        F.col("t.merchant_risk_level"),
        F.col("t.merchant_size"),

        F.col("t.branch_region"),
        F.col("t.branch_type"),

        F.col("t.payment_channel"),
        F.col("t.payment_type"),
        F.col("t.device_type"),

        F.col("t.transaction_size_band"),

        F.col("t.year"),
        F.col("t.quarter"),
        F.col("t.month"),

        # ----------------------------------------------------
        # Transaction metrics
        # ----------------------------------------------------

        F.col("t.transaction_amount"),
        F.col("t.fraud_flag"),
        F.col("t.fraud_exposure"),
        F.col("t.risk_adjusted_amount"),
        F.col("t.risk_exposure"),
        F.col("t.balance_exposure"),
        F.col("t.interest_exposure"),
        F.col("t.credit_utilization"),
        F.col("t.transaction_to_income_ratio"),
        F.col("t.weekend_risk"),
        F.col("t.high_risk_transaction"),
        F.col("t.analytical_score"),

        # ----------------------------------------------------
        # Customer metrics
        # ----------------------------------------------------

        F.col("ca.customer_txns"),
        F.col("ca.customer_volume"),
        F.col("ca.customer_avg_amount"),
        F.col("ca.customer_max_amount"),
        F.col("ca.customer_fraud_events"),
        F.col("ca.customer_fraud_exposure"),
        F.col("ca.customer_risk_volume"),
        F.col("ca.customer_balance_exposure"),
        F.col("ca.customer_interest_exposure"),
        F.col("ca.customer_score"),

        # ----------------------------------------------------
        # Account metrics
        # ----------------------------------------------------

        F.col("ca.account_txns"),
        F.col("ca.account_volume"),
        F.col("ca.account_avg_amount"),
        F.col("ca.account_fraud_events"),
        F.col("ca.account_fraud_exposure"),
        F.col("ca.account_avg_utilization"),
        F.col("ca.account_max_utilization"),
        F.col("ca.account_balance_exposure"),
        F.col("ca.account_interest_exposure"),
        F.col("ca.account_risk_volume"),
        F.col("ca.account_score"),

        # ----------------------------------------------------
        # Merchant metrics
        # ----------------------------------------------------

        F.col("ma.merchant_txns"),
        F.col("ma.merchant_volume"),
        F.col("ma.merchant_avg_amount"),
        F.col("ma.merchant_max_amount"),
        F.col("ma.merchant_fraud_events"),
        F.col("ma.merchant_fraud_exposure"),
        F.col("ma.merchant_risk_volume"),
        F.col("ma.merchant_score"),
        F.col("ma.merchant_revenue"),

        # ----------------------------------------------------
        # Payment metrics
        # ----------------------------------------------------

        F.col("p.payment_txns"),
        F.col("p.payment_volume"),
        F.col("p.payment_avg_amount"),
        F.col("p.payment_fraud_events"),
        F.col("p.payment_fraud_exposure"),
        F.col("p.payment_risk_volume"),
        F.col("p.payment_score"),
    )
)


# ============================================================
# Regional + Fraud Join
#
# Explicit rf_ prefixes prevent ambiguous references later.
# ============================================================

section("Regional and fraud analytical join")


regional_spine = (
    analytical_spine

    .select(
        "branch_region",
        "branch_type",
        "customer_segment",
        "income_band",
        "txn_merchant_category",
        "merchant_region",
        "year",
        "month",
    )

    .dropDuplicates()

    .repartition(
        SHUFFLE_PARTITIONS,
        "branch_region",
        "customer_segment",
        "year",
        "month",
    )
)


regional_fraud = (
    regional_spine.alias("s")

    .join(
        regional_month.alias("r"),
        (
            (F.col("s.branch_region") == F.col("r.branch_region"))
            &
            (F.col("s.branch_type") == F.col("r.branch_type"))
            &
            (
                F.col("s.customer_segment")
                == F.col("r.customer_segment")
            )
            &
            (
                F.col("s.income_band")
                == F.col("r.income_band")
            )
            &
            (F.col("s.year") == F.col("r.year"))
            &
            (F.col("s.month") == F.col("r.month"))
        ),
        "left",
    )

    .join(
        fraud_month.alias("f"),
        (
            (
                F.col("s.txn_merchant_category")
                == F.col("f.txn_merchant_category")
            )
            &
            (
                F.col("s.merchant_region")
                == F.col("f.merchant_region")
            )
            &
            (F.col("s.year") == F.col("f.year"))
            &
            (F.col("s.month") == F.col("f.month"))
        ),
        "left",
    )

    .select(
        F.col("s.branch_region").alias(
            "rf_branch_region"
        ),

        F.col("s.branch_type").alias(
            "rf_branch_type"
        ),

        F.col("s.customer_segment").alias(
            "rf_customer_segment"
        ),

        F.col("s.income_band").alias(
            "rf_income_band"
        ),

        F.col("s.txn_merchant_category").alias(
            "rf_merchant_category"
        ),

        F.col("s.merchant_region").alias(
            "rf_merchant_region"
        ),

        F.col("s.year").alias(
            "rf_year"
        ),

        F.col("s.month").alias(
            "rf_month"
        ),

        # Regional
        F.col("r.regional_txns").alias(
            "rf_regional_txns"
        ),

        F.col("r.regional_volume").alias(
            "rf_regional_volume"
        ),

        F.col("r.regional_avg_amount").alias(
            "rf_regional_avg_amount"
        ),

        F.col("r.regional_fraud_events").alias(
            "rf_regional_fraud_events"
        ),

        F.col("r.regional_fraud_exposure").alias(
            "rf_regional_fraud_exposure"
        ),

        F.col("r.regional_risk_volume").alias(
            "rf_regional_risk_volume"
        ),

        F.col("r.regional_aum").alias(
            "rf_regional_aum"
        ),

        F.col("r.regional_operating_cost").alias(
            "rf_regional_operating_cost"
        ),

        F.col("r.regional_score").alias(
            "rf_regional_score"
        ),

        # Fraud
        F.col("f.fraud_txns").alias(
            "rf_fraud_txns"
        ),

        F.col("f.fraud_volume").alias(
            "rf_fraud_volume"
        ),

        F.col("f.fraud_exposure").alias(
            "rf_fraud_exposure"
        ),

        F.col("f.fraud_risk_volume").alias(
            "rf_fraud_risk_volume"
        ),

        F.col("f.fraud_score").alias(
            "rf_fraud_score"
        ),
    )
)


# ============================================================
# Final Join
# ============================================================

section("Final analytical join")


final_joined = (
    analytical_spine.alias("a")

    .join(
        regional_fraud.alias("rf"),
        (
            (
                F.col("a.branch_region")
                == F.col("rf.rf_branch_region")
            )
            &
            (
                F.col("a.branch_type")
                == F.col("rf.rf_branch_type")
            )
            &
            (
                F.col("a.customer_segment")
                == F.col("rf.rf_customer_segment")
            )
            &
            (
                F.col("a.income_band")
                == F.col("rf.rf_income_band")
            )
            &
            (
                F.col("a.txn_merchant_category")
                == F.col("rf.rf_merchant_category")
            )
            &
            (
                F.col("a.merchant_region")
                == F.col("rf.rf_merchant_region")
            )
            &
            (F.col("a.year") == F.col("rf.rf_year"))
            &
            (F.col("a.month") == F.col("rf.rf_month"))
        ),
        "left",
    )
)


# ============================================================
# Final Aggregation
# ============================================================

section("Final multidimensional aggregation")


final_aggregated = (
    final_joined

    .repartition(
        SHUFFLE_PARTITIONS,
        "year",
        "quarter",
        "branch_region",
        "customer_segment",
        "account_type",
        "txn_merchant_category",
    )

    .sortWithinPartitions(
        "year",
        "quarter",
        "branch_region",
        "customer_segment",
        "account_type",
        "txn_merchant_category",
    )

    .groupBy(
        "year",
        "quarter",
        "branch_region",
        "branch_type",
        "customer_segment",
        "income_band",
        "risk_rating",
        "account_type",
        "account_status",
        "txn_merchant_category",
        "merchant_region",
        "merchant_risk_level",
        "merchant_size",
        "payment_channel",
        "payment_type",
        "device_type",
        "transaction_size_band",
    )

    .agg(

        # ----------------------------------------------------
        # Transaction
        # ----------------------------------------------------

        F.count("*").alias(
            "rows"
        ),

        F.sum("transaction_amount").alias(
            "transaction_volume"
        ),

        F.avg("transaction_amount").alias(
            "avg_transaction_amount"
        ),

        F.max("transaction_amount").alias(
            "max_transaction_amount"
        ),

        F.sum("fraud_flag").alias(
            "fraud_events"
        ),

        F.sum("fraud_exposure").alias(
            "fraud_exposure"
        ),

        F.sum("risk_adjusted_amount").alias(
            "risk_adjusted_volume"
        ),

        F.sum("risk_exposure").alias(
            "risk_exposure"
        ),

        F.sum("balance_exposure").alias(
            "balance_exposure"
        ),

        F.sum("interest_exposure").alias(
            "interest_exposure"
        ),

        F.avg("credit_utilization").alias(
            "avg_credit_utilization"
        ),

        F.sum("transaction_to_income_ratio").alias(
            "income_ratio_sum"
        ),

        F.sum("weekend_risk").alias(
            "weekend_risk"
        ),

        F.sum("high_risk_transaction").alias(
            "high_risk_transactions"
        ),

        F.sum("analytical_score").alias(
            "analytical_score"
        ),

        # ----------------------------------------------------
        # Customer
        # ----------------------------------------------------

        F.sum("customer_txns").alias(
            "customer_txns"
        ),

        F.sum("customer_volume").alias(
            "customer_volume"
        ),

        F.avg("customer_avg_amount").alias(
            "customer_avg_amount"
        ),

        F.sum("customer_fraud_events").alias(
            "customer_fraud_events"
        ),

        F.sum("customer_fraud_exposure").alias(
            "customer_fraud_exposure"
        ),

        F.sum("customer_risk_volume").alias(
            "customer_risk_volume"
        ),

        F.sum("customer_balance_exposure").alias(
            "customer_balance_exposure"
        ),

        F.sum("customer_interest_exposure").alias(
            "customer_interest_exposure"
        ),

        F.sum("customer_score").alias(
            "customer_score"
        ),

        # ----------------------------------------------------
        # Account
        # ----------------------------------------------------

        F.sum("account_txns").alias(
            "account_txns"
        ),

        F.sum("account_volume").alias(
            "account_volume"
        ),

        F.avg("account_avg_amount").alias(
            "account_avg_amount"
        ),

        F.sum("account_fraud_events").alias(
            "account_fraud_events"
        ),

        F.sum("account_fraud_exposure").alias(
            "account_fraud_exposure"
        ),

        F.avg("account_avg_utilization").alias(
            "account_avg_utilization"
        ),

        F.max("account_max_utilization").alias(
            "account_max_utilization"
        ),

        F.sum("account_balance_exposure").alias(
            "account_balance_exposure"
        ),

        F.sum("account_interest_exposure").alias(
            "account_interest_exposure"
        ),

        F.sum("account_risk_volume").alias(
            "account_risk_volume"
        ),

        F.sum("account_score").alias(
            "account_score"
        ),

        # ----------------------------------------------------
        # Merchant
        # ----------------------------------------------------

        F.sum("merchant_txns").alias(
            "merchant_txns"
        ),

        F.sum("merchant_volume").alias(
            "merchant_volume"
        ),

        F.avg("merchant_avg_amount").alias(
            "merchant_avg_amount"
        ),

        F.max("merchant_max_amount").alias(
            "merchant_max_amount"
        ),

        F.sum("merchant_fraud_events").alias(
            "merchant_fraud_events"
        ),

        F.sum("merchant_fraud_exposure").alias(
            "merchant_fraud_exposure"
        ),

        F.sum("merchant_risk_volume").alias(
            "merchant_risk_volume"
        ),

        F.sum("merchant_score").alias(
            "merchant_score"
        ),

        F.avg("merchant_revenue").alias(
            "merchant_revenue"
        ),

        # ----------------------------------------------------
        # Payment
        # ----------------------------------------------------

        F.sum("payment_txns").alias(
            "payment_txns"
        ),

        F.sum("payment_volume").alias(
            "payment_volume"
        ),

        F.avg("payment_avg_amount").alias(
            "payment_avg_amount"
        ),

        F.sum("payment_fraud_events").alias(
            "payment_fraud_events"
        ),

        F.sum("payment_fraud_exposure").alias(
            "payment_fraud_exposure"
        ),

        F.sum("payment_risk_volume").alias(
            "payment_risk_volume"
        ),

        F.sum("payment_score").alias(
            "payment_score"
        ),

        # ----------------------------------------------------
        # Regional
        # ----------------------------------------------------

        F.sum("rf_regional_txns").alias(
            "regional_txns"
        ),

        F.sum("rf_regional_volume").alias(
            "regional_volume"
        ),

        F.avg("rf_regional_avg_amount").alias(
            "regional_avg_amount"
        ),

        F.sum("rf_regional_fraud_events").alias(
            "regional_fraud_events"
        ),

        F.sum("rf_regional_fraud_exposure").alias(
            "regional_fraud_exposure"
        ),

        F.sum("rf_regional_risk_volume").alias(
            "regional_risk_volume"
        ),

        F.sum("rf_regional_aum").alias(
            "regional_aum"
        ),

        F.sum("rf_regional_operating_cost").alias(
            "regional_operating_cost"
        ),

        F.sum("rf_regional_score").alias(
            "regional_score"
        ),

        # ----------------------------------------------------
        # Fraud branch
        # ----------------------------------------------------

        F.sum("rf_fraud_txns").alias(
            "category_fraud_txns"
        ),

        F.sum("rf_fraud_volume").alias(
            "category_fraud_volume"
        ),

        F.sum("rf_fraud_exposure").alias(
            "category_fraud_exposure"
        ),

        F.sum("rf_fraud_risk_volume").alias(
            "category_fraud_risk_volume"
        ),

        F.sum("rf_fraud_score").alias(
            "category_fraud_score"
        ),
    )
)


# ============================================================
# Global Sort
#
# This intentionally creates a substantial final sort.
# ============================================================

section("Global analytical sort")


result = (
    final_aggregated
    .orderBy(
        F.col("year"),
        F.col("quarter"),
        F.col("branch_region"),
        F.col("customer_segment"),
        F.col("income_band"),
        F.col("account_type"),
        F.col("txn_merchant_category"),
        F.col("transaction_volume").desc(),
        F.col("risk_exposure").desc(),
    )
)


# ============================================================
# ONE terminal action
#
# No count()
# No show()
# No collect()
#
# This is important for Qualification Tool analysis because
# we want one execution of the complete lineage.
# ============================================================

section("Writing result")


(
    result
    .write
    .mode("overwrite")
    .saveAsTable(OUTPUT_TABLE)
)


print()
print("=" * 90)
print("SPARK ETL V9 COMPLETE")
print("=" * 90)
print(f"Source transactions : {TRANSACTION_TABLE}")
print(f"Output table        : {OUTPUT_TABLE}")
print(f"Shuffle partitions  : {SHUFFLE_PARTITIONS}")
print()
print("Single terminal write completed.")
print("No post-write count/show/collect was executed.")
print("=" * 90)


spark.stop()