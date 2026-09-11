#****************************************************************************
# (C) Cloudera, Inc. 2020-2026
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#****************************************************************************
#
# Spark RAPIDS Benchmark - ETL v8
#
# Purpose:
#
#   v8 intentionally changes the workload topology from v7.
#
#   v7:
#
#       transaction-grain fact
#           -> multiple aggregations
#           -> repeated joins back to the full transaction-grain dataset
#           -> final aggregation
#
#   v8:
#
#       transaction-grain fact
#           -> dimension enrichment
#           -> high-cardinality intermediate reduction
#           -> multiple analytical reductions over the reduced dataset
#           -> joins between reduced analytical datasets
#           -> regional reduction
#           -> final aggregation
#           -> global sort
#
#   The objective is to create a deeper analytical DAG while reducing the
#   amount of raw transaction-grain data carried through later stages.
#
#   This intentionally emphasizes legitimate Spark SQL operations that are
#   generally strong RAPIDS candidates:
#
#       Exchange
#       Sort
#       HashAggregate
#       SortMergeJoin
#       Project
#       Filter
#       arithmetic expressions
#
#   The workload remains at the existing transaction scale rather than
#   artificially increasing the source dataset.
#
#****************************************************************************

import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import cml.data_v1 as cmldata


class BankingETLv8:

    def __init__(
        self,
        connection_name,
        database,
        storage
    ):
        self.connection_name = connection_name
        self.database = database
        self.storage = storage

    ############################################################
    # Spark configuration
    ############################################################

    def createSparkConnection(self):

        os.makedirs(
            "/home/cdsw/spark-rapids-qualification-tool/spark-event-logs-dir",
            exist_ok=True
        )

        spark = (
            SparkSession.builder
            .appName("Spark-ETL-v8")
            .config("spark.driver.cores", 4)
            .config("spark.driver.memory", "4g")
            .config("spark.dynamicAllocation.enabled", "true")
            .config("spark.executor.cores", 4)
            .config("spark.executor.memory", "16g")
            .config("spark.sql.shuffle.partitions", 800)
            .config(
                "spark.kerberos.access.hadoopFileSystems",
                self.storage
            )
            .config(
                "spark.eventLog.dir",
                "file:///home/cdsw/spark-rapids-qualification-tool/spark-event-logs-dir"
            )
            .getOrCreate()
        )

        # Force distributed joins rather than allowing small analytical
        # datasets to become broadcast joins.
        spark.conf.set(
            "spark.sql.autoBroadcastJoinThreshold",
            -1
        )

        spark.conf.set(
            "spark.sql.shuffle.partitions",
            800
        )

        return spark

    ############################################################
    # Run ETL
    ############################################################

    def run(self, spark):

        ########################################################
        # Read source tables
        ########################################################

        transactions = (
            spark.table(
                f"{self.database}.TRS_v25"
            )
            .filter(
                F.col("transaction_id") < F.lit(12500000)
            )
        )

        customers = spark.table(
            f"{self.database}.CUSTOMERS_skewed"
        )

        accounts = spark.table(
            f"{self.database}.ACCOUNTS_skewed"
        )

        merchants = spark.table(
            f"{self.database}.MERCHANTS_skewed"
        )

        branches = spark.table(
            f"{self.database}.BRANCHES_skewed"
        )

        calendar = spark.table(
            f"{self.database}.CALENDAR"
        )

        ########################################################
        # Dimension enrichment
        ########################################################

        base = (
            transactions.alias("t")
            .join(
                customers.alias("c"),
                F.col("t.customer_id") == F.col("c.customer_id"),
                "inner"
            )
            .join(
                accounts.alias("a"),
                F.col("t.account_id") == F.col("a.account_id"),
                "inner"
            )
            .join(
                merchants.alias("m"),
                F.col("t.merchant_id") == F.col("m.merchant_id"),
                "inner"
            )
            .join(
                branches.alias("b"),
                F.col("t.branch_id") == F.col("b.branch_id"),
                "inner"
            )
            .join(
                calendar.alias("cal"),
                F.col("t.transaction_date") == F.col("cal.calendar_date"),
                "inner"
            )
            .select(
                F.col("t.transaction_id"),
                F.col("t.customer_id"),
                F.col("t.account_id"),
                F.col("t.merchant_id"),
                F.col("t.branch_id"),
                F.col("t.transaction_amount"),
                F.col("t.fraud_flag"),
                F.col("t.transaction_date"),

                F.col("c.customer_segment"),
                F.col("c.credit_score"),
                F.col("c.risk_rating"),

                F.col("a.account_type"),

                F.col("m.merchant_category"),
                F.col("m.region").alias("merchant_region"),

                F.col("b.state").alias("branch_state"),
                F.col("b.region").alias("branch_region"),

                F.col("cal.year"),
                F.col("cal.month"),
                F.col("cal.quarter")
            )
        )

        ########################################################
        # Native numeric expressions
        ########################################################

        base = (
            base
            .withColumn(
                "transaction_bucket",
                F.when(
                    F.col("transaction_amount") < 50,
                    "LOW"
                )
                .when(
                    F.col("transaction_amount") < 500,
                    "MEDIUM"
                )
                .otherwise("HIGH")
            )
            .withColumn(
                "risk_score",
                (
                    F.col("transaction_amount")
                    * F.col("credit_score")
                )
                / (
                    F.col("credit_score") + F.lit(1)
                )
            )
            .withColumn(
                "fraud_amount",
                F.when(
                    F.col("fraud_flag") == 1,
                    F.col("transaction_amount")
                )
                .otherwise(F.lit(0))
            )
            .withColumn(
                "risk_adjusted_amount",
                F.col("transaction_amount")
                * (
                    F.col("credit_score") + F.lit(1)
                )
                / (
                    F.col("credit_score") + F.lit(1000)
                )
            )
            .withColumn(
                "amount_squared",
                F.col("transaction_amount")
                * F.col("transaction_amount")
            )
            .withColumn(
                "amount_cubed",
                F.col("transaction_amount")
                * F.col("amount_squared")
            )
            .withColumn(
                "risk_squared",
                F.col("risk_score")
                * F.col("risk_score")
            )
            .withColumn(
                "calc_a",
                (
                    F.col("amount_squared")
                    + F.col("risk_squared")
                ) / F.lit(3.0)
            )
            .withColumn(
                "calc_b",
                (
                    F.col("amount_cubed")
                    - F.col("amount_squared")
                )
                / (
                    F.col("credit_score") + F.lit(2.0)
                )
            )
            .withColumn(
                "calc_c",
                (
                    F.col("risk_score")
                    * F.col("risk_squared")
                )
                + F.col("transaction_amount")
            )
            .withColumn(
                "calc_d",
                (
                    F.col("amount_squared")
                    * F.col("risk_score")
                )
                / (
                    F.col("credit_score") + F.lit(10.0)
                )
            )
            .withColumn(
                "calc_e",
                (
                    F.col("amount_cubed")
                    + F.col("risk_score")
                )
                / (
                    F.col("transaction_amount") + F.lit(1.0)
                )
            )
            .withColumn(
                "compute_intensity",
                (
                    F.col("calc_a")
                    + F.col("calc_b")
                    + F.col("calc_c")
                    + F.col("calc_d")
                    + F.col("calc_e")
                    + F.col("risk_adjusted_amount")
                )
            )
        )

        ########################################################
        # Keep the majority of rows.
        ########################################################

        base = (
            base
            .filter(
                (F.col("transaction_amount") > 5)
                & (F.col("credit_score") > 300)
            )
        )

        ########################################################
        # V8 TOPOLOGY CHANGE
        #
        # First reduce the transaction-grain dataset to a
        # customer/merchant/month analytical grain.
        #
        # This creates a high-cardinality Exchange + Aggregate
        # while substantially reducing the amount of data that
        # subsequent stages need to process.
        ########################################################

        customer_merchant_month = (
            base
            .repartition(
                800,
                "customer_id",
                "merchant_id",
                "year",
                "month"
            )
            .sortWithinPartitions(
                "customer_id",
                "merchant_id",
                "year",
                "month"
            )
            .groupBy(
                "customer_id",
                "merchant_id",
                "year",
                "month",
                "customer_segment",
                "merchant_category",
                "merchant_region"
            )
            .agg(
                F.count("*").alias("txn_count"),
                F.sum("transaction_amount").alias("volume"),
                F.avg("transaction_amount").alias("avg_amount"),
                F.max("transaction_amount").alias("max_amount"),
                F.min("transaction_amount").alias("min_amount"),
                F.stddev("transaction_amount").alias("stddev_amount"),

                F.sum("fraud_flag").alias("fraud_events"),
                F.sum("fraud_amount").alias("fraud_amount"),

                F.sum("risk_score").alias("risk_score_sum"),
                F.avg("risk_score").alias("risk_score_avg"),

                F.sum("risk_adjusted_amount").alias(
                    "risk_adjusted_volume"
                ),

                F.sum("amount_squared").alias(
                    "amount_squared_sum"
                ),

                F.sum("amount_cubed").alias(
                    "amount_cubed_sum"
                ),

                F.sum("compute_intensity").alias(
                    "compute_intensity_sum"
                ),

                F.avg("calc_a").alias("calc_a_avg"),
                F.avg("calc_b").alias("calc_b_avg"),
                F.max("calc_c").alias("calc_c_max"),
                F.max("calc_d").alias("calc_d_max"),
                F.max("calc_e").alias("calc_e_max")
            )
        )

        ########################################################
        # Analytical reduction 1
        #
        # Customer/month perspective.
        #
        # This is performed against the reduced dataset rather
        # than against the original transaction grain.
        ########################################################

        customer_month = (
            customer_merchant_month
            .repartition(
                800,
                "customer_id",
                "year",
                "month"
            )
            .sortWithinPartitions(
                "customer_id",
                "year",
                "month"
            )
            .groupBy(
                "customer_id",
                "customer_segment",
                "year",
                "month"
            )
            .agg(
                F.sum("txn_count").alias("customer_txns"),
                F.sum("volume").alias("customer_volume"),
                F.avg("avg_amount").alias("customer_avg_amount"),
                F.max("max_amount").alias("customer_max_amount"),
                F.min("min_amount").alias("customer_min_amount"),

                F.sum("fraud_events").alias(
                    "customer_fraud_events"
                ),

                F.sum("fraud_amount").alias(
                    "customer_fraud_amount"
                ),

                F.sum("risk_adjusted_volume").alias(
                    "customer_risk_volume"
                ),

                F.sum("compute_intensity_sum").alias(
                    "customer_compute"
                ),

                F.avg("calc_a_avg").alias(
                    "customer_calc_a"
                ),

                F.max("calc_c_max").alias(
                    "customer_calc_c_peak"
                )
            )
        )

        ########################################################
        # Analytical reduction 2
        #
        # Merchant/month perspective.
        ########################################################

        merchant_month = (
            customer_merchant_month
            .repartition(
                800,
                "merchant_id",
                "year",
                "month"
            )
            .sortWithinPartitions(
                "merchant_id",
                "year",
                "month"
            )
            .groupBy(
                "merchant_id",
                "merchant_category",
                "merchant_region",
                "year",
                "month"
            )
            .agg(
                F.sum("txn_count").alias("merchant_txns"),
                F.sum("volume").alias("merchant_volume"),
                F.avg("avg_amount").alias("merchant_avg_amount"),
                F.max("max_amount").alias("merchant_max_amount"),

                F.sum("fraud_events").alias(
                    "merchant_fraud_events"
                ),

                F.sum("fraud_amount").alias(
                    "merchant_fraud_amount"
                ),

                F.sum("risk_adjusted_volume").alias(
                    "merchant_risk_volume"
                ),

                F.sum("compute_intensity_sum").alias(
                    "merchant_compute"
                ),

                F.avg("calc_b_avg").alias(
                    "merchant_calc_b"
                ),

                F.max("calc_d_max").alias(
                    "merchant_calc_d_peak"
                )
            )
        )

        ########################################################
        # Analytical reduction 3
        #
        # Customer/account/month perspective.
        #
        # We derive this from the reduced dataset by joining
        # account ownership back to the customer/merchant/month
        # analytical grain.
        ########################################################

        account_map = (
            base
            .select(
                "customer_id",
                "account_id",
                "account_type"
            )
            .dropDuplicates()
        )

        customer_merchant_month_account = (
            customer_merchant_month.alias("x")
            .join(
                account_map.alias("a"),
                F.col("x.customer_id")
                == F.col("a.customer_id"),
                "inner"
            )
            .select(
                F.col("x.customer_id"),
                F.col("x.year"),
                F.col("x.month"),
                F.col("x.txn_count"),
                F.col("x.volume"),
                F.col("x.fraud_events"),
                F.col("x.fraud_amount"),
                F.col("x.risk_adjusted_volume"),
                F.col("x.compute_intensity_sum"),
                F.col("a.account_id"),
                F.col("a.account_type")
            )
        )

        account_month = (
            customer_merchant_month_account
            .repartition(
                800,
                "account_id",
                "year",
                "month"
            )
            .sortWithinPartitions(
                "account_id",
                "year",
                "month"
            )
            .groupBy(
                "account_id",
                "account_type",
                "year",
                "month"
            )
            .agg(
                F.sum("txn_count").alias("account_txns"),
                F.sum("volume").alias("account_volume"),

                F.sum("fraud_events").alias(
                    "account_fraud_events"
                ),

                F.sum("fraud_amount").alias(
                    "account_fraud_amount"
                ),

                F.sum("risk_adjusted_volume").alias(
                    "account_risk_volume"
                ),

                F.sum("compute_intensity_sum").alias(
                    "account_compute"
                )
            )
        )

        ########################################################
        # Join analytical reductions
        #
        # Unlike v7, these joins are between substantially reduced
        # analytical datasets rather than repeatedly joining
        # aggregates back to the full transaction-grain dataset.
        ########################################################

        customer_merchant = (
            customer_merchant_month.alias("x")
            .join(
                customer_month.alias("c"),
                [
                    "customer_id",
                    "year",
                    "month",
                    "customer_segment"
                ],
                "inner"
            )
            .join(
                merchant_month.alias("m"),
                [
                    "merchant_id",
                    "year",
                    "month",
                    "merchant_category",
                    "merchant_region"
                ],
                "inner"
            )
            .select(
                F.col("x.customer_id"),
                F.col("x.merchant_id"),
                F.col("x.year"),
                F.col("x.month"),
                F.col("x.customer_segment"),
                F.col("x.merchant_category"),
                F.col("x.merchant_region"),

                F.col("x.txn_count"),
                F.col("x.volume"),
                F.col("x.avg_amount"),
                F.col("x.max_amount"),
                F.col("x.fraud_events"),
                F.col("x.fraud_amount"),
                F.col("x.risk_score_sum"),
                F.col("x.risk_adjusted_volume"),
                F.col("x.compute_intensity_sum"),

                F.col("c.customer_volume"),
                F.col("c.customer_avg_amount"),
                F.col("c.customer_fraud_amount"),
                F.col("c.customer_risk_volume"),
                F.col("c.customer_compute"),

                F.col("m.merchant_volume"),
                F.col("m.merchant_avg_amount"),
                F.col("m.merchant_fraud_amount"),
                F.col("m.merchant_risk_volume"),
                F.col("m.merchant_compute")
            )
        )

        ########################################################
        # Join account-level analytical result.
        #
        # The account mapping is intentionally joined after the
        # customer/merchant analytical reductions.
        ########################################################

        account_enriched = (
            customer_merchant.alias("cm")
            .join(
                account_map.alias("a"),
                F.col("cm.customer_id")
                == F.col("a.customer_id"),
                "inner"
            )
            .join(
                account_month.alias("am"),
                [
                    "account_id",
                    "account_type",
                    "year",
                    "month"
                ],
                "inner"
            )
            .select(
                F.col("cm.*"),
                F.col("a.account_id"),
                F.col("a.account_type"),

                F.col("am.account_txns"),
                F.col("am.account_volume"),
                F.col("am.account_fraud_events"),
                F.col("am.account_fraud_amount"),
                F.col("am.account_risk_volume"),
                F.col("am.account_compute")
            )
        )

        ########################################################
        # Regional reduction
        #
        # This creates another wide Exchange + Aggregate over
        # the already-reduced analytical dataset.
        ########################################################

        regional = (
            account_enriched
            .repartition(
                800,
                "merchant_region",
                "customer_segment",
                "account_type",
                "year",
                "month"
            )
            .sortWithinPartitions(
                "merchant_region",
                "customer_segment",
                "account_type",
                "year",
                "month"
            )
            .groupBy(
                "merchant_region",
                "customer_segment",
                "account_type",
                "year",
                "month"
            )
            .agg(
                F.sum("txn_count").alias(
                    "regional_txns"
                ),

                F.sum("volume").alias(
                    "regional_volume"
                ),

                F.avg("avg_amount").alias(
                    "regional_avg_amount"
                ),

                F.max("max_amount").alias(
                    "regional_max_amount"
                ),

                F.sum("fraud_events").alias(
                    "regional_fraud_events"
                ),

                F.sum("fraud_amount").alias(
                    "regional_fraud_amount"
                ),

                F.sum("customer_volume").alias(
                    "regional_customer_volume"
                ),

                F.sum("merchant_volume").alias(
                    "regional_merchant_volume"
                ),

                F.sum("account_volume").alias(
                    "regional_account_volume"
                ),

                F.sum("customer_compute").alias(
                    "regional_customer_compute"
                ),

                F.sum("merchant_compute").alias(
                    "regional_merchant_compute"
                ),

                F.sum("account_compute").alias(
                    "regional_account_compute"
                ),

                F.sum("risk_adjusted_volume").alias(
                    "regional_risk_volume"
                )
            )
        )

        ########################################################
        # Final analytical aggregation
        ########################################################

        result = (
            account_enriched.alias("e")
            .join(
                regional.alias("r"),
                [
                    "merchant_region",
                    "customer_segment",
                    "account_type",
                    "year",
                    "month"
                ],
                "inner"
            )
            .groupBy(
                "year",
                "month",
                "merchant_region",
                "customer_segment",
                "account_type",
                "merchant_category"
            )
            .agg(
                F.sum("txn_count").alias(
                    "transactions"
                ),

                F.sum("volume").alias(
                    "total_amount"
                ),

                F.avg("avg_amount").alias(
                    "avg_amount"
                ),

                F.max("max_amount").alias(
                    "max_amount"
                ),

                F.sum("fraud_events").alias(
                    "fraud_events"
                ),

                F.sum("fraud_amount").alias(
                    "fraud_amount"
                ),

                F.sum("risk_score_sum").alias(
                    "risk_score_sum"
                ),

                F.sum("risk_adjusted_volume").alias(
                    "risk_adjusted_total"
                ),

                F.sum("customer_volume").alias(
                    "customer_volume"
                ),

                F.sum("merchant_volume").alias(
                    "merchant_volume"
                ),

                F.sum("account_volume").alias(
                    "account_volume"
                ),

                F.sum("regional_volume").alias(
                    "regional_volume"
                ),

                F.sum("customer_compute").alias(
                    "customer_compute"
                ),

                F.sum("merchant_compute").alias(
                    "merchant_compute"
                ),

                F.sum("account_compute").alias(
                    "account_compute"
                ),

                F.sum("regional_customer_compute").alias(
                    "regional_customer_compute"
                ),

                F.sum("regional_merchant_compute").alias(
                    "regional_merchant_compute"
                ),

                F.sum("regional_account_compute").alias(
                    "regional_account_compute"
                ),

                F.sum("regional_risk_volume").alias(
                    "regional_risk_volume"
                )
            )
        )

        ########################################################
        # Final global sort
        ########################################################

        result = (
            result
            .orderBy(
                F.desc("total_amount"),
                F.desc("fraud_amount"),
                F.desc("regional_volume"),
                F.desc("risk_score_sum"),
                F.asc("year"),
                F.asc("month"),
                F.asc("merchant_region"),
                F.asc("customer_segment")
            )
        )

        return result

    ############################################################
    # Save
    ############################################################

    def save(self, df):

        df.write.mode(
            "overwrite"
        ).saveAsTable(
            f"{self.database}.ETL_V8_RESULT"
        )

        print("ETL v8 complete")

        print(
            f"Output rows: {df.count():,}"
        )


############################################################
# Main
############################################################

def main():

    USERNAME = os.environ["PROJECT_OWNER"]

    DATABASE = (
        "DEMO_pauldefusco"
    )

    CONNECTION_NAME = (
        "se-aws-edl"

    )

    STORAGE = (
        "s3a://goes-se-sandbox/data"
    )

    job = BankingETLv8(
        CONNECTION_NAME,
        DATABASE,
        STORAGE
    )

    start_time = time.time()

    spark = job.createSparkConnection()

    output = job.run(
        spark
    )

    job.save(
        output
    )

    end_time = time.time()

    elapsed = end_time - start_time

    print(
        f"\nTotal ETL job time: {elapsed:.2f} seconds "
        f"({elapsed / 60:.2f} minutes)"
    )


if __name__ == "__main__":
    main()
