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
# Spark RAPIDS Benchmark - ETL v7
#
# Purpose:
#   This version is engineered from the v5 qualification profile.
#
#   v5 qualification showed that most task time was concentrated in:
#       Scan parquet -> Filter -> ColumnarToRow
#
#   The reason is that v4 fans the same transaction-level lineage into many
#   independent aggregation branches. Spark can therefore revisit the fact
#   scan/joins for each branch.
#
#   v6 instead uses a sequential analytical pipeline:
#       fact/dimension joins
#           -> selective filter
#           -> customer-month aggregation
#           -> shuffle SortMergeJoin back to fact
#           -> merchant-quarter aggregation
#           -> shuffle SortMergeJoin back to fact
#           -> account-month aggregation
#           -> shuffle SortMergeJoin back to fact
#           -> regional aggregation
#           -> second regional SortMergeJoin
#           -> final multi-dimensional aggregation
#           -> expensive global sort
#
#   The intent is NOT to manufacture unsupported work. The intent is to
#   increase the proportion of legitimate, RAPIDS-supported:
#       Exchange / Sort / SortMergeJoin / HashAggregate / Project / Filter
#   operations relative to raw Parquet scanning.
#
#****************************************************************************

import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import cml.data_v1 as cmldata


class BankingETLv7:

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
            .appName("Spark-ETL-v7")
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

        # Force the dimension joins and analytical joins to remain shuffle
        # joins so the qualification tool sees the intended SortMergeJoin /
        # Exchange workload.
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
            # Push the reduction into the fact-table scan. The benchmark
            # data generator uses a monotonically increasing transaction_id,
            # so Parquet row-group statistics can prune roughly half of the
            # source data before the expensive joins and expressions.
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
        # Fact + dimensions
        #
        # Important v5 change:
        #   This is one continuous fact pipeline rather than several
        #   independent branches starting from "base".
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
        # Heavy GPU-friendly expressions
        #
        # These are deliberately numeric and built from native Spark
        # expressions so they remain good RAPIDS candidates.
        ########################################################

        base = (
            base
            .withColumn(
                "transaction_bucket",
                F.when(F.col("transaction_amount") < 50, "LOW")
                 .when(F.col("transaction_amount") < 500, "MEDIUM")
                 .otherwise("HIGH")
            )
            .withColumn(
                "risk_score",
                (F.col("transaction_amount") * F.col("credit_score"))
                / (F.col("credit_score") + F.lit(1))
            )
            .withColumn(
                "fraud_amount",
                F.when(F.col("fraud_flag") == 1, F.col("transaction_amount"))
                 .otherwise(F.lit(0))
            )
            .withColumn(
                "risk_adjusted_amount",
                F.col("transaction_amount")
                * (F.col("credit_score") + F.lit(1))
                / (F.col("credit_score") + F.lit(1000))
            )
            .withColumn(
                "fraud_risk_amount",
                F.when(
                    F.col("fraud_flag") == 1,
                    F.col("transaction_amount") * F.col("risk_score")
                ).otherwise(F.lit(0))
            )
            .withColumn("amount_squared", F.col("transaction_amount") * F.col("transaction_amount"))
            .withColumn("amount_cubed", F.col("transaction_amount") * F.col("amount_squared"))
            .withColumn("risk_squared", F.col("risk_score") * F.col("risk_score"))

            # V7 compute-density workload. These are intentionally composed
            # from native numeric Spark expressions so RAPIDS can target the
            # Project expression evaluation rather than introducing UDFs.
            # The extra arithmetic is performed on the surviving rows so the
            # benchmark remains compute-heavy despite the narrower scan.
            .withColumn("calc_01", (F.col("amount_squared") + F.col("risk_squared")) / F.lit(3.0))
            .withColumn("calc_02", (F.col("amount_cubed") - F.col("amount_squared")) / (F.col("credit_score") + 2.0))
            .withColumn("calc_03", (F.col("risk_score") * F.col("risk_squared")) + F.col("transaction_amount"))
            .withColumn("calc_04", (F.col("amount_squared") * F.col("risk_score")) / (F.col("credit_score") + 10.0))
            .withColumn("calc_05", (F.col("amount_cubed") + F.col("risk_score")) / (F.col("transaction_amount") + 1.0))
            .withColumn("calc_06", (F.col("calc_01") * F.col("calc_02")) + F.col("risk_adjusted_amount"))
            .withColumn("calc_07", (F.col("calc_03") + F.col("calc_04")) / (F.col("credit_score") + 5.0))
            .withColumn("calc_08", F.col("calc_05") * F.col("calc_06") + F.col("fraud_risk_amount"))
            .withColumn("calc_09", (F.col("calc_07") * F.col("calc_08")) / (F.col("risk_score") + 1.0))
            .withColumn("calc_10", F.col("calc_09") + F.col("amount_squared") + F.col("risk_squared"))
            .withColumn("calc_11", (F.col("calc_10") * F.col("risk_adjusted_amount")) / (F.col("transaction_amount") + 10.0))
            .withColumn("calc_12", (F.col("calc_11") + F.col("calc_06")) * (F.col("credit_score") + 1.0))
            .withColumn("calc_13", F.col("calc_12") / (F.col("risk_score") + 2.0))
            .withColumn("calc_14", F.col("calc_13") * F.col("calc_04") + F.col("calc_08"))
            .withColumn("calc_15", (F.col("calc_14") + F.col("amount_cubed")) / (F.col("credit_score") + 20.0))
            .withColumn("calc_16", F.col("calc_15") * F.col("calc_10") + F.col("risk_squared"))
            .withColumn("calc_17", (F.col("calc_16") + F.col("calc_12")) / (F.col("transaction_amount") + 25.0))
            .withColumn("calc_18", F.col("calc_17") * F.col("calc_07") + F.col("fraud_risk_amount"))
            .withColumn("calc_19", (F.col("calc_18") + F.col("calc_15")) / (F.col("credit_score") + 50.0))
            .withColumn("calc_20", F.col("calc_19") * F.col("calc_16") + F.col("risk_adjusted_amount"))
            .withColumn("compute_intensity",
                        F.col("calc_20") + F.col("calc_18") + F.col("calc_14")
                        + F.col("calc_10") + F.col("calc_06"))
        )

        ########################################################
        # Filter
        #
        # Keep most of the fact rows so the downstream shuffles
        # remain substantial. This is intentionally not a highly
        # selective filter.
        ########################################################

        base = (
            base
            .filter(
                (F.col("transaction_amount") > 5)
                & (F.col("credit_score") > 300)
            )
        )

        ########################################################
        # Analytical Stage 1
        #
        # Customer / month aggregation.
        #
        # High-cardinality grouping key creates a substantial
        # shuffle while retaining many output groups.
        ########################################################

        customer_month = (
            base
            .groupBy(
                "customer_id",
                "customer_segment",
                "risk_rating",
                "year",
                "month"
            )
            .agg(
                F.count("*").alias("cm_txn_count"),
                F.sum("transaction_amount").alias("cm_volume"),
                F.avg("transaction_amount").alias("cm_avg_amount"),
                F.max("transaction_amount").alias("cm_max_amount"),
                F.min("transaction_amount").alias("cm_min_amount"),
                F.stddev("transaction_amount").alias("cm_stddev"),
                F.sum("fraud_flag").alias("cm_fraud_events"),
                F.sum("fraud_amount").alias("cm_fraud_amount"),
                F.sum("risk_adjusted_amount").alias("cm_risk_amount"),
                F.sum("amount_squared").alias("cm_amount_squared"),
                F.sum("amount_cubed").alias("cm_amount_cubed"),
                F.max("risk_squared").alias("cm_max_risk_squared"),
                F.sum("compute_intensity").alias("cm_compute_intensity"),
                F.avg("calc_20").alias("cm_calc_20_avg"),
                F.max("calc_18").alias("cm_calc_18_max")
            )
        )

        ########################################################
        # Rejoin Stage 1
        #
        # Broadcast is disabled, so this becomes a shuffle
        # SortMergeJoin.
        ########################################################

        enriched = (
            base.alias("b")
            .join(
                customer_month.alias("cm"),
                [
                    "customer_id",
                    "customer_segment",
                    "risk_rating",
                    "year",
                    "month"
                ],
                "inner"
            )
            .select(
                F.col("b.*"),
                F.col("cm.cm_txn_count"),
                F.col("cm.cm_volume"),
                F.col("cm.cm_avg_amount"),
                F.col("cm.cm_max_amount"),
                F.col("cm.cm_stddev"),
                F.col("cm.cm_fraud_events"),
                F.col("cm.cm_fraud_amount"),
                F.col("cm.cm_risk_amount"),
                F.col("cm.cm_amount_squared"),
                F.col("cm.cm_amount_cubed"),
                F.col("cm.cm_max_risk_squared"),
                F.col("cm.cm_compute_intensity"),
                F.col("cm.cm_calc_20_avg"),
                F.col("cm.cm_calc_18_max")
            )
        )

        ########################################################
        # Explicit repartition + local sort.
        #
        # This is intentional: the benchmark should contain
        # substantial supported Sort / Exchange activity rather
        # than relying only on the join planner to introduce it.
        ########################################################

        enriched = (
            enriched
            .repartition(
                800,
                "merchant_id",
                "merchant_region",
                "year",
                "quarter"
            )
            .sortWithinPartitions(
                "merchant_id",
                "merchant_region",
                "year",
                "quarter"
            )
        )

        ########################################################
        # Analytical Stage 2
        #
        # Merchant / quarter aggregation.
        ########################################################

        merchant_quarter = (
            enriched
            .groupBy(
                "merchant_id",
                "merchant_category",
                "merchant_region",
                "year",
                "quarter"
            )
            .agg(
                F.count("*").alias("mq_txn_count"),
                F.sum("transaction_amount").alias("mq_volume"),
                F.avg("transaction_amount").alias("mq_avg_amount"),
                F.max("transaction_amount").alias("mq_max_amount"),
                F.min("transaction_amount").alias("mq_min_amount"),
                F.stddev("transaction_amount").alias("mq_stddev"),
                F.sum("fraud_flag").alias("mq_fraud_events"),
                F.sum("fraud_amount").alias("mq_fraud_amount"),
                F.sum("cm_volume").alias("mq_customer_month_volume"),
                F.avg("cm_avg_amount").alias("mq_customer_month_avg"),
                F.max("cm_max_risk_squared").alias("mq_max_risk"),
                F.sum("risk_adjusted_amount").alias("mq_risk_amount"),
                F.sum("compute_intensity").alias("mq_compute_intensity"),
                F.avg("calc_19").alias("mq_calc_19_avg"),
                F.max("calc_16").alias("mq_calc_16_max")
            )
        )

        ########################################################
        # Rejoin Stage 2
        ########################################################

        enriched = (
            enriched.alias("e")
            .join(
                merchant_quarter.alias("mq"),
                [
                    "merchant_id",
                    "merchant_category",
                    "merchant_region",
                    "year",
                    "quarter"
                ],
                "inner"
            )
            .select(
                F.col("e.*"),
                F.col("mq.mq_txn_count"),
                F.col("mq.mq_volume"),
                F.col("mq.mq_avg_amount"),
                F.col("mq.mq_max_amount"),
                F.col("mq.mq_stddev"),
                F.col("mq.mq_fraud_events"),
                F.col("mq.mq_fraud_amount"),
                F.col("mq.mq_customer_month_volume"),
                F.col("mq.mq_customer_month_avg"),
                F.col("mq.mq_max_risk"),
                F.col("mq.mq_risk_amount"),
                F.col("mq.mq_compute_intensity"),
                F.col("mq.mq_calc_19_avg"),
                F.col("mq.mq_calc_16_max")
            )
        )

        ########################################################
        # Explicit repartition + sort for the next analytical
        # aggregation.
        ########################################################

        enriched = (
            enriched
            .repartition(
                800,
                "account_id",
                "account_type",
                "year",
                "month",
                "branch_region"
            )
            .sortWithinPartitions(
                "account_id",
                "account_type",
                "year",
                "month",
                "branch_region"
            )
        )

        ########################################################
        # Analytical Stage 3
        #
        # Account / month / region aggregation.
        ########################################################

        account_month = (
            enriched
            .groupBy(
                "account_id",
                "account_type",
                "year",
                "month",
                "branch_region"
            )
            .agg(
                F.count("*").alias("am_txn_count"),
                F.sum("transaction_amount").alias("am_volume"),
                F.avg("transaction_amount").alias("am_avg_amount"),
                F.max("transaction_amount").alias("am_max_amount"),
                F.min("transaction_amount").alias("am_min_amount"),
                F.stddev("transaction_amount").alias("am_stddev"),
                F.sum("fraud_flag").alias("am_fraud_events"),
                F.sum("fraud_amount").alias("am_fraud_amount"),
                F.sum("risk_adjusted_amount").alias("am_risk_amount"),
                F.sum("mq_volume").alias("am_merchant_quarter_volume"),
                F.avg("mq_avg_amount").alias("am_merchant_quarter_avg"),
                F.max("mq_max_risk").alias("am_max_risk"),
                F.sum("compute_intensity").alias("am_compute_intensity"),
                F.avg("calc_17").alias("am_calc_17_avg"),
                F.max("calc_14").alias("am_calc_14_max")
            )
        )

        ########################################################
        # Rejoin Stage 3
        ########################################################

        enriched = (
            enriched.alias("e")
            .join(
                account_month.alias("am"),
                [
                    "account_id",
                    "account_type",
                    "year",
                    "month",
                    "branch_region"
                ],
                "inner"
            )
            .select(
                F.col("e.*"),
                F.col("am.am_txn_count"),
                F.col("am.am_volume"),
                F.col("am.am_avg_amount"),
                F.col("am.am_max_amount"),
                F.col("am.am_stddev"),
                F.col("am.am_fraud_events"),
                F.col("am.am_fraud_amount"),
                F.col("am.am_risk_amount"),
                F.col("am.am_merchant_quarter_volume"),
                F.col("am.am_merchant_quarter_avg"),
                F.col("am.am_max_risk"),
                F.col("am.am_compute_intensity"),
                F.col("am.am_calc_17_avg"),
                F.col("am.am_calc_14_max")
            )
        )

        ########################################################
        # Explicit repartition + sort before regional aggregation.
        ########################################################

        enriched = (
            enriched
            .repartition(
                800,
                "branch_region",
                "merchant_region",
                "customer_segment",
                "account_type",
                "year",
                "quarter"
            )
            .sortWithinPartitions(
                "branch_region",
                "merchant_region",
                "customer_segment",
                "account_type",
                "year",
                "quarter"
            )
        )

        ########################################################
        # Analytical Stage 4
        #
        # Regional aggregation. This is deliberately performed
        # after multiple enrich/repartition stages so the final
        # aggregation operates on a wide analytical record.
        ########################################################

        regional = (
            enriched
            .groupBy(
                "branch_region",
                "merchant_region",
                "customer_segment",
                "account_type",
                "year",
                "quarter"
            )
            .agg(
                F.count("*").alias("regional_txns"),
                F.sum("transaction_amount").alias("regional_volume"),
                F.avg("transaction_amount").alias("regional_avg"),
                F.max("transaction_amount").alias("regional_max"),
                F.stddev("transaction_amount").alias("regional_stddev"),
                F.sum("fraud_flag").alias("regional_fraud"),
                F.sum("fraud_amount").alias("regional_fraud_amount"),
                F.sum("cm_volume").alias("regional_customer_volume"),
                F.sum("mq_volume").alias("regional_merchant_volume"),
                F.sum("am_volume").alias("regional_account_volume"),
                F.avg("cm_avg_amount").alias("regional_customer_avg"),
                F.avg("mq_avg_amount").alias("regional_merchant_avg"),
                F.avg("am_avg_amount").alias("regional_account_avg"),
                F.max("am_max_risk").alias("regional_max_risk"),
                F.sum("compute_intensity").alias("regional_compute_intensity"),
                F.avg("calc_15").alias("regional_calc_15_avg"),
                F.max("calc_12").alias("regional_calc_12_max")
            )
        )

        ########################################################
        # Rejoin Stage 4
        ########################################################

        enriched = (
            enriched.alias("e")
            .join(
                regional.alias("r"),
                [
                    "branch_region",
                    "merchant_region",
                    "customer_segment",
                    "account_type",
                    "year",
                    "quarter"
                ],
                "inner"
            )
            .select(
                F.col("e.*"),
                F.col("r.regional_txns"),
                F.col("r.regional_volume"),
                F.col("r.regional_avg"),
                F.col("r.regional_max"),
                F.col("r.regional_stddev"),
                F.col("r.regional_fraud"),
                F.col("r.regional_fraud_amount"),
                F.col("r.regional_customer_volume"),
                F.col("r.regional_merchant_volume"),
                F.col("r.regional_account_volume"),
                F.col("r.regional_customer_avg"),
                F.col("r.regional_merchant_avg"),
                F.col("r.regional_account_avg"),
                F.col("r.regional_max_risk"),
                F.col("r.regional_compute_intensity"),
                F.col("r.regional_calc_15_avg"),
                F.col("r.regional_calc_12_max")
            )
        )

        ########################################################
        # Final analytical aggregation
        #
        # Multiple grouping dimensions deliberately create another
        # substantial HashAggregate + Exchange stage.
        ########################################################

        result = (
            enriched
            .groupBy(
                "year",
                "quarter",
                "month",
                "branch_state",
                "branch_region",
                "merchant_region",
                "customer_segment",
                "merchant_category",
                "account_type",
                "transaction_bucket"
            )
            .agg(
                F.count("*").alias("transactions"),
                F.sum("transaction_amount").alias("total_amount"),
                F.avg("transaction_amount").alias("avg_amount"),
                F.max("transaction_amount").alias("max_amount"),
                F.stddev("transaction_amount").alias("amount_stddev"),
                F.sum("fraud_flag").alias("fraud_events"),
                F.sum("fraud_amount").alias("fraud_amount"),
                F.sum("risk_score").alias("risk_score_sum"),
                F.avg("risk_score").alias("avg_risk_score"),
                F.sum("risk_adjusted_amount").alias("risk_adjusted_total"),
                F.sum("amount_squared").alias("amount_squared_sum"),
                F.sum("amount_cubed").alias("amount_cubed_sum"),
                F.sum("cm_volume").alias("customer_month_volume"),
                F.sum("mq_volume").alias("merchant_quarter_volume"),
                F.sum("am_volume").alias("account_month_volume"),
                F.sum("regional_volume").alias("regional_volume"),
                F.avg("regional_avg").alias("regional_avg"),
                F.max("regional_max_risk").alias("regional_max_risk"),
                F.sum("compute_intensity").alias("compute_intensity_sum"),
                F.avg("calc_20").alias("compute_intensity_avg"),
                F.max("calc_18").alias("compute_peak"),
                F.sum("cm_compute_intensity").alias("customer_compute_sum"),
                F.sum("mq_compute_intensity").alias("merchant_compute_sum"),
                F.sum("am_compute_intensity").alias("account_compute_sum"),
                F.sum("regional_compute_intensity").alias("regional_compute_sum")
            )
        )

        ########################################################
        # Final sort
        #
        # Sorting a relatively large final analytical result is
        # intentional. Unlike v4, the preceding stages perform
        # substantial reduction before this global sort.
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
                F.asc("branch_region"),
                F.asc("merchant_region")
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
            f"{self.database}.ETL_V7_RESULT"
        )

        print("ETL v7 complete")

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
        "pdf0714-aw-dl"
    )

    STORAGE = (
        "s3a://goes-se-sandbox/data"
    )

    job = BankingETLv7(
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
