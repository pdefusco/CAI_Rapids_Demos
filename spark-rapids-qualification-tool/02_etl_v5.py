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
# Spark RAPIDS Benchmark - ETL v5
#
# Workload:
#   - Large fact table scan
#   - Multiple dimension joins
#   - Heavy expressions
#   - Multiple independent aggregation branches
#   - Fact-side daily aggregation and rejoin
#   - Customer analytics
#   - Merchant analytics
#   - Account analytics
#   - Branch/time analytics
#   - Monthly/category analytics
#   - Multiple aggregate rejoins
#   - Final multi-dimensional aggregation
#   - Final global sort
#
# This version is intentionally more shuffle-heavy than ETL v3 so that
# the resulting Spark event log contains substantially more aggregation,
# exchange, sort, and join work for the NVIDIA Spark RAPIDS Qualification Tool.
#
#****************************************************************************

import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import cml.data_v1 as cmldata


class BankingETLv5:

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

            .appName(
                "Spark-ETL-v5"
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
                800
            )

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

        # Disable broadcast joins so that the dimension and aggregate joins
        # generate explicit shuffle/exchange work in the event log.
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

    def run(self, spark):

        ########################################################
        # Read source tables
        ########################################################

        transactions = spark.table(
            f"{self.database}.TRS_v25"
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
        # Join fact + dimensions
        ########################################################

        base = (

            transactions.alias("t")

            .join(
                customers.alias("c"),

                F.col("t.customer_id")
                ==
                F.col("c.customer_id"),

                "inner"
            )

            .join(
                accounts.alias("a"),

                F.col("t.account_id")
                ==
                F.col("a.account_id"),

                "inner"
            )

            .join(
                merchants.alias("m"),

                F.col("t.merchant_id")
                ==
                F.col("m.merchant_id"),

                "inner"
            )

            .join(
                branches.alias("b"),

                F.col("t.branch_id")
                ==
                F.col("b.branch_id"),

                "inner"
            )

            .join(
                calendar.alias("cal"),

                F.col("t.transaction_date")
                ==
                F.col("cal.calendar_date"),

                "inner"
            )

            .select(

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

                F.col("m.region")
                .alias(
                    "merchant_region"
                ),

                F.col("b.state")
                .alias(
                    "branch_state"
                ),

                F.col("b.region")
                .alias(
                    "branch_region"
                ),

                F.col("cal.year"),

                F.col("cal.month"),

                F.col("cal.quarter")
            )
        )

        ########################################################
        # Heavy expressions
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

                .otherwise(
                    "HIGH"
                )
            )

            .withColumn(
                "risk_score",

                (
                    F.col("transaction_amount")
                    *
                    F.col("credit_score")
                )
                /
                (
                    F.col("credit_score")
                    + 1
                )
            )

            .withColumn(
                "fraud_amount",

                F.when(
                    F.col("fraud_flag") == 1,
                    F.col("transaction_amount")
                )

                .otherwise(0)
            )

            .withColumn(
                "risk_adjusted_amount",

                F.col("transaction_amount")
                *
                (
                    F.col("credit_score") + 1
                )
                /
                (
                    F.col("credit_score")
                    + F.lit(1000)
                )
            )

            .withColumn(
                "fraud_risk_amount",

                F.when(
                    F.col("fraud_flag") == 1,

                    F.col("transaction_amount")
                    *
                    F.col("risk_score")
                )

                .otherwise(0)
            )

            .withColumn(
                "amount_squared",

                F.col("transaction_amount")
                *
                F.col("transaction_amount")
            )
        )

        ########################################################
        # Aggregation Layer 1
        # Customer behavior
        ########################################################

        customer_metrics = (

            base

            .groupBy(
                "customer_id",
                "customer_segment",
                "risk_rating"
            )

            .agg(

                F.count("*")
                .alias(
                    "customer_txn_count"
                ),

                F.sum(
                    "transaction_amount"
                )
                .alias(
                    "customer_total_spend"
                ),

                F.avg(
                    "transaction_amount"
                )
                .alias(
                    "customer_avg_spend"
                ),

                F.max(
                    "risk_score"
                )
                .alias(
                    "customer_risk_score"
                ),

                F.sum(
                    "fraud_flag"
                )
                .alias(
                    "customer_fraud_count"
                ),

                F.sum(
                    "fraud_amount"
                )
                .alias(
                    "customer_fraud_amount"
                ),

                F.stddev(
                    "transaction_amount"
                )
                .alias(
                    "customer_spend_stddev"
                ),

                F.max(
                    "amount_squared"
                )
                .alias(
                    "customer_max_amount_squared"
                )
            )
        )

        ########################################################
        # Aggregation Layer 2
        # Merchant behavior
        ########################################################

        merchant_metrics = (

            base

            .groupBy(
                "merchant_id",
                "merchant_category",
                "merchant_region"
            )

            .agg(

                F.count("*")
                .alias(
                    "merchant_txn_count"
                ),

                F.sum(
                    "transaction_amount"
                )
                .alias(
                    "merchant_volume"
                ),

                F.avg(
                    "transaction_amount"
                )
                .alias(
                    "merchant_avg_transaction"
                ),

                F.max(
                    "transaction_amount"
                )
                .alias(
                    "merchant_max_transaction"
                ),

                F.sum(
                    "fraud_flag"
                )
                .alias(
                    "merchant_fraud_count"
                ),

                F.sum(
                    "fraud_amount"
                )
                .alias(
                    "merchant_fraud_amount"
                ),

                F.stddev(
                    "transaction_amount"
                )
                .alias(
                    "merchant_spend_stddev"
                )
            )
        )

        ########################################################
        # Aggregation Layer 3
        # Account behavior
        ########################################################

        account_metrics = (

            base

            .groupBy(
                "account_id",
                "account_type",
                "merchant_region"
            )

            .agg(

                F.count("*")
                .alias(
                    "account_txn_count"
                ),

                F.sum(
                    "transaction_amount"
                )
                .alias(
                    "account_total_spend"
                ),

                F.avg(
                    "transaction_amount"
                )
                .alias(
                    "account_avg_spend"
                ),

                F.max(
                    "transaction_amount"
                )
                .alias(
                    "account_max_spend"
                ),

                F.min(
                    "transaction_amount"
                )
                .alias(
                    "account_min_spend"
                ),

                F.stddev(
                    "transaction_amount"
                )
                .alias(
                    "account_spend_stddev"
                ),

                F.sum(
                    "fraud_amount"
                )
                .alias(
                    "account_fraud_amount"
                ),

                F.max(
                    "risk_adjusted_amount"
                )
                .alias(
                    "account_max_risk_adjusted"
                )
            )
        )

        ########################################################
        # Aggregation Layer 4
        # Branch / quarterly behavior
        ########################################################

        branch_metrics = (

            base

            .groupBy(
                "branch_state",
                "branch_region",
                "year",
                "quarter"
            )

            .agg(

                F.count("*")
                .alias(
                    "branch_txn_count"
                ),

                F.sum(
                    "transaction_amount"
                )
                .alias(
                    "branch_volume"
                ),

                F.avg(
                    "transaction_amount"
                )
                .alias(
                    "branch_avg_transaction"
                ),

                F.max(
                    "transaction_amount"
                )
                .alias(
                    "branch_max_transaction"
                ),

                F.sum(
                    "fraud_flag"
                )
                .alias(
                    "branch_fraud_count"
                ),

                F.sum(
                    "fraud_amount"
                )
                .alias(
                    "branch_fraud_amount"
                )
            )
        )

        ########################################################
        # Aggregation Layer 5
        # Monthly merchant/customer analytics
        ########################################################

        monthly_metrics = (

            base

            .groupBy(
                "year",
                "month",
                "merchant_category",
                "customer_segment"
            )

            .agg(

                F.count("*")
                .alias(
                    "monthly_txn_count"
                ),

                F.sum(
                    "transaction_amount"
                )
                .alias(
                    "monthly_volume"
                ),

                F.avg(
                    "transaction_amount"
                )
                .alias(
                    "monthly_avg_transaction"
                ),

                F.max(
                    "transaction_amount"
                )
                .alias(
                    "monthly_max_transaction"
                ),

                F.sum(
                    "fraud_flag"
                )
                .alias(
                    "monthly_fraud_count"
                ),

                F.sum(
                    "fraud_amount"
                )
                .alias(
                    "monthly_fraud_amount"
                ),

                F.avg(
                    "risk_score"
                )
                .alias(
                    "monthly_avg_risk"
                )
            )
        )

        ########################################################
        # Aggregation Layer 6
        # Fact-side daily customer aggregation.
        #
        # Unlike a simple dimension lookup, this creates a new
        # aggregated fact dataset and then rejoins it to the
        # transaction-level data.
        ########################################################

        daily_customer_metrics = (

            base

            .groupBy(
                "customer_id",
                "transaction_date"
            )

            .agg(

                F.count("*")
                .alias(
                    "daily_customer_txn_count"
                ),

                F.sum(
                    "transaction_amount"
                )
                .alias(
                    "daily_customer_volume"
                ),

                F.avg(
                    "transaction_amount"
                )
                .alias(
                    "daily_customer_avg"
                ),

                F.sum(
                    "fraud_flag"
                )
                .alias(
                    "daily_customer_fraud"
                ),

                F.sum(
                    "fraud_amount"
                )
                .alias(
                    "daily_customer_fraud_amount"
                ),

                F.max(
                    "risk_score"
                )
                .alias(
                    "daily_customer_max_risk"
                )
            )
        )

        ########################################################
        # Aggregation Layer 7
        # Merchant/category/time analytics
        ########################################################

        merchant_month_metrics = (

            base

            .groupBy(
                "merchant_id",
                "merchant_category",
                "merchant_region",
                "year",
                "month"
            )

            .agg(

                F.count("*")
                .alias(
                    "merchant_month_txn_count"
                ),

                F.sum(
                    "transaction_amount"
                )
                .alias(
                    "merchant_month_volume"
                ),

                F.avg(
                    "transaction_amount"
                )
                .alias(
                    "merchant_month_avg"
                ),

                F.sum(
                    "fraud_amount"
                )
                .alias(
                    "merchant_month_fraud"
                ),

                F.max(
                    "risk_score"
                )
                .alias(
                    "merchant_month_max_risk"
                )
            )
        )

        ########################################################
        # Rejoin aggregate datasets
        #
        # All broadcast joins are disabled above. These joins are
        # therefore intentionally represented as shuffle joins.
        ########################################################

        enriched = (

            base

            .join(
                customer_metrics,
                [
                    "customer_id",
                    "customer_segment",
                    "risk_rating"
                ],
                "inner"
            )

            .join(
                merchant_metrics,
                [
                    "merchant_id",
                    "merchant_category",
                    "merchant_region"
                ],
                "inner"
            )

            .join(
                account_metrics,
                [
                    "account_id",
                    "account_type",
                    "merchant_region"
                ],
                "inner"
            )

            .join(
                branch_metrics,
                [
                    "branch_state",
                    "branch_region",
                    "year",
                    "quarter"
                ],
                "inner"
            )

            .join(
                monthly_metrics,
                [
                    "year",
                    "month",
                    "merchant_category",
                    "customer_segment"
                ],
                "inner"
            )

            .join(
                daily_customer_metrics,
                [
                    "customer_id",
                    "transaction_date"
                ],
                "inner"
            )

            .join(
                merchant_month_metrics,
                [
                    "merchant_id",
                    "merchant_category",
                    "merchant_region",
                    "year",
                    "month"
                ],
                "inner"
            )
        )

        ########################################################
        # Second analytical aggregation over the enriched data
        ########################################################

        regional_metrics = (

            enriched

            .groupBy(
                "year",
                "quarter",
                "branch_region",
                "merchant_region",
                "customer_segment",
                "account_type"
            )

            .agg(

                F.count("*")
                .alias(
                    "regional_transactions"
                ),

                F.sum(
                    "transaction_amount"
                )
                .alias(
                    "regional_volume"
                ),

                F.avg(
                    "customer_total_spend"
                )
                .alias(
                    "regional_avg_customer_value"
                ),

                F.avg(
                    "merchant_volume"
                )
                .alias(
                    "regional_avg_merchant_volume"
                ),

                F.sum(
                    "customer_fraud_count"
                )
                .alias(
                    "regional_customer_fraud"
                ),

                F.sum(
                    "merchant_fraud_count"
                )
                .alias(
                    "regional_merchant_fraud"
                ),

                F.sum(
                    "account_fraud_amount"
                )
                .alias(
                    "regional_account_fraud_amount"
                ),

                F.sum(
                    "daily_customer_fraud_amount"
                )
                .alias(
                    "regional_daily_fraud_amount"
                ),

                F.max(
                    "merchant_month_max_risk"
                )
                .alias(
                    "regional_max_risk"
                )
            )
        )

        ########################################################
        # Third analytical aggregation
        ########################################################

        result = (

            enriched

            .groupBy(
                "year",
                "quarter",
                "month",
                "branch_state",
                "branch_region",
                "customer_segment",
                "merchant_category",
                "merchant_region",
                "account_type",
                "transaction_bucket"
            )

            .agg(

                F.count("*")
                .alias(
                    "transactions"
                ),

                F.sum(
                    "transaction_amount"
                )
                .alias(
                    "total_amount"
                ),

                F.avg(
                    "customer_total_spend"
                )
                .alias(
                    "avg_customer_value"
                ),

                F.avg(
                    "merchant_volume"
                )
                .alias(
                    "avg_merchant_volume"
                ),

                F.avg(
                    "account_total_spend"
                )
                .alias(
                    "avg_account_value"
                ),

                F.sum(
                    "customer_fraud_count"
                )
                .alias(
                    "fraud_events"
                ),

                F.sum(
                    "merchant_fraud_count"
                )
                .alias(
                    "merchant_fraud_events"
                ),

                F.sum(
                    "account_fraud_amount"
                )
                .alias(
                    "account_fraud_amount"
                ),

                F.sum(
                    "daily_customer_volume"
                )
                .alias(
                    "daily_customer_volume"
                ),

                F.sum(
                    "monthly_volume"
                )
                .alias(
                    "monthly_volume"
                ),

                F.sum(
                    "merchant_month_volume"
                )
                .alias(
                    "merchant_month_volume"
                ),

                F.avg(
                    "risk_score"
                )
                .alias(
                    "avg_risk_score"
                ),

                F.max(
                    "merchant_month_max_risk"
                )
                .alias(
                    "max_merchant_month_risk"
                ),

                F.sum(
                    "amount_squared"
                )
                .alias(
                    "amount_squared_sum"
                )
            )
        )

        ########################################################
        # Rejoin the regional aggregate to create another
        # shuffle-heavy analytical stage.
        ########################################################

        result = (

            result.alias("r")

            .join(

                regional_metrics.alias("rm"),

                (
                    (F.col("r.year") == F.col("rm.year"))
                    &
                    (F.col("r.quarter") == F.col("rm.quarter"))
                    &
                    (F.col("r.branch_region") == F.col("rm.branch_region"))
                    &
                    (F.col("r.merchant_region") == F.col("rm.merchant_region"))
                    &
                    (F.col("r.customer_segment") == F.col("rm.customer_segment"))
                    &
                    (F.col("r.account_type") == F.col("rm.account_type"))
                ),

                "left"
            )

            .select(
                F.col("r.*"),

                F.col(
                    "rm.regional_transactions"
                ),

                F.col(
                    "rm.regional_volume"
                ),

                F.col(
                    "rm.regional_avg_customer_value"
                ),

                F.col(
                    "rm.regional_avg_merchant_volume"
                ),

                F.col(
                    "rm.regional_customer_fraud"
                ),

                F.col(
                    "rm.regional_merchant_fraud"
                ),

                F.col(
                    "rm.regional_account_fraud_amount"
                ),

                F.col(
                    "rm.regional_daily_fraud_amount"
                ),

                F.col(
                    "rm.regional_max_risk"
                )
            )
        )

        ########################################################
        # Final sort
        ########################################################

        result = (

            result

            .orderBy(
                F.desc(
                    "total_amount"
                ),
                F.desc(
                    "fraud_events"
                ),
                F.desc(
                    "regional_volume"
                )
            )
        )

        return result

    ############################################################

    def save(self, df):

        df.write.mode(
            "overwrite"
        ).saveAsTable(
            f"{self.database}.ETL_v5_RESULT"
        )

        print(
            "ETL v5 complete"
        )

        print(
            f"Output rows: {df.count():,}"
        )


############################################################

def main():

    USERNAME = os.environ["PROJECT_OWNER"]

    DATABASE = (
        f"DEMO_pauldefusco"
    )

    CONNECTION_NAME = (
        "pdf0714-aw-dl"
    )

    STORAGE = (
        "s3a://goes-se-sandbox/data"
    )

    job = BankingETLv5(
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
        f"\nTotal ETL job time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)"
    )


if __name__ == "__main__":
    main()
