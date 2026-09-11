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
# Spark RAPIDS Benchmark - ETL v1
#
# Workload:
#   - Fact-to-dimension joins
#   - Distributed joins
#   - Wide aggregations
#   - Distinct aggregations
#   - Sort
#
#****************************************************************************

import os

import cml.data_v1 as cmldata

from pyspark.sql import functions as F
from pyspark.sql import SparkSession



class BankingETLv1:


    def __init__(self, connection_name, database, storage):

        self.connection_name = connection_name
        self.database = database
        self.storage = storage



    ############################################################
    # Spark Configuration
    ############################################################

    def createSparkConnection(self):

        spark = (

            SparkSession.builder

            .appName("Spark-ETL-v1")

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
                "8g"
            )

            .config(
                "spark.sql.shuffle.partitions",
                800
            )

            .config(
                "spark.kerberos.access.hadoopFileSystems",
                self.storage
            )

            .getOrCreate()

        )


        #
        # Force shuffle-based joins
        # Better for RAPIDS qualification workloads
        #

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
        # Fact -> Dimension joins
        ########################################################

        enriched = (

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


            ####################################################
            # Explicit projection
            # Prevent duplicate column ambiguity
            ####################################################

            .select(

                F.col("t.transaction_id"),

                F.col("t.customer_id"),

                F.col("t.account_id"),

                F.col("t.merchant_id"),

                F.col("t.branch_id"),

                F.col("t.transaction_amount"),

                F.col("t.fraud_flag"),

                F.col("t.payment_channel"),

                F.col("t.transaction_date"),


                # Customer

                F.col("c.customer_segment"),

                F.col("c.credit_score"),

                F.col("c.risk_rating"),


                # Account

                F.col("a.account_type"),


                # Merchant

                F.col("m.merchant_category"),

                F.col("m.region")
                 .alias("merchant_region"),


                # Branch

                F.col("b.state")
                 .alias("branch_state"),

                F.col("b.region")
                 .alias("branch_region"),


                # Calendar

                F.col("cal.year"),

                F.col("cal.quarter"),

                F.col("cal.month")

            )

        )



        ########################################################
        # Derived columns
        ########################################################

        enriched = (

            enriched

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

                "fraud_amount",

                F.when(

                    F.col("fraud_flag") == 1,

                    F.col("transaction_amount")

                )

                .otherwise(0)

            )


            .withColumn(

                "high_value_flag",

                F.when(

                    F.col("transaction_amount") > 1000,

                    1

                )

                .otherwise(0)

            )

        )



        ########################################################
        # Aggregation
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

                "account_type",

                "merchant_category",

                "transaction_bucket"

            )


            .agg(

                F.count("*")
                .alias("transaction_count"),


                F.countDistinct(
                    "customer_id"
                )
                .alias("unique_customers"),


                F.countDistinct(
                    "merchant_id"
                )
                .alias("unique_merchants"),


                F.sum(
                    "transaction_amount"
                )
                .alias("total_amount"),


                F.avg(
                    "transaction_amount"
                )
                .alias("avg_transaction_amount"),


                F.sum(
                    "fraud_flag"
                )
                .alias("fraud_transactions"),


                F.sum(
                    "fraud_amount"
                )
                .alias("fraud_amount"),


                F.sum(
                    "high_value_flag"
                )
                .alias("high_value_transactions")

            )

        )



        ########################################################
        # Sort
        ########################################################

        result = result.orderBy(
            F.desc("total_amount")
        )


        return result



    ############################################################

    def save(self, df):

        df.write.mode(
            "overwrite"
        ).saveAsTable(

            f"{self.database}.ETL_V1_RESULT"

        )


        print(
            "ETL v1 complete"
        )


        print(
            f"Output rows: {df.count():,}"
        )



############################################################

def main():


    USERNAME = os.environ["PROJECT_OWNER"]

    DATABASE = f"DEMO_{USERNAME}"

    CONNECTION_NAME = "pdf0714-aw-dl"

    STORAGE = (
        "s3a://goes-se-sandbox/data"
    )



    job = BankingETLv1(

        CONNECTION_NAME,

        DATABASE,

        STORAGE

    )


    spark = job.createSparkConnection()


    output = job.run(
        spark
    )


    job.save(
        output
    )



if __name__ == "__main__":

    main()
