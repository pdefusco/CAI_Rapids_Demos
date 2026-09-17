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
# Spark RAPIDS Benchmark - ETL v3
#
# Workload:
#   - Large fact table scan
#   - Multiple dimension joins
#   - Multi-stage aggregations
#   - Customer behavioral metrics
#   - Merchant analytics
#   - Rejoins
#   - Heavy expressions
#   - Final aggregation
#
#****************************************************************************


import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

import cml.data_v1 as cmldata



class BankingETLv3:


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

        # -----------------------------------------------------------------
        # NOTE: Update this path to match where the RAPIDS jar actually
        # resolves to on disk. Find it with:
        #   find / -name "rapids-4-spark*26.02.0*.jar" 2>/dev/null
        # -----------------------------------------------------------------
        RAPIDS_JAR_PATH = "/home/cdsw/.ivy2/jars/com.nvidia_rapids-4-spark_2.12-26.02.0.jar"

        spark = (

            SparkSession.builder

            .appName("Spark-Rapids-ETL-v3")

            # ------------------------------------------------------------------
            # Resources
            # ------------------------------------------------------------------
            .config("spark.dynamicAllocation.enabled", "false")
            .config("spark.executor.instances", "12")
            .config("spark.executor.cores", "4")
            .config("spark.executor.memory", "16g")
            .config("spark.executor.memoryOverhead", "3g")
            .config("spark.driver.memory", "10g")

            # ------------------------------------------------------------------
            # GPU Configuration
            # ------------------------------------------------------------------
            .config("spark.executor.resource.gpu.amount", "1")
            .config("spark.task.resource.gpu.amount", 0.250)
            .config(
                "spark.executor.resource.gpu.discoveryScript",
                "/home/cdsw/getGpusResources.sh"
            )
            .config(
                "spark.executor.resource.gpu.vendor",
                "nvidia.com"
            )
            .config(
                "spark.shuffle.manager",
                "com.nvidia.spark.rapids.spark351.RapidsShuffleManager"
            )

            # ------------------------------------------------------------------
            # Explicit RAPIDS jar classpath (fixes ClassNotFoundException on
            # RapidsShuffleManager -- spark.jars.packages alone does not
            # reliably reach the executor classpath in time for shuffle
            # manager initialization)
            # ------------------------------------------------------------------
            .config(
                "spark.driver.extraClassPath",
                RAPIDS_JAR_PATH
            )
            .config(
                "spark.executor.extraClassPath",
                RAPIDS_JAR_PATH
            )

            # ------------------------------------------------------------------
            # Spark RAPIDS
            # ------------------------------------------------------------------
            .config(
                "spark.plugins",
                "com.nvidia.spark.SQLPlugin"
            )
            .config(
                "spark.jars.packages",
#                "com.nvidia:rapids-4-spark_2.12:25.08.0"
                "com.nvidia:rapids-4-spark_2.12:26.02.0"
            )
          .config(
              "spark.rapids.shims-provider-override",
              "com.nvidia.spark.rapids.shims.spark351.SparkShimServiceProvider",
            )
            .config(
                "spark.kryo.registrator",
                "com.nvidia.spark.rapids.GpuKryoRegistrator"
            )

            .config(
                "spark.rapids.sql.enabled",
                "true"
            )

            .config(
                "spark.rapids.sql.incompatibleOps.enabled",
                "true"
            )

            .config(
                "spark.rapids.sql.udfCompiler.enabled",
                "true"
            )

            .config(
                "spark.rapids.sql.concurrentGpuTasks",
                "2"
            )

            .config(
                "spark.rapids.memory.pinnedPool.size",
                "2g"
            )

            .config(
                "spark.rapids.sql.explain",
                "ALL"
            )

            .config(
                "spark.rapids.sql.variableFloatAgg.enabled",
                "true"
            )

            .config(
                "spark.rapids.sql.castFloatToString.enabled",
                "true"
            )

            .config(
                "spark.rapids.sql.castStringToFloat.enabled",
                "true"
            )

            .config(
                "spark.rapids.sql.csv.read.float.enabled",
                "true"
            )

            .config(
                "spark.rapids.sql.format.csv.enabled",
                "true"
            )

            .config(
                "spark.rapids.sql.format.csv.read.enabled",
                "true"
            )

#            .config(
#                "spark.rapids.shims-provider-override",
#                "com.nvidia.spark.rapids.shims.spark330.SparkShimServiceProvider"
#            )

            # ------------------------------------------------------------------
            # Spark Optimizations
            # ------------------------------------------------------------------
            .config("spark.sql.adaptive.enabled", "true")
            .config(
                "spark.sql.adaptive.advisoryPartitionSizeInBytes",
                "1g"
            )

            .config(
                "spark.sql.files.maxPartitionBytes",
                "256m"
            )

            .config(
                "spark.locality.wait",
                "0"
            )

            .config(
                "spark.sql.shuffle.partitions",
                "800"
            )

            .config(
                "spark.eventLog.enabled",
                "true"
            )

            # ------------------------------------------------------------------
            # Storage
            # ------------------------------------------------------------------
#            .config(
#                "spark.hadoop.fs.s3a.custom.signers",
#                "RazS3SignerPlugin:org.apache.ranger.raz.hook.s3.RazS3SignerPlugin:org.apache.ranger.raz.hook.s3.RazS3SignerPluginInitializer"
#            )
#
#            .config(
#                "spark.hadoop.fs.s3a.s3.signing-algorithm",
#                "RazS3SignerPlugin"
#            )
#
#            .config(
#                "spark.hadoop.fs.s3a.aws.credentials.provider",
#                "org.apache.ranger.raz.hook.s3.RazCredentialProvider"
#            )

#            .config(
#                "spark.kubernetes.executor.podTemplateFile",
#                "/tmp/spark-executor.json"
#            )

            .config(
                "spark.kerberos.access.hadoopFileSystems",
                self.storage
            )

            .getOrCreate()

        )

        spark.conf.set(
            "spark.sql.autoBroadcastJoinThreshold",
            -1
        )

        return spark



    ############################################################

    def run(self, spark):


        transactions = spark.table(
            f"{self.database}.TRX"
        )


        customers = spark.table(
            f"{self.database}.CUSTOMERS"
        )


        accounts = spark.table(
            f"{self.database}.ACCOUNTS"
        )


        merchants = spark.table(
            f"{self.database}.MERCHANTS"
        )


        branches = spark.table(
            f"{self.database}.BRANCHES"
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
                )

            )

        )



        ########################################################
        # Join aggregates back
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

        )



        ########################################################
        # Final analytical aggregation
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


                F.sum(
                    "customer_fraud_count"
                )
                .alias(
                    "fraud_events"
                )

            )

        )



        ########################################################
        # Final sort
        ########################################################


        result = result.orderBy(

            F.desc(
                "total_amount"
            )

        )


        return result



    ############################################################

    def save(self, df):


        df.write.mode(

            "overwrite"

        ).saveAsTable(

            f"{self.database}.ETL_V3_RESULT"

        )


        print(
            "ETL v3 complete"
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
        "s3a://pdf0714-buk-d7392db3/data"
    )



    job = BankingETLv3(

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
