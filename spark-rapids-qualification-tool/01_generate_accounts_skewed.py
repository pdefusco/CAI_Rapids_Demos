# ****************************************************************************
# (C) Cloudera, Inc. 2020-2026
# All rights reserved.
#
# Skewed synthetic-data variant for the Spark RAPIDS Qualification Tool demo.
# Based on the original generators in:
# https://github.com/pdefusco/CAI_Rapids_Demos/tree/main/spark-rapids-qualification-tool
#
# The goal is to preserve row counts and schemas while introducing controlled,
# reproducible skew in join keys and selected categorical attributes.
# ****************************************************************************

import os
from pyspark.sql import functions as F
import cml.data_v1 as cmldata


class AccountDimension:

    def __init__(self, connection_name, database):
        self.connection_name = connection_name
        self.database = database

    def createSparkConnection(self):
        conn = cmldata.get_connection(self.connection_name)
        spark = conn.get_spark_session()
        spark.conf.set("spark.sql.shuffle.partitions", "800")
        from pyspark import SparkContext
        SparkContext.setSystemProperty("spark.executor.cores", "5")
        SparkContext.setSystemProperty("spark.executor.memory", "10g")
        return spark

    def generateAccounts(self, spark):
        customers = spark.table(f"{self.database}.CUSTOMERS_skewed")

        base_accounts = customers.select("customer_id").withColumn(
            "account_sequence", F.lit(1)
        )
        extra_accounts = (
            customers.filter((F.col("customer_id") % 2) == 0)
            .select("customer_id")
            .withColumn("account_sequence", F.lit(2))
        )
        accounts = base_accounts.union(extra_accounts)

        hot_branch = (
            F.when(
                F.rand(seed=201) < 0.70,
                (F.rand(seed=202) * 250).cast("long") + 1
            )
            .otherwise(
                (F.rand(seed=203) * 5000).cast("long") + 1
            )
        )

        accounts = (
            accounts
            .withColumn("account_id", F.monotonically_increasing_id())
            .withColumn(
                "account_type",
                F.when(F.col("account_sequence") == 1, "Checking")
                 .otherwise("Savings")
            )
            .withColumn("currency", F.lit("USD"))
            .withColumn(
                "account_status",
                F.when(F.col("customer_id") % 100 == 0, "Dormant")
                 .otherwise("Active")
            )
            .withColumn("opened_year", 2000 + (F.col("customer_id") % 25))
            .withColumn("branch_id", hot_branch)
            .withColumn("current_balance", F.round(F.rand(seed=10) * 200000, 2))
            .withColumn("credit_limit", F.round(F.rand(seed=20) * 50000, 2))
            .withColumn("interest_rate", F.round(F.rand(seed=30) * 8 + 1, 2))
            .drop("account_sequence")
            .repartition(800)
        )
        return accounts

    def saveTable(self, df):
        df.write.mode("overwrite").saveAsTable(f"{self.database}.ACCOUNTS_skewed")
        print(f"Accounts table created; rows: {df.count():,}")
        df.show(10, False)


def main():
    username = os.environ["PROJECT_OWNER"]
    database = f"DEMO_{username}"
    generator = AccountDimension("se-aws-edl", database)
    spark = generator.createSparkConnection()
    accounts = generator.generateAccounts(spark)
    generator.saveTable(accounts)


if __name__ == "__main__":
    main()
