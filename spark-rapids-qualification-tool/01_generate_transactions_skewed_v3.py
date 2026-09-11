# ****************************************************************************
# (C) Cloudera, Inc. 2020-2026
# All rights reserved.
#
# Stronger-skew synthetic-data variant for the Spark RAPIDS Qualification Tool demo.
#
# This version preserves the v2 schema, row count, and temporal distribution while
# increasing realistic concentration in foreign-key activity. It does NOT change
# Spark partitioning or ETL topology.
# ****************************************************************************
import os
from pyspark.sql import functions as F
import cml.data_v1 as cmldata

# ============================================================
# Stronger skew controls
# ============================================================
# v2 baseline:
#   customer 25% into 1% of keys
#   account  25% into 1% of keys
#   merchant 35% into 1% of keys
#   branch   50% into 5% of keys
#
# v3 experiment:
#   customer 40% into 0.5% of keys
#   account  40% into 0.5% of keys
#   merchant 50% into 0.5% of keys
#   branch   65% into 2% of keys
#
# This creates stronger hot populations without collapsing activity onto a
# single key or intentionally creating a pathological partitioning scheme.
CUSTOMER_HOTKEY_RATE = 0.40
ACCOUNT_HOTKEY_RATE = 0.40
MERCHANT_HOTKEY_RATE = 0.50
BRANCH_HOTKEY_RATE = 0.65

CUSTOMER_HOTKEY_FRACTION = 0.005
ACCOUNT_HOTKEY_FRACTION = 0.005
MERCHANT_HOTKEY_FRACTION = 0.005
BRANCH_HOTKEY_FRACTION = 0.02


class TransactionFactGenerator:
    def __init__(self, connection_name, database):
        self.connection_name = connection_name
        self.database = database

    def createSparkConnection(self):
        from pyspark import SparkContext
        SparkContext.setSystemProperty("spark.executor.cores", "5")
        SparkContext.setSystemProperty("spark.executor.memory", "20g")
        SparkContext.setSystemProperty("spark.driver.cores", "5")
        SparkContext.setSystemProperty("spark.driver.memory", "20g")

        conn = cmldata.get_connection(self.connection_name)
        spark = conn.get_spark_session()

        return spark

    def skewed_key(self, rate, fraction, cardinality, seed_hot, seed_cold):
        hot = F.rand(seed=seed_hot) < F.lit(rate)
        hot_id = (
            F.rand(seed=seed_hot + 1000)
            * F.lit(cardinality * fraction)
        ).cast("long")
        cold_id = (
            F.rand(seed=seed_cold) * F.lit(cardinality)
        ).cast("long")
        return F.when(hot, hot_id).otherwise(cold_id)

    def generateTransactions(self, spark, rows=2500000000):
        customers = 20000000
        accounts = 30000000
        merchants = 500000
        branches = 5000

        transactions = (
            spark.range(rows)
            .withColumnRenamed("id", "transaction_id")
            # Controlled stronger foreign-key skew.
            .withColumn(
                "customer_id",
                self.skewed_key(
                    CUSTOMER_HOTKEY_RATE, CUSTOMER_HOTKEY_FRACTION,
                    customers, 1, 2
                )
            )
            .withColumn(
                "account_id",
                self.skewed_key(
                    ACCOUNT_HOTKEY_RATE, ACCOUNT_HOTKEY_FRACTION,
                    accounts, 3, 4
                )
            )
            .withColumn(
                "merchant_id",
                self.skewed_key(
                    MERCHANT_HOTKEY_RATE, MERCHANT_HOTKEY_FRACTION,
                    merchants, 5, 6
                )
            )
            .withColumn(
                "branch_id",
                self.skewed_key(
                    BRANCH_HOTKEY_RATE, BRANCH_HOTKEY_FRACTION,
                    branches, 7, 8
                )
            )
            # Keep temporal distribution uniform.
            .withColumn(
                "transaction_date",
                F.date_add(
                    F.lit("2024-01-01"),
                    (F.rand(seed=9) * 730).cast("int")
                )
            )
            .withColumn(
                "transaction_timestamp",
                F.expr("""
                timestampadd(
                    SECOND,
                    cast(rand(10)*86400 as int),
                    transaction_date
                )
                """)
            )
            .withColumn(
                "merchant_category",
                F.expr("""
                CASE
                    WHEN merchant_id % 12 = 0 THEN 'Retail'
                    WHEN merchant_id % 12 = 1 THEN 'Restaurant'
                    WHEN merchant_id % 12 = 2 THEN 'Fuel'
                    WHEN merchant_id % 12 = 3 THEN 'Travel'
                    WHEN merchant_id % 12 = 4 THEN 'Healthcare'
                    WHEN merchant_id % 12 = 5 THEN 'Groceries'
                    WHEN merchant_id % 12 = 6 THEN 'Utilities'
                    WHEN merchant_id % 12 = 7 THEN 'Electronics'
                    WHEN merchant_id % 12 = 8 THEN 'Entertainment'
                    WHEN merchant_id % 12 = 9 THEN 'Gaming'
                    WHEN merchant_id % 12 = 10 THEN 'Crypto'
                    ELSE 'Insurance'
                END
                """)
            )
            .withColumn(
                "transaction_amount",
                F.when(
                    F.col("merchant_category") == "Groceries",
                    F.rand(seed=11) * 150
                )
                .when(
                    F.col("merchant_category") == "Travel",
                    F.rand(seed=12) * 3000
                )
                .when(
                    F.col("merchant_category") == "Electronics",
                    F.rand(seed=13) * 2500
                )
                .when(
                    F.col("merchant_category") == "Crypto",
                    F.rand(seed=14) * 10000
                )
                .otherwise(F.rand(seed=15) * 500)
            )
            .withColumn(
                "payment_channel",
                F.expr("""
                CASE
                    WHEN transaction_id % 4 = 0 THEN 'ONLINE'
                    WHEN transaction_id % 4 = 1 THEN 'MOBILE'
                    WHEN transaction_id % 4 = 2 THEN 'POS'
                    ELSE 'ATM'
                END
                """)
            )
            .withColumn(
                "payment_type",
                F.expr("""
                CASE
                    WHEN transaction_id % 3 = 0 THEN 'CREDIT_CARD'
                    WHEN transaction_id % 3 = 1 THEN 'DEBIT_CARD'
                    ELSE 'TRANSFER'
                END
                """)
            )
            .withColumn(
                "device_type",
                F.expr("""
                CASE
                    WHEN transaction_id % 5 = 0 THEN 'MOBILE'
                    WHEN transaction_id % 5 = 1 THEN 'TABLET'
                    ELSE 'DESKTOP'
                END
                """)
            )
            .withColumn(
                "fraud_probability",
                F.when(F.col("merchant_category") == "Crypto", 0.10)
                 .when(F.col("merchant_category") == "Gaming", 0.05)
                 .when(F.col("merchant_category") == "Electronics", 0.02)
                 .otherwise(0.005)
            )
            .withColumn(
                "fraud_flag",
                F.when(
                    F.rand(seed=20) < F.col("fraud_probability"),
                    1
                ).otherwise(0)
            )
            .withColumn("latitude", F.rand(seed=30) * 180 - 90)
            .withColumn("longitude", F.rand(seed=31) * 360 - 180)
            .drop("fraud_probability")
        )
        return transactions

    def saveTable(self, df):
        # Preserve TRS_v14 as the v10 control dataset.
        df.write.mode("overwrite").saveAsTable(
            f"{self.database}.TRS_v25"
        )
        #print(f"Transactions table created; rows: {df.count():,}")
        #df.show(10, False)


def main():
    username = os.environ["PROJECT_OWNER"]
    database = f"DEMO_{username}"
    generator = TransactionFactGenerator("se-aws-edl", database)
    spark = generator.createSparkConnection()
    print("https://spark-"+os.environ["CDSW_ENGINE_ID"]+"."+os.environ["CDSW_DOMAIN"])
    transactions = generator.generateTransactions(spark, rows=2500000000)
    generator.saveTable(transactions)


if __name__ == "__main__":
    main()
