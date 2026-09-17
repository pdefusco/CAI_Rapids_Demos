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


class MerchantDimension:

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

    def generateMerchants(self, spark, rows=500000):
        r = F.rand(seed=301)

        merchants = (
            spark.range(rows)
            .withColumnRenamed("id", "merchant_id")
            .withColumn(
                "state",
                F.expr("""
                CASE
                    WHEN merchant_id % 100 < 35 THEN 'CA'
                    WHEN merchant_id % 100 < 50 THEN 'NY'
                    WHEN merchant_id % 100 < 60 THEN 'TX'
                    WHEN merchant_id % 100 < 68 THEN 'FL'
                    WHEN merchant_id % 100 < 75 THEN 'WA'
                    WHEN merchant_id % 100 < 81 THEN 'IL'
                    WHEN merchant_id % 100 < 86 THEN 'AZ'
                    WHEN merchant_id % 100 < 90 THEN 'GA'
                    WHEN merchant_id % 100 < 93 THEN 'MA'
                    WHEN merchant_id % 100 < 95 THEN 'NC'
                    WHEN merchant_id % 100 < 97 THEN 'OH'
                    WHEN merchant_id % 100 < 98 THEN 'PA'
                    WHEN merchant_id % 100 < 99 THEN 'VA'
                    ELSE 'CO'
                END
                """)
            )
            .withColumn(
                "region",
                F.when(F.col("state").isin("CA", "WA"), "WEST")
                 .when(F.col("state").isin("TX", "AZ", "CO"), "SOUTHWEST")
                 .when(F.col("state").isin("NY", "NJ", "PA", "MA"), "NORTHEAST")
                 .when(F.col("state").isin("IL", "OH"), "MIDWEST")
                 .otherwise("SOUTHEAST")
            )
            .withColumn(
                "merchant_category",
                F.when(r < 0.30, "Retail")
                 .when(r < 0.50, "Restaurant")
                 .when(r < 0.65, "Fuel")
                 .when(r < 0.72, "Travel")
                 .when(r < 0.79, "Healthcare")
                 .when(r < 0.85, "Groceries")
                 .when(r < 0.90, "Utilities")
                 .when(r < 0.94, "Electronics")
                 .when(r < 0.96, "Entertainment")
                 .when(r < 0.98, "Gaming")
                 .when(r < 0.99, "Crypto")
                 .otherwise("Insurance")
            )
            .withColumn(
                "merchant_name",
                F.concat(F.lit("MERCHANT_"), F.col("merchant_id"))
            )
            .withColumn(
                "risk_level",
                F.when(F.col("merchant_category").isin("Gaming", "Crypto"), "HIGH")
                 .when(F.col("merchant_category").isin("Electronics", "Travel"), "MEDIUM")
                 .otherwise("LOW")
            )
            .withColumn("annual_revenue", F.round(F.rand(seed=10) * 100000000, 2))
            .withColumn(
                "merchant_size",
                F.when(F.col("annual_revenue") < 1000000, "SMALL")
                 .when(F.col("annual_revenue") < 10000000, "MEDIUM")
                 .otherwise("LARGE")
            )
            .withColumn("opened_year", 1990 + (F.col("merchant_id") % 35))
            .withColumn(
                "active",
                F.when(F.col("merchant_id") % 100 == 0, False).otherwise(True)
            )
            .repartition(800)
        )
        return merchants

    def saveTable(self, df):
        df.write.mode("overwrite").saveAsTable(f"{self.database}.MERCHANTS_skewed")
        print(f"Merchant table created; rows: {df.count():,}")
        df.show(10, False)


def main():
    username = os.environ["PROJECT_OWNER"]
    database = f"DEMO_{username}"
    generator = MerchantDimension("se-aws-edl", database)
    spark = generator.createSparkConnection()
    merchants = generator.generateMerchants(spark, rows=500000)
    generator.saveTable(merchants)


if __name__ == "__main__":
    main()
