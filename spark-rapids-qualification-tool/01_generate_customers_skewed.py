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


class CustomerDimension:

    def __init__(self, connection_name, database):
        self.connection_name = connection_name
        self.database = database

    def createSparkConnection(self):
        conn = cmldata.get_connection(self.connection_name)
        spark = conn.get_spark_session()
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {self.database}")
        spark.conf.set("spark.sql.shuffle.partitions", "800")
        from pyspark import SparkContext
        SparkContext.setSystemProperty("spark.executor.cores", "5")
        SparkContext.setSystemProperty("spark.executor.memory", "10g")
        return spark

    def generateCustomers(self, spark, rows=2000000):
        r_segment = F.rand(seed=101)
        r_income = F.rand(seed=102)

        customers = (
            spark.range(rows)
            .withColumnRenamed("id", "customer_id")
            .withColumn("age", (F.rand(seed=1) * 60 + 18).cast("int"))
            .withColumn("credit_score", (F.rand(seed=2) * 550 + 300).cast("int"))
            .withColumn(
                "state",
                F.expr("""
                CASE
                    WHEN customer_id % 100 < 35 THEN 'CA'
                    WHEN customer_id % 100 < 50 THEN 'NY'
                    WHEN customer_id % 100 < 60 THEN 'TX'
                    WHEN customer_id % 100 < 68 THEN 'FL'
                    WHEN customer_id % 100 < 75 THEN 'WA'
                    WHEN customer_id % 100 < 81 THEN 'IL'
                    WHEN customer_id % 100 < 86 THEN 'AZ'
                    WHEN customer_id % 100 < 90 THEN 'GA'
                    WHEN customer_id % 100 < 93 THEN 'MA'
                    WHEN customer_id % 100 < 95 THEN 'NC'
                    WHEN customer_id % 100 < 97 THEN 'OH'
                    WHEN customer_id % 100 < 98 THEN 'PA'
                    WHEN customer_id % 100 < 99 THEN 'VA'
                    ELSE 'CO'
                END
                """)
            )
            .withColumn(
                "city",
                F.concat(F.lit("CITY_"), (F.col("customer_id") % 500).cast("string"))
            )
            .withColumn(
                "income_band",
                F.when(r_income < 0.45, "LOW")
                 .when(r_income < 0.70, "LOW_MID")
                 .when(r_income < 0.85, "MID")
                 .when(r_income < 0.95, "HIGH")
                 .otherwise("ULTRA")
            )
            .withColumn(
                "estimated_income",
                (F.rand(seed=5) * 250000 + 30000).cast("double")
            )
            .withColumn(
                "customer_segment",
                F.when(r_segment < 0.50, "Retail")
                 .when(r_segment < 0.75, "Gold")
                 .when(r_segment < 0.90, "Premier")
                 .otherwise("Private")
            )
            .withColumn("tenure_years", (F.rand(seed=7) * 30).cast("int"))
            .withColumn(
                "risk_rating",
                F.when(F.col("credit_score") > 760, "LOW")
                 .when(F.col("credit_score") > 680, "MEDIUM")
                 .otherwise("HIGH")
            )
            .repartition(800)
        )
        return customers

    def saveTable(self, df):
        df.write.mode("overwrite").saveAsTable(f"{self.database}.CUSTOMERS_skewed")
        print(f"Customer table created; rows: {df.count():,}")
        df.show(10, False)


def main():
    username = os.environ["PROJECT_OWNER"]
    database = f"DEMO_{username}"
    generator = CustomerDimension("se-aws-edl", database)
    spark = generator.createSparkConnection()
    customers = generator.generateCustomers(spark, rows=2000000)
    generator.saveTable(customers)


if __name__ == "__main__":
    main()
