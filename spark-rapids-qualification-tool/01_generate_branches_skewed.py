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


class BranchDimension:

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

    def generateBranches(self, spark, rows=5000):
        branches = (
            spark.range(rows)
            .withColumnRenamed("id", "branch_id")
            .withColumn(
                "branch_name",
                F.concat(F.lit("BRANCH_"), F.col("branch_id"))
            )
            .withColumn(
                "state",
                F.expr("""
                CASE
                    WHEN branch_id % 100 < 35 THEN 'CA'
                    WHEN branch_id % 100 < 50 THEN 'NY'
                    WHEN branch_id % 100 < 60 THEN 'TX'
                    WHEN branch_id % 100 < 68 THEN 'FL'
                    WHEN branch_id % 100 < 75 THEN 'WA'
                    WHEN branch_id % 100 < 81 THEN 'IL'
                    WHEN branch_id % 100 < 86 THEN 'AZ'
                    WHEN branch_id % 100 < 90 THEN 'GA'
                    WHEN branch_id % 100 < 93 THEN 'MA'
                    WHEN branch_id % 100 < 95 THEN 'NC'
                    WHEN branch_id % 100 < 97 THEN 'OH'
                    WHEN branch_id % 100 < 98 THEN 'PA'
                    WHEN branch_id % 100 < 99 THEN 'VA'
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
                "branch_type",
                F.when(F.col("branch_id") % 10 == 0, "Investment")
                 .when(F.col("branch_id") % 5 == 0, "Commercial")
                 .otherwise("Retail")
            )
            .withColumn("employee_count", (F.rand(seed=10) * 450 + 50).cast("int"))
            .withColumn("assets_under_management", F.round(F.rand(seed=20) * 2000000000, 2))
            .withColumn("annual_operating_cost", F.round(F.rand(seed=30) * 100000000, 2))
            .withColumn("opened_year", 1980 + (F.col("branch_id") % 45))
            .withColumn("manager_id", F.col("branch_id") + 1000000)
            .withColumn(
                "status",
                F.when(F.col("branch_id") % 200 == 0, "Renovation").otherwise("Open")
            )
            .repartition(50)
        )
        return branches

    def saveTable(self, df):
        df.write.mode("overwrite").saveAsTable(f"{self.database}.BRANCHES_skewed")
        print(f"Branches table created; rows: {df.count():,}")
        df.show(10, False)


def main():
    username = os.environ["PROJECT_OWNER"]
    database = f"DEMO_{username}"
    generator = BranchDimension("se-aws-edl", database)
    spark = generator.createSparkConnection()
    branches = generator.generateBranches(spark, rows=5000)
    generator.saveTable(branches)


if __name__ == "__main__":
    main()
