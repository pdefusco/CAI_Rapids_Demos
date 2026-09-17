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

import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import cml.data_v1 as cmldata


class CustomerDimension:

    def __init__(self, connection_name, database):

        self.connection_name = connection_name
        self.database = database

    ##############################################################

    def createSparkConnection(self):

        conn = cmldata.get_connection(self.connection_name)

        spark = conn.get_spark_session()

        spark.sql(
            f"CREATE DATABASE IF NOT EXISTS {self.database}"
        )

        spark.conf.set(
            "spark.sql.shuffle.partitions",
            "800"
        )

        from pyspark import SparkContext
        SparkContext.setSystemProperty('spark.executor.cores', '5')
        SparkContext.setSystemProperty('spark.executor.memory', '10g')

        return spark

    ##############################################################

    def generateCustomers(
            self,
            spark,
            rows=5000000):

        customers = (

            spark.range(rows)

            .withColumnRenamed("id", "customer_id")

            #####################################################
            # Demographics
            #####################################################

            .withColumn(
                "age",
                (F.rand(seed=1) * 60 + 18).cast("int")
            )

            .withColumn(
                "credit_score",
                (F.rand(seed=2) * 550 + 300).cast("int")
            )

            #####################################################
            # Geography
            #####################################################

            .withColumn(
                "state",
                F.expr("""
                CASE (customer_id % 15)

                    WHEN 0 THEN 'CA'
                    WHEN 1 THEN 'NY'
                    WHEN 2 THEN 'TX'
                    WHEN 3 THEN 'FL'
                    WHEN 4 THEN 'WA'
                    WHEN 5 THEN 'IL'
                    WHEN 6 THEN 'AZ'
                    WHEN 7 THEN 'GA'
                    WHEN 8 THEN 'MA'
                    WHEN 9 THEN 'NC'
                    WHEN 10 THEN 'OH'
                    WHEN 11 THEN 'PA'
                    WHEN 12 THEN 'VA'
                    WHEN 13 THEN 'NJ'
                    ELSE 'CO'

                END
                """)
            )

            .withColumn(
                "city",
                F.concat(
                    F.lit("CITY_"),
                    (F.col("customer_id") % 500).cast("string")
                )
            )

            #####################################################
            # Financial
            #####################################################

            .withColumn(
                "income_band",
                F.expr("""
                CASE

                    WHEN customer_id % 5 = 0 THEN 'LOW'
                    WHEN customer_id % 5 = 1 THEN 'LOW_MID'
                    WHEN customer_id % 5 = 2 THEN 'MID'
                    WHEN customer_id % 5 = 3 THEN 'HIGH'
                    ELSE 'ULTRA'

                END
                """)
            )

            .withColumn(
                "estimated_income",

                (
                    F.rand(seed=5) * 250000 + 30000

                ).cast("double")
            )

            #####################################################
            # Customer Metadata
            #####################################################

            .withColumn(
                "customer_segment",
                F.expr("""

                CASE

                    WHEN customer_id % 4 = 0 THEN 'Retail'
                    WHEN customer_id % 4 = 1 THEN 'Gold'
                    WHEN customer_id % 4 = 2 THEN 'Premier'
                    ELSE 'Private'

                END

                """)
            )

            .withColumn(
                "tenure_years",
                (F.rand(seed=7) * 30).cast("int")
            )

            .withColumn(
                "risk_rating",
                F.expr("""

                CASE

                    WHEN credit_score > 760 THEN 'LOW'

                    WHEN credit_score > 680 THEN 'MEDIUM'

                    ELSE 'HIGH'

                END

                """)
            )

            #####################################################
            # Partitioning
            #####################################################

            .repartition(800)

        )

        return customers

    ##############################################################

    def saveTable(
            self,
            df):

        df.write.mode("overwrite").saveAsTable(
            f"{self.database}.CUSTOMERS"
        )

        print()

        print("Customer table created")

        print(f"Rows: {df.count():,}")

        print()

        df.show(10, False)


##############################################################

def main():

    USERNAME = os.environ["PROJECT_OWNER"]

    DATABASE = f"DEMO_{USERNAME}"

    CONNECTION_NAME = "se-aws-edl"

    generator = CustomerDimension(

        CONNECTION_NAME,

        DATABASE

    )

    spark = generator.createSparkConnection()

    customers = generator.generateCustomers(

        spark,

        rows=2000000

    )

    generator.saveTable(customers)


##############################################################

if __name__ == "__main__":

    main()
