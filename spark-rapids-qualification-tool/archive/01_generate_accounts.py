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

from pyspark.sql import functions as F
import cml.data_v1 as cmldata


class AccountDimension:

    def __init__(self, connection_name, database):

        self.connection_name = connection_name
        self.database = database

    ############################################################

    def createSparkConnection(self):

        conn = cmldata.get_connection(self.connection_name)

        spark = conn.get_spark_session()

        spark.conf.set("spark.sql.shuffle.partitions", "800")

        from pyspark import SparkContext
        SparkContext.setSystemProperty('spark.executor.cores', '5')
        SparkContext.setSystemProperty('spark.executor.memory', '10g')

        return spark

    ############################################################

    def generateAccounts(self, spark):

        customers = spark.table(f"{self.database}.CUSTOMERS")

        #
        # Roughly 50% of customers receive a second account.
        #
        base_accounts = customers.select(
            "customer_id"
        ).withColumn(
            "account_sequence",
            F.lit(1)
        )

        extra_accounts = (
            customers
            .filter((F.col("customer_id") % 2) == 0)
            .select("customer_id")
            .withColumn(
                "account_sequence",
                F.lit(2)
            )
        )

        accounts = base_accounts.union(extra_accounts)

        ########################################################

        accounts = (

            accounts

            .withColumn(
                "account_id",
                F.monotonically_increasing_id()
            )

            .withColumn(
                "account_type",
                F.expr("""

                    CASE account_sequence

                        WHEN 1 THEN 'Checking'
                        ELSE 'Savings'

                    END

                """)
            )

            .withColumn(
                "currency",

                F.lit("USD")
            )

            .withColumn(
                "account_status",

                F.expr("""

                CASE

                    WHEN customer_id % 100 = 0
                    THEN 'Dormant'

                    ELSE 'Active'

                END

                """)
            )

            .withColumn(
                "opened_year",

                2000 + (F.col("customer_id") % 25)
            )

            .withColumn(
                "branch_id",

                (F.col("customer_id") % 5000) + 1
            )

            ####################################################
            # Financial Information
            ####################################################

            .withColumn(
                "current_balance",

                F.round(

                    F.rand(seed=10) * 200000,

                    2

                )
            )

            .withColumn(
                "credit_limit",

                F.round(

                    F.rand(seed=20) * 50000,

                    2

                )
            )

            .withColumn(
                "interest_rate",

                F.round(

                    F.rand(seed=30) * 8 + 1,

                    2

                )
            )

            ####################################################

            .drop(
                "account_sequence"
            )

            .repartition(
                800
            )

        )

        return accounts

    ############################################################

    def saveTable(self, df):

        df.write.mode("overwrite").saveAsTable(
            f"{self.database}.ACCOUNTS"
        )

        print()

        print("Accounts table created")

        print()

        print("Rows:", df.count())

        print()

        df.show(10, False)


############################################################

def main():

    USERNAME = os.environ["PROJECT_OWNER"]

    DATABASE = f"DEMO_{USERNAME}"

    CONNECTION_NAME = "se-aws-edl"


    generator = AccountDimension(

        CONNECTION_NAME,

        DATABASE

    )

    spark = generator.createSparkConnection()

    accounts = generator.generateAccounts(

        spark

    )

    generator.saveTable(accounts)


############################################################

if __name__ == "__main__":

    main()
