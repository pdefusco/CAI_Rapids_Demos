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


class TransactionFactGenerator:

    def __init__(self, connection_name, database):

        self.connection_name = connection_name
        self.database = database


    ############################################################

    def createSparkConnection(self):

        conn = cmldata.get_connection(
            self.connection_name
        )

        spark = conn.get_spark_session()

#        spark.conf.set(
#            "spark.sql.shuffle.partitions",
#            "800"
#        )
        
        from pyspark import SparkContext
        SparkContext.setSystemProperty('spark.executor.cores', '5')
        SparkContext.setSystemProperty('spark.executor.memory', '20g')
        SparkContext.setSystemProperty('spark.driver.cores', '5')
        SparkContext.setSystemProperty('spark.driver.memory', '20g')
        
        return spark


    ############################################################

    def generateTransactions(
            self,
            spark,
            rows=25000000):

        customers = 20000000
        accounts = 30000000
        merchants = 500000
        branches = 5000


        transactions = (

            spark.range(rows)

            .withColumnRenamed(
                "id",
                "transaction_id"
            )


            ####################################################
            # Foreign Keys
            ####################################################

            .withColumn(

                "customer_id",

                (
                    F.rand(seed=1) * customers

                ).cast("long")

            )


            .withColumn(

                "account_id",

                (
                    F.rand(seed=2) * accounts

                ).cast("long")

            )


            .withColumn(

                "merchant_id",

                (
                    F.rand(seed=3) * merchants

                ).cast("long")

            )


            .withColumn(

                "branch_id",

                (
                    F.rand(seed=4) * branches

                ).cast("long")

            )


            ####################################################
            # Transaction Date
            ####################################################

            .withColumn(

                "transaction_date",

                F.date_add(

                    F.lit("2024-01-01"),

                    (
                        F.rand(seed=5) * 730

                    ).cast("int")

                )

            )


            .withColumn(

                "transaction_timestamp",

                F.expr("""

                timestampadd(

                    SECOND,

                    cast(rand()*86400 as int),

                    transaction_date

                )

                """)

            )


            ####################################################
            # Merchant Category
            ####################################################

            .withColumn(

                "merchant_category",

                F.expr("""

                CASE

                WHEN merchant_id % 12 = 0
                    THEN 'Retail'

                WHEN merchant_id % 12 = 1
                    THEN 'Restaurant'

                WHEN merchant_id % 12 = 2
                    THEN 'Fuel'

                WHEN merchant_id % 12 = 3
                    THEN 'Travel'

                WHEN merchant_id % 12 = 4
                    THEN 'Healthcare'

                WHEN merchant_id % 12 = 5
                    THEN 'Groceries'

                WHEN merchant_id % 12 = 6
                    THEN 'Utilities'

                WHEN merchant_id % 12 = 7
                    THEN 'Electronics'

                WHEN merchant_id % 12 = 8
                    THEN 'Entertainment'

                WHEN merchant_id % 12 = 9
                    THEN 'Gaming'

                WHEN merchant_id % 12 = 10
                    THEN 'Crypto'

                ELSE 'Insurance'

                END

                """)

            )


            ####################################################
            # Amount
            ####################################################

            .withColumn(

                "transaction_amount",

                F.when(

                    F.col("merchant_category")
                    == "Groceries",

                    F.rand(seed=10)*150

                )

                .when(

                    F.col("merchant_category")
                    == "Travel",

                    F.rand(seed=11)*3000

                )

                .when(

                    F.col("merchant_category")
                    == "Electronics",

                    F.rand(seed=12)*2500

                )

                .when(

                    F.col("merchant_category")
                    == "Crypto",

                    F.rand(seed=13)*10000

                )

                .otherwise(

                    F.rand(seed=14)*500

                )

            )


            ####################################################
            # Payment Information
            ####################################################

            .withColumn(

                "payment_channel",

                F.expr("""

                CASE

                    WHEN transaction_id % 4 = 0
                        THEN 'ONLINE'

                    WHEN transaction_id % 4 = 1
                        THEN 'MOBILE'

                    WHEN transaction_id % 4 = 2
                        THEN 'POS'

                    ELSE 'ATM'

                END

                """)

            )


            .withColumn(

                "payment_type",

                F.expr("""

                CASE

                    WHEN transaction_id % 3 = 0
                        THEN 'CREDIT_CARD'

                    WHEN transaction_id % 3 = 1
                        THEN 'DEBIT_CARD'

                    ELSE 'TRANSFER'

                END

                """)

            )


            .withColumn(

                "device_type",

                F.expr("""

                CASE

                    WHEN transaction_id % 5 = 0
                        THEN 'MOBILE'

                    WHEN transaction_id % 5 = 1
                        THEN 'TABLET'

                    ELSE 'DESKTOP'

                END

                """)

            )


            ####################################################
            # Fraud Logic
            ####################################################

            .withColumn(

                "fraud_probability",

                F.when(

                    F.col("merchant_category")
                    == "Crypto",

                    0.10

                )

                .when(

                    F.col("merchant_category")
                    == "Gaming",

                    0.05

                )

                .when(

                    F.col("merchant_category")
                    == "Electronics",

                    0.02

                )

                .otherwise(

                    0.005

                )

            )


            .withColumn(

                "fraud_flag",

                F.when(

                    F.rand(seed=20)
                    < F.col("fraud_probability"),

                    1

                )

                .otherwise(0)

            )


            ####################################################
            # Location
            ####################################################

            .withColumn(

                "latitude",

                F.rand(seed=30)*180-90

            )

            .withColumn(

                "longitude",

                F.rand(seed=31)*360-180

            )


            .drop(

                "fraud_probability"

            )


#            .repartition(800)

        )


        return transactions



    ############################################################

    def saveTable(self, df):

        df.write.mode(
            "overwrite"
        ).saveAsTable(

            f"{self.database}.TRS_v13"

        )


        print()

        print("Transactions table created")

        print(
            f"Rows: {df.count():,}"
        )

        df.show(10, False)



############################################################

def main():

    USERNAME = os.environ["PROJECT_OWNER"]

    DATABASE = f"DEMO_{USERNAME}"

    CONNECTION_NAME = "se-aws-edl"



    generator = TransactionFactGenerator(

        CONNECTION_NAME,

        DATABASE

    )


    spark = generator.createSparkConnection()


    transactions = generator.generateTransactions(

        spark,

        rows=25000000

    )


    generator.saveTable(

        transactions

    )



############################################################

if __name__ == "__main__":

    main()
