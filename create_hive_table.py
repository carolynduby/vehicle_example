from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from  pyspark.sql.functions import abs

spark = SparkSession\
    .builder\
    .appName("PythonExample")\
    .getOrCreate()

# Add the data file to hdfs.
!hdfs dfs -put resources/car_details.csv /tmp

# read data from csv format and load into data frame
vehicles = spark.read.format('csv').options(header='true').options(inferSchema='true').load("/tmp/car_details.csv")
vehicles.show()

# create the hive database
spark.sql("CREATE DATABASE IF NOT EXISTS automotive")
spark.sql("show databases").show()

# write the data frame to a hive table
#vehicles.write.mode("overwrite").saveAsTable("automotive.vehicles")

#spark.sql("REFRESH TABLE automotive.vehicles")
