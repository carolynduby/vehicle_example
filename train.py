from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from  pyspark.sql.functions import abs

spark = SparkSession\
    .builder\
    .appName("PythonExample")\
    .getOrCreate()

# read the vehicle data
hive_df = spark.sql("select * from automotive.vehicles")
hive_df.show()

features = hive_df.select("year", "km_driven")


# extract features
assembler = VectorAssembler(
    inputCols=features.columns,
    outputCol="features")

output = assembler.transform(hive_df).select('features','selling_price')
output.show()

# split
train,test = output.randomSplit([0.75, 0.25])

# train
lin_reg = LinearRegression(featuresCol = 'features', labelCol='selling_price')
linear_model = lin_reg.fit(train)
print("Coefficients: " + str(linear_model.coefficients))
print("\nIntercept: " + str(linear_model.intercept))

trainSummary = linear_model.summary

print("RMSE: %f" % trainSummary.rootMeanSquaredError)
print("\nr2: %f" % trainSummary.r2)

#evaluate

predictions = linear_model.transform(test)
x =((predictions['selling_price']-predictions['prediction'])/predictions['selling_price'])*100
predictions = predictions.withColumn('Accuracy',abs(x))
predictions.select("prediction","selling_price","Accuracy","features").show()