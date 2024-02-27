import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").appName("test").getOrCreate()
df = spark.read.option("header", "true").csv("data/taxi+_zone_lookup.csv")
df.show()

df.write.parquet("outputs/zones")