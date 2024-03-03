import pyspark
from pyspark.sql import SparkSession
import os

spark = SparkSession.builder.master("local[*]").appName("test").getOrCreate()

os.getcwd()

data_path = "data/taxi+_zone_lookup.csv"
data_path = r"Documents\GitHub\Courses-Projects-Current\2024 Advanced Analytics with PySpark\data\taxi+_zone_lookup.csv"

df = spark.read.option("header", "true").csv(data_path)
df.show()

df.write.parquet("outputs/zones2")