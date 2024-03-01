#%% Setting Up The Data
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *

spark = SparkSession.builder.master("local[*]").getOrCreate()
prev = spark.read.csv("data/linkage/block*.csv")
print(prev)
print(prev.show(5))

parsed = spark.read.option("header", "true").option("nullValue", "?").option("inferSchema", "true").\
    csv("data/linkage/block*.csv")
print(parsed.printSchema())

# define the schema ahead-of-time to provide a performance boost
schema = StructType([StructField("id_1", IntegerType(), False),
                     StructField("id_2", StringType(), False),
                     StructField("cmp_fname_c1", DoubleType(), False)])
spark.read.schema(schema).csv("data/linkage/block*.csv")

# it's also possible to define the scheme with data definition language (DDSL) statements
schema = "id_1 INT, id_2 INT, cmp_fname_c1 DOUBLE"

# return the first element
print(parsed.first())

# return the number of objects
print(parsed.count())

# return an array with all the row objects
# print(parsed.collect())

# write the dataframe to persistent storage
# parsed.write.format("parquet").save("outputs/blocks")

#%% Analysing Data With the DataFrame API
from pyspark.sql.functions import avg, col, expr, stddev

print(parsed.printSchema())
print(parsed.show(5))
print(parsed.count())

# cache the dataset for performance benefits
parsed.cache()

# return the relative fraction of matches vs non-matches
parsed.groupby("is_match").count().orderBy(col("count").desc()).show()

# calculate the mean and standard deviation of a variable
parsed.agg(avg("cmp_sex"), stddev("cmp_sex")).show()

# convert the dataframe to a (temporary) database table
parsed.createOrReplaceTempView("linkage")

spark.sql("""
    SELECT is_match, COUNT(*) cnt
    FROM linkage
    GROUP BY is_match
    ORDER BY cnt DESC
""").show()

# use HiveQL in queries
# hive_spark = SparkSession.builder.master("local[4]").enableHiveSupport().getOrCreate()

#%% Fast Summary Statistics For DataFrames
# generate summary statistics
summary = parsed.describe()
summary.show()

# view a subset of the table
summary.select("summary", "cmp_fname_c1", "cmp_fname_c2").show()

# compute summary statistics partitioned by the target variable
matches = parsed.where("is_match = true")
match_summary = matches.describe()
match_summary.show()

misses = parsed.filter(col("is_match") == False)
miss_summary = misses.describe()
miss_summary.show()

#%% Pivoting And Reshaping DataFrames
# convert into a pandas dataframe
summary_p = summary.toPandas()
print(summary_p.head())
print(summary_p.shape)

# transpose the rows and columns
summary_p = summary_p.set_index("summary").transpose().reset_index()
summary_p = summary_p.rename(columns={"index": "field"})
summary_p = summary_p.rename_axis(None, axis=1)
print(summary_p.shape)

# convert it back to a Spark DataFrame
summary_sdf = spark.createDataFrame(summary_p)
summary_sdf.show()
summary_sdf.printSchema()

# convert the values from strings to numbers
for col in summary_sdf.columns:
    if col == "field":
        continue
    summary_sdf = summary_sdf.withColumn(col, summary_sdf[col].cast(DoubleType()))
summary_sdf.printSchema()


# generate a function to perform the above logic
def pivot_summary(desc):
    desc_p = desc.toPandas()

    desc_p = desc_p.set_index("summary").transpose().reset_index()
    desc_p = desc_p.rename(columns={"index": "field"})
    desc_p = desc_p.rename_axis(None, axis=1)

    desc_sdf = spark.createDataFrame(desc_p)

    for col in desc_sdf.columns:
        if col == "field":
            continue
        desc_sdf = desc_sdf.withColumn(col, desc_sdf[col].cast(DoubleType()))

    return desc_sdf


match_summary_sdf = pivot_summary(match_summary)
miss_summary_sdf = pivot_summary(miss_summary)
print(match_summary_sdf.show())
print(miss_summary_sdf.show())

#%% Joining DataFrames And Selecting Features
match_summary_sdf.createOrReplaceTempView("match_desc")
miss_summary_sdf.createOrReplaceTempView("miss_desc")

spark.sql("""
    SELECT a.field, a.count + b.count total, a.mean - b.mean delta
    FROM match_desc a INNER JOIN miss_desc b ON a.field = b.field
    WHERE a.field NOT IN ("id_1", "id_2")
    ORDER BY delta DESC, total DESC
""").show()

#%% Scoring And Model Evaluation
# sum the values from several fields
good_feats = ["cmp_lname_c1", "cmp_plz", "cmp_by", "cmp_bd", "cmp_bm"]
sum_exp = " + ".join(good_feats)
print(sum_exp)

# use this to calculating the score, replacing null values with 0s
scored = parsed.fillna(0, subset=good_feats).withColumn('score', expr(sum_exp)).select('score', 'is_match')
scored.show()


# compute a cross-tab to analyse the compromise between the TPR and TNR
def crossTabs(scored, t):
    return scored.selectExpr(f"score >= {t} as above", "is_match").groupBy("above")\
        .pivot("is_match", ("true", "false")).count()


# a high threshold means almost all the non-matches are filtered out but over 90% of the matches are retained
print(crossTabs(scored, 4.0).show())
# a low threshold means all the known matches are captured, but at a substantial number of FPs
print(crossTabs(scored, 2.0).show())
