#%% Setting Up The Data
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max, min, split
from pyspark.sql.types import IntegerType, StringType

spark_config = (SparkConf().setMaster("local[*]")
                # .set("spark.executor.memory", "2g")
                # .set("spark.memoryFraction", "0.9")
                # .set("spark.executorCores", 1)
                )

spark = SparkSession.builder.config(conf=spark_config).getOrCreate()

raw_user_artist_path = "data/audioscrobbler/user_artist_data.txt"
raw_user_artist_data = spark.read.text(raw_user_artist_path)
raw_user_artist_data.show(5)
raw_artist_data = spark.read.text("data/audioscrobbler/artist_data.txt")
raw_artist_data.show(5, truncate=False)
raw_artist_alias = spark.read.text("data/audioscrobbler/artist_alias.txt")
raw_artist_alias.show(5)


# split lines by space characters and parse the values as integers
def parseFile(in_df, col_str, split_col, pattern, idx, cast_type=IntegerType(), limit=-1):
    out_df = in_df.withColumn(col_str, split(split_col, pattern, limit).getItem(idx).cast(cast_type))
    return out_df


user_artist_df = parseFile(raw_user_artist_data, 'user', raw_user_artist_data['value'], ' ', 0)
user_artist_df = parseFile(user_artist_df, 'artist', raw_user_artist_data['value'], ' ', 1)
user_artist_df = parseFile(user_artist_df, 'count', raw_user_artist_data['value'], ' ', 2)
user_artist_df = user_artist_df.drop('value')
user_artist_df.show()

user_artist_df.select([min("user"), max("user"), min("artist"), max("artist")]).show()

# process the artist data
artist_by_id = parseFile(raw_artist_data, 'id', col('value'), '\s+', 0, limit=2)
artist_by_id = parseFile(artist_by_id, 'name', col('value'), '\s+', 1, cast_type=StringType(), limit=2)
artist_by_id = artist_by_id.drop('value')
artist_by_id.show()

# process the artist alias data
artist_alias = parseFile(raw_artist_alias, 'artist', col('value'), '\s+', 0)
artist_alias = parseFile(artist_alias, 'alias', col('value'), '\s+', 1, cast_type=StringType())
artist_alias = artist_alias.drop('value')
artist_alias.show()

artist_by_id.filter(artist_by_id.id.isin(1092764, 1000311)).show()

#%% Building A First Model
from pyspark.sql.functions import broadcast, when
from pyspark.ml.recommendation import ALS

# convert all artist ids to a canonical id if one exists
train_data = user_artist_df.join(broadcast(artist_alias), 'artist', how='left')
train_data.show()

# get the artist's alias if it exists, otherwise get the original artist
train_data = train_data.withColumn('artist', when(col('alias').isNull(), col('artist')).otherwise(col('alias')))
# cast the artist id back to an integer type
train_data = train_data.withColumn('artist', col('artist').cast(IntegerType())).drop('alias')
train_data.show()
# cache and count the number of rows
train_data.cache()
train_data.count()

# build an Alternating Least Squares (ALS) model
model = ALS(seed=0, maxIter=5, implicitPrefs=True, userCol='user', itemCol='artist', ratingCol='count')
model.fit(train_data)
