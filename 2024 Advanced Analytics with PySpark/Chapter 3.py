#%% Setting Up The Data
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max, min, split
from pyspark.sql.types import IntegerType, StringType

spark_config = (SparkConf().setMaster("local[*]")
                # .set("spark.executor.memory", "5g") # smaller VM
                # .set("spark.driver.memory", "5g")
                .set("spark.executor.memory", "10g") # home
                .set("spark.driver.memory", "10g")
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
als_algo = ALS(seed=0, maxIter=5, implicitPrefs=True, userCol='user', itemCol='artist', ratingCol='count')
als_model = als_algo.fit(train_data)

als_model.userFactors.show(1, truncate=False)

#%% Spot Checking Recommendations
# determine if the recommendations are sensible by examining a user, plays and recommendations for that user
user_id = 2093760
# locate all lines with the specified user id and collect the dataset
existing_artist_ids = train_data.filter(train_data.user == user_id).select("artist").collect()
existing_artist_ids = [i[0] for i in existing_artist_ids]
# filter for those artists
artist_by_id.filter(col('id').isin(existing_artist_ids)).show()

# make some recommendations - not that this approach is suitable for batch scoring but not real-time use cases
user_subset = train_data.select("user").where(col("user") == user_id).distinct()
top_preds = als_model.recommendForUserSubset(user_subset, 5)
top_preds.show()
# convert to a pandas dataframe
top_preds_p = top_preds.toPandas()
print(top_preds_p)
# extract the artist ids from the recommendations column
recommended_artist_ids = [i[0] for i in top_preds_p.recommendations[0]]
# filter the artist_by_id column by those ids to get the artist names
artist_by_id.filter(col('id').isin(recommended_artist_ids)).show()

#%% Computing AUC
import random
from pyspark.sql.functions import coalesce, count, lit, mean
from pyspark.sql.functions import sum as _sum

all_data = user_artist_df.join(broadcast(artist_alias), 'artist', how='left')\
    .withColumn('artist', when(col('alias').isNull(), col('artist')).otherwise(col('alias')))\
    .withColumn('artist', col('artist').cast(IntegerType())).drop('alias')
all_data.show(5)

# split the data into a training set and cross-validation set
train_data, cv_data = all_data.randomSplit([0.9, 0.1], seed=54321)
train_data.cache()
cv_data.cache()

# select the distinct artist ids
artist_id_set = all_data.select("artist").distinct()
print(artist_id_set.count())
# and convert it to a list
artist_id_set = [i[0] for i in artist_id_set.collect()]
# and broadcast it
# b_artist_id_set = broadcast(artist_id_set)
# artist_id_set.show(5)

# fit and train an ALS model
als_algo = ALS(seed=0, maxIter=5, implicitPrefs=True, userCol='user', itemCol='artist', ratingCol='count')
als_model = als_algo.fit(train_data)


# create a function calculating the area under the curve
def area_under_curve(positive_data, artist_ids, predict_function):
    pos_preds = predict_function(positive_data).select("user", "artist", "prediction")\
        .withColumnRenamed("prediction", "pos_pred")

    def gen_negative_data(user_artist_tuples):
        user_neg_artists = []
        for user, pos_artist_ids in user_artist_tuples:
            pos_artist_id_set = set(pos_artist_ids)
            neg_artists = set()
            while len(neg_artists) < len(pos_artist_id_set):
                artist_id = artist_ids[random.randint(0, len(artist_ids) - 1)]
                if artist_id not in pos_artist_id_set:
                    neg_artists.add(artist_id)
            user_neg_artists.extend([(user, artist_id) for artist_id in neg_artists])

        return user_neg_artists

    user_artist_rdd = positive_data.select("user", "artist").rdd.groupByKey().mapValues(list).collect()
    neg_data = spark.createDataFrame(gen_negative_data(user_artist_rdd), schema=["user", "artist"])\
        .withColumn("count", lit(1))

    neg_preds = predict_function(neg_data).select("user", "artist", "prediction")\
        .withColumnRenamed("prediction", "neg_pred")
    joined_preds = pos_preds.join(neg_preds, "user").select("user", "pos_pred", "neg_pred").cache()

    all_counts = joined_preds.groupBy("user").agg(count(lit(1)).alias("total")).select("user", "total")
    adj_counts = joined_preds.filter(col("pos_pred") > col("neg_pred")).groupBy("user") \
        .agg(count("user").alias("adjusted")).select("user", "adjusted")

    mean_auc = all_counts.join(adj_counts, ["user"], "left_outer") \
        .select(col("user"), (coalesce(col("adjusted"), lit(0)) / col("total")).alias("auc")) \
        .agg(mean("auc")).collect()[0][0]

    joined_preds.unpersist()

    return mean_auc


mean_auc = area_under_curve(cv_data, artist_id_set, als_model.transform)
print(mean_auc)


# benchmark the result against the auc if the globally most-played artists are recommended to every user
def predict_most_played(in_data):
    play_counts = in_data.groupBy("artist").agg(_sum("count").alias("prediction")).select("artist", "prediction")
    joined_data = in_data.join(play_counts, "artist", "left_outer")
    return joined_data


benchmark_auc = area_under_curve(cv_data, artist_id_set, predict_most_played)
print(benchmark_auc)

# https://github.com/sryza/aas/tree/pyspark-edition
#%% Hyperparameter Tuning
from itertools import product
from pprint import pprint

ranks = [5, 30]
reg_params = [4., 0.0001]
alphas = [1., 40.]
param_grid = list(product(*[ranks, reg_params, alphas]))

results = []
for params in param_grid[:1]:
    rank = params[0]
    reg_param = params[1]
    alpha = params[2]

    als_algo = ALS().setSeed(10).setImplicitPrefs(True).setRank(rank).setRegParam(reg_param).setAlpha(alpha)\
        .setMaxIter(20).setUserCol("user").setItemCol("artist").setRatingCol("count")
    als_model = als_algo.fit(train_data)

    auc = area_under_curve(cv_data, artist_id_set, als_model.transform)

    # free up model resources
    als_model.userFactors.unpersist()
    als_model.itemFactors.unpersist()

    results.append((auc, (rank, reg_param, alpha)))

# sort by descending AUC
results.sort(key=lambda x: x[0], reverse=True)
pprint(results)

#%% Making Recommendations
# select 100 users from the dataset
selected_users = all_data.select("user").distinct().limit(100)


def make_recommendations(model, user_id, num_recs):
    user_subset = train_data.select("user").where(col("user") == user_id).distinct()
    recs = model.recommendForUserSubset(user_subset, num_recs)
    return recs






als_algo = ALS(seed=0, maxIter=5, implicitPrefs=True, userCol='user', itemCol='artist', ratingCol='count')
