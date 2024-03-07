#%% Preparing The Data
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col

spark_config = (SparkConf().setMaster("local[*]")
                # .set("spark.executor.memory", "5g") # smaller VM
                # .set("spark.driver.memory", "5g")
                .set("spark.executor.memory", "10g") # home
                .set("spark.driver.memory", "10g")
                )

spark = SparkSession.builder.config(conf=spark_config).getOrCreate()

# import the data without column names and print the schema
data_no_header = spark.read.option("inferSchema", True).option("header", False).csv("data/covtype/covtype.data")
data_no_header.printSchema()

# add column names based on the covtype.info file
colnames = ["elevation", "aspect", "slope", "horiz_dist_hydrology", "vert_dist_hydrology", "horiz_dist_roadways",
            "hillshade_9am", "hillshade_12pm", "hillshade_3pm", "horiz_dist_fire_points"] + \
           [f"wilderness_area_{i}" for i in range(4)] +\
           [f"soil_type_{i}" for i in range(40)] +\
           ["cover_type"]

# cast the target variable as a double as required by MLlib APIs
data = data_no_header.toDF(*colnames).withColumn("cover_type", col("cover_type").cast(DoubleType()))
data.show(5)

#%%First Decision Tree
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
import pandas as pd
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame

# split data into a training and test set
(train_data, test_data) = data.randomSplit([0.9, 0.1])
train_data.cache()
test_data.cache()

# collect all the input features into a single column (required by MLlib)
input_cols = colnames[:-1]
vector_assembler = VectorAssembler(inputCols=input_cols, outputCol="feature_vector")
train_vector = vector_assembler.transform(train_data)
train_vector.select("feature_vector").show(truncate=False)

# build a decision tree classifier
# add the name of a new column to store predictions
clf = DecisionTreeClassifier(seed=1234, labelCol="cover_type", featuresCol="feature_vector",
                             predictionCol="prediction")
dt_model = clf.fit(train_vector)
print(dt_model.toDebugString)

# view model feature importance scores
feat_imp = pd.DataFrame(dt_model.featureImportances.toArray(), index=input_cols, columns=['importance'])
feat_imp.sort_values(by='importance', ascending=False, inplace=True)
feat_imp.iloc[:11, :]

# make predictions on the training data
preds = dt_model.transform(train_vector)
preds.select("cover_type", "prediction", "probability").show(10, truncate=False)

# evaluate model accuracy
test_vector = vector_assembler.transform(test_data)
test_preds = dt_model.transform(test_vector)

evaluator = MulticlassClassificationEvaluator(labelCol="cover_type", predictionCol="prediction")
print(evaluator.setMetricName("accuracy").evaluate(test_preds))
print(evaluator.setMetricName("f1").evaluate(test_preds))

# construct and view a confusion matrix
conf_mat = test_preds.groupBy("cover_type").pivot("prediction", range(1, 8)).count().na.fill(0.0).orderBy("cover_type")
conf_mat.show()


# compare against a baseline model where the probability of predicting a particular class is proportionate with its
# prevalance in the data
def class_probabilities(data):
    total = data.count()
    out_probs = data.groupBy("cover_type").count().orderBy("cover_type").select(col("count").cast(DoubleType()))\
        .withColumn("count_prop", col("count")/total).select("count_prop").collect()
    return out_probs


train_prior_probs = class_probabilities(train_data)
test_prior_probs = class_probabilities(test_data)
pd.DataFrame(train_prior_probs)
pd.DataFrame(test_prior_probs)

train_prior_probs = [prob[0] for prob in train_prior_probs]
test_prior_probs = [prob[0] for prob in test_prior_probs]
# sum products of pairs in train_prior_probs and test_prior_probs - this is the accuracy
sum([train_prob * test_prob for train_prob, test_prob in zip(train_prior_probs, test_prior_probs)])

#%% Tuning Decison Trees

