#%% Preparing The Data
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col

spark_config = (SparkConf().setMaster("local[*]")
                # .set("spark.executor.memory", "5g") # smaller VM
                # .set("spark.driver.memory", "5g")
                .set("spark.executor.memory", "10g")
                .set("spark.driver.memory", "10g")
                )

spark = SparkSession.builder.config(conf=spark_config).getOrCreate()

# import the data without column names and print the schema
data_no_header = spark.read.option("inferSchema", True).option("header", False).csv("data/covertype/covtype.data")
data_no_header.printSchema()
data_no_header.show(5, truncate=False)

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
print(sum([train_prob * test_prob for train_prob, test_prob in zip(train_prior_probs, test_prior_probs)]))

#%% Tuning Decison Trees
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pprint import pprint
from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorIndexer
from pyspark.sql.types import IntegerType

# encapsulate the feature creation and classifier in a single pipleine
vector_assembler = VectorAssembler(inputCols=input_cols, outputCol="featureVector")
clf = DecisionTreeClassifier(seed=1234, labelCol="cover_type", featuresCol="featureVector", predictionCol="prediction")
pipeline = Pipeline(stages=[vector_assembler, clf])

# define the combinations of hyperparameters to be tested and the evaluation metric
hp_grid = (ParamGridBuilder().addGrid(clf.impurity, ["gini", "entropy"]).addGrid(clf.maxDepth, [1, 20])
           .addGrid(clf.maxBins, [40, 300]).addGrid(clf.minInfoGain, [0.0, 0.05]).build())

multi_class_evaluator = (MulticlassClassificationEvaluator().setLabelCol("cover_type").setPredictionCol("prediction")
                         .setMetricName("accuracy"))

# use train-validation split rather than k-fold cv as it's less expensive and cv doesn't add much value with large
# data sets
# hold out 10% of the training data as a validation set
validator = TrainValidationSplit(seed=1234, estimator=pipeline, evaluator=multi_class_evaluator,
                                 estimatorParamMaps=hp_grid, trainRatio=0.9)
validator_model = validator.fit(train_data)

# examine the hyperparameters of the "best" model
best_model = validator_model.bestModel
pprint(best_model.stages[1].extractParamMap())

# examine the accuracy for each hyperparameter combination
metrics = validator_model.validationMetrics
params = validator_model.getEstimatorParamMaps()
metrics_and_params = list(zip(metrics, params))

metrics_and_params.sort(key=lambda x: x[0], reverse=True)
metrics_and_params_df = pd.DataFrame(metrics_and_params)

# examine the accuracy of the best model for the cv set and the test set
metrics.sort(reverse=True)
print(metrics[0])

multi_class_evaluator.evaluate(best_model.transform(test_data))


#%% Categorical Features Revisited
# "undo" the one-hot encoding
def unencode_onehot(in_data):
    wilderness_cols = ["wilderness_area_" + str(i) for i in range(4)]
    wilderness_assembler = VectorAssembler().setInputCols(wilderness_cols).setOutputCol("wilderness")

    unhot_udf = udf(lambda v: v.toArray().tolist().index(1))

    with_wilderness = wilderness_assembler.transform(in_data).drop(*wilderness_cols)\
        .withColumn("wilderness", unhot_udf(col("wilderness")))

    soil_cols = ["soil_type_" + str(i) for i in range(40)]
    soil_assembler = VectorAssembler().setInputCols(soil_cols).setOutputCol("soil")

    with_soil = soil_assembler.transform(with_wilderness).drop(*soil_cols)\
        .withColumn("soil", unhot_udf(col("soil")))

    out_data = with_soil.withColumn("wilderness", col("wilderness").cast(IntegerType()))
    out_data = out_data.withColumn("soil", col("soil").cast(IntegerType()))

    return out_data


unenc_train_data = unencode_onehot(train_data)
unenc_train_data.printSchema()

# check that the unencoding worked
unenc_train_data.groupBy("wilderness").count().show()

cols = unenc_train_data.columns
input_cols = [col for col in cols if col != "cover_type"]

vector_assembler = VectorAssembler().setInputCols(input_cols).setOutputCol("featureVector")
# turn input into properly-labelled categorical feature columns
# note the max categories >= 40 because soil has 40 values
vector_indexer = VectorIndexer().setMaxCategories(40).setInputCol("featureVector").setOutputCol("indexedVector")

clf = DecisionTreeClassifier().setLabelCol("cover_type").setFeaturesCol("indexedVector").setPredictionCol("prediction")
pipeline = Pipeline().setStages([vector_assembler, vector_indexer, clf])

hp_grid = (ParamGridBuilder().addGrid(clf.impurity, ["gini", "entropy"]).addGrid(clf.maxDepth, [1, 20])
           .addGrid(clf.maxBins, [40, 300]).addGrid(clf.minInfoGain, [0.0, 0.05]).build())

multi_class_evaluator = (MulticlassClassificationEvaluator().setLabelCol("cover_type").setPredictionCol("prediction")
                         .setMetricName("accuracy"))

# use train-validation split rather than k-fold cv as it's less expensive and cv doesn't add much value with large
# data sets
# hold out 10% of the training data as a validation set
validator = TrainValidationSplit(seed=1234, estimator=pipeline, evaluator=multi_class_evaluator,
                                 estimatorParamMaps=hp_grid, trainRatio=0.9)
validator_model = validator.fit(unenc_train_data)

# examine the accuracy for each hyperparameter combination
metrics = validator_model.validationMetrics
params = validator_model.getEstimatorParamMaps()
metrics_and_params = list(zip(metrics, params))

metrics_and_params.sort(key=lambda x: x[0], reverse=True)
metrics_and_params_df = pd.DataFrame(metrics_and_params)

# examine the accuracy of the best model for the cv set and the test set
metrics.sort(reverse=True)
print(metrics[0])

unenc_test_data = unencode_onehot(test_data)
multi_class_evaluator.evaluate(best_model.transform(unenc_test_data))

#%% Random Forests
from pyspark.ml.classification import RandomForestClassifier

clf = RandomForestClassifier(seed=1234, labelCol="cover_type", featuresCol="indexedVector", predictionCol="prediction")


