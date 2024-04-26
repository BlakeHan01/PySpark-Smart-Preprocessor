import json
from pyspark.sql import SparkSession

from tooling.open_ai import OPENAI

# Create a SparkSession
spark = SparkSession.builder.appName("CSV Ingestion").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Read the CSV file into a DataFrame
df = spark.read.csv(
    "unclean_sample.csv",
    header=True,
    inferSchema=True,
)

#### DEMO part for normalizer
# Show the DataFrame
df.select("num_critic_for_reviews").show(3)

# column normalizer
from profiling.column_profiling import column_normalizer_profiler


client = OPENAI()
df = column_normalizer_profiler(df, client)

df.select(
    "num_critic_for_reviews_vector", "normalized_num_critic_for_reviews"
).show(3)
