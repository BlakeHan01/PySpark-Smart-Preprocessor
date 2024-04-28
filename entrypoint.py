import json
from pyspark.sql import SparkSession

from tooling.open_ai import OPENAI

from profiling.column_profiling import column_normalizer_profiler, column_date_extraction_profiler



def demo_normalizer(df):
    #### DEMO part for normalizer
    # Show the DataFrame
    df.select("num_critic_for_reviews").show(3)

    # column normalizer



    client = OPENAI()
    df = column_normalizer_profiler(df, client)


    df.select(
        "num_critic_for_reviews_vector", "normalized_num_critic_for_reviews"
    ).show(3)

def demo_date_extraction(df):
    #### DEMO part for date extraction
    # Show the DataFrame
    df.select("movie_title", "title_date").show(3)

    client = OPENAI()
    df = column_date_extraction_profiler(df, client)


    df.select(
        "movie_title", "title_date", "title_date_year_extracted"
    ).show(3)


if __name__ == "__main__":
    # Create a SparkSession
    spark = SparkSession.builder.appName("CSV Ingestion").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Read the CSV file into a DataFrame
    df = spark.read.csv(
        "unclean_sample.csv",
        header=True,
        inferSchema=True,
    )

    # demo_normalizer(df)
    demo_date_extraction(df)