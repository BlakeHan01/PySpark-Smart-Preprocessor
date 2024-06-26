import json
from pyspark.sql import SparkSession
from profiling.column_profiling import column_textdata_profiler
from tooling.open_ai import OPENAI

from profiling.column_profiling import (
    column_normalizer_profiler,
    column_date_extraction_profiler,
    imputation_profiler,
)


def demo_normalizer(df):
    #### DEMO part for normalizer
    # Show the DataFrame
    # df.show()

    # column normalizer



    client = OPENAI()
    df = column_normalizer_profiler(df, client)


    df.show()
    return df

def demo_date_extraction(df):
    #### DEMO part for date extraction
    # Show the DataFrame
    df.select("movie_title", "title_date").show(3)

    client = OPENAI()
    df = column_date_extraction_profiler(df, client)

    df.select(
        "movie_title", "title_date", "title_date_year_extracted"
    ).show(3)
    return df


def demo_imputation(df):

    df.show()
    client = OPENAI()
    df = imputation_profiler(df, client)

    df.show()
    return df


def demo_textdata_profiler(df):

    # message = column_textdata_profiler(df)
    # client = OPENAI()
    # response = client.chat_completion(message, temperature=0)

    # Convert JSON string to dictionary
    result_df = column_textdata_profiler(df)
    result_df.show()
    return result_df


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
    
    print("Imputation: ")
    df = demo_imputation(df)
    print("Date Extraction: ")
    df = demo_date_extraction(df)
    print("Text Data Profiler: ")
    df = demo_textdata_profiler(df)
    print("Normalizer: ")
    df = demo_normalizer(df)
