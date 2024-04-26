import asyncio
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, FloatType, DoubleType, DateType, TimestampType
from tooling.open_ai import OPENAI
from datetime import datetime, date
import json


def column_normalizer_profiler(df) -> str:
    """
    Return the message to be passed into the chat_completion function
    """
    numeric_cols = [
        f.name
        for f in df.schema.fields
        if isinstance(f.dataType, (IntegerType, FloatType, DoubleType))
    ]
    stats = df.select(numeric_cols).describe().collect()

    result_string = ""
    for stat in stats:
        result_string += (
            f"{stat['summary']}: "
            + ", ".join([f"{col}: {stat[col]}" for col in numeric_cols])
            + "\n"
        )

    message = (
        "Given the below columns, with their count, mean, standard deviation, min, and max, "
        "return the column names as candidates for normalization in JSON format as 'column_name': list[column_names], "
        "and 'explanation': 'your explanation'\n" + result_string
    )
    return message

def column_date_extraction_profiler(df) -> str:
    """
    Return the message to be passed into the chat_completion function
    """
    cols = [
        f.name
        for f in df.schema.fields
    ]
    result_string = ""
    for c in cols:
        result_string += str(c) + '\n'

    message = (
        "Given the below columns"
        "We can extract year, month, day, hour, minute, second, duration of two dates and day of the week. "
        "Return the column names as candidates for datetime extraction in JSON format as 'column_name': list[column_names], "
        "and 'explanation': 'your explanation'\n" + result_string
    )
    return message

##################################################################
# TEST
##################################################################

def test_normalizer_profiler(spark):
    # Sample dataframe creation
    data = [
        ("25", "50000", "New York"),
        ("30", "60000", "Los Angeles"),
        ("35", "70000", "Chicago"),
        ("40", "80000", "Houston"),
        ("45", "90000", "Phoenix"),
    ]

    df = spark.createDataFrame(data, ["Age", "Income", "City"])

    # Convert the string numeric values to integers
    df = df.withColumn("Age", df["Age"].cast(IntegerType()))
    df = df.withColumn("Income", df["Income"].cast(IntegerType()))

    message = column_normalizer_profiler(df)
    client = OPENAI()
    response = client.chat_completion(message, temperature=0)

    # Convert JSON string to dictionary
    result_dict = json.loads(response)
    print(result_dict)

def test_date_extraction_profiler(spark):
    """
        Test date extraction profiler
    """
    df = spark.createDataFrame([
    ("1", "New York", date(2000, 1, 1), datetime(2000, 1, 1, 12, 0), date(2001, 1, 1)),
    ("2", "Houston", date(2000, 2, 1), datetime(2000, 1, 2, 12, 0), date(2002, 1, 1)),
    ("3", "Phoenix", date(2000, 3, 1), datetime(2000, 1, 3, 12, 0), date(2003, 1, 1))
    ], ["id", "city", "start_date", "start_datetime", "end_date"])

    message = column_date_extraction_profiler(df)
    client = OPENAI()
    response = client.chat_completion(message, temperature=0)
    result_dict = json.loads(response)
    print(result_dict)

if __name__ == "__main__":
    # Initialize SparkSession
    spark = (
        SparkSession.builder.master("local")
        .appName("Column Profiler")
        .getOrCreate()
    )
    test_normalizer_profiler(spark)
    test_date_extraction_profiler(spark)

    
