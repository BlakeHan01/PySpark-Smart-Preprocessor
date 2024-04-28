import asyncio
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, FloatType, DoubleType, DateType, TimestampType, StringType
from tooling.open_ai import OPENAI
from datetime import datetime, date
import json
from preprocessing_functions.column_processing import normalizer, date_extraction


def column_normalizer_profiler(df, client) -> str:
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
    result_dict = {}
    while "column_name" not in result_dict:
        response = client.chat_completion(message, temperature=0)

        # Convert JSON string to dictionary
        result_dict = json.loads(response)

        explaination = result_dict["explanation"]
        column_names_arr = result_dict["column_name"]
        print(f"Column names to normalize: {column_names_arr}")
        print(f"Here's why: {explaination}")
    # Take user input from terminal
    user_input = input("Enter columns to normalize(comma seperated): ")
    choice = input(
        "Enter 1 for min_max_normalization or 2 for standardization: "
    )
    if "," in choice:
        input_list = user_input.split(",")
    else:
        input_list = [user_input]
    for column_name in input_list:
        df = normalizer(
            df, column_name, f"normalized_{column_name}", int(choice), 0, 1
        )
    return df


def column_date_extraction_profiler(df, client) -> str:
    """
    Date extraction process
    """
    cols = [
        f.name
        for f in df.schema.fields
    ]
    # print(str(cols))
    
    # first_row = df.first()

    message = (
        "Given the below column names" + str(cols) +
        "Return the column names as candidates for datetime extraction in JSON format as 'column_name': list[column_names], "
        "and 'explanation': 'your explanation'\n" 
    )
    result_dict = {}
    while "column_name" not in result_dict:
        response = client.chat_completion(message, temperature=0)
        # print(response)

        # Convert JSON string to dictionary
        result_dict = json.loads(response)

        explaination = result_dict["explanation"]
        column_names_arr = result_dict["column_name"]
        print(f"Column names to process: {column_names_arr}")
        print(f"Here's why: {explaination}")

    # Take user input from terminal
    choice_prompt = "Enter your choice of extraction: 'year', 'month', 'day', 'hour', 'minute', 'second', 'weekday'"
    if len(column_names_arr) >= 2:
        choice_prompt += ", or you can enter 'duration' to compute the duration between two dates"
    choice_prompt += ": "
    choice = input(choice_prompt)

    if choice != 'duration':
        user_input = input("Enter columns to you want to process(comma seperated): ")
        if "," in choice:
            input_list = user_input.split(",")
        else:
            input_list = [user_input]
        for column_name in input_list:
            df = date_extraction(
                df, column_name, f"{column_name}_{choice}_extracted", choice=choice)
    else:
        user_input = input("Enter two columns to you want to compute the duration (comma seperated): ")
        input_list = user_input.split(",")
        while len(input_list) != 2:
            user_input = input("Reenter two columns to you want to compute the duration (comma seperated): ")
            input_list = user_input.split(",")
        col1, col2 = input_list
        df = date_extraction(
                df, col1, f"{col1}_{col2}_duration", choice=choice, another_col=col2)
    return df

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
    ("1", "New York", "2000-01-01", "2000-01-01 12:00", "2001-01-01"),
    ("2", "Houston", "2000-02-01", "2000-01-01 12:00", "2001-02-01"),
    ("3", "Phoenix", "2000-03-01", "2000-01-01 12:00", "2001-03-01")
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
    # test_normalizer_profiler(spark)
    # test_date_extraction_profiler(spark)
