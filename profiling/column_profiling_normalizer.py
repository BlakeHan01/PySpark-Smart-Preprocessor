import asyncio
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from preprocessing_functions.column_processing import normalizer
from pyspark.sql.types import IntegerType, FloatType, DoubleType
from tooling.open_ai import OPENAI
import json


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


if __name__ == "__main__":
    # Initialize SparkSession
    spark = (
        SparkSession.builder.master("local")
        .appName("Column Profiler")
        .getOrCreate()
    )

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
