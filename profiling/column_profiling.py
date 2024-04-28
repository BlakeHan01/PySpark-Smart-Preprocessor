import asyncio
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace
from pyspark.sql.types import IntegerType, FloatType, DoubleType, DateType, TimestampType, StringType
from tooling.open_ai import OPENAI
from pyspark.sql import DataFrame
from datetime import datetime, date
import json
from preprocessing_functions.column_processing import normalizer, date_extraction, handle_text_data


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


def column_textdata_profiler(df: DataFrame) -> str:
    """
    Profiles each text data column in the DataFrame for null values and all most common values,
    handling ties properly.

    Parameters:
    df (DataFrame): The DataFrame to be profiled.

    Returns:
    str: A prompt containing the profiling results and asking for further processing advice.
    """
    result_info = []
    text_cols = [
        f.name for f in df.schema.fields if isinstance(f.dataType, StringType)
    ]

    for column in text_cols:
        null_count = df.filter(col(column).isNull()).count()

        # Count rows with non-alphabetic characters
        non_alpha_count = df.filter(col(column).rlike("[^a-zA-Z ]")).count()

        # Append the information to the results list
        result_info.append(
            f"Column '{column}' has {null_count} null values and {non_alpha_count} rows with non-alphabetic characters."
        )

    # Join all column results into a single string
    result_string = "\n".join(result_info)

    prompt = (
        "Given the below columns, with their count of null values and number of rows where it has non alphabetic characters, "
        "return only the text value column names as candidates for text data processing to remove null or "
        "remove strange the non alphabetic characters in JSON format as 'column_name': list[column_names], "
        "and 'explanation': 'your explanation'\n" + result_string
    )
    result_dict = {}
    client = OPENAI()
    while "column_name" not in result_dict:
        response = client.chat_completion(prompt, temperature=0)

        # Convert JSON string to dictionary
        result_dict = json.loads(response)

        columns_to_process = result_dict["column_name"]
        explanation = result_dict["explanation"]
        print(f"Columns recommended for processing: {columns_to_process}")
        print(f"Explanation: {explanation}")

    # Ask the user for their choice on what to do with the recommended columns
    operation_columns = input(
        "Enter columns to perform operations(comma seperated):"
    )
    print("1: Remove null values")
    print("2: Remove non-alphabetic characters")
    print("3: Both")
    operation_choice = input(
        "Enter your choice for each column in the same order(1, 2, or 3, comma seperated): "
    )

    # Split user input into lists
    columns = operation_columns.split(",")
    operations = operation_choice.split(",")

    # Apply the operations to the columns as per user input
    for column, operation in zip(columns, operations):
        df = handle_text_data(df, column.strip(), operation.strip())

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
