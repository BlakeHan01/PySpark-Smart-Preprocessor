import asyncio
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, FloatType, DoubleType, DateType, TimestampType
from tooling.open_ai import OPENAI
from datetime import datetime, date
import json
from preprocessing_functions.column_processing import normalizer


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

    def imputation_profiler(df) -> str:
        """
        Analyze the DataFrame to determine the proportion of missing values per column and
        suggest columns for imputation based on a threshold of missing data.

        Returns:
        str: A message formatted with columns recommended for imputation and their statistics.
        """

        data_missing = {}
        fields = df.schema.fields
        for field in fields:
            col_name = field.name
            row_cnt = df.count()
            cnt_NULL = df.select(col_name).where(col(col_name).isNull() | isnan(col(col_name))).count()
            ratio = cnt_NULL / row_cnt
            data_missing[col_name] = ratio

        # Threshold for recommending imputation could be set here (e.g., 0.2 for 20% missing)
        threshold = 0.8
        columns_for_imputation = [
            col_name for col_name, missing_ratio in data_missing.items() if missing_ratio > 0.0
            ]
        columns_may_drop = [
            col_name for col_name, missing_ratio in data_missing.items() if missing_ratio > threshold
            ]
        # Formatting results
        result_string = ""
        for col_name, missing_ratio in data_missing.items():
            result_string += f
            "{col_name}: {missing_ratio * 100:.2f}% missing data\n"

        message = (
            "Based on the analysis, the following columns are candidates for imputation "
            "(there is some missing data in each column):\n"
            + ", ".join(columns_for_imputation) + "\n"
            + "Detailed missing data percentages per column:\n"
            + result_string
            + "If the ratio of missing data of some column is larger than threshold \n"
            + "these column will be dropped in the imputation, by default, the value of threshold is 0.8\n"
            + "For the remaining columns, you can choose the following strategy to replace the Null value:\n"
            + "[min_value], [max_value], [mode_value], by default, the strategy is [mode_value]"
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

def test_imputation_profiler(spark):


    df = spark.createDataFrame([
        ("AK", "99504", 2.516, "a"),
        (None, None, 30.709, "b"),
        ("NY", "35010", 6.849, "c"),
        (None, "99645", None, "d"),
        (None, "35127", 42.966, "e"),
        (None, "99504", None, "f"),
    ], ['State', 'Zipcode', 'value', 'tmp'])

    message = imputation_profiler(df)
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
    test_imputation_profiler(spark)
