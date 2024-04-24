import asyncio
from pyspark.sql import SparkSession
from tooling.open_ai import OPENAI
import json

from column_processing import handle_text_data
from pyspark.sql.functions import col, explode, split, regexp_replace, lower
from pyspark.sql.types import StringType

def profile_text_data(df):
    """
    Profiles text columns of a DataFrame using the `handle_text_data` function
    and formats the profile into a string suitable for GPT analysis.
    
    Parameters:
    df (pyspark.sql.DataFrame): The DataFrame to be profiled.
    
    Returns:
    str: A formatted string containing the profiling results.
    """
    text_columns_info = []
    fields = df.schema.fields

    for field in fields:
        if isinstance(field.dataType, StringType):
            col_name = field.name
            
            processed_df = handle_text_data(df, col_name, col_name + "_processed")
            
            # Tokenize processed text and get word counts
            word_df = processed_df.withColumn('word', explode(split(col(col_name + "_processed"), "\\s+")))
            total_word_count = word_df.select('word').count()
            unique_word_count = word_df.select('word').distinct().count()
            
            text_columns_info.append((col_name, total_word_count, unique_word_count))

    result_string = "\n".join([
        f"Column '{col_info[0]}' has {col_info[1]} total words and {col_info[2]} unique words."
        for col_info in text_columns_info
    ])

    prompt = (
        "Given the below columns, with their details from profiling, "
        "return the column names as candidates that can potentially benefit from tokenization/lemmatization in JSON format as 'column_name': list[column_names], "
        "and 'explanation': 'your explanation'\n" + result_string
    )

    return prompt

if __name__ == "__main__":
    # Initialize SparkSession
    spark = (
        SparkSession.builder.master("local")
        .appName("Text Column Profiler")
        .getOrCreate()
    )

    # Sample dataframe creation with text data
    data = [
        ("John Doe", "35", "New York"),
        ("Jane Smith", "30", "Los Angeles"),
        ("Alice Johnson", "45", "Chicago"),
        ("Bob Brown", "40", "Houston"),
        ("Carol White", "25", "Phoenix"),
    ]
    df = spark.createDataFrame(data, ["Name", "Age", "City"])

    # Profiling text data in the DataFrame
    prompt = profile_text_data(df)

    client = OPENAI()
    response = client.chat_completion(prompt, temperature=0.3)

    # Display the GPT response
    result_dict = json.loads(response)
    print(result_dict)