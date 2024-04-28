from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType
from pyspark.sql.functions import (
    col,
    isnan,
    when,
    desc,
    to_date,
    regexp_replace,
)
from pyspark.ml.feature import Tokenizer, StopWordsRemover, RegexTokenizer
from pyspark.ml import Pipeline

spark = SparkSession.builder.getOrCreate()


def normalizer(
    df,
    column_name: str,
    column_new_name,
    choice: int,
    range_start: float = 0,
    range_end: float = 1,
):
    """
    Normalize the values in a column of a DataFrame using either Min-Max Scaling or Standardization.

    Args:
        df (DataFrame): The input DataFrame.
        column_name (str): The name of the column to be normalized.
        column_new_name: The name of the new column that will store the normalized values.
        range_start (float, optional): The start of the range for Min-Max Scaling. Defaults to 0.
        range_end (float, optional): The end of the range for Min-Max Scaling. Defaults to 1.
        min_max_scaling (bool, optional): Whether to apply Min-Max Scaling. Defaults to True.
        standardization (bool, optional): Whether to apply Standardization. Defaults to True.

    Returns:
        DataFrame: The DataFrame with the normalized column.

    """
    # Step 1: Create a vector column
    assembler = VectorAssembler(
        inputCols=[column_name], outputCol=column_name + "_vector"
    )
    vector_df = assembler.transform(df)

    # Step 2: Apply Min-Max Scaling
    if choice == 1:
        min_max_scaler = MinMaxScaler(
            inputCol=column_name + "_vector",
            outputCol=column_new_name,
            min=range_start,
            max=range_end,
        )
        vector_df = min_max_scaler.fit(vector_df).transform(vector_df)

    # Step 2: Apply Standardization
    if choice == 2:
        standard_scaler = StandardScaler(
            inputCol=column_name + "_vector",
            outputCol=column_new_name,
            withMean=True,
            withStd=True,
        )
        vector_df = standard_scaler.fit(vector_df).transform(vector_df)

    return vector_df


def date_extraction(
    df, colname: str, new_colname: str, choice=None, another_col=None
):
    """
    Extract year, month, day... from the date column
    choice:   'year', 'month', 'day', 'hour', 'minute', 'second' => y/m/d/h/M/S
            'duration' => duration between the date of two columns
            'weekday' => weekday if 1-5, weekend if 6-7

    Args:
        df (DataFrame): The input DataFrame
        colname (str): The date column to be extracted
        new_colname (str): The column to store the extracted data
        choice (str): Choice of extraction type
        another_col (str): The column to be compared with when choice is 'duration'

    Return:
        DataFrame: The DataFrame with extracted data column
        
    """
    if choice is None:
        return df
    
    df.withColumn(colname, to_date(col(colname), "yyyy-MM-dd"))

    date_functions = {
        'year': F.year,
        'month': F.month,
        'day': F.dayofmonth,
        'hour': F.hour,
        'minute': F.minute,
        'second': F.second
    }

    if choice in date_functions:
        func = date_functions[choice]
        output = df.withColumn(new_colname, func(colname))
    elif choice == 'duration' and another_col is not None:
        output = df.withColumn(
            new_colname, 
            F.datediff(df[another_col], df[colname])
        )
    elif choice == 'weekday':
        output = df.withColumn(
            new_colname, 
            F.when(F.dayofweek(df[colname]).isin([6, 7]), 'weekend')
            .otherwise('weekday')
        )
    else:
        output = df

    return output


def Imputation(df, threthold=0.8, replace_strate="mode_value"):
    """
    docstring
    """
    # User can define the threshold of the ratio of Null or Nan in each column
    # by default, if the ratio of Null or Nan in one column > 0.8, then it will be dropped
    # replace_strate defines the replace strategy of Null or Nan in each column
    # by defalut, the strategy uses the mode value in each column to replace the Null/Nan
    # If the data type of a column is numerical, then the user can specify one strategy from 3 kinds of values in each column
    fields = df.schema.fields
    for field in fields:
        col_name = field.name
        is_numerical = isinstance(field.dataType, NumericType)
        drop_col = []
        row_cnt = df.count()
        cnt_NULL = (
            df.select(col_name)
            .where(col(col_name).isNull() | isnan(col(col_name)))
            .count()
        )
        ratio = cnt_NULL / row_cnt
        if ratio > threthold:
            drop_col.append(col_name)
        else:
            if not is_numerical:
                mode_value = (
                    df.select(col_name)
                    .filter(col(col_name).isNotNull() & (~isnan(col(col_name))))
                    .orderBy(desc(col_name))
                    .groupBy(col_name)
                    .count()
                    .first()[col_name]
                )
                df = df.withColumn(
                    col_name,
                    when(
                        col(col_name).isNull() | isnan(col(col_name)),
                        mode_value,
                    ).otherwise(col(col_name)),
                )
            else:
                if replace_strate == "mode_value":
                    mode_value = (
                        df.select(col_name)
                        .filter(
                            col(col_name).isNotNull() & (~isnan(col(col_name)))
                        )
                        .orderBy(desc(col_name))
                        .groupBy(col_name)
                        .count()
                        .first()[col_name]
                    )
                    df = df.withColumn(
                        col_name,
                        when(
                            col(col_name).isNull() | isnan(col(col_name)),
                            mode_value,
                        ).otherwise(col(col_name)),
                    )
                elif replace_strate == "min_value":
                    min_value = (
                        df.select(col_name)
                        .filter(
                            col(col_name).isNotNull() & (~isnan(col(col_name)))
                        )
                        .orderBy(col_name, ascending=True)
                        .first()[col_name]
                    )
                    df = df.withColumn(
                        col_name,
                        when(
                            col(col_name).isNull() | isnan(col(col_name)),
                            min_value,
                        ).otherwise(col(col_name)),
                    )
                elif replace_strate == "max_value":
                    max_value = (
                        df.select(col_name)
                        .filter(
                            col(col_name).isNotNull() & (~isnan(col(col_name)))
                        )
                        .orderBy(col_name, ascending=False)
                        .first()[col_name]
                    )
                    df = df.withColumn(
                        col_name,
                        when(
                            col(col_name).isNull() | isnan(col(col_name)),
                            max_value,
                        ).otherwise(col(col_name)),
                    )

    df = df.drop(*drop_col)

    return df


def handle_text_data(df, column_name, operation):
    """
    Processes a given text column in a DataFrame by removing null values and/or non-alphabetic characters.

    Parameters:
    df (DataFrame): The DataFrame containing the data to be processed.
    column_name (str): The column name on which to perform operations.
    operation (str): The operation to perform: '1' to remove nulls, '2' to remove non-alphabetic characters, '3' for both.

    Returns:
    DataFrame: The processed DataFrame with the operations applied.
    """
    # Operation '1' or '3' includes removing null values
    if operation in {"1", "3"}:
        df = df.filter(col(column_name).isNotNull())

    # Operation '2' or '3' includes removing non-alphabetic characters
    if operation in {"2", "3"}:
        df = df.withColumn(
            column_name, regexp_replace(col(column_name), "[^a-zA-Z ]", "")
        )

    return df
