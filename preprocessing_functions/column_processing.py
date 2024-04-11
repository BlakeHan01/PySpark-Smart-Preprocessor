from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()


def normalizer(df, column_name: str, column_new_name, range_start: float = 0, range_end: float = 1, min_max_scaling: bool = True, standardization: bool = True):
    if min_max_scaling:
        # Apply Min-Max Scaling
        min_max_scaler = MinMaxScaler(inputCol=column_name, outputCol=column_new_name, min=range_start, max=range_end)
        df = min_max_scaler.fit(df).transform(df)
    if standardization:
        # Apply Standardization
        standard_scaler = StandardScaler(inputCol=column_name, outputCol=column_new_name, withMean=True, withStd=True)
        df = standard_scaler.fit(df).transform(df)

    return df

def date_extraction(df, colname: str, new_colname: str, choice=None, anothercol=None):
    # extract year, month, day... from the date
    # choice:   'year', 'month', 'day', 'hour', 'minute', 'second' => y/M/d/h/m/s
    #           'duration' => duration between the date of two columns
    #           'weekday' => weekday if 1-5, weekend if 6-7 

    if choice is None:
        return df
    if choice == 'year':
        output = df.withColumn(new_colname, F.year(colname))
    elif choice == 'month':
        output = df.withColumn(new_colname, F.month(colname))
    elif choice == 'day':
        output = df.withColumn(new_colname, F.dayofmonth(colname))
    elif choice == 'hour':
        output = df.withColumn(new_colname, F.hour(colname))
    elif choice == 'minute':
        output = df.withColumn(new_colname, F.minute(colname))
    elif choice == 'second':
        output = df.withColumn(new_colname, F.second(colname))
    elif choice == 'duration' and anothercol is not None:
        output = df.withColumn(new_colname, F.datediff(anothercol, colname))
    elif choice == 'weekday':
        output = df.withColumn(new_colname, 'weekend' if F.dayofweek.isin([6, 7]) else 'weekday')
    return output
