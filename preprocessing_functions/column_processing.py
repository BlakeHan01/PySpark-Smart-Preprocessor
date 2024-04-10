from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.getOrCreate()


def normalizer(df, column_name: str, column_new_name, range_start: float, range_end: float, min_max_scaling: bool = True, standardization: bool = True):
    if min_max_scaling:
        # Apply Min-Max Scaling
        min_max_scaler = MinMaxScaler(inputCol=column_name, outputCol=column_new_name, min=range_start, max=range_end)
        df = min_max_scaler.fit(df).transform(df)
    if standardization:
        # Apply Standardization
        standard_scaler = StandardScaler(inputCol=column_name, outputCol=column_new_name, withMean=True, withStd=True)
        df = standard_scaler.fit(df).transform(df)

    return df