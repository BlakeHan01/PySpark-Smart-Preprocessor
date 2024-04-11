from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType
from pyspark.sql.functions import col, isnan, when, desc
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
    # choice:   'year', 'month', 'day', 'hour', 'minute', 'second' => y/m/d/h/M/S
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
        output = df.withColumn(new_colname, F.lit('weekend' if F.dayofweek in [6, 7] else 'weekday'))
    return output


def Imputation(df, threthold = 0.8, replace_strate = "mode_value"):
    fields = df.schema.fields
    for field in fields:
        col_name = field.name
        is_numerical = isinstance(field.dataType, NumericType)
        drop_col = []
        row_cnt = df.count()
        cnt_NULL = df.select(col_name).where(col(col_name).isNull() | isnan(col(col_name))).count()
        ratio = cnt_NULL / row_cnt
        if ratio > threthold:
            drop_col.append(col_name)
        else:
            if not is_numerical:
                mode_value = df.select(col_name).filter(col(col_name).isNotNull() & (~isnan(col(col_name)))).orderBy(desc(col_name)).groupBy(col_name).count().first()[col_name]
                df = df.withColumn(col_name, when(col(col_name).isNull() | isnan(col(col_name)), mode_value).otherwise(col(col_name)))
            else:
                if replace_strate = "mode_value":
                    mode_value = df.select(col_name).filter(col(col_name).isNotNull() & (~isnan(col(col_name)))).orderBy(desc(col_name)).groupBy(col_name).count().first()[col_name]
                    df = df.withColumn(col_name, when(col(col_name).isNull() | isnan(col(col_name)), mode_value).otherwise(col(col_name)))
                elif replace_strate = "min_value":
                    min_value = df.select(col_name).filter(col(col_name).isNotNull() & (~isnan(col(col_name)))).orderBy(col_name, ascending = True).first()[col_name]
                    df = df.withColumn(col_name, when(col(col_name).isNull() | isnan(col(col_name)), min_value).otherwise(col(col_name)))
                elif replace_strate = "max_value":
                    max_value = df.select(col_name).filter(col(col_name).isNotNull() & (~isnan(col(col_name)))).orderBy(col_name, ascending = False).first()[col_name]
                    df = df.withColumn(col_name, when(col(col_name).isNull() | isnan(col(col_name)), max_value).otherwise(col(col_name)))

    df = df.drop(*drop_col)


    return df

