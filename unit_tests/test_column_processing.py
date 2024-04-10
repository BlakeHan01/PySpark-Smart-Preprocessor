import pytest
from pyspark.sql import SparkSession
from preprocessing_functions.column_processing import normalizer


@pytest.fixture
def spark_session():
    spark = SparkSession.builder.appName("Test").getOrCreate()
    yield spark
    spark.stop()


def test_normalization(spark_session):
    data = [(1, 10), (2, 20), (3, 30)]
    df = spark_session.createDataFrame(data, ["id", "value"])
    normalized_df = normalizer(df, "value", "normalized_value", 0, 1)
    assert "normalized_value" in normalized_df.columns
    normalized_values = normalized_df.select("normalized_value").collect()
    for row in normalized_values:
        assert 0 <= row.normalized_value <= 1
