import pytest
from pyspark.sql import SparkSession
from preprocessing_functions.column_processing import (
    normalizer,
    date_extraction,
    Imputation,
)
from datetime import datetime, date

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

def test_date_extraction(spark_session):
    df = spark_session.createDataFrame([
    (date(2000, 1, 1), datetime(2000, 1, 1, 12, 0), date(2001, 1, 1)),
    (date(2000, 2, 1), datetime(2000, 1, 2, 12, 0), date(2002, 1, 1)),
    (date(2000, 3, 1), datetime(2000, 1, 3, 12, 0), date(2003, 1, 1))
    ], ['date', 'datetime', 'date2'])
    # test year (month, day)
    extracted_df = date_extraction(df, 'date', 'new_col', 'year')
    assert 'new_col' in extracted_df.columns
    new_values = extracted_df.select('new_col','date').collect()
    for row in new_values:
        assert row.new_col == row.date.year
    # test hour (minute, second)
    extracted_df = date_extraction(df, 'datetime', 'new_col', 'hour')
    assert 'new_col' in extracted_df.columns
    new_values = extracted_df.select('new_col', 'datetime').collect()
    for row in new_values:
        assert row.new_col == row.datetime.hour
    # test duration
    extracted_df = date_extraction(df, 'date', 'new_col', 'duration', 'date2')
    assert 'new_col' in extracted_df.columns
    new_values = extracted_df.select('new_col', 'date', 'date2').collect()
    for row in new_values:
        assert row.new_col == (row.date2-row.date).days
    # test weekday
    extracted_df = date_extraction(df, 'date', 'new_col', 'weekday')
    assert 'new_col' in extracted_df.columns
    new_values = extracted_df.select('new_col', 'date').collect()
    for row in new_values:
        assert row.new_col == 'weekend' if row.date.isoweekday in [6, 7] else 'weekday'


def test_imputation(spark_session):
    df = spark_session.createDataFrame(
        [
            ("AK", "99504", 2.516),
            (None, None, 30.709),
            ("NY", "35010", 6.849),
            (None, "99645", None),
            (None, "35127", 42.966),
            (None, "99504", None),
        ],
        ["State", "Zipcode", "value"],
    )

    imputed_df = Imputation(df, threthold=0.5, replace_strate="min_value")
    # imputed_df.show()
    # to check whether the State column is dropped
    assert "State" not in imputed_df.columns

    assert "Zipcode" in imputed_df.columns
    assert "value" in imputed_df.columns
    # to check whether the None value in the remaining columns is replaced by some strategy
    assert (
        imputed_df.filter(
            col("Zipcode").isNull() | col("value").isNull()
        ).count()
        == 0
    )

    rows = imputed_df.collect()

    valid_zipcodes = {"99504", "35010", "99645", "35127"}
    # None value of Non-numerical column data needs to be replaced by the mode value strategy
    assert (
        imputed_df.select("Zipcode").where(col("Zipcode") == "99504").count()
        == 3
    )
    # Numerical data can be replaced by the user defined strategy
    assert imputed_df.select("value").where(col("value") == 2.516).count() == 3
    for row in rows:
        assert row["Zipcode"] in valid_zipcodes
