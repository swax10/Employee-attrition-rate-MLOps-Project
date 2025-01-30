#clean_data.py
# Import necessary libraries
import pandas as pd
import logging
from zenml import step
from typing import Tuple
from src.data_cleaning import DataCleaning, DataPreProcessStrategy, DataDivideStrategy

@step
def clean_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Data cleaning step"""
    try:
        logging.info("Column Names: %s", df.columns)
        logging.info("Data Shape: %s", df.shape)
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        processed_data = data_cleaning.handle_data()
        
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed")
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error in cleaning data: %s", e)
        raise e
