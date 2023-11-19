#clean_data.py
# Import necessary libraries
import pandas as pd
import logging
from zenml import step
from typing_extensions import Annotated
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from typing import Tuple



@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "Y_train"],
    Annotated[pd.Series, "Y_test"],
]:
    try:
        print("Column Names:", df.columns)  # Add this line
        print("Data Shape:", df.shape)  # Add this line
        logging.info("Data Before Cleaning: ", df.shape)

        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        processed_data = data_cleaning.handle_data()
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, Y_train, Y_test = data_cleaning.handle_data()
        
        logging.info("Data cleaning completed")
        
        return X_train, X_test, Y_train, Y_test
    except Exception as e:
        logging.error(f"Error in cleaning data: {str(e)}")
        raise e
