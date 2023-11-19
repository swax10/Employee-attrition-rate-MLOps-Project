import logging
import pandas as pd
from src.data_cleaning import DataCleaning, DataPreProcessStrategy

# Updated get_data_for_test() function
def get_data_for_test():
    try:
        df = pd.read_csv("./data/HR-Employee-Attrition.csv")
        print("Original Data sample:")
        print(df.head())
        # Create a DataPreProcessStrategy instance with encoder
        preprocess_strategy = DataPreProcessStrategy() 
        # Data cleaning with preprocessing
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.handle_data()
        print("Preprocessed Data:")
        print(df.head())
        # Drop 'Attrition' column from test data
        df.drop(["Attrition"], axis=1, inplace=True)
        print("Data Shape for Inference:", df.shape)  # Add this line to print the shape
        df = df.sample(100)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e
