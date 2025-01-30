import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

class Model(ABC):
    """
    Abstract base class for all models.
    """
    @abstractmethod
    def train(self, X_train:pd.DataFrame, Y_train:pd.Series):
        """
        Trains the model
        Args:
            X_train: Training data
            Y_train: Training labels
        Returns:
            None
        """
        pass    

class LogisticRegressionModel(Model):
    """
    Logistic Regression model implementation.
    """
    def train(self, X_train:pd.DataFrame, Y_train:pd.Series, **kwargs):
        """
        Trains the logistic regression model
        Args:
            X_train: Training data
            Y_train: Training labels
        Returns:
            Trained model
        """
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model = LogisticRegression(max_iter=1000, **kwargs)
            model.fit(X_train_scaled, Y_train)
            logging.info("Model training completed")
            return model
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e
