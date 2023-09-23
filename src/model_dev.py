import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression

class Model(ABC):
    @abstractmethod
    def train(self, X_train, Y_train):
        pass    

class LogisticRegressionModel(Model):
    def train(self, X_train, Y_train, **kwargs):
        try:
            clf = LogisticRegression(**kwargs)
            clf.fit(X_train, Y_train)
            logging.info("Model training completed")
            return clf  # Return the trained model
        except Exception as e:
            logging.error(f"Error in training the model: {e}")
            raise e
