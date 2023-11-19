import logging
import numpy as np
from sklearn.metrics import classification_report
from abc import ABC, abstractmethod

class Evaluation(ABC):
    """
    Abstract Class defining the strategy for evaluating model performance
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

class ClassificationReport(Evaluation):
    
    def calculate_scores(self,y_true: np.ndarray, y_pred: np.ndarray)-> float:
        try:
            logging.info("Calculate Classification Report")
            report = classification_report(y_true, y_pred, output_dict=True)
            logging.info(f"Classification Report:\n{report}")
            return report
        except Exception as e:
            logging.error(f"Error in calculating Classification Report: {e}")
            raise e
