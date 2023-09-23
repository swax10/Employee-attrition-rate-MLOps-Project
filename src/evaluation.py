import logging
import numpy as np
from sklearn.metrics import classification_report

class ClassificationReport:
    @staticmethod
    def calculate_scores(y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculate Classification Report")
            report = classification_report(y_true, y_pred, output_dict=True)
            logging.info(f"Classification Report:\n{report}")
            return report
        except Exception as e:
            logging.error(f"Error in calculating Classification Report: {e}")
            raise e
