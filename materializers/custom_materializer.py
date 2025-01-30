from zenml.materializers.base_materializer import BaseMaterializer
from sklearn.linear_model import LogisticRegression
import joblib

class SklearnMaterializer(BaseMaterializer):
    """Materializer for scikit-learn models."""
    
    ASSOCIATED_TYPES = (LogisticRegression, )
    
    def load(self, data_type: type) -> LogisticRegression:
        """Load the model from artifacts."""
        return joblib.load(self.uri)
    
    def save(self, model: LogisticRegression) -> None:
        """Save the model to artifacts."""
        joblib.dump(model, self.uri)
