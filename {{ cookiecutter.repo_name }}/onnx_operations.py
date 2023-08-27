import json

from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn

class OnnxConverter:
    def __init__(self):
        pass
    
    @staticmethod
    def convert_to_onnx(model, X_train):
        """Convert the Scikit-learn model to ONNX format."""
        initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        return onnx_model

    @staticmethod
    def save_onnx_model(onnx_model, filename="model.onnx"):
        """Save the converted ONNX model to a file."""
        with open(filename, "wb") as f:
            f.write(onnx_model.SerializeToString())

    @staticmethod
    def save_vectorizer_vocabulary(vectorizer, filename="vocabulary.json"):
        """Save the vocabulary of the TF-IDF vectorizer to a file."""
        with open(filename, "w") as f:
            json.dump(vectorizer.vocabulary_, f)