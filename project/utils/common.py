
import joblib
from tensorflow.keras.models import load_model

def save_artifact(obj, path):
    joblib.dump(obj, path)


def load_artifact(filepath):
    """
    Load a saved artifact (e.g., model, preprocessor).

    Args:
        filepath (str): Path to the saved artifact.

    Returns:
        The loaded artifact.
    """
    if filepath.endswith(".pkl"):
        # Load non-Keras artifacts (e.g., preprocessor, XGBoost model)
        return joblib.load(filepath)
    elif filepath.endswith(".h5") or filepath.endswith(".keras"):
        # Load Keras models with custom objects
        return load_model(filepath)
    else:
        raise ValueError(f"Unsupported file type for {filepath}")
