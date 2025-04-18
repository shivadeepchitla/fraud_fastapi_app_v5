
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold ,train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from project.services.preprocessing import get_preprocessing_pipeline
from project.utils.common import save_artifact, load_artifact
import json
from tensorflow.keras import models, layers
from fastapi import Request
from tensorflow.keras.saving import register_keras_serializable
import joblib
from tensorflow.keras.models import load_model
import pickle
import os
from focal_loss import BinaryFocalLoss
from tensorflow.keras.losses import Loss
import io

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "resources", "data", "carclaims.csv")
MODEL_DIR = "project/resources/models"
PREDICT_DATA_PATH = os.path.join(BASE_DIR, "resources", "data", "new_data.csv")
PREDICT_RESULT_PATH = os.path.join(BASE_DIR, "resources", "data", "predictions_results.csv")
TOP_K_INDICES_PATH = os.path.join("project", "resources", "models", "top_k_indices.json")
best_thresh = 0.5


def save_top_k_indices(indices):
    with open(TOP_K_INDICES_PATH, "w") as f:
        json.dump(indices, f)

@register_keras_serializable()
class CustomBinaryFocalLoss(Loss):
    def __init__(self, gamma=2.0, alpha=0.35, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_mean(self.alpha * tf.pow(1. - pt_1, self.gamma) * tf.math.log(pt_1)) - \
               tf.reduce_mean((1 - self.alpha) * tf.pow(pt_0, self.gamma) * tf.math.log(1. - pt_0))

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


def build_improved_model(input_dim, learning_rate=1e-3):
    model = Sequential()
    model.add(Dense(128, input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Dense(16))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.15))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=CustomBinaryFocalLoss(gamma=2, alpha=0.35),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

def train_and_save_models():
    df = pd.read_csv(DATA_PATH)
    df.drop(columns=["PolicyNumber", "RepNumber", "Year", "Age"], inplace=True)
    df["FraudFound"] = df["FraudFound"].map({"Yes": 1, "No": 0})
    y = df["FraudFound"]
    X = df.drop(columns=["FraudFound"])
    # Identify column types
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Define encoding strategies
    one_hot_cols = [
        "Month", "MonthClaimed", "DayOfWeek", "DayOfWeekClaimed", "Sex", "MaritalStatus",
        "AccidentArea", "Fault", "PolicyType", "VehicleCategory", "PoliceReportFiled",
        "WitnessPresent", "AgentType", "BasePolicy", "WeekOfMonth", "WeekOfMonthClaimed"
    ]

    ordinal_cols = [
        "VehiclePrice", "Days:Policy-Accident", "Days:Policy-Claim", "PastNumberOfClaims",
        "AgeOfVehicle", "AgeOfPolicyHolder", "NumberOfSuppliments", "AddressChange-Claim",
        "NumberOfCars", "Deductible", "DriverRating"
    ]

    
    
    # ---------------------------
    # Train/Test Split & Oversampling
    # ---------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    preprocessor = get_preprocessing_pipeline(numeric_cols, ordinal_cols, one_hot_cols)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    

    # Apply ADASYN to numeric+encoded inputs only
    print("Class distribution before ADASYN:")
    print(y_train.value_counts(normalize=True))
    adasyn = ADASYN(random_state=42, sampling_strategy=0.8)
    X_train_res, y_train_res = adasyn.fit_resample(X_train_proc, y_train)
    # Calculate the oversampling ratio
    oversampling_ratio = len(y_train_res) / len(y_train)
    print("Class distribution after ADASYN:")
    print(pd.Series(y_train_res).value_counts(normalize=True))
    
    # ---------------------------
    # Optimal Feature Selection
    # ---------------------------
    # Mutual Info and DNN-based Feature Selection
    mi_scores = mutual_info_classif(X_train_res, y_train_res, random_state=42)
    sorted_indices = np.argsort(mi_scores)[::-1]

    best_auc = 0
    best_k = 0
    best_features = None

    for k in range(20, X_train_res.shape[1], 10):
        top_k_indices = sorted_indices[:k]
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for train_idx, val_idx in skf.split(X_train_res[:, top_k_indices], y_train_res):
            model = models.Sequential([
                layers.Dense(64, activation="relu", input_shape=(k,)),
                layers.Dense(32, activation="relu"),
                layers.Dense(1, activation="sigmoid")
            ])
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
            model.fit(X_train_res[train_idx][:, top_k_indices], y_train_res[train_idx], epochs=5, batch_size=64, verbose=0)
            auc = model.evaluate(X_train_res[val_idx][:, top_k_indices], y_train_res[val_idx], verbose=0)[1]
            aucs.append(auc)
        avg_auc = np.mean(aucs)
        if avg_auc > best_auc:
            best_auc = avg_auc
            best_k = k
            best_features = top_k_indices.tolist()  # Convert to list for saving

    # Save best feature indices
    save_top_k_indices(best_features)
    feature_names = preprocessor.get_feature_names_out()

    # Get the top 15 indices from the sorted list
    top_15_indices = sorted_indices[:15]
    # Fetch the feature names
    top_15_features = [feature_names[i] for i in top_15_indices]
    # Print them
    res_top_15 = []
    print("Top 15 features based on importance:")
    for rank, feat in enumerate(top_15_features, 1):
        res_top_15.append((rank, feat))
    # Apply to train and test data
    X_train_final = X_train_res[:, best_features]
    X_test_final = X_test_proc[:, best_features]

    # Save the preprocessor and top-k indices  
    # Save X_test_final and y_test for later use
    pd.DataFrame(X_test_final).to_csv(os.path.join(MODEL_DIR, "X_test_final.csv"), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(MODEL_DIR, "y_test.csv"), index=False)
    
    X_train, X_val, y_train, y_val = train_test_split(
    X_train_final, y_train_res, test_size=0.2, stratify=y_train_res, random_state=42 # Use X_train_res instead of X_train_proc and y_train_res instead of y_train
    )

    # Final models
    dnn_final = build_improved_model(input_dim=X_train.shape[1])
    dnn_final.fit(X_train, y_train, epochs=40, batch_size=256, validation_data=(X_val, y_val),
          callbacks=[EarlyStopping(patience=6, restore_best_weights=True)])

    # Count of class labels to compute scale_pos_weight
    neg, pos = np.bincount(y_train)
    scale = neg / pos
    xgb = XGBClassifier(
        n_estimators=500,              # More trees
        learning_rate=0.05,            # Slower learning
        max_depth=6,                   # Prevents overfitting
        min_child_weight=5,            # Minimum instances in a leaf
        gamma=0.2,                     # Minimum loss reduction to make a split
        subsample=0.8,                 # Row subsample ratio
        colsample_bytree=0.8,          # Feature subsample ratio
        reg_alpha=0.5,                 # L1 regularization
        reg_lambda=1.0,                # L2 regularization
        scale_pos_weight=scale,        # Handles imbalance (major key!)
        objective='binary:logistic',
        eval_metric='auc',             # Better for fraud detection than logloss
        use_label_encoder=False,
        random_state=42,
        verbosity=1
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    
    # Capture DNN model summary
    model_summary = io.StringIO()
    dnn_final.summary(print_fn=lambda x: model_summary.write(x + "\n"))
    model_summary_str = model_summary.getvalue()
    # Final models
    # ensemble = VotingClassifier(estimators=[("dnn", dnn_final), ("xgb", xgb)], voting="soft")
    # ensemble.fit(X_train_final, y_train_res)

    save_artifact(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))
    dnn_final.save(os.path.join(MODEL_DIR, "dnn_model.keras"))
    save_artifact(xgb, os.path.join(MODEL_DIR, "xgb_model.pkl"))
    # save_artifact(ensemble, os.path.join(MODEL_DIR, "ensemble.pkl"))
    return {"features_used": best_k,"top_15_features_by_importance":res_top_15, "auc": best_auc, "oversampling_ratio": oversampling_ratio, "model_summary": model_summary_str}

def evaluate_soft_voting():
    """
    Perform soft voting dynamically using DNN and XGBoost models.

    Args:
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for the test set.
        preprocessor: Preprocessing pipeline.
        dnn_model: Trained DNN model.
        xgb_model: Trained XGBoost model.
        best_features (list): Indices of selected features.

    Returns:
        dict: A dictionary containing the confusion matrix and classification report.
    """
    preprocessor = load_artifact(os.path.join(MODEL_DIR, "preprocessor.pkl"))
    dnn_model = load_artifact(os.path.join(MODEL_DIR, "dnn_model.keras"))
    xgb_model = load_artifact(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    
    # Load X_test_final and y_test
    X_test_final = pd.read_csv(os.path.join(MODEL_DIR, "X_test_final.csv")).values
    y_test = pd.read_csv(os.path.join(MODEL_DIR, "y_test.csv")).values.ravel()

    # Step 1: Preprocess and apply feature selection for DNN
    print("starting dnn eval")
    dnn_probs = dnn_model.predict(X_test_final).ravel()  # DNN probabilities
    print("end of dnn eval")

    print("starting xgb eval")
    # Step 2: Preprocess and apply feature selection for XGBoost
    xgb_probs = xgb_model.predict_proba(X_test_final)[:, 1]  # XGBoost probabilities
    print("end of xgb eval")

    # Step 3: Average probabilities (soft voting)
    ensemble_probs = 0.8 * dnn_probs + 0.2 * xgb_probs  # Weighted average

    # Step 4: Tune threshold for optimal F1 score
    prec, rec, thresholds = precision_recall_curve(y_test, ensemble_probs)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-6)
    global best_thresh
    best_thresh = thresholds[np.argmax(f1_scores)]
    print(f"\nOptimal Threshold (F1): {best_thresh:.4f}")

    # Step 5: Apply threshold to generate final predictions
    y_pred_ensemble = (ensemble_probs >= best_thresh).astype(int)

    # Step 6: Generate classification report
    class_report = classification_report(y_test, y_pred_ensemble, output_dict=True)

    # Step 7: Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred_ensemble)

    # Return results as a dictionary
    return {
        "confusion_matrix": cm.tolist(),  # Convert to list for JSON serialization
        "classification_report": class_report,
        "best_threshold": best_thresh
    }
def predict_fraud(input_data):
    
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([input_data])

        # Load preprocessor and top-k indices
        preprocessor = load_artifact(os.path.join(MODEL_DIR, "preprocessor.pkl"))
        with open(TOP_K_INDICES_PATH, "r") as f:
            top_k_idx = json.load(f)

        # Load the trained ensemble model
        model = load_artifact(os.path.join(MODEL_DIR, "ensemble.pkl"))

        # Preprocess the input data
        X_proc = preprocessor.transform(df)
        X_final = X_proc[:, top_k_idx]

        # Make prediction
        fraud_probability = float(model.predict_proba(X_final)[0][1])
        return fraud_probability

    except Exception as e:
        # Handle errors gracefully
        print(f"Error in predict_fraud: {e}")
        return {"error": "Failed to process input data or make prediction."}


def predict_from_csv(input_csv_path, output_csv_path=None):
    """
    Predict fraud for each datapoint in the input CSV and add a column with predicted values.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str, optional): Path to save the updated CSV file with predictions. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with actual, predicted, and fraud probability columns.
    """
    try:
        # Load the input CSV file
        df = pd.read_csv(input_csv_path)
        df.drop(columns=["PolicyNumber", "RepNumber", "Year", "Age"], inplace=True)

        # Remove empty rows
        df.dropna(inplace=True)

        # Ensure the input data matches the training format
        if "FraudFound" not in df.columns:
            raise ValueError("Input CSV must contain the 'FraudFound' column for actual values.")

        # Separate features and actual labels
        y_actual = df["FraudFound"]
        X = df.drop(columns=["FraudFound"])

        # Load preprocessor and top-k indices
        preprocessor = load_artifact(os.path.join(MODEL_DIR, "preprocessor.pkl"))
        with open(TOP_K_INDICES_PATH, "r") as f:
            top_k_idx = json.load(f)

        # Preprocess the input data
        X_processed = preprocessor.transform(X)
        X_final = X_processed[:, top_k_idx]

        # Load the trained DNN and XGBoost models
        dnn_model = load_artifact(os.path.join(MODEL_DIR, "dnn_model.keras"))
        xgb_model = load_artifact(os.path.join(MODEL_DIR, "xgb_model.pkl"))

        # Make predictions using both models
        dnn_probs = dnn_model.predict(X_final).ravel()  # DNN probabilities
        xgb_probs = xgb_model.predict_proba(X_final)[:, 1]  # XGBoost probabilities

        # Combine predictions using soft voting
        ensemble_probs = 0.8 * dnn_probs + 0.2 * xgb_probs  # Weighted average
        print("best threshold used for predictions on new data", best_thresh)
        y_pred = (ensemble_probs >= best_thresh).astype(int)  # Convert probabilities to binary predictions

        # Add predictions to the original DataFrame
        df["Predicted"] = y_pred
        df["FraudProbability"] = ensemble_probs

        # Save the updated DataFrame to a new CSV file if output path is provided
        if output_csv_path:
            df.to_csv(output_csv_path, index=False)

        return df,best_thresh

    except Exception as e:
        # Handle errors gracefully
        print(f"Error in predict_from_csv: {e}")
        return {"error": "Failed to process input CSV or make predictions."}
        
if __name__ == "__main__":
    result = train_and_save_models()
    print("Training completed.")
    print("Details:", result)