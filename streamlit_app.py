import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from project.services.model_service import (
    evaluate_soft_voting,
    train_and_save_models,
    predict_from_csv,
)
import io

# Streamlit app
st.title("Fraud Detection - Vehicle Insurance")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Train Model", "Test Model For Optimal Threshold", "Predict Fraud"])

# Train Model Section
if options == "Train Model":
    st.header("Train Model")

    # Initialize session state for the button
    if "training_in_progress" not in st.session_state:
        st.session_state.training_in_progress = False

    # Disable the button if training is in progress
    if st.session_state.training_in_progress:
        st.warning("Training is in progress. Please wait...")
    else:
        if st.button("Start Training"):
            st.session_state.training_in_progress = True  # Set training state to True
            try:
                import time
                start_time = time.time()

                # Call the training function
                result = train_and_save_models()

                # Calculate training duration
                duration = time.time() - start_time

                # Display success message
                st.success("Training completed successfully!")
                st.write(f"Training Duration: {duration:.2f} seconds")

                # Display training details
                st.subheader("Training Details")
                st.json(result)  # Display features used, AUC, etc.

            except Exception as e:
                st.error(f"Error during training: {e}")
            finally:
                st.session_state.training_in_progress = False  # Reset training state
# Evaluate Model Section
elif options == "Test Model For Optimal Threshold":
    st.header("Test Model For Optimal Threshold")
    if st.button("Test & Get Threshold"):
        try:
            # Call evaluate_soft_voting to get results
            results = evaluate_soft_voting()

            # Extract confusion matrix and classification report
            confusion_matrix = results["confusion_matrix"]
            classification_report = results["classification_report"]

            # Display confusion matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)

            # Display classification report
            st.subheader("Best Prediction Threshold")
            st.text(results["best_threshold"])
            st.subheader("Classification Report")
            st.table(pd.DataFrame(classification_report))

        except Exception as e:
            st.error(f"Error during evaluation: {e}")

# Predict Fraud Section
elif options == "Predict Fraud":
    st.header("Predict Fraud")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            # Save uploaded file to a temporary location
            input_csv_path = "uploaded_file.csv"
            with open(input_csv_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Call predict_from_csv
            output_df, threshold = predict_from_csv(input_csv_path)

            # Display predictions
            st.subheader("Threshold for fraud detection")
            st.text(threshold)
            st.subheader("Predictions")
            st.dataframe(output_df)

            # Option to download the updated CSV
            output_csv_path = "predictions_results.csv"
            output_df.to_csv(output_csv_path, index=False)
            with open(output_csv_path, "rb") as f:
                st.download_button(
                    label="Download Predictions as CSV",
                    data=f,
                    file_name="predictions_results.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error during prediction: {e}")