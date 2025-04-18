
from fastapi import APIRouter, HTTPException
from project.services.model_service import predict_fraud
from project.services.model_service import predict_from_csv
import os
router = APIRouter()

# Paths for input and output CSV files
PREDICT_DATA_PATH = os.path.join("project", "resources", "data", "new_data.csv")
PREDICT_RESULT_PATH = os.path.join("project", "resources", "data", "predictions_results.csv")


@router.post("/predict-from-csv/")
async def predict_from_csv_endpoint(input_csv_path: str = PREDICT_DATA_PATH, output_csv_path: str = PREDICT_RESULT_PATH):
    """
    Endpoint to predict fraud for each datapoint in the input CSV and save the results.

    Args:
        input_csv_path (str): Path to the input CSV file. Defaults to PREDICT_DATA_PATH.
        output_csv_path (str): Path to save the updated CSV file with predictions. Defaults to PREDICT_RESULT_PATH.

    Returns:
        dict: A message indicating success and the path to the output file.
    """
    try:
        # Call the predict_from_csv function
        result_df = predict_from_csv(input_csv_path, output_csv_path)

        # Check if the result contains an error
        if isinstance(result_df, dict) and "error" in result_df:
            raise HTTPException(status_code=500, detail=result_df["error"])

        return {
            "message": "Predictions successfully saved at below path",
            "output_csv_path": output_csv_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process the CSV file: {str(e)}")
