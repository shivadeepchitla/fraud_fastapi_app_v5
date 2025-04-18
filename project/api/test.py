
from fastapi import APIRouter, HTTPException
from project.services.model_service import evaluate_soft_voting

router = APIRouter()

@router.post("/")
async def test_model():
    """
    Endpoint to evaluate the model using soft voting.

    Returns:
        dict: A dictionary containing the confusion matrix and classification report.
    """
    
        # Call evaluate_soft_voting
    results = evaluate_soft_voting()
    return results
    print(f"Error during evaluation: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Error during evaluation: {str(e)}")