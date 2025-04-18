
from fastapi import APIRouter
from project.services.model_service import train_and_save_models

router = APIRouter()

@router.post("/")
def train_model():
    result = train_and_save_models()
    return {"message": "Training completed", "details": result}
