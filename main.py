
from fastapi import FastAPI
from project.api import train, test, predict

app = FastAPI(
    title="Fraud Detection API",
    description="Train, test and predict using DNN + XGBoost + Soft Voting Ensemble",
    version="1.0"
)

app.include_router(train.router, prefix="/train")
app.include_router(test.router, prefix="/test")
app.include_router(predict.router, prefix="/predict")
