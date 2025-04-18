
# Fraud Detection FastAPI App

## Features
- Preprocessing using ColumnTransformer
- Feature importance via Mutual Info + DNN AUC
- Model training using both DNN and XGBoost
- Soft Voting Ensemble
- FastAPI endpoints for training, testing, and prediction

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the server:
```bash
uvicorn main:app --reload
```

3. Use `/docs` for Swagger UI
