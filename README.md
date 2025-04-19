# Fraud Detection FastAPI App

## Features
- Preprocessing using ColumnTransformer
- Feature importance via Mutual Info + DNN AUC
- Model training using both DNN and XGBoost
- Soft Voting Ensemble
- FastAPI endpoints for training, testing, and prediction
- Streamlit app for interactive visualization of training, evaluation, and predictions

## How to Run

- Navigate root folder (where you downloaded this code folder) before running the below commands
- follow Streamlit steps below.
- Prefer Streamlit over uvicorn run.
- If you want to work with RestAPIs , prefer uvicorn

### FastAPI Server
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the server:
   ```bash
   uvicorn main:app --reload
   ```

3. Use `/docs` for Swagger UI to test the API.

---

### Streamlit App
1. Install dependencies (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

3. Open the URL provided by Streamlit (usually `http://localhost:8501`) in your browser.

4. Use the Streamlit app to:
   - Train models interactively.
   - Visualize confusion matrices and classification reports.
   - Upload CSV files for fraud prediction on new data.