# mlops-iris-api
# MLOps Iris Prediction API

This project demonstrates an end-to-end MLOps pipeline using:

- Scikit-learn ML model
- FastAPI for serving predictions
- Docker for containerization
- Pytest for testing
- GitHub Actions for CI/CD

## Model
The model is trained using the Iris dataset and predicts the class of a flower based on four features.

## API Endpoint

POST /predict

Example request:

{
  "features": [5.1, 3.5, 1.4, 0.2]
}

Example response:

{
  "prediction": 0
}

## Running the API locally

```bash
uvicorn app.main:app --reload

