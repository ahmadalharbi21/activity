from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

pipeline = joblib.load('preprocessor_pipeline.joblib')
model = joblib.load('kmeans_model.joblib')

def rename_clusters(cluster_id):
    cluster_mapping = {
        1: "Running",
        0: "Concluded",
        2: "Blockbuster"
    }
    return cluster_mapping.get(cluster_id, "Unknown")

class InputFeatures(BaseModel):
    run_length: int
    ongoing: str
    content_rating: str
    imdb_rating: float
    total_ratings: int

@app.post("/predict")
def predict(input_features: InputFeatures):
    test_data = pd.DataFrame({
        'run_length': [input_features.run_length],
        'ongoing': [1 if input_features.ongoing.lower() == "yes" else 0],
        'Content Rating': [input_features.content_rating],
        'IMDb Rating': [input_features.imdb_rating],
        'Total Ratings': [input_features.total_ratings]
    })

    try:
        processed_data = pipeline.transform(test_data)
        prediction = model.predict(processed_data)[0]
        predicted_label = rename_clusters(prediction)
        return {"predicted_category": predicted_label}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
