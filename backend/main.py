from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
import uuid

from core.data_handler import DataHandler
from core.automl import AutoMLPipeline

app = FastAPI(title="AutoML Framework API")

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for datasets and results (for demo purposes)
# In production, use a database and distributed file storage (e.g. S3)
storage = {
    "datasets": {},
    "results": {}
}

@app.get("/")
def read_root():
    return {"message": "AutoML Framework API is running"}

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        df = await DataHandler.read_csv_file(file)
        schema = DataHandler.infer_schema(df)
        
        dataset_id = str(uuid.uuid4())
        storage["datasets"][dataset_id] = df
        
        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "schema": schema
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/{dataset_id}")
async def train_model(
    dataset_id: str,
    target_column: str = Form(...),
    task_type: str = Form(...),
    models: str = Form(...),  # Comma separated string e.g., "random_forest,xgboost"
    use_tuning: bool = Form(False),
    tuning_trials: int = Form(10)
):
    if dataset_id not in storage["datasets"]:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    if task_type not in ["classification", "regression"]:
        raise HTTPException(status_code=400, detail="task_type must be classification or regression")
        
    df = storage["datasets"][dataset_id]
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    
    try:
        # Note: For production, this should be sent to a Celery worker queue
        pipeline = AutoMLPipeline(
            task_type=task_type, 
            target_column=target_column, 
            model_names=model_list,
            use_tuning=use_tuning,
            tuning_trials=tuning_trials
        )
        results = pipeline.fit(df)
        
        job_id = str(uuid.uuid4())
        storage["results"][job_id] = {
            "status": "completed",
            "metrics": results,
            "task_type": task_type,
            "target_column": target_column
        }
        
        return {
            "job_id": job_id,
            "status": "completed",
            "message": "Training completed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    if job_id not in storage["results"]:
        raise HTTPException(status_code=404, detail="Job results not found")
    return storage["results"][job_id]
