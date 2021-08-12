import uuid
from logging import getLogger
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks
from src.ml.transform import Data
from src.ml.transform import prep 
from src.backend import background_job, store_data_job

logger = getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health() -> Dict[str, str]:
    return {
        "health": "ok",
    }

@router.get("/metadata")
def metadata() -> Dict[str, Any]:
    return {
        "data_type": "str",
        "data_structure": (1, ),
        "data_sample": "Lorem Ipsum is simply dummy text of the printing and typesetting industry.",
        "prediction_type": "float32",
        "prediction_structure": (1,2),
        "prediction_sample": [0.97093159, 0.01558308],
    }


@router.post("/predict")
def predict(data: Data, background_tasks: BackgroundTasks) -> Dict[str, str]:
    job_id = str(uuid.uuid4())
    results = {
        "job_id": job_id
    }
    data = prep.transform(data.data)
    background_job.save_data_job(data, job_id, background_tasks, True)
    logger.info(f"add redis job_id: {job_id} data: {data}")
    return results 

@router.get("/job/{job_id}")
def prediction_resuls(job_id: str):
    results = {"job_id": job_id}
    data = store_data_job.get_data_redis(job_id)
    results["prediction"] = data 
    return results 