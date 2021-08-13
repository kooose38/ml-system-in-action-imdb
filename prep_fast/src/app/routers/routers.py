from logging import getLogger
from typing import Any, Dict
import time 

from fastapi import APIRouter
from src.ml.prediction import Data, classifier
from src.ml.outlier_detection import outlier_detector
from src.configurations import ServiceConfigurations 

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


@router.get("/label")
def label() -> Dict[int, str]:
    return classifier.label


@router.get("/predict/test")
def predict_test() -> Dict[str, Any]:
    input_ids = classifier.transform(Data().data)
    is_outlier, outlier_score = outlier_detector.predict(input_ids)
    results = {
        "is_outlier": is_outlier,
        "outlier_score": outlier_score
    }

    for service, url in ServiceConfigurations().services.items():
        logger.info(f"request to {url}")
        response = classifier.predict(input_ids, url)
        results[service] = response
    return response 


@router.get("/predict/test/label")
def predict_test_label() -> Dict[str, str]:
    input_ids = classifier.transform(Data().data)
    is_outlier, outlier_score = outlier_detector.predict(input_ids)
    results = {
        "is_outlier": is_outlier,
        "outlier_score": outlier_score
    }

    for service, url in ServiceConfigurations().services.items():
        logger.info(f"request to {url}")
        response = classifier.predict_label(input_ids, url)
        results[service] = response 
    return results 


@router.post("/predict")
def predict(data: Data, job_id: str) -> Dict[str, Any]:
    input_ids = classifier.transform(data.data)
    is_outlier, outlier_score = outlier_detector.predict(input_ids)
    results = {
        "job_id": job_id,
        "is_outlier": is_outlier,
        "outlier_score": outlier_score
    }

    start = time.time()
    for service, url in ServiceConfigurations().services.items():
        logger.info(f"request to {url}")
        response = classifier.predict(input_ids, url)
        results[service] = response
        logger.info(f"input: {data.data} output: {response}")
    elapsed = 1000*(time.time()-start)
    logger.info(f"[{job_id}] [{results}] [{elapsed} ms] [{data.data}] [{input_ids}]")
    return response 


@router.post("/predict/label")
def predict_label(data: Data, job_id: str) -> Dict[str, str]:
    input_ids = classifier.tranform(data.data)
    is_outlier, outlier_score = outlier_detector.predict(input_ids)
    results = {
        "job_id": job_id,
        "is_outlier": is_outlier,
        "outlier_score": outlier_score
    }

    start = time.time()
    for service, url in ServiceConfigurations().services.items():
        logger.info(f"request to {url}")
        response = classifier.predict_label(input_ids, url)
        results[service] = response 
        logger.info(f"input: {data.data} output: {response}")
    elapsed = 1000*(time.time()-start)
    logger.info(f"[{job_id}] [{results}] [{elapsed} ms] [{data.data}] [{input_ids}]")
    return results 