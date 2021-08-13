from logging import getLogger
from typing import Any, Tuple

import numpy as np
import requests 
import json 
from src.configurations import ServiceConfiguraionsOutlier

logger = getLogger(__name__)

class OutlierDetector(object):
    def __init__(self, model_path: str):
       self.endpoint = model_path 
       self.headers = {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
       }

    def predict(self, data: Any) -> Tuple[bool, list]:
        np_data = np.array(data).astype(np.float32)
        data = {"data": np_data}
        is_outlier, outlier_score = requests.post(self.endpoint, data=json.dumps(data), headers=self.headers)
        outlier_score = list(outlier_score)
        return is_outlier, outlier_score

outlier_detector = OutlierDetector(
   model_path=ServiceConfiguraionsOutlier().outlier_url
)
