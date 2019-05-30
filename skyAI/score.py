import json
import numpy as np
import os
from keras.models import load_model

from azureml.core.model import Model

def init():
    global model
    # retrieve the path to the model file using the model name
    model_path = Model.get_model_path('sky')
    model = load_model(model_path)
    

def run(raw_data):
    try:
        data = np.array(json.loads(raw_data)['data'])
        # make prediction
        y_hat = model.predict(data)
        # you can return any data type as long as it is JSON-serializable
        return y_hat.tolist()
    except Exception as e:
        error = str(e)
        return error
