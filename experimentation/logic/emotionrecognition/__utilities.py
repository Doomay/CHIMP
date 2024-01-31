import tempfile
import os
from os import path, getcwd, makedirs

import numpy as np
from numpy.random import RandomState
import json
import heapq

from mlflow import log_artifact
from mlflow import MlflowClient
from mlflow.models import get_model_info
from mlflow.artifacts import download_artifacts
from mlflow import search_runs


def split_data(data, fraction: float, random_state: RandomState):
    """
    Splits the data two sets of unequal proportions, in a deterministic manner.

    :param data: The data to split into two sets.
    :param fraction: The proportion of the data split. Must be a number between 0 and 1.
    :param random_state: The random state that determines how the data will be split.

    :return: Returns two sets of data, the first one being of the size of the given proportions, and the other one being
    the remaining data.
    """
    mask = random_state.random(len(data['image_data'])) < fraction

    return _apply_mask(data, mask), _apply_mask(data, ~mask)


def _apply_mask(data, mask):
    """
    A utility function that filters the data object based on the given mask. The mask and the data object need to be of
    equal lengths.

    :param data: The data object which to filter. Needs to have the following dictionary keys: 'image_data', 'class_',
    'category'.
    :param mask: The mask to apply to the data object.

    :return: Returns the data filtered by the given mask.
    """
    data = data.copy()
    data['image_data'] = data['image_data'][mask]
    data['class_'] = data['class_'][mask]
    data['category'] = data['category'][mask]

    return data


def save_data_object(data_object: dict, artifact_path: str):
    """
    Utility method for the emotion recognition module to save data as an artifact to the MLFlow artifact server. Splits
    data into the prediction data (image), labels (emotions), and encoding for the labels (numeric identification of
    emotion).

    :param data_object: A dictionary with data partials (predicted data, labels, encoded labels) to save into the MLFlow
    artifact server.
    :param artifact_path: The path at which to store the data partials as artifacts.
    """
    for data_entry_key in data_object.keys():
        _save_data_item(data_object[data_entry_key], artifact_filename=data_entry_key, artifact_path=artifact_path)


def _save_data_item(data_item: np.ndarray, artifact_filename, artifact_path):
    """
    Utility method for the emotion recognition module to save individual data partials (images, labels, etc.) into the
    MLFlow artifact server as a npy file.

    :param data_item: The individual data partial to store to the MLFlow server.
    :param artifact_filename: The file name with which to store the data partial artifact.
    :param artifact_path: The path at which to store the data partial artifact.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = path.join(tmpdir, f"{artifact_filename}.npy")

        np.save(file=local_file, arr=data_item)
        log_artifact(local_file, artifact_path)
        
stored_models = []  # Define a list to store the models

def fetch_models(model, evaluation_metric_input = None):
    """
    Utility method for the emotion recognition module to fetch the run ID's of models after their run has finished.

    :param model: The run ID of the model
    :param evaluation_metric_input: The input number of the publish metric
    """
    with open(path.join(getcwd(), 'config.json'), 'r') as f:
        config = json.load(f)
        evaluation_metric = config['best_model_publish_metric']
        
    global stored_models
    stored_models.append({f"model": model, f"{evaluation_metric}": evaluation_metric_input})
    if evaluation_metric_input is not None:
        print(f"Model saved successfully:", model, f"with {evaluation_metric}", evaluation_metric_input)
    else:
        print(f"Model saved successfully:", model, f"({evaluation_metric} not provided)")

    return {"model": model, f"{evaluation_metric}": evaluation_metric_input}

def rate_fetched_models():
    """
    Utility method for the emotion recognition module to rate the latest models that have been made.

    :return: The best-rated model.
    """
    with open(path.join(getcwd(), 'config.json'), 'r') as f:
        config = json.load(f)
        evaluation_metric = config['best_model_publish_metric']
        
    client = MlflowClient()
    global stored_models

    if not stored_models:
        print("No models to rate.")
        return None
    
    #best_n_function = heapq.nlargest if evaluation_metric != 'loss' and evaluation_metric != 'val_loss' \
         # else heapq.nsmallest

    best_model_id = None
    best_evaluation_metric = float(0)
    for model_entry in stored_models:
        model = model_entry["model"]
        val = model_entry.get(evaluation_metric)  # Get evaluation_metric if provided, otherwise None

        if val is None:
            model_info = client.get_metric_history(model, evaluation_metric)
            val = model_info[0].value
            print("Model info for", model, ":", model_info)

        if val > best_evaluation_metric:
            best_evaluation_metric = val
            best_model_id = model

    print("Models rated successfully.")
    return best_model_id

def model_to_production(run_id):
    """
    Utility method for the emotion recognition module to put the best model of the latest models into production.
    
    :param run_id: The run ID of the model
    """
    with open(path.join(getcwd(), 'config.json'), 'r') as f:
        config = json.load(f)
    client = MlflowClient()
    model_version_info = client.search_model_versions(f"run_id = '{run_id}'")[0]
    client.transition_model_version_stage(config['model_name'], model_version_info.version,
                                                  stage='production', archive_existing_versions=True)
    print("Run: ", run_id, "has been transitioned to production" )


def remove_files(folder_path):
    """
    Utility method for the emotion recognition module to run through the pictures of each dataset and delete them, leaving the folders.
    
    :param folder_path: The path of the folder that it should to start in.
    """
    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error removing file: {file_path}")
                
                
