import os
from os import chdir, listdir, path, getcwd, makedirs, remove
import shutil
from shutil import rmtree as remove_directory
from threading import Thread
from zipfile import ZipFile
from PIL import Image
from io import BytesIO
import json
import uuid

from flask import Response, request, abort, jsonify, send_file, after_this_request
from werkzeug.utils import secure_filename

from logic.emotionrecognition.pipelines import build_emotion_recognition_pipeline
from logic.emotionrecognition.__utilities import rate_fetched_models, model_to_production, remove_files


# region Training
PRINT_TENSORFLOW_INFO = True


class TrainingPipelineSingleton:
    _is_training: bool = False
    _thread = None
    pipeline = None

    def invoke(self):
        if self.pipeline is None:
            self._load_pipeline()

        if not self._is_training:
            self._thread = Thread(target=self._invoke_async, daemon=True)
            self._thread.start()

    def _load_pipeline(self):
        with open(path.join(getcwd(), 'config.json'), 'r') as f:
            self.pipeline = build_emotion_recognition_pipeline(config=json.load(f))

    def _invoke_async(self):
        self._is_training = True

        if PRINT_TENSORFLOW_INFO:
            self._print_tensorflow_info()
        self.pipeline.run()

        self._is_training = False

    @staticmethod
    def _print_tensorflow_info():
        import tensorflow as tf

        print("Version of Tensorflow: ", tf.__version__)
        print("Cuda Availability: ", tf.test.is_built_with_cuda())
        print("GPU  Availability: ", tf.config.list_physical_devices('GPU'))
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


_training_pipeline = TrainingPipelineSingleton()


def _train_model():
    # Make asynchronous pipeline call
    _training_pipeline.invoke()

    return Response(response='Pong!', status=200)
# endregion


# region Calibration
def is_file_allowed(fname: str):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() == 'zip'

def _calibrate_model():
    # Check if user id defined
    if 'user_id' not in request.args:
        return abort(400, 'No user specified.')

    # Check if files present in request
    if len(request.files) == 0:
        return abort(400, 'No files uploaded.')

    if 'zipfile' not in request.files:
        return abort(400, 'Different file expected.')

    # Check if file is a valid zip
    file = request.files['zipfile']

    if file.filename == '':
        return abort(400, 'No file selected.')
    if not is_file_allowed(file.filename):
        return abort(400, 'File type not allowed. Must be a zip.')

    # Save zip file
    file_name = secure_filename(file.filename)
    folder_path = path.join(getcwd(), 'uploads', request.args.get('user_id'))
    makedirs(folder_path, exist_ok=True)

    file_path = path.join(folder_path, file_name)
    file.save(file_path)

    # Unpack zip file
    with ZipFile(file_path, 'r') as zipfile:
        zipfile.extractall(folder_path)

    # Call calibration upon folder with the given user id
    with open(path.join(getcwd(), 'config.json'), 'r') as f:
        config = json.load(f)
        config['data_directory'] = folder_path

        pipeline = build_emotion_recognition_pipeline(config=config, do_calibrate_base_model=True)
        pipeline.run(run_name=request.args.get('user_id', '', str))

    # Remove data folder
    remove_directory(config['data_directory'], ignore_errors=True)

    return jsonify(success=True)
# endregion

# region snapshot uploading
def _snapshot_upload():
    # Check if snapshot is a valid
    file = request.files['snapshot']
    
    unique_identifier = str(uuid.uuid4())
    # Save snapshot
    file_name = secure_filename(file.filename)
    folder_path = path.join(getcwd(), 'uploads', request.args.get('user_id'))
    makedirs(folder_path, exist_ok=True)

    # Convert blob to image
    img = Image.open(BytesIO(file.read()))
    
    img = img.convert('RGB')

    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    for emotion in emotions:
        directory = path.join(folder_path, emotion)
        makedirs(directory, exist_ok=True)
        
    # Create a directory for each correction value
    correction_directory = path.join(folder_path, request.args.get('emotion'))
    makedirs(correction_directory, exist_ok=True)
    img_name = f"{unique_identifier}_{file_name}.jpg"

    # Save the image in the correction-specific directory
    img.save(path.join(correction_directory, img_name), format='JPEG')
    
    print("User ID:", request.args.get('user_id'))
    print("Emotion:", request.args.get('emotion'))
    
    return jsonify(success=True)
# endregion

#region datasets
def download_dataset():
    user_id = request.args.get('user_id')
    dataset_folder = path.join(getcwd(), 'uploads', user_id)
    zip_filename = f"{user_id}_dataset"
    owd = os.getcwd()
    
    # Remove original zip
    existing_zips = [f for f in listdir(dataset_folder) if f.endswith('.zip')]
    for existing_zip in existing_zips:
        remove(path.join(dataset_folder, existing_zip))
    
    # Change directory and make the zipfile
    chdir(dataset_folder) 
    shutil.make_archive(zip_filename, 'zip', '.')

    # Specify the full path to the zip file
    zip_file_path = f"{dataset_folder}/{zip_filename}.zip"

    # After the request has been made, delete the pictures.
    @after_this_request
    def remove_dataset_folder(response):
        remove_files(dataset_folder)
        os.chdir(owd)
        return response

    return send_file(zip_file_path, as_attachment=True)


def upload_and_extract_zip(file, user_id):
    if file.filename == '':
        return abort(400, 'No file selected.')
    if not is_file_allowed(file.filename):
        return abort(400, 'File type not allowed. Must be a zip.')

    file_name = secure_filename(file.filename)
    folder_path = path.join(getcwd(), 'uploads', user_id)
    makedirs(folder_path, exist_ok=True)

    file_path = path.join(folder_path, file_name)
    file.save(file_path)

    # Unpack zip file
    with ZipFile(file_path, 'r') as zipfile:
        zipfile.extractall(folder_path)

    return folder_path

def load_dataset():
    file = request.files['zipfile']
    user_id = request.args.get('user_id')

    # Use helper function upload_and_extract_zip
    upload_and_extract_zip(file, user_id)
    return jsonify(success=True)

#end region

#region pipeline
def _pipeline_run():
    file = request.files['zipfile']
    user_id = request.args.get('user_id')
    
    
    with open(path.join(getcwd(), 'config.json'), 'r') as f:
        config = json.load(f)
        
        # Run old model - old data
        pipeline = build_emotion_recognition_pipeline(config=config)
        pipeline.run(run_name=str(1))
        
        folder_path = upload_and_extract_zip(file, user_id)
        config['data_directory'] = folder_path
        
        # Run new model - old data
        pipeline = build_emotion_recognition_pipeline(config=config, do_calibrate_base_model=True)
        pipeline.run(run_name=str(2))
        
        # Run new model - new data
        pipeline = build_emotion_recognition_pipeline(config=config)
        pipeline.run(run_name=str(3))
        
        # Rate model
        best_rated_model = rate_fetched_models()
        print("Best Rated Model:", best_rated_model)

        # Put best model in production
        model_to_production(best_rated_model)
        
    # Remove data folder
    remove_directory(config['data_directory'], ignore_errors=True)
        
        
    return jsonify(success=True)

#endregion

    
# Flask route handler
def add_as_route_handler(app):
    global _train_model, _calibrate_model, _snapshot_upload, _pipeline_run, download_dataset, load_dataset

    _train_model = app.route('/model/train', methods=['POST'])(_train_model)
    _calibrate_model = app.route('/model/calibrate', methods=['POST'])(_calibrate_model)
    _snapshot_upload = app.route('/snapshot/upload', methods=['POST'])(_snapshot_upload)
    _pipeline_run = app.route('/snapshot/train', methods=['POST'])(_pipeline_run)
    download_dataset = app.route('/download/dataset', methods=['GET', 'POST'])(download_dataset)
    load_dataset = app.route('/upload/dataset', methods=['POST'])(load_dataset)

    return app
