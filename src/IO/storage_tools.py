# https://cloud.google.com/storage/docs/reference/libraries
# https://googleapis.dev/python/storage/latest/client.html

import logging

import joblib
from google.api_core.exceptions import NotFound
from google.cloud import storage
import os
import time
import datetime

def create_bucket(bucket_name):
    log = logging.getLogger()

    storage_client = storage.Client()
    if bucket_name not in [x.name for x in storage_client.list_buckets()]:
        bucket = storage_client.create_bucket(bucket_name)

        log.info("Bucket {} created".format(bucket.name))
    else:
        log.info("Bucket {} already exists".format(bucket_name))


def upload_file_to_bucket(model_file_name, bucket_name):
    log = logging.getLogger()
    log.warning(f'uploading {model_file_name} to {bucket_name}')
    client = storage.Client()
    b = client.get_bucket(bucket_name)
    blob = storage.Blob(model_file_name, b)
    with open(model_file_name, "rb") as model_file:
        blob.upload_from_file(model_file)


def delete_model(ticker, bucket_name):
    client = storage.Client()
    b = client.get_bucket(bucket_name)
    blob = storage.Blob(f'{ticker}.pkl', b)
    blob.delete()

def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)

def get_model_from_bucket(model_filename, bucket_name):
    log = logging.getLogger()
    client = storage.Client()
    b = client.get_bucket(bucket_name)
    blob = storage.Blob(f'{model_filename}', b)
    try:
        with open(f'{model_filename}', 'wb') as file_obj:  # critical resource should use tempfile...
            print(model_filename)
            print('****')
            print(time.ctime(os.path.getmtime(model_filename)))
            print(modification_date(model_filename))
            print("&&&&&&")
            client.download_blob_to_file(blob, file_obj)
        with open(f'{model_filename}', 'rb') as file_obj:
            model = joblib.load(file_obj)
    except NotFound as e:
        log.warning(f'model {model_filename} not found\n')
        model = None

    return model

# def get_model_from_bucket(model_filename, bucket_name):
#     log = logging.getLogger()
#     client = storage.Client()
#     b = client.get_bucket(bucket_name)
#     blob = storage.Blob(f'{model_filename}', b)
#     try:
#         with open(f'{model_filename}', 'wb') as file_obj:  # critical resource should use tempfile...
#             print(model_filename)
#             print('****')
#             print((datetime.datetime.now() - modification_date(model_filename)).days)
#             client.download_blob_to_file(blob, file_obj)
#             if ((datetime.datetime.now() - modification_date(model_filename)).days) > 1:
#                 delete_model(model_filename, bucket_name)
#                 print('@@@')
#             else:
#                 with open(f'{model_filename}', 'rb') as file_obj:
#                     model = joblib.load(file_obj)
#     except NotFound as e:
#         log.warning(f'model {model_filename} not found\n')
#         model = None
#
#     return model