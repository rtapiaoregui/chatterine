#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 11:15:52 2019

@author: rita

Code to load a model and predict a label with it.

"""
import os
from google.cloud import storage
from sklearn.externals import joblib


def model_loader(model_pth):
    """
    Determines where a model is supposed to be found and loads it.
    """
    if os.environ.get('SERVER_TYPE', '') == 'GCP': 
        local_model = '/tmp/model.joblib'
        filename = os.path.basename(model_pth)
    
        gcs = storage.Client()
        bucket = gcs.get_bucket(os.environ.get('MODELS_STORAGE_BUCKET'))
        blob = bucket.get_blob(filename)
        blob.download_to_filename(local_model)
        model = joblib.load(local_model)
        os.remove(local_model)

    else:
        model = joblib.load(model_pth)
        
    return model



def predict(new_obs, paths, l):
    """ 
    Predicts a label and returns the score of the prediction 
    alongside the predicted label
    """
    model_pth = os.path.join(paths.get('models_pth'), l+'.joblib')  
    model = model_loader(model_pth)
 
    return model.predict(new_obs)[0], model.predict_proba(new_obs)

