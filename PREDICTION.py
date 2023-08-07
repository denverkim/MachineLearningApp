# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 02:03:47 2023

@author: USER
"""

import joblib
def predict(data):
    rf = joblib.load("rf_model.sav")
    return rf.predict(data)