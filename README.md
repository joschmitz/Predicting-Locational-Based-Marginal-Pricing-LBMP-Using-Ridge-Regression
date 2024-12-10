To uses this code you can download the datasets from NYISO or you can start with the cleaned dataset I have included for the one New York City location (61761_Real_Time.csv).
Prepare_ISO_data.ipynb will prepare the real time data you download from NYISO.
Prepare_V_data.ipynb will prepare the data from NYISO to verify and analyze the results of the model.
Main.ipynb will generate the RRMP model.
You may also just use the RRmP_mdoel.pkl and RRMP_scaler.pkl to generate prediction without having to run the creation code. Just use this code to 
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

# Load your saved Ridge regression model
ridge_model = joblib.load('RRMP_model.pkl')

# Load the scaler used during training
scaler = joblib.load('RRMP_scaler.pkl')

This model was presented at NAPS conference in El Paso, TX October 14, 2024.  It is published here: https://ieeexplore.ieee.org/abstract/document/10741722
