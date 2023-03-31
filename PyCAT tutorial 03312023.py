#!/usr/bin/env python
# coding: utf-8

# # Importing all the relevant libraries

# In[6]:


import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Directory where pycat is located
sys.path.append(r'E:\My data\Ph.D. research\CCN\CCN test\CCN_calibration\CCN Analysis packet\GitHub upload\PyCAT')

import pycat


# # Locating the directory of calibration data - set the directory where the calibration data is saved on your machine

# In[7]:


calibration_dir = r'E:\My data\Ph.D. research\CCN\AAC\Data\BC and SMPS 3082 testing\SMPS 3082 CCN measurements\3080_3082Data\3080'


# # Feeding in the calibration SMPS data for number size distributions and metadata repositories - provide the name of the SMPS file in the ``filename`` field. You can change the names of ``dataFile`` and ``timeFile`` variables if you want to

# In[9]:


filename = '3080_AS_Cal_03Aug2022.txt'

dataFile = 'timeData.txt'     # metadata for the SMPS timeseries data
timeFile = 'timeVal.txt'      # metadata for the timestamps of the upscan start points


# # Calibration execution using the SMPS datafile

# In[10]:


# ``offset`` variable is for manually correcting for any deviation between N_CN and N_CCN curves
# ``offset`` moves the CCN curve with respect to the CN curve
# (+)ve offset --> moves the CCN curve backward
# (-)ve offset --> moves the CCN curve forward
data_generator = pycat.SMPS_data_process(calibration_dir, filename, dataFile, timeFile, offset=1, downscan='no')

# Set the starting and end diameter size between which you want to obtain the size distribution and activation data
# ``flow_rate`` correction might be required - default value is 1.0
preprocessed_data, calibration_sheet, calibration_fit =                                data_generator.data_generation(start_dia=15., end_dia=155., flow_rate=1.)

## argument 2 --> %age error threshold for selecting out the calculated supersaturation
calibrator_class = pycat.CCNC_Calibration(preprocessed_data, 20.)


# # Model fitting parameters for CCNC calibration curve

# In[11]:


# Calculation of the linear fit parameters between instrument and calibrated supersaturations
slope_calib, intercept_calib = calibrator_class.calibration_fitting()
calib_fit_params = [slope_calib[0], intercept_calib]


# # Preparing a database with the calibration metadata

# In[12]:


# This part is just for sanity check - can be completely ignored
calib_final_data = calibrator_class.compute_kappa()


# # Writing the calibration data in the excel file

# In[14]:


calib_dir = 'calib_dir'
try:
    # Making a new save directory for storing calibration data, if the directory does not already exist
    # Changing into the save directory to actually save the file
    os.chdir(os.mkdir(os.path.join(calibration_dir, calib_dir), mode=0o666))
except:
    # If the calibration save directory already exists
    # Change into the save directory to overwrite the calibration datafile
    os.chdir(os.path.join(calibration_dir, calib_dir))

data_writer = pd.ExcelWriter('Calibration.xlsx', engine='xlsxwriter')

preprocessed_data.to_excel(data_writer, sheet_name='Calibration data')
calibration_sheet.to_excel(data_writer, sheet_name='Calibration - activation data')

data_writer.save()


# ========================================================================================================================
# 
# ========================================================================================================================

# # The main code after CCNC calibration begins here!!

# In[16]:


analysis_dir = r'E:\My data\Ph.D. research\CCN\AAC\Data\BC and SMPS 3082 testing\SMPS 3082 CCN measurements\3080_3082Data\3080'

filebase = '3080_Sucrose_03Aug2022'

analysis_file = filebase+'.txt'                      # raw datafile

analysis_CPC_data = 'SMPS_'+filebase+'.txt'          # metadata for the SMPS timeseries data
analysis_timestamps = 'Timestamps_'+filebase+'.txt'  # metadata for the timestamps of the upscan start points


# # CCN activity analysis; supersaturation correction with the calibration data

# In[17]:


## 'offset' is an important parameter --> see the description in the calibration part of the process
## 'sigmoids' is the number of sigmoids --> default = 1
SMPS_analysis_gen = fb.SMPS_data_process(analysis_dir, analysis_file, analysis_CPC_data,
                                         analysis_timestamps, offset=0, sigmoids=1, downscan='yes')
analysis_data, analysis_sheet, analysis_fit = SMPS_analysis_gen.data_generation(start_dia=20, end_dia=150., flow_rate=1.05)

## Instrument (calibrated) supersaturation estimation
analysis_data = fb.calibration(analysis_data, calib_fit_params)


# # Writing the analysis data in the excel file

# In[20]:


analysis_save = 'analysis_dir'
try:
    # Making a new save directory for storing analysis data, if the directory does not already exist
    # Changing into the save directory to actually save the file
    os.chdir(os.mkdir(os.path.join(analysis_dir, analysis_save), mode=0o666))
except:
    # If the analysis save directory already exists
    # Change into the save directory to overwrite the analysis datafile
    os.chdir(os.path.join(analysis_dir, analysis_save))

data_writer = pd.ExcelWriter(filebase+'.xlsx', engine='xlsxwriter')

# Size-resolved activation ratio and sigmoid fitting
analysis_sheet.to_excel(data_writer, sheet_name=filebase+' - ratio')
analysis_fit.to_excel(data_writer, sheet_name=filebase+' - fit')

# CCN analysis using CCN activation data

analysis_data.to_excel(data_writer, sheet_name=filebase+' - data')

data_writer.save()


# In[ ]:




