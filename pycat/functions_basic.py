# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 19:56:28 2020

@author: Kanishk Gohil
"""

import math
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import islice
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression


def processing(filename, newFile, timeFile, references, timestring, delete_strings):
    
    """ This code contains all the necessary functions used in SMPS data
    processingvand CCN analysis classes. This function performs the task of
    processing the raw datafile for generating the usable datafiles in the
    main code. The filenames for the usable files and others are provided
    as arguments.
    
    Parameters
    ----------
    filename : name of the file in which unprocessed raw SMPS data is stored
    newFile : name of the file in which processed SMPS data is written
    timeFile : timestamps at which SMPS datasets were collected
    references : flags to mark lines in 'filename' between which
    data is written
    timestring : variable containing time column name in the SMPS files
    delete_strings : flags for the strings that will be deleted in the final
    dataset before writing in 'newFile'
    
    """
    
    with open(filename, 'r', encoding='utf-8',
                 errors='ignore') as My_File:
        lines = My_File.readlines()
        
    with open(filename, 'r', encoding='utf-8',
                 errors='ignore') as My_File:
        for num, line in enumerate(My_File, 1):        
            if any(keyword in line for keyword in references):
                if 'Comment' in line:               # 'Comment' is one the keywords in the file - it marks the end of the raw file
                    endLine=(num-1) 
                else:
                    startLine=num
            if (timestring in line):
                time_line = num-1
    
        time_array = lines[time_line]
        time_array = time_array.replace('\t', '\n')
    
    # writing into the file containing only the timestamps - somewhat like metadata
    with open(timeFile, 'w') as file:
        file.write(time_array)
    
    # writing into the SMPS datafile with the relevant CPC data                    
    with open(filename, 'r', encoding='utf-8',
                 errors='ignore') as My_File:    
        lines = islice(My_File, startLine-1, endLine)
        with open(newFile, 'w') as SMPS_file:
            for line in lines:    
                for string in delete_strings:    
                    line = line.replace(string, '')    
                SMPS_file.write(line)
                
def CCN_time_ref(dfCCN, timeFile):
    
    """ Alignment of the CCN and SMPS (CN) datasets
    
    Parameters
    ----------
    dfCCN : pandas dataframe
        CCN dataframe generated from the CCN input file
    timeFile : .txt file
        text file containing timestamps for data collection at every
        2.25 minutes interval
    
    Returns
    -------
    CCNtimeVals : numpy array
        array containing the timestamps from the CCN dataframe
    CCNrefIndex : int
        the reference index determined from the CCN dataframe for which
        the CCN data will be read while aligning CN and CCN datasets
    
    """
    
    dfTime = pd.read_csv(timeFile)
    CCNtimeVals = np.array(dfTime['Start Time'].tolist())
    
    references = [datetime.strptime(time, '%H:%M:%S') for time in dfCCN.Time]
    timestamps = [ref.second + ref.minute*60. +ref.hour*3600. for ref in references]

    refTime = min(timestamps)
    for index in range(len(CCNtimeVals)):
        timestamp = datetime.strptime(CCNtimeVals[index], '%H:%M:%S')
        time = timestamp.second + timestamp.minute*60. + timestamp.hour*3600.
        print ('Testing!', time, refTime)
        if time <= refTime:
            print ('We are in continue!!', type(CCNtimeVals[index]))    
            continue
        else:
            print ('We are in break!!', type(CCNtimeVals[index]))    
            CCNrefIndex = index
            break
    
    return CCNtimeVals, CCNrefIndex

def gaussian(x, amp, cen, wid):
    
    """ Applies Gaussian fit to the downscans for aligning CN and CCN datasets
        against each other
    
    Parameters
    ----------
    x : float
        Maximum parcel supersaturation
    amp, cen, wid : floats
        Gaussian parameters' - maximum height, mean, FWHM respectively
    
    Returns
    -------
    Gaussian fit to the data
    
    """
    
    return (1./ (np.sqrt(2 * np.pi * wid )).real * np.exp(-(x-cen)**2 / (2 * wid))).real

def Sigmoid(x, *A):
    
    ## This function has been replicated from SMCA_CCNAnalysis.py
    ## In case any changes need to be made, the above code should be referenced
    ## The last commented line in this function should be used if nothing else seem to work
    ## The following functional solution looks stable for now
    
    """ Applies Sigmoid fit to the size-resolved activation ratio data.
        Can fit multiple Sigmoids to the activation data.
    
    Parameters
    ----------
    x : float
        Maximum parcel supersaturation
    A* : float array
        Sigmoid parameters' - slope, midpoint position and sigmoid maximum
    
    Returns
    -------
    Sigmoid fit to the data
    
    """
    
    import numpy as np
    
    if len(A) == 3:
        
        return A[0] /(1 + np.exp(-A[2] * (x - A[1])))
    
    elif len(A) > 3:
        
        i = 0
        while (i < len(A)):
            
            if i == 0:
                y = A[i] /(1 + np.exp(-A[i+2] * (x - A[i+1])))
                i += 3
            else:
                y += (A[i] - A[i-3]) /(1 + np.exp(-A[i+2] * (x - A[i+1])))
                i += 3
                
        return y

def sigmoid_fit(diameters, activation_ratios, num_sigmoids):
    
    """ Processes the size-resolved activation ratio data for sigmoid fitting
    
    Parameters
    ----------
    diameters : float array
        diameters array
    activation_ratios : float array
        activation ratio array
    num_sigmoids : int
        number of sigmoids to fit on the activation ratio data
    
    Returns
    -------
    ratios_fit : float array
        overall Sigmoid fit to the size-resolved activation data
    
    """
    
    ratio = 1./num_sigmoids
    y_reference = np.max(activation_ratios)
    x_reference = np.max(diameters)
    slope_reference = 1.
    
    y_guess = [(i + 1)*y_reference/3 for i in range(num_sigmoids)]
    x_guess = [0.25*x_reference*(i + 1) for i in range(num_sigmoids)]
    slope_guess = [slope_reference + ((-1)**(i+1))*ratio for i in range(num_sigmoids)]
    
    p0 = []        
    for i in range(len(y_guess)):
        
        p0.append(y_guess[i])
        p0.append(x_guess[i])
        p0.append(slope_guess[i])
    
    popt, pcov = curve_fit(Sigmoid, diameters, activation_ratios, p0, maxfev=15000)
    diameters_fit = np.linspace(min(diameters), max(diameters), 101)
    ratios_fit = Sigmoid(diameters_fit, *popt)
    
    return ratios_fit

def inflection_index(X, count):
    
    """ located the first inflection points of the downscan datasets
    
    Parameters
    ----------
    X : int array
        diameters array mapped to their indices to generate an array
    count : float array
        downscan number concentrations
    
    Returns
    -------
    idInflection : int
        index of the inflection point on the number concentration array
    
    """
    
    mean = sum(X*count)/sum(count)
    sigma = (np.sqrt(sum(count*(X-mean)**2))).real/sum(count)
    
    try:
        popt, popv = curve_fit(gaussian, X, count, p0=[1,mean,sigma], maxfev=15000)
        countDeriv = np.diff(gaussian(X,*popt))
        idInflection = np.argmax(countDeriv)
        
        return idInflection
    
    except (RuntimeError, TypeError):
        
        return 9.969e36

def surface_tension_temp(T,a=241.322,b=1.26,c=0.0589,d=0.5,e=0.56917,Tc=647.096):
    
    """ temperature-dependent surface tension calculation
    
    Parameters
    ----------
    T : int array
        diameters array mapped to their indices to generate an array
    a, b, c, d, e : float, optional
        model parameters of sigma parameterization
    Tc : float, optional
        triple point temperature
    
    Returns
    -------
    sigma : float
        surface tension --> returned in SI units
    
    """
    
    """ This function returns the surface tension of water estimated as a
    function of instrument temperature. The empirical relationship between the
    temperature and surface tension has been used as prescribed by
    Kalova and Mares (2018)."""
    
    tau = 1 - T/Tc
    sigma = a*tau**b * (1 - c*tau**d- e*tau)
    
    return sigma*10**-3

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def ChargeCorrection(ratio, dia, temperature):
    
    """ applies multiple charge correction on the SMPS-derived dataset
    
    Parameters
    ----------
    ratio : int array
        activation ratio data
    dia : float array
        particle size array
    temperature : float
        temperature at which measurements were taken
    
    Returns
    -------
    ratioCorrected : float array
        corrected activation ratio data after multiple charge correction
    
    """
    
    """ This function performs multiple charge correction for the activation
    ratio data of a given sample. The charge correction is done for +2 and +3
    charged particles, specifically."""
    
    # charge correction by Gunn's method
    def charge_correction_Gunn(n, dp, e=1.6E-19, k=1.38E-23, e0=8.85E-12):
        phi = e/(4*np.pi**2*e0*k*temperature*(dp*10**-9))**0.5
        A = 2*np.pi*e0*k*temperature*(dp*10**-9)/e**2
        B = 2*A
        return phi*np.exp( -(n - A*np.log(0.875))/B )
    
    # charge correction by Fuchs' method
    def charge_correction_Fuchs(A, dp):
        return sum( [A[i]*(math.log(dp,10))**i for i in range(len(A))] )
    
    
    N = [0,1,2,3,-2,-1]         # number of possible charges on the particles
    ratioCorrected = []         # multiple charge corrected activation ratio
    
    for index in range(len(dia)):
        
        dp = dia[index]
        for n in N:
        
            if n == 1:
                
                f1 = 10**( charge_correction_Fuchs([-2.3484, 0.6044, 0.48, 0.0013, -0.1544 ,0.0320], dp) )
#                f1 = charge_correction_Gunn(n, dp)
                
            elif n == 2:
                
                f2 = 10**( charge_correction_Fuchs([-44.4756, 79.3772, -62.89, 26.4492, -5.748, 0.5059], dp) )
#                f2 = charge_correction_Gunn(n, dp)
                
            elif n == -1:
                
                f_1 = 10**( charge_correction_Fuchs([-2.3197, 0.6175, 0.6201, -0.1105, -0.126, 0.0297], dp) )
                f_1 = charge_correction_Gunn(n, dp)
                
            elif n == -2:
                
                f_2 = 10**( charge_correction_Fuchs([-26.3328, 35.9904, -21.4608, 7.0867, -1.3088, 0.1051], dp) )
                f_2 = charge_correction_Gunn(n, dp)
                
            elif n == 0:
                
                f0 = 10**( charge_correction_Fuchs([-0.0003, -0.1014, 0.3073, -0.3372, 0.1023, -0.0105], dp) )
                f0 = charge_correction_Gunn(n, dp)
                
            else:
                
                f3 = charge_correction_Gunn(n, dp)
            
#        ratioCorrected.append( f1*ratio[index] + f2*ratio[index] + f3*ratio[index] + f0*ratio[index] )
        ratioCorrected.append( ratio[index] - f1*ratio[index] - f2*ratio[index] )
        
    return ratioCorrected

def SS_correction(df, kappa_analytical=0.6046, Mw=0.018, R=8.314, rho=1000):
    
    """ calibrates CCNC supersaturation using ammonium sulfate activation data
    
    Parameters
    ----------
    df : pandas dataframe
        dataframe containing measured ammonium sulfate activation data
    kappa_analytical : float
        single-hygroscopicity parameter of ammonium sulfate
    Mw : float
        molar mass of water
    R : float
        universal gas constant
    rho : float
        density of water
    
    Returns
    -------
    df : pandas dataframe
        dataframe containing additional column for calibrated supersaturation
        to use for analyzing measurements for other aerosols
    
    """
    
    corrected_SS = []
    
    for index, row in df.iterrows():
        
        sigma = row['Surface tension']
        temp = row['Temperature']
        # This is a hokey solution to the following line - change it somehow
        dia = row['Activation diameter'][0]*10**-9 if isinstance(row['Activation diameter'], list) else row['Activation diameter']*10**-9
        
        A = (4*sigma*Mw)/(R*temp*rho)
        
        lnS = np.sqrt((4*A**3)/(27*kappa_analytical*dia**3))
        S = np.exp(lnS)-1.
        
        corrected_SS.append(round(S*100, 3))
        
    df['Corrected supersaturation'] = pd.Series(corrected_SS, index=df.index)
    
    return df

def calibration(df, params_):
    
    SS = df['Supersaturation']
    
    instrument_SS = [round(params_[0]*ss + params_[1], 3) for ss in SS]
    
    df['Instrument supersaturation'] = pd.Series(instrument_SS, index=df.index)
    
    return df

def scan_extraction(file, keywords, num=1, count=1, txtfile='AAC_SCANFILE_', extension='.txt'):
    
    # initiation of the arrays to hold the line numbers for 'SCAN' and 'END OF SCAN' keywords
    # the maximum number of SCANS possible in the datasets is 2
    # by default this function creates 2 AAC scanfiles
    # subject to change depending on further requirements
    
    scan_linenums = []
    endscan_linenums = []
    
    with open(file, 'rt') as AAC_data:
        
        for number, line in enumerate(AAC_data, 1):
            
            if any(key in line for key in keywords):
                
                endscan_linenums.append(number) if 'END OF SCAN' in line else scan_linenums.append(number)
                
        scan_linenums.sort(), endscan_linenums.sort()
        
    for index in range(len(scan_linenums)):
        
        with open(file, 'rt') as AAC_data:
                                
            scan_datalines = islice(AAC_data, scan_linenums[index], endscan_linenums[index]-1)
            scan_tmpfile = open(txtfile+str(index+1)+extension, 'w')
            for line in scan_datalines:
                
                scan_tmpfile.write(line)
            
            scan_tmpfile.close()
    
    return (len(scan_linenums))

def mean_free_path(T, P, lamb_0=67.3, T_0=296.15, P_0=101325, S=110.4):
    
    """ calculates mean free path of the aerosol particles
    
    Parameters
    ----------
    T : float
        measurement temperature
    P : float
        measurement pressure
    lamb_0 : float
        reference mean free path
    T_0 : float
        reference temperature
    P_0 : float
        reference pressure
    S : float
        Sutherland constant of air
    
    Returns
    -------
    mean_free_path : float
        mean free path at T and P
    
    """
    
    return (lamb_0 * ((T_0 + S) / (T + S)) * (P_0 / P) * ((T / T_0)**2))

def viscosity(T, mu_0=1.83245E-5, T_0=296.15, S=110.4):
    
    """ calculates viscosity of the air
    
    Parameters
    ----------
    T : float
        measurement temperature
    mu_0 : float
        reference viscosity
    T_0 : float
        reference temperature
    S : float
        Sutherland constant of air
    
    Returns
    -------
    viscosity : float
        viscosity at T
    
    """
    
    return (mu_0 * ((T_0 + S) / (T + S)) * ((T / T_0)**1.5))

def diffusivity(T, C, mu, d, k=1.38E-23):
    
    """ calculates diffusivity of the aerosol particles
    
    Parameters
    ----------
    T : float
        measurement temperature
    mu : float
        viscosity
    d : float
        particle size
    k : float
        Boltzmann's constant
    
    Returns
    -------
    diffusivity : float
        diffusivity at T
    
    """
    
    return k * T * C / (3 * np.pi * mu * d)
            
#def calibration_ss(df, Ms=0.13214, Mw=0.018, R=8.314, rhow=1000, rhos=1770):
#    
#    a0 = 1.8853
#    a1 = 2.46422E-2
#    a2 =-3.69417E-4
#    a3 = 2.86862E-6
#    a4 =-8.5855E-9
#    
#    corrected_S = []
#    
#    for index, row in df.iterrows():
#        sigma = row['Surface tension']
#        temp = row['Temperature']
#        dia_b = row['Activation diameter']      # base diameter value in the nm units
#        dia = row['Activation diameter']*10**-9
#        
#        A = (4*sigma*Mw)/(R*temp*rhow)
#        
#        vhf = a0 +\
#              a1 * dia_b +\
#              a2 * dia_b**2 +\
#              a3 * dia_b**3 +\
#              a4 * dia_b**4
#        kappa_analytical = (vhf*rhos*Mw)/(rhow*Ms)
#        
#        lnS = np.sqrt((4*A**3)/(27*kappa_analytical*dia**3))
#        S = np.exp(lnS)-1.
#        
#        corrected_S.append(round(S*100, 3))
#        
#    df['Corrected supersaturation'] = pd.Series(corrected_S, index=df.index)
#    
#    return df
