# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 23:06:53 2021

@author: Kanishk Gohil
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from . import functions_basic as fb

class CCNC_Calibration(object):
    
    """
    All quantities in SI units.
    """
    
    Ms=0.13214      ## Molecular mass of (NH4)2SO4
    Mw=0.018        ## Molecular mass of H2O
    R=8.314         ## Universal (ideal) gas constant
    rhos=1770       ## Density of (NH4)2SO4
    rhow=1000       ## Density of H2O
    
    """
    Attributes
    ----------
    data : array floats
        Measured ammonium sulfate CCN data at different supersaturations.
    error_buffer : floats
        Permissible offset between set and instrument supersaturations to
        retain datapoints within the database.
    
    Methods
    -------
    postprocessing(temp_data):
        Has a local scope and is called from within 'calibration_fitting'.
        Appends the calibrated supersaturation data to measured database
    calibration_fitting():
        Fits set and instrument supersaturation datasets, plots the datasets,
        and computes fitting parameters to use with CCN analysis
    compute_kappa():
        Computes ammonium sulfate kappa to validate calibrated supersaturation.
    
    """
    
    def __init__(self, data, error_buffer):
        
        """ Initialize CCNC calibration.
        
        Parameters
        ----------
        data : array_like sequence floats
            Measured CCN data.
            1. Ammonium sulfate data if calibrating
            2. Any other species if doing CCN analysis using Kohler theory
        error_buffer : floats
            Offset for retaining set and instrument supersaturations in data.
        """
        
        self.data = data
        self.error_buffer = error_buffer
    
    def postprocessing(self, temp_data):
        
        """ Processing of the dataset containing the calibrated
        supersaturations based on an offset buffer. Retention of the datapoints
        for which the offset between set and instrument supersaturations is
        found to be under the buffer value.
        
        Parameters
        ----------
        temp_data : pandas dataframe
            The dataframw containing the measurement ammonium sulfate data.
        
        Returns
        -------
        temp_data : pandas dataframe
            The appended dataframe containing the retained calibrated
            supersaturation datapoints.
        
        """
        
        temp_data = fb.SS_correction(temp_data)
        
        for index, row in temp_data.iterrows():
            error = abs(row['Corrected supersaturation'] - row['Supersaturation']) / row['Supersaturation']
            
            if error*100. > self.error_buffer:                   ## Error threshold --> can be changed depending on requirement
                temp_data = temp_data.drop(index)
                
        temp_data.reset_index()
        temp_data.index += 1
        
        return temp_data
    
    def calibration_fitting(self):
        
        """ Fits and plots the set and intrument supersaturations.
            Performs linear regression on the set and instrument
            supersaturations for calibration.
        
        Returns
        -------
        model.coef_, mode.intercept_ : float
            Fit parameters of the linear regression fir across the set and
            instrument supersaturations.
        
        """
        
        self.data = self.postprocessing(self.data)
        
        plt.figure(figsize=(8,6))
    
        ax1 = plt.subplot(221)
        ax1.scatter(self.data['Supersaturation'], self.data['Corrected supersaturation'])
        ax1.set_xlabel('Instrument supersaturation')
        ax1.set_ylabel('Corrected supersaturation')
        
        ax2 = plt.subplot(222)
        ax2 = sns.boxplot(x='Supersaturation', y='Corrected supersaturation', data=self.data)
        ax2.set_xlabel('Instrument supersaturation')
        ax2.yaxis.label.set_visible(False)
        
        instrument = list(set(self.data['Supersaturation']))
        corrected = []
        for ss in instrument:
            measured = np.array(self.data.loc[self.data.Supersaturation == ss]['Corrected supersaturation'])
            q75, q25 = np.percentile(measured, [75 ,25])
            measured = measured[measured >= q25]
            measured = measured[measured <= q75]
            corrected.append(round(np.mean(measured[~np.isnan(measured)]), 3))
        
#       print (instrument, corrected)
        model = LinearRegression()
        model.fit(np.array(instrument).reshape(-1,1),
                  np.nan_to_num(corrected, nan=np.mean(instrument), posinf=np.mean(instrument), neginf=np.mean(instrument)))
        instrument_line = np.linspace(min(instrument), max(instrument), len(instrument))
        corrected_line = model.predict(np.array(instrument_line).reshape(-1,1))
        
        ax3 = plt.subplot(212)
        ax3.scatter(instrument, corrected, color='red')
        ax3.plot(instrument_line, corrected_line, 'b--', lw=0.6)
#        ax3.text(0.5, 0.7, 'Fit score (R$^2$) ='+str(round(r2_score(np.array(instrument_line).reshape(-1,1), corrected_line), 4)))
        corrected.sort()
        corrected_line.sort()
        
#       print (corrected, corrected_line)
        if np.isnan(np.sum(corrected)):
            nanindices = np.argwhere(np.isnan(corrected))[0]
            for index in sorted(nanindices, reverse=True):
                corrected = np.delete(corrected, index)
                corrected_line = np.delete(corrected_line, index)
        else:
            pass
        
        ax3.text(0.5, 0.7, 'Fit score (R$^2$) ='+str(round(r2_score(np.array(corrected),
                                                           np.array(corrected_line)), 4)))
        ax3.set_xlabel('Instrument supersaturation')
        ax3.set_ylabel('Corrected supersaturation')
        
        plt.show()
        
        return model.coef_, model.intercept_
    
    def compute_kappa(self):
        
        """ Computes kappa from the size-resolved measurements of the CCN
        from the critical dry diameter and critical supersaturation using the
        Kohler theory.
        
        Returns
        -------
        self.data : pandas dataframe
            Appended with the kappa column in the dataframe. Takes in the raw
            dataframe as the input.
        
        """
        
        kappas = []
        
        for index, row in self.data.iterrows():
            
            temperature = row['Temperature']
            sigma = row['Surface tension']
                
            A = (4*sigma*self.Mw)/(self.R*temperature*self.rhow)
            
            dia = round(row['Activation diameter'][0], 3)*10**-9 if isinstance(row['Activation diameter'], list) else round(row['Activation diameter'], 3)*10**-9
            s = 1 + round(row['Corrected supersaturation']*10**-2, 4)
            
            kappa = ((4*A**3)/(27*(dia**3)*(np.log(s))**2))
            
            kappas.append(round(kappa, 1))
                    
        self.data['kappa'] = pd.Series(kappas, index=self.data.index)
        
        return self.data
    
    # The following method requires work
    # Not sure what it does
    # The above methods are sufficient for calibrating the CCNC

#    def SS_compute(diameter, temperature,):
#        
#        a0 = 1.8853
#        a1 = 2.46422E-2
#        a2 =-3.69417E-4
#        a3 = 2.86862E-6
#        a4 =-8.5855E-9
#        
#        sigma = 0.072
#        temp = temperature
#        dia_b = round(diameter)      # base diameter value in the nm units
#        dia = diameter*10**-9
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
#        corrected_S = np.exp(lnS)-1.
#        
#        return round(100*(corrected_S), 3)
