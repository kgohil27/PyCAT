# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:56:51 2021

@author: Kanishk Gohil
"""

import numpy as np

import functions_basic as fb

from sklearn.linear_model import LinearRegression

class TF_tool(object):
    
    ac=2.33
    bc=0.966
    cc=0.4985
    
    Leff = 46       # in m, for AAC
    
    Qa = 0.872      # in lpm
    
    a = -1.73
    b = -0.0316
    c = 1.999
    
    def __init__(self, d, T, P, Qs):
        
        self.d  = d
        self.T  = T
        self.P  = P
        self.Qs = Qs
    
    def slip_correction(self):
        
        self.lamb = fb.mean_free_path(self.T, self.P)
        
        self.C = (1 + (self.lamb / self.d)*(self.ac + self.bc*np.exp(-self.cc*self.d / self.lamb)))
        
        return self.C
    
    def counting_efficiency(self):
        
        """
        Counting efficiency of the AAC-CPC coupling
        Tavakoli and Olfert (2014)
        Resembles a logarithmic distribution
        Approximate using a linear plot for simplicity - needs some discussion!!
        """
        
        # Aerosol size
        X = np.linspace(95, 910, 51)        # particle size in nm
        
        # Corresponding counting efficiency
        y = np.linspace(10, 42, 51)         # efficiency in %age
        
        model = LinearRegression().fit(X.reshape((-1, 1)), y)
        
        slope, intercept = model.coef_[0], model.intercept_
        
        return (slope*self.d + intercept) / 100
    
    def logDlogtau(self):
        
        d = self.d*1E-9
        lamb = fb.mean_free_path(self.T, self.P) * 1E-9
        
        K = (2*d + self.ac*lamb +\
             self.bc*lamb*(1 - self.cc*d/lamb)*np.exp(-self.cc*d/lamb))**-1
        
        dlDdltau = self.slip_correction() * d * K
        
        return dlDdltau
    
    def beta_NI(self):
        
        self.mu = fb.viscosity(self.T)
        
        D = fb.diffusivity(self.T, self.slip_correction(), self.mu, self.d*1E-9)
        
        delta = self.Leff * D / (self.Qa*1.66667E-5)
        
        if delta >= 0.007:
            lamb_d = 0.819*np.exp(-11.5 * delta) + \
                     0.0975*np.exp(-70.1 * delta) + \
                     0.0325*np.exp(-179 * delta)
        elif delta < 0.007:
            lamb_d = 1 - 5.5*delta**(2/3) + 3.77*delta + 0.814*delta**(4/3)
        
        lamb_e = 0.8
        
        lamb_omega = lamb_d * lamb_e                    ## Transmission efficiency
        
        mu_omega = self.a*(self.d**self.b) + self.c     ## Transfer function width factor
        
        beta = self.Qa / self.Qs
        
        beta_ = lamb_omega * mu_omega * ( np.log((1 + beta/mu_omega) / (1 - beta/mu_omega)) + \
                                        ( np.log(1 - (beta/mu_omega)**2) )*mu_omega/beta )
        
        return beta_

class AAC_TF_analysis(object):
    
    def __init__(self, aac_scan):
        
        self.aac_scan = aac_scan
    
    def CN_number_conc(self):
        
        df = self.aac_scan
        
        ## AAC analysis starts from this cell
        ## AAC_CCN code needs to run first to get the required data
        ## Further analysis to be made
        
        # Comparison between measured and calculated relaxation times (tau)
        
        columns = df.columns
                
        aac_dia = np.array(df[columns[1]])      # aerodynamic sizes
        aac_conc = np.array(df[columns[3]])     # CN number concentration (dNdlogDa (#/cc))
        aac_tau = np.array(df[columns[4]])      # measured relaxation time
        aac_spd = np.array(df[columns[15]])     # rotational speed of AAC cylinder
        aac_sheath = np.array(df[columns[16]])  # sheath flowrate
        aac_tmp = np.array(df[columns[17]])     # temperature inside AAC cylinder
        aac_prs = np.array(df[columns[18]])     # pressure across AAC cylinder
        
        CN_conc = []
        
        for index in range(len(aac_dia)):
            
            dia = aac_dia[index]
            
            T = aac_tmp[index] + 273.15
            P = aac_prs[index]
            
            Qsh = aac_sheath[index]
            
            dNdlogD = aac_conc[index]               # measured number concentration
            
            analyzer = TF_tool(dia, T, P, Qsh)
            
#            Cc = analyzer.slip_correction()         # Cunningham's slip correction factor
            
            eta = analyzer.counting_efficiency()    # counting efficiency
            
            dlogDdlogtau = analyzer.logDlogtau()    # empirical variable
            
            beta_ = analyzer.beta_NI()              # non-ideal beta
            
            N = dNdlogD * eta * dlogDdlogtau * beta_ / np.log(10)
            
            CN_conc.append(N)
        
        return CN_conc
      
