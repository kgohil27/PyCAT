# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:56:51 2021
@author: Kanishk Gohil
"""

import numpy as np

from . import functions_basic as fb

from datetime import datetime
from sklearn.linear_model import LinearRegression

from scipy.optimize import curve_fit
from scipy.signal import find_peaks

class TF_tool(object):
    
    """
    Attributes
    ----------
    d, T, P, Qs : array floats
        Measured aerodynamic diameters, temperatures, pressures and sheath
        flow rates of the AAC-CCN measurement dataset at the given
        instrument supersaturation.
    
    Methods
    -------
    slip_correction():
        Computes the Cunningham's slip correction factor for the given set of
        d, T, P, Qs.
    counting_efficiency():
        Computes the size-dependent counting efficiency of the Condensation
        Particle Counter (CPC) instrument. Above ~40nm particle size, the
        counting efficiency is ~100%.
    logDlogtau():
        Computes the rate of change of aerodynamic diameter with the particle
        relaxation time - computed on the semi-log scale.
    beta_NI():
        Computes the transfer function parameter to incorporate non-ideal
        particle behavior.
    beta_I():
        Computes the transfer function parameter to incorporate ideal particle
        behavior - beta_I is not called for analysis.
    
    """
    
    ac=2.33
    bc=0.966
    cc=0.4985
    
    Leff = 46       # in m, for AAC
    
    Qa = 0.872      # in lpm
    
    a = -1.73
    b = -0.0316
    c = 1.999
    
    def __init__(self, d, T, P, Qs):
        
        """ Initializes transfer function analysis.
        
        Parameters
        ----------
        d : float
            Particle aerodynamic diameter.
        T : floats
            Temperature of the measurement.
        P : floats
            Pressure of the measurement.
        Qs : floats
            Sheath flow rate reported for the measurement on the AAC.
        """
        
        self.d  = d
        self.T  = T
        self.P  = P
        self.Qs = Qs
    
    def slip_correction(self):
        
        """ Computes the Cunningham's slip correction of the particle at the
        given temperature and pressure.
        
        Returns
        -------
        self.C : float
            Cunningham's slip correction factor value.
        
        """
        
        self.lamb = fb.mean_free_path(self.T, self.P)
        
        self.C = (1 + (self.lamb / self.d)*(self.ac + self.bc*np.exp(-self.cc*self.d / self.lamb)))
        
        return self.C
    
    def counting_efficiency(self):
        
        """
        Counting efficiency of the AAC-CPC coupling
        Based on Wiedensohler et al (1997)
        """
        
        """ Computes size-dependent counting efficiency of the counter
        instrument at different aerodynamic diameter.
        
        Returns
        -------
        eta : float
            Counting efficiency of the Condensation Particle Counter (CPC).
        
        """
        
        # Aerosol size
        dia = self.d                        # particle size in nm
        
        # Corresponding counting efficiency
        a = 1.17
        b = 1.0
        D1 = 7.5
        D2 = 2.0
        eta = b - a*(1 + np.exp((dia - D1) / D2))**-1         # efficiency in %age
                
        return eta
    
    def logDlogtau(self):
        
        """ Computes rate of change of aerodynamic diameter with respect to
        particle relaxation time. This quantity is necessary for derivation of
        particle counts from lognormal number distributions.
        
        Returns
        -------
        dlDdltau : float
            Log-log rate of change of the aerodynamic diameter with respect to
            the particle relaxation time.
        
        """
        
        d = self.d*1E-9
        lamb = fb.mean_free_path(self.T, self.P) * 1E-9
        
        K = (2*d + self.ac*lamb +\
             self.bc*lamb*(1 - self.cc*d/lamb)*np.exp(-self.cc*d/lamb))**-1
        
        dlDdltau = self.slip_correction() * d * K
        
        return dlDdltau
    
    def beta_NI(self):
        
        """ Computes transfer function beta parameter to incorporate non-ideal
        particle behavior.
        
        Returns
        -------
        beta_ : float
            Transfer function beta parameter that is based on the transmission
            efficiency and transfer function width factor.
        
        """
        
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
    
    def beta_I(self):
        
        """ Computes transfer function beta parameter to incorporate ideal
        particle behavior.
        
        Returns
        -------
        beta_ : float
            Transfer function beta parameter that does not explicitly account
            for the transmission efficiency and transfer function width factor.
        
        """
        
        beta = self.Qa / self.Qs
        
        beta_ = np.log((1 + beta) / (1 - beta)) + (np.log(1 - beta**2))/beta
        
        return beta_

class AAC_data_derivation(object):
    
    """
    Computes CN and CCN number concentrations from raw measurement data.
    
    Attributes
    ----------
    aac_scan : array floats
        Measured AAC-derived raw CN number concentration data.
    ccn_scan : array floats
        Measured raw CCN number concentration data.
    
    Methods
    -------
    CN_number_conc():
        Computes the CN number concentration using the raw measurement data
        by performing transfer function analysis.
    CCN_number_conc(time_col, conc_col, temp_col, ss_col, offset):
        Computes the CCN number concentration using the raw CCNC measurements.
        Number concentration derivation based on the CN number concentrations.
    
    """
    
    def __init__(self, aac_scan, ccn_scan):
        
        """ Initializes AAC data derivation. 'TF_tool' is instantiated from
        within this class to derive the number counts.
        
        Parameters
        ----------
        aac_scan : floats array
            Raw measurement data containing particle count from the AAC-CCN
            size-resolved measurements.
        ccn_scan : floats array
            Raw measurement data containing particle count from the CCNC.
        """
        
        self.aac_scan = aac_scan
        self.ccn_scan = ccn_scan
    
    def CN_number_conc(self):
        
        """ Computes CN number concentration from the raw measurement data.
        
        Returns
        -------
        CN_conc : float array
            CN concentration derived from the raw size-resolved AAC-based
            number concentration measurements.
        """
        
        aac_df = self.aac_scan
        
        ## AAC analysis starts from this cell
        ## AAC_CCN code needs to run first to get the required data
        ## Further analysis to be made
        
        # Comparison between measured and calculated relaxation times (tau)
        
        columns = aac_df.columns
                
        aac_dia = np.array(aac_df[columns[1]])      # aerodynamic sizes
        aac_conc = np.array(aac_df[columns[3]])     # CN number concentration (dNdlogDa (#/cc))
        aac_tau = np.array(aac_df[columns[4]])      # measured relaxation time
        aac_spd = np.array(aac_df[columns[15]])     # rotational speed of AAC cylinder
        aac_sheath = np.array(aac_df[columns[16]])  # sheath flowrate
        aac_tmp = np.array(aac_df[columns[17]])     # temperature inside AAC cylinder
        aac_prs = np.array(aac_df[columns[18]])     # pressure across AAC cylinder
        
        CN_conc = []
        
        for index in range(len(aac_dia)):
            
            dia = aac_dia[index]
            
            T = aac_tmp[index] + 273.15
            P = aac_prs[index]
            
            Qsh = aac_sheath[index]
            
            dNdlogD = aac_conc[index]               # measured number concentration
            
            analyzer = TF_tool(dia, T, P, Qsh)
            
#            Cc = analyzer.slip_correction()         # Cunningham's slip correction factor
            
            eta = analyzer.counting_efficiency()    # counting efficiency - CPC 3776 standard
#            eta = 1.                                 # counting efficiency
            
            dlogDdlogtau = analyzer.logDlogtau()    # empirical variable
            
            beta_ = analyzer.beta_NI()              # non-ideal beta
#            beta_ = analyzer.beta_I()               # ideal beta
            
            N = dNdlogD * eta * dlogDdlogtau * beta_ / np.log(10)
            
            CN_conc.append(N)
        
        return CN_conc
    
    def CCN_number_conc(self, time_col = 'Time', conc_col = 'CCN Number Conc',
                        temp_col='T1 Read', ss_col='Current SS', offset=1):
        
        """ Computes CN number concentration from the raw measurement data.
        
        Parameters
        ----------
        time_col : string
            Name of the column containing the timestamps of the measurements.
        conc_col : string
            Name of the column containing the lognormal number concentrations
            of the measurements.
        temp_col : string
            Name of the column containing the temperatures of the measurements.
        ss_col : string
            Name of the column containing the instrument supersaturations
            of the measurements.
        offset : float
            Offset between the times at which CN and CCN measurements reported.
        
        Returns
        -------
        CCN_distribution : float array
            Number size distributions of the CCN.
        temp : float array
            Temperature array of the number concentration measurements.
        SS_array : float array
            Array containing the set of the instrument supersaturations.
        """
        
        aac_df = self.aac_scan
        ccn_df = self.ccn_scan
        
        standard_time = 15.             # the average time difference between 2 consecutive AAC readings
    
        CCN_distribution = []           # initialization of the CCN density array
        temperatures = []
        SS_array = []
        
        for index in range(len(aac_df)):
            
            # time of the starting measurement
            start_time = np.array(aac_df[time_col])[index-1] if index > 0 else np.array(aac_df[time_col])[0]
            
            # time of the previous measurement
            end_time = np.array(aac_df[time_col])[index] if index > 0 else np.array(aac_df[time_col])[0]
            
            timestamp_start = datetime.strptime(start_time, '%H:%M:%S')
            timestamp_end = datetime.strptime(end_time, '%H:%M:%S')
            
            timevalue_start = timestamp_start.second + \
                              timestamp_start.minute*60. + \
                              timestamp_start.hour*3600.
            timevalue_end = timestamp_end.second + \
                            timestamp_end.minute*60. + \
                            timestamp_end.hour*3600.
            
            timevalue_difference = timevalue_end - timevalue_start
            
            """ CCN number count will be estimated using the quantities derived above.
            The relevent datapoints in the CCN dataframe will be deduced with the help
            of the corresponding times. The 'start time' for every iteration will
            be extracted out of the AAC dataframe."""
            
            CCN_times = np.array(ccn_df[time_col])
            CCN_density = np.array(ccn_df[conc_col])
            
#            if index == 0:
            
            CCN_start_index = CCN_times.tolist().index(start_time)      # the start index for the given iteration
            CCN_count = CCN_density[CCN_start_index + offset*int(standard_time)]
            CCN_distribution.append(CCN_count)                          # CCN number distribution for the measurement timestamp
            
            temperatures.append(np.array(ccn_df[temp_col])[CCN_start_index])
            SS_array.append(np.array(ccn_df[ss_col])[CCN_start_index])
            
#            else:
#                
#                CCN_start_index = CCN_times.tolist().index(start_time)      # the start index for the given iteration
#                CCN_end_index = CCN_times.tolist().index(end_time)          # the end index for the given iteration
#                CCN_count = np.sum(CCN_density[CCN_start_index:CCN_end_index])*standard_time/timevalue_difference
#                CCN_distribution.append(CCN_count)                          # CCN number distribution for the measurement timestamp
        
        temp = np.mean(temperatures) + 273.15
        
        return np.array(CCN_distribution), temp, fb.surface_tension_temp(temp), np.mean(SS_array)

class uncertainty_AAC_data(object):
    
    """
    Computes uncertainties in CN and CCN number concentrations from raw
    measurement data.
    
    Attributes
    ----------
    diameter : array floats
        Aerodynamic diameters
    dry_count : array floats
        CN number concentration data.
    droplet_count : array floats
        CCN number concentration data.
    ratio : array floats
        Size-resolved activation ratio data.
    
    Methods
    -------
    count_uncertainty():
        Computes the uncertainties in the CN and CCN number concentrations.
    dia_uncertainty(aac_scan):
        Computes the aerodynamic diameters.
    
    """
    
    def __init__(self, diameter, dry_count, droplet_count, ratio):
        
        """ Initializes AAC data uncertainty analysis.
        
        Parameters
        ----------
        diameter : floats array
            Aerodynamic diameters.
        dry_count : floats array
            Dry particle counts.
        droplet_count : floats array
            Droplet counts.
        ratio : floats array
            Size-resolved activation ratio counts.
        """
        
        self.diameter = diameter
        self.dry_count = dry_count
        self.droplet_count = droplet_count
        self.ratio = ratio
    
    def count_uncertainty(self):
        
        """ Computes uncertainty in the dry particle counts, droplet counts
        and size-resolved activation ratio.
        
        Returns
        -------
        error_CN : float array
            Uncertainties in the dry particle counts.
        error_CCN : float array
            Uncertainties in droplet counts.
        error_ratio : float array
            Uncertainties in the size-resolved activation ratios.
        """
        
        # flow rate uncertainties
        err_Q_cn = 0.04
        err_Q_ccn = 0.05
        
        error_CN2 = [1/val + err_Q_cn**2 for val in self.dry_count]
        error_CCN2 = [1/val + err_Q_ccn**2 for val in self.droplet_count]
        
        error_CN = [np.sqrt(val) for val in error_CN2]
        error_CCN = [np.sqrt(val) for val in error_CCN2]
        error_CN = [error_CN[index]*self.dry_count[index] for index in range(len(self.dry_count))]
        error_CCN = [error_CCN[index]*self.droplet_count[index] for index in range(len(self.droplet_count))]
        
        error_ratio2 = [error_CN2[index] + error_CCN2[index] for index in range(len(error_CN2))]
        error_ratio = [np.sqrt(val) for val in error_ratio2]
        error_ratio = [error_ratio[index]*self.ratio[index] for index in range(len(self.ratio))]
        
        return error_CN, error_CCN, error_ratio
    
    def dia_uncertainty(self, aac_scan):
        
        """ Computes uncertainty in the aerodynamic diameters.
        
        Parameters
        ----------
        aac_scan : floats array
            Aerodynamic diameters.
        
        Returns
        -------
        uncertain_dia : float array
            Uncertainties in the particle aerodynamic diameters.
        """
        
        # Uncertainty analysis for the transfer function of the AAC
        # Such an analysis will help with understanding the possible uncertainty in the particle size
        # This will be useful with the overall CCN analysis
        
        columns = aac_scan.columns

        aac_tau_measured = np.array(aac_scan[columns[4]])
        aac_dia = np.array(aac_scan[columns[1]])
        aac_spd = np.array(aac_scan[columns[15]])
        aac_tmp = np.array(aac_scan[columns[17]])
        aac_prs = np.array(aac_scan[columns[18]])
        aac_conc = np.array(aac_scan[columns[3]])
        aac_sheath = np.array(aac_scan[columns[16]])
        
        aac_lamb_T = []
        for index in range(len(aac_tmp)):
            T = aac_tmp[index] + 273.15
            P = aac_prs[index]
            lamb = fb.mean_free_path(T, P)
            aac_lamb_T.append(lamb)
        
        dQ = 0.01    # unit = Lpm
        dr = 2      # unit = um
        dw = 2      # unit = rpm
        dL = 2      # unit = mm
        
        r1 = 56     # unit = mm
        r2 = 60     # unit = mm
        r = 58      # unit = mm
        L = 206     # unit = mm
        
        # Uncertainty in tau = dtau_tau
        # Propagation of error = (dtau_tau)**2
        
        # The 2 expressions are totally different from each other --> at least this is observed
        # Using a different route, uncertainty could be obtained by taking the square root of the 1st expression
        
        ac = 2.33
        bc = 0.966
        cc = 0.4985
        
        uncertain_dia = []
        
        resolution_array = []
        
        for index in range(len(aac_dia)):
            
            sheath = aac_sheath[index]      # unit = Lpm
            speed = aac_spd[index]          # unit = rps
            
            speed = speed*60                # unit = rpm
            
            dtau_tau = (dQ / sheath) - 2*(dw / speed) - 2*(dr*1e-3 / r) - (dL / L)
            dtau_tau2 = (dQ / sheath)**2 + 4*(dw / speed)**2 + 4*(dr*1e-3 / r)**2 + (dL / L)
            
            dtau_tau2 = np.sqrt(dtau_tau2)
            
            resolution = 1 / np.abs(round(dtau_tau2, 3))
            
            ## The minimum admissible resolution must be about 4.5
            
            resolution_array.append(round((1 / np.abs(round(dtau_tau2, 3))), 3))
            
            # The uncertainty in the relaxation time requires modification
            # This will lead to the uncertainty estimate in the Dae
            
            round_dtau_tau = abs(dtau_tau2)
            
            lamb = aac_lamb_T[index]
            dia = aac_dia[index]
            factor = (dia + ac*lamb + bc*lamb*np.exp(-cc * dia/lamb)) /\
                     (2*dia + ac*lamb + bc*lamb*(1 - cc*dia/lamb)*np.exp(-cc * dia/lamb))
                     
            ddia_dia = round(round_dtau_tau * factor, 3)
            
            uncertain_dia.append(ddia_dia)
        
        return uncertain_dia
      
