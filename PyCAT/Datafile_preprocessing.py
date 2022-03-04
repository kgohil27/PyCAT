# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 20:35:50 2021

@author: Kanishk Gohil
"""

import os
import re
import glob
import shutil
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.signal import find_peaks
from matplotlib import container
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

import matplotlib as mpl
mpl.style.use('default')

import matplotlib.pyplot as plt
from . import functions_basic as fb
importlib.reload(fb)
from . import AAC_data_extraction as de
importlib.reload(de)

class SMPS_data_process(object):
    
    def __init__(self, basedir, basefile, data, time, offset=0, sigmoids=1, downscan='no'):
        
        self.basedir = basedir
        self.basefile = basefile
        self.data = data
        self.time = time
        self.offset = offset
        self.sigmoids = sigmoids
        self.downscan = downscan
    
    def data_generation(self, start_dia=0., end_dia=9.9e36, flow_rate=1.):
        
        os.chdir(self.basedir)
        
        references = ['Raw Data - ', 'Comment']
        timestring = 'Start Time'
        
        delete_strings = ['Raw Data - ', '(s)', ' #']   # useless strings in the raw datafile that will be deleted
        
        fb.processing(self.basefile, self.data, self.time, references, timestring, delete_strings)
        
        with open(self.time, 'r') as timefile:
            times = timefile.readlines()
        
        times = [time.rstrip() for time in times]
        times = [time for time in times if time]
        times = times[1:]
        
        activation_dataframe_cols = np.append(['Diameters'], times)
        
        CCN_activation_dataframe = pd.DataFrame(columns=activation_dataframe_cols)
        CCN_activation_fit = pd.DataFrame(columns=activation_dataframe_cols)
        
        CNdf = pd.read_csv(self.data, sep='\t')
        
        extension = 'csv'
        allcsv = [i for i in glob.glob('*.{}'.format(extension))]
        dfCCN = pd.concat([pd.read_csv(f, skiprows=4) for f in allcsv]).reset_index(drop=True)
        dfCCN.columns = dfCCN.columns.str.strip()
        
#        time0, timelen = np.array(dfCCN.Time)[0], len(np.array(dfCCN.Time))
#        time_tmp = []
#        a = datetime.strptime(time0, "%H:%M:%S")
#        for i in range(timelen):
#            if i == 0:
#                time_tmp.append(time0)
#            else:
#                b = a + timedelta(seconds=i)
#                b = b.time().strftime("%H:%M:%S")
#                time_tmp.append(b)
#        
#        dfCCN.Time = time_tmp
        
        CCNtimeVals, CCNrefIndex = fb.CCN_time_ref(dfCCN, self.time)
        print (CCNrefIndex)
        
        database = pd.DataFrame(columns=['Time stamp',
                                         'Scan #',
                                         'Activation diameter',
                                         'Supersaturation',
                                         'Temperature',
                                         'Surface tension'])
        
        ## starting index of the CN dataset
        s = CNdf.columns[1]
        beginIndex = int(re.findall('\d+', s)[0])
        
        for timeIndex in range(len(CCNtimeVals)):
            
            try:
            
                """ This is the extraction of the index in the CN dataset which
                corresponds to the inflection point in the rising half of the downscan
                gaussian region."""
                
                self.time = np.array(CNdf['Time'].tolist())
                count = np.array(CNdf['Counts'+str(timeIndex+beginIndex+CCNrefIndex)].tolist())
                diameter = np.array(CNdf['Diameter'+str(timeIndex+beginIndex+CCNrefIndex)].tolist())
                index = np.where(self.time==120)[0][0]
                
                print ('Scan number = '+str(timeIndex))
        
                idInflection = fb.inflection_index(self.time[index:], count[index:])
                
                if idInflection <= len(self.time[index:]):
                    """ This is the extraction of the index in the CCN dataset which
                    corresponds to the inflection point in the rising half of the downscan
                    gaussian region, relative to the CN measurements. The extraction of the
                    index for the CCN dataset is marginally more complex because the CCN
                    dataset first needs to be aligned with respect to the CN dataset with
                    the help of the timestamps of their downscan measurements."""
                    
                    timestr = CCNtimeVals[timeIndex+CCNrefIndex]
                    pt = datetime.strptime(timestr,'%H:%M:%S')
                    tots = pt.second + pt.minute*60 + pt.hour*3600
                    timeInflection = self.time[index:][idInflection] + tots
                    
                    timesCCN = dfCCN.Time.tolist()
                    timesStrCCN = timesCCN
                    for ind in range(len(timesCCN)):
                        pt = datetime.strptime(timesCCN[ind],'%H:%M:%S')
                        tots = pt.second + pt.minute*60 + pt.hour*3600
                        timesCCN[ind] = tots
                        
                    timeDiffCCN = np.abs(np.array(timesCCN) - timeInflection)
                    minDiffIndex = timeDiffCCN.argmin()
                    timeInfCCN = timesStrCCN[minDiffIndex]
                    timeInfCCN = str(timedelta(seconds=timeInfCCN))
                            
                    dfCCN_reduced = dfCCN[minDiffIndex:]
                    CCN_conc = dfCCN_reduced['CCN Number Conc'].values.tolist()
                    X = np.linspace(1,len(CCN_conc[:20]),len(CCN_conc[:20]),endpoint=True)
            
                    indexInterim = fb.inflection_index(X, CCN_conc[:20])
                    
                    if indexInterim <= len(X):
                        xval = np.linspace(min(X), max(X), len(X), endpoint=True)[indexInterim]
                        idInflection = min(range(len(X)), key=lambda i: abs(X[i]-xval)) + self.offset
                
                        """ This is the cropping of the correct CCN dataset for the CN dataset
                        under consideration, by aligning the inflection points of the downscan
                        measurements of both datasets."""
                        
                        startIndex = dfCCN[dfCCN.Time == timestr].index[0]
                        trueStartIndex = startIndex + idInflection
                        trueCCN = dfCCN['CCN Number Conc'].values.tolist()[trueStartIndex:trueStartIndex+135]
                        
                        #####################################################################################
                
                        supersaturation = list(dfCCN['Current SS'][dfCCN.Time == timestr])[0]
                        temperature = np.mean(dfCCN['T1 Read'].values.tolist()[trueStartIndex+20:trueStartIndex+135])+273.15
                        sigma = fb.surface_tension_temp(temperature)
                        
                        trueCN = []
                        trueDia = []
                        self.data = len(count)
                        trueData = self.data/10
                        for i in range(int(trueData)):
                            index = 10*i
                            total = sum(count[index-10:index])
                            mean = np.mean(diameter[index-10:index])
                            trueCN.append(total*flow_rate)   # flow rate conversion is 1.05-put in manually, can be estimated
                            trueDia.append(mean)
                        trueDia = [x for x in trueDia if str(x) != 'nan']
                        
                        if self.downscan == 'no':
                            plot_index = 120
                            X_CCN = np.logspace(np.log10(8.06), np.log10(352.3), 120)
                        elif self.downscan == 'yes':
                            plot_index = 135
                            X_CCN = np.linspace(1, plot_index, plot_index, endpoint=True)
                                                
                        try:
                            plt.xscale('log')
                            plt.plot(X_CCN, [1.1*val for val in trueCN][:plot_index], color='C0', marker='s',
                                     markerfacecolor='none', ls='none', label='CN')
                            plt.plot(X_CCN, trueCCN[:plot_index], color='C1', marker='^',
                                     markerfacecolor='none', ls='none', label='CCN')
                            plt.xticks(fontsize=15)
                            plt.yticks(fontsize=15)
                            plt.xlabel('Mobility diameter (nm)', fontsize=15)
                            plt.ylabel('Number distribution', fontsize=15)
                            plt.show()
                            
                            Ratio = []
                            for index in range(len(trueCN)):
                                CCN = trueCCN[index]
                                CN = trueCN[index]
                                if CN == 0:
                                    CN += 1
                                Ratio.append(CCN/CN)
                            
                            start_ind = fb.find_nearest(trueDia, start_dia)
                            end_ind = fb.find_nearest(trueDia, end_dia)
                                
                            diaPlot = trueDia[start_ind-1:end_ind]
                            RatioPlot = Ratio[start_ind-1:end_ind]
                            RatioPlot = [(value - min(RatioPlot))/(max(RatioPlot) - min(RatioPlot)) for value in RatioPlot]
            #                CCN_activation_dataframe[times[timeIndex]] = RatioPlot
                            RatioPlotCorrected = fb.ChargeCorrection(RatioPlot,diaPlot,temperature)
                            RatioPlotCorrected = [(value - min(RatioPlotCorrected))/\
                                                  (max(RatioPlotCorrected) - min(RatioPlotCorrected))\
                                                  for value in RatioPlotCorrected]
                            CCN_activation_dataframe[times[timeIndex]] = RatioPlotCorrected
                            
                            try:
                                # "self.sigmoids" dictates the number of sigmoids that need to be fit to the activation data
                                # Default value --> self.sigmoids=1
                                RatioFit = fb.sigmoid_fit(diaPlot, RatioPlotCorrected, self.sigmoids)  # 1 sigmoid or more!!
                                inflectionInds, _ = find_peaks(np.gradient(RatioFit))            
                                dia_fit = np.linspace(min(diaPlot), max(diaPlot), 101)
                                diameters = np.array(dia_fit)[inflectionInds]
#                                plt.plot(diaPlot, RatioPlot, ls='None', marker='^',
#                                         markerfacecolor='none', color='red')
                                plt.plot(diaPlot, RatioPlotCorrected, ls='None', marker='s',
                                         markerfacecolor='none', color='green')
                                plt.plot(dia_fit, RatioFit, 'black', '--')
                                plt.plot(np.linspace(20, 120, 10), [0.5]*10, 'r--')
                                plt.xlim([min(diaPlot)-5, max(diaPlot)+5])
                                plt.ylim([-0.05, 1.1])
                                plt.xticks(fontsize=15)
                                plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
                                plt.xlabel('Mobility diameter (nm)', fontsize=15)
                                plt.ylabel('$N_{CCN}/N_{CN}$', fontsize=15)
                                plt.show()
                                if len(diameters) != 0:
                                    if len(diameters) == 1:
                                        to_append = [timestr, timeIndex, round(diameters[0],3), supersaturation, temperature, sigma]
                                        database.loc[len(database)] = to_append
                                    elif len(diameters > 1):
                                        diameters = [round(dia,3) for dia in diameters]
                                        to_append = [timestr, timeIndex, diameters, supersaturation, temperature, sigma]
                                        database.loc[len(database)] = to_append
                                CCN_activation_fit[times[timeIndex]] = RatioFit
                                
                            except (RuntimeError, ValueError):
                                print ('Optimal parameters for the sigmoid not found and/or sigmoid could not be optimized.')
                        except ValueError:
                            pass
                    else:
                        continue
                else:
                    continue
            except KeyError:
                break
        
        CCN_activation_dataframe['Diameters'] = diaPlot
        CCN_activation_fit['Diameters'] = dia_fit
        
        return database, CCN_activation_dataframe, CCN_activation_fit

class AAC_data_process(object):
    
    def __init__(self, basedir, basefile, sigmoids=1, get_errors='Y'):
        
        self.basedir = basedir
        self.basefile = basefile
        self.sigmoids = sigmoids
        self.get_errors = get_errors
    
    def data_generation(self, end_size=9.9e36, num_scans=1, offset=0):
        
        os.chdir(self.basedir)
        keys = ['SCAN', 'END OF SCAN']
        
        file_base = os.path.splitext(self.basefile)[0]
        target_file_txt = shutil.copyfile(self.basefile, file_base + '.txt')
        
        proc_datafile = 'AAC_SCANFILE_'
        
        num_scans = fb.scan_extraction(target_file_txt, keys, txtfile=proc_datafile)
        
        for i in range(num_scans):
            AAC_scan = pd.read_csv(proc_datafile+str(i+1)+'.txt', delimiter='\t')
            AAC_scan.set_index(AAC_scan.columns[0], inplace=True)
        
        '''
        The following part of the code - 
        1. AAC_scan read
        2. Extract relavant variables
        3. TF analysis - for number concentration
        4. Return number concentration
        '''
        
        extension = 'csv'
        allcsv = [file for file in glob.glob('*.{}'.format(extension))]
        CCN_scan = pd.concat([pd.read_csv(file, skiprows=4) for file in allcsv]).reset_index(drop=True)
        CCN_scan.columns = CCN_scan.columns.str.strip()
        
        data_extractor = de.AAC_data_derivation(AAC_scan, CCN_scan)
        CN_conc = data_extractor.CN_number_conc()                                ## CN number concentration derived
        CCN_conc, temp, sigma, SS = data_extractor.CCN_number_conc(offset=offset)             ## CCN number concentration derived
        
        diameters = AAC_scan['Size (nm)']
        max_size_index = fb.find_nearest(diameters, end_size)
        diameters_reduced = diameters[:max_size_index]
        
        ratio = [CCN_conc[index]/CN_conc[index] for index in range(len(CCN_conc))]
        ratio_reduced = ratio[:max_size_index]
        ratio_reduced = [(val - min(ratio_reduced)) /\
                         (max(ratio_reduced) - min(ratio_reduced)) for val in ratio_reduced]
        
        ratio_fit = fb.sigmoid_fit(diameters_reduced, ratio_reduced, self.sigmoids)  # 1 sigmoid or more!!
        diameters_fit = np.linspace(min(diameters_reduced), max(diameters_reduced), 101)
        inflectionInds, _ = find_peaks(np.gradient(ratio_fit))
        dp50 = np.array(diameters_fit)[inflectionInds]
        
        uncertainty_calculator = de.uncertainty_AAC_data(diameters, CN_conc, CCN_conc, ratio)
        error_CN, error_CCN, error_ratio = uncertainty_calculator.count_uncertainty()
        error_ratio_reduced = error_ratio[:max_size_index]
        error_dia = uncertainty_calculator.dia_uncertainty(AAC_scan)
        error_dia_reduced = np.array(error_dia[:max_size_index])
        
        # number concentration plots
        plt.plot(diameters, CN_conc, color='C0', ls='none', marker='s', label='CN')
        plt.plot(diameters, CCN_conc, color='C1', ls='none', marker='^', label='CCN')
        plt.xticks(fontsize=12.5)
        plt.yticks(fontsize=12.5)
        plt.show()
        
        # activation ratio plots
        plt.errorbar(diameters_reduced, ratio_reduced, yerr=error_ratio_reduced, xerr=error_dia_reduced,
                     ls='None', marker='o', markerfacecolor='C2', ecolor='C2', markeredgecolor='C2',
                     markersize=3.5, capsize=4, elinewidth=2, markeredgewidth=2)
        plt.plot(diameters_fit, ratio_fit, color='C2', ls='-')
        plt.xlim([min(diameters_reduced)-5, max(diameters_reduced)+5])
        plt.ylim(-0.05, 1.1)
        plt.xticks(fontsize=12.5)
        plt.yticks(np.arange(0., 1.2, 0.2), fontsize=12.5)
        plt.show()
        
        if len(dp50) != 0:
            if len(dp50) == 1:
                CCN_data = [SS, dp50[0], temp, sigma]
            elif len(dp50) > 1:
                CCN_data = [SS, dp50, temp, sigma]
        
        if self.get_errors.upper() == 'Y':
            return CCN_data, diameters_reduced, ratio_reduced,\
        [error_dia, error_CN, error_CCN], [error_dia_reduced, error_ratio_reduced]
        elif self.get_errors.upper() == 'N':
            return CCN_data, diameters_reduced, ratio_reduced
        
