import re
import socket

import pandas as pd
import numpy as np
import math
from datetime import datetime
import time
from collections import Counter
import copy

import dash
from dash.dependencies import Output, Input, State
from dash import dcc, html, dash_table
import plotly
import plotly.express as px
import plotly.graph_objects as go

import threading

import logging 
logging.getLogger('werkzeug').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter('ignore', np.RankWarning)

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split




####################################################################
#****************  FILL THIS PART OUT BEFORE RACE  ****************#
####################################################################

#TIMEZONE OF TRACK (EST', 'CST', 'MST', 'PST')
timeZone = 'CST'

#IF INDY500 SET TRUE
INDY500_filter = None

#SET IF RACE OR P/Q SESSIONS
race = True

#SET IF SC/RC OR OVAL/SPEEDWAY
road_course = True

#SPECIAL DRIVER ENTRY FOR INDY500
special_entry = False

#ENTRY DRIVERS
drivers = np.array([2,3,4,5,6,7,8,9,10,11,12,14,15,16,18,20,21,26,27,28,29,30,45,48,51,60,77])

#TRACK LENGTH (miles)
track_distance = 4.01 

#LAPTIME FILTER AND AVG LAP STINT ESTIMATION
LAP_FILTER = 115 #seconds
MAX_LAPS = 14 #laps
GPL_CUTOFF = 1.4

#TIME & SCORE IP
ip = "indycar.livetiming.net"
port_score = 50006

####################################################################
####################################################################





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#***********************  OTHER PARAMETERS  ***********************#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#DASH PLOTS SIZE
width_plot = 1600
height_plot = 700

#FUEL CAPACITY
capacity = 18.3 #gallons

#HYPERPARAMS
MIN_LAP_FILTER = LAP_FILTER-20
CUTOFF_PROB = 0.8 #0-1 prob
max_DAG = 2

params = {'w_Lap1out': 1.5, 'w_Lap2out': 1.25, 'minLapTime':LAP_FILTER, 'maxL':MAX_LAPS, 'prob':CUTOFF_PROB,
         'adjAfterLaps':10, 'minLapW':1.4, 'firstMinLap':12}

if race:
    CONT_IMPROV = 0.025
else:
    CONT_IMPROV = 0.1

DIFF_MULT = 1.25
LAP_OUT_IGNORE = 0.75
SESS_PROGRESS = 0.01

scatter_color = {15: 'blue', 30:'red', 45:'black'}
time_format = "%H:%M:%S.%f"


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                              STUFF                               #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#MAIN DASHBOARD INIT
app = dash.Dash(__name__)
server = app.server


#>>>>>>>>>>>>>>> WHERE STUFF GOES <<<<<<<<<<<<<<<#

lap_times = {}
telem_dict = {}
DAG_dict = {}
TYRE_STINTS = {}

#>>>>>>>>>>>>>>>------------------<<<<<<<<<<<<<<<#


drivers_dict = []
for car in drivers:
    car_dict = {'label': 'Car '+str(car), 'value': car}
    drivers_dict.append(car_dict)



def timeZone_offset(timeZone):
    sec_float = 3600.0
    if timeZone == 'CST':
        time_diff_s = sec_float*5
    elif timeZone == 'MST':
        time_diff_s = sec_float*6
    elif timeZone == 'PST':
        time_diff_s = sec_float*7
    elif timeZone == 'EST':
        time_diff_s = sec_float*4
    else:
        time_diff_s = sec_float*4
    return time_diff_s



class Tyre_P:
    
    def __init__(self, car, lap_times, params):
        
        self.laps_pitted = []
        columns = ['Lap', 'LapTime', 'Status', 'TimeOfDay', 'TyreType', 'Speed', 'PitsTotal', 'LastPitLap']
        self.new_df = pd.DataFrame(lap_times[car], columns=columns)
        #self.new_df.Status.values[self.new_df.Status =='P'] = 1
        #self.new_df.Status.values[self.new_df.Status !=1] = 0
        
        laps = np.zeros(len(self.new_df))
        ones = np.ones(len(self.new_df))
        self.new_df['Status W'] = ones
        self.new_df['Tyre W'] = laps
        self.new_df['LapDiff'] = laps
        self.new_df['Proba'] = laps
        self.new_df.Proba.values[0] = 0.05
        self.LapFilter = params['minLapTime']
        self.minLapTime = self.LapFilter
        self.LSL = 0
        self.cont_diff = 0
        self.extended_tyreage = 0
        self.LSL_idx = []
        self.stints_num = 0
        self.stint_laps = []
        self.stint_info = []
        self.trend_comp = {}
        slice_df = self.new_df.LapTime[self.new_df.LapTime.values < self.LapFilter]
        if slice_df.empty:
            return
        else:
            self.maxLTime = max(slice_df)
        self.adj_breakline = []
        self.DAG = []
        self.tyre_tp = []
        self.med_laptimes = []
        self.blacks_DAG = []
        self.reds_DAG = []
        self.for_sure_pit = False
        self.stint_on_blk = {}
        self.stint_on_red = {}
        
    def apply_pits_w(self, params):
        
        for iter_num in range(0, len(self.new_df)):
            if self.new_df.Status.values[iter_num] == 0 and iter_num>2:
                if self.new_df['Status W'].values[iter_num-1] == params['w_Lap1out']:
                    self.new_df['Status W'].values[iter_num] = params['w_Lap2out']

                if self.new_df.Status.values[iter_num-2] == 1:
                    self.laps_pitted.append(self.new_df.Lap.values[iter_num])
                    self.new_df['Status W'].values[iter_num] = params['w_Lap1out']
                
    def lap_filter(self, params):
        self.new_df = self.new_df[self.new_df.LapTime < self.LapFilter].reset_index(drop=True)
    
    def diffs(self):
        
        self.dips = self.new_df['LapTime'].diff(periods=1).fillna(0)
        for i in range(0,len(self.dips)):
            self.dips.values[i] = -math.tanh(self.dips.values[i])*DIFF_MULT
            
    def tyre_age(self, i, params):
        maxLap = params['maxL']
        allTimeW = params['minLapW']
        #Array of age probs
        self.age_Probs = np.linspace(0.01, 1, maxLap+1)
        
        #If no tyre change was assigned, extend high tyre age probability to next lap
        self.LSP = self.new_df.Lap.values[i]-self.LSL
        if self.LSP > maxLap:
            self.new_df['Tyre W'].values[i] = self.age_Probs[maxLap]
            self.extended_tyreage +=1
        else:
            #Assign tyre age probabilities
            self.new_df['Tyre W'].values[i] = self.age_Probs[self.LSP]
        
        #Assign normalized differentials
        self.new_df.LapDiff.values[i] = self.dips.values[i]
        
    def fastest_lap(self, i, params):
        
        #Skip first n laps from minlap
        if self.new_df.Lap.values[i] < params['firstMinLap']:
            return

        #All-time lowest laptime condition
        if self.new_df.LapTime.values[i] <  self.minLapTime:
            self.minLapTime = self.new_df.LapTime.values[i]
            self.minLapTimeL = self.new_df.Lap.values[i]

            #Check for min lap to start cosidering for probability adjustment
            if self.LSP>params['adjAfterLaps']:
                self.new_df.LapDiff.values[i] = self.new_df.LapDiff.values[i]*params['minLapW']
                
    def track_cont_descent(self, i):
        if self.new_df.LapDiff.values[i] > 0:
            self.cont_diff += 1
        else:
            self.cont_diff = 0    
            
    def track_laptime_stats(self, stint_num, break_point):
        
        stint_idx = stint_num -1
        if stint_idx==0:
            curr_df = self.new_df[:break_point] 
        else:
            start = self.LSL_idx[stint_idx-1]+1
            curr_df = self.new_df[start:break_point]
            
        med_laptime = np.median(curr_df.LapTime.values)
        return med_laptime   

    def combine_probs(self, i):
        
        #Session progression increase chance of tyre change
        if self.new_df.Lap[i]> params['maxL']:
            self.new_df.LapDiff.values[i] = self.new_df.LapDiff.values[i]*(1 + (i*SESS_PROGRESS))
        
        p_tyre = self.new_df['Tyre W'].values[i]
        p_diff = self.new_df.LapDiff.values[i]
        p_pit = self.new_df['Status W'].values[i]
            
                
        #Combined probabilities
        if p_diff > 0:
            self.new_df.Proba.values[i] = round(p_tyre * p_diff *p_pit,2)
        else:
            reduced_prob = round(self.new_df.Proba.values[i-1] + (p_tyre*p_diff),2)
            if reduced_prob >= 0 and self.new_df.Lap.values[i-1] not in self.stint_laps:
                self.new_df.Proba.values[i] = reduced_prob
            else:
                self.new_df.Proba.values[i] = 0 
                
        #Continuous improvment
        if self.cont_diff >1 and i>0:
            if self.new_df.Proba.values[i-1] < params['prob']:
                if self.new_df.Proba.values[i-1] > self.new_df.Proba.values[i]:
                    self.new_df.Proba.values[i] = self.new_df.Proba.values[i-1]+CONT_IMPROV 
                else:
                    self.new_df.Proba.values[i] = self.new_df.Proba.values[i]+CONT_IMPROV
                
        #if i==67:
            #print('Lap 86 (not sure which car) proba: {}'.format(self.new_df.Proba.values[i]))
                
        #If no stint although tyres are pass maximum stint
        if self.extended_tyreage>0:
            self.new_df.Proba.values[i] = self.new_df.Proba.values[i] + (self.extended_tyreage*0.025)
            
        #if i==67:
            #print('Lap 86 (not sure which car) proba: {}'.format(self.new_df.Proba.values[i]))
            
        #Pit stop in and out lap diff ignore
        if i>1:
            p_diff_prev = self.new_df.LapDiff.values[i-1]
            if p_diff_prev < -0.9 and p_diff > 0.9:
                self.new_df.Proba.values[i] = self.new_df.Proba.values[i]*LAP_OUT_IGNORE
                
        
                
    def evaluate(self, i, params):
        #If clear A->P or P->A tyre change
        if i>0:
            prev_tyre = self.new_df['TyreType'].values[i-1]
            curr_tyre = self.new_df['TyreType'].values[i]
            if prev_tyre != curr_tyre:
                self.for_sure_pit = True
                self.new_df.Proba.values[i] = 1
                
        #Probability treshold for assigning tyre change
        if self.new_df.Proba.values[i] >= params['prob']:
            if self.for_sure_pit:
                if len(self.LSL_idx)>0:
                    if self.LSL_idx[-1] == (i-1) or self.LSL_idx[-1] == (i-2):
                        self.LSL_idx[-1] = i
                        self.stint_info[-1] = [self.new_df.Lap.values[i], self.new_df.TimeOfDay.values[i]]
                        self.stint_laps[-1] = self.new_df.Lap.values[i]
                        self.for_sure_pit = False
                    else:
                        self.LSL = self.new_df.Lap.values[i]
                        self.LSL_idx.append(i)
                        time_stamp = self.new_df.TimeOfDay.values[i]
                        self.stint_info.append([self.LSL, time_stamp])
                        self.stint_laps.append(self.LSL)
                        self.stints_num = len(self.stint_laps)
                        med_laptime = self.track_laptime_stats(self.stints_num, i)
                        self.med_laptimes.append(med_laptime)
                else:
                    self.LSL = self.new_df.Lap.values[i]
                    self.LSL_idx.append(i)
                    time_stamp = self.new_df.TimeOfDay.values[i]
                    self.stint_info.append([self.LSL, time_stamp])
                    self.stint_laps.append(self.LSL)
                    self.stints_num = len(self.stint_laps)
                    med_laptime = self.track_laptime_stats(self.stints_num, i)
                    self.med_laptimes.append(med_laptime)
            else:
                self.LSL = self.new_df.Lap.values[i]
                self.LSL_idx.append(i)
                time_stamp = self.new_df.TimeOfDay.values[i]
                self.stint_info.append([self.LSL, time_stamp])
                self.stint_laps.append(self.LSL)
                self.stints_num = len(self.stint_laps)
                med_laptime = self.track_laptime_stats(self.stints_num, i)
                self.med_laptimes.append(med_laptime)
            self.extended_tyreage = 0

    def break_trends(self, break_point, stint_idx, max_trend_idx):
        
        median_lt = self.med_laptimes[stint_idx]
        dev_lt = median_lt*1.025

        if stint_idx==0:
            trend1 = self.new_df[:break_point] #+1
            trend1 = trend1[trend1.LapTime.values < dev_lt]
            trend2 = self.new_df[break_point:max_trend_idx]
            trend2 = trend2[trend2.LapTime.values < dev_lt]

        else:
            start = self.LSL_idx[stint_idx-1]+1
            trend1 = self.new_df[start:break_point]
            trend1 = trend1[trend1.LapTime.values < dev_lt]
            trend2 = self.new_df[break_point+2:max_trend_idx-1]
            trend2 = trend2[trend2.LapTime.values < dev_lt]

        return trend1, trend2
    
    def trend_decomp(self, stint_idx):

        tyre_types_obs = []
        scores = []
        offset = []

        max_obs = (len(self.new_df) - self.LSL_idx[stint_idx])-1
        if max_obs == 0:
            return

        if max_obs > 5:
            max_obs = 5
            
        min_obs = -5
        test_min_obs = len(self.new_df) - max_obs
        if test_min_obs < abs(min_obs):
            test_min_obs -= 1
            min_obs = -test_min_obs
        
        observations_id = np.arange(min_obs,max_obs)
        
        for i in range(min_obs,max_obs):

            #Perturbed breakpoint
            break_point = self.LSL_idx[stint_idx] + i

            #Record tyre type at that observ`ation
            tyre_types_obs.append(self.new_df['TyreType'].values[break_point])

            #Select range to break trend
            max_trend_idx = max_obs + self.LSL_idx[stint_idx]

            trend1, trend2 = self.break_trends(break_point, stint_idx, max_trend_idx)

            if len(trend2) <2 or len(trend1) <2:
                continue

            coeffb, interceptb = np.polyfit(trend1.index.values, trend1.LapTime.values,1)
            coeffa, intercepta = np.polyfit(trend2.index.values, trend2.LapTime.values,1)

            if coeffa>0 and coeffb>0:
                pos = (2+coeffa+coeffb) 
            else:
                if coeffa>0:
                    pos = 1+coeffa
                elif coeffb>0:
                    pos = 1+coeffb
                else:
                    pos = coeffa+coeffb
                    
            offset.append(i)
            scores.append(pos)

        #Check the tyre array for when tyre change happens
        change_idx = [i for i in range(1,len(tyre_types_obs)) if tyre_types_obs[i]!=tyre_types_obs[i-1]]

        if change_idx:
            tyre_change_id = change_idx[0]
            new_breakline = self.LSL_idx[stint_idx] + observations_id[tyre_change_id]
        else:
            if len(scores) == 0:
                new_breakline = self.LSL_idx[stint_idx]
            else:
                obs = np.argmax(scores)
                new_breakline = self.LSL_idx[stint_idx] + offset[obs]
        
        #print('Almost ready to add breakline. Checking conditions')
        #print('My current stint ({}) lap_idx is: {}. I want to add: {}'.format(stint_idx+1, 
                                                                               #self.LSL_idx[stint_idx], new_breakline))
        
        if self.adj_breakline:
            if self.adj_breakline[-1] != new_breakline:
                self.adj_breakline.append(new_breakline)
        else:
            self.adj_breakline.append(new_breakline)

           
    def post_hoc(self, car):
        
        for stint_idx in range(0, self.stints_num):
            #print('Running post-hoc for stint {}'.format(stint_idx+1))
            self.trend_decomp(stint_idx)
            
            
    def plot_stints(self, car):  

        if not np.array(self.new_df.LapTime.values).any():
            return go.Figure()

        minLapPlot = min(self.new_df.LapTime.values)
        
        #Plot DAG        
        timing_df = self.new_df[['Lap', 'LapTime']]
        fig1 = px.line(timing_df, x="Lap", y="LapTime") 
        fig_data = fig1.data
        
        if self.stints_num<1:
            if len(self.new_df.LapTime.values) > 2:
                
                get_start_count = 0
                for i in range(1, len(self.new_df)):
                    lap_s = self.new_df.LapTime.values[i]
                    if self.new_df.LapTime.values[get_start_count] - lap_s >0:
                        get_start_count = i
                    else:
                        get_start_count = i-1
                        break
                
                trend_stint1 = px.scatter(self.new_df[get_start_count:], x="Lap", y="LapTime", trendline="ols",
                                         trendline_color_override='red')
                fig_data += trend_stint1.data
        
        if self.stints_num>0:
            for i in range(0, self.stints_num):
                
                #4 stints proba but 3 breaklines identified
                #if len(self.adj_breakline) != self.stints_num and (i+1) ==  len(self.adj_breakline):
                    #break
                
                if (i+1) == self.stints_num:
                    max_trend_idx = len(self.new_df)
                else:
                    max_trend_idx = self.LSL_idx[i]
                
                trend1, trend2 = self.break_trends(self.LSL_idx[i], i, max_trend_idx)

                trend_b = px.scatter(trend1, x="Lap", y="LapTime", trendline="ols", trendline_color_override='red')
                fig_data += trend_b.data
                
                if (i+1) == self.stints_num:
                    trend_a = px.scatter(trend2, x="Lap", y="LapTime", trendline="ols", trendline_color_override='red')
                    fig_data += trend_a.data
            
        fig_f = go.Figure(data=(fig_data))
        
        if self.stints_num>0:
            for i in range(0, self.stints_num):
                
                x1b = self.new_df.Lap.values[self.LSL_idx[i]]
                fig_f.add_shape(type='line', x0=x1b, y0=minLapPlot,
                                x1=x1b, y1=self.maxLTime, 
                                line=dict(color='Orange',), xref='x', yref='y')
                
                #print('Number of stints: {} and numner of breaklines: {}'.format(self.stints_num, len(self.adj_breakline)))
                
                if len(self.adj_breakline) != self.stints_num and (i+1) ==  len(self.adj_breakline):
                    break
                elif len(self.adj_breakline)==0:
                    break
                else:
                    x0b = self.new_df.Lap.values[self.adj_breakline[i]]
                    if x0b!=x1b:
                        fig_f.add_shape(type='line', x0=x0b, y0=minLapPlot,
                                    x1=x0b, y1=self.maxLTime, 
                                    line=dict(color='Orange',), xref='x', yref='y')

                        fig_f.add_shape(type="rect", x0=x0b, y0=minLapPlot,
                                        x1=x1b, y1=self.maxLTime,
                                        line=dict(color='Yellow',),
                                        fillcolor='Yellow', opacity = 0.5) 
            
        fig_f.update_layout(title='Car#'+str(car)+' DEG', xaxis_title="Laps", yaxis_title="Lap Time (s)",
                            width=width_plot, height=height_plot)        
        return fig_f
    
    
    def get_DAG(self):

        got_DAG = 0
                  
        if self.stints_num<1:
            if len(self.new_df.LapTime.values) > 2:
                
                get_start_count = 0
                for i in range(1, len(self.new_df)):
                    lap_s = self.new_df.LapTime.values[i]
                    if self.new_df.LapTime.values[get_start_count] - lap_s >0:
                        get_start_count = i
                    else:
                        get_start_count = i-1
                        break
                        
                #Get DAG
                coeff, intercept = np.polyfit(self.new_df.index.values[get_start_count:], 
                                              self.new_df.LapTime.values[get_start_count:],1)
                if coeff>0 and coeff<max_DAG:
                    self.DAG.append(coeff)
                    got_DAG += 1
                    
                    #Which tyre
                    b = Counter(self.new_df['TyreType'].values)
                    type_t = b.most_common()[0][0]
                    self.tyre_tp.append(type_t)
                    
                    if type_t == 'A':
                        self.stint_on_red[1] = self.new_df[['Lap', 'LapTime']][get_start_count:]
                    else:
                        self.stint_on_blk[1] = self.new_df[['Lap', 'LapTime']][get_start_count:]

        if self.stints_num>0:
            for i in range(0, self.stints_num):
                #print("Stint")
                #print(i+1)
                    
                if (i+1) == self.stints_num:
                    max_trend_idx = len(self.new_df)
                else:
                    max_trend_idx = self.LSL_idx[i]
                
                trend1, trend2 = self.break_trends(self.LSL_idx[i], i, max_trend_idx)
                if len(trend1)>2:
                    coeffb, interceptb = np.polyfit(trend1.index.values, trend1.LapTime.values,1)
                    if coeffb>0 and coeffb < max_DAG:
                        self.DAG.append(coeffb)
                        got_DAG += 1
                        
                        #Which tyre
                        b = Counter(trend1['TyreType'].values)
                        type_t = b.most_common()[0][0]
                        self.tyre_tp.append(type_t)
                        
                        if type_t == 'A':
                            self.stint_on_red[i+1] = trend1[['Lap', 'LapTime']]
                        else:
                            self.stint_on_blk[i+1] = trend1[['Lap', 'LapTime']]
                
                if (i+1) == self.stints_num:
                    if len(trend2)>2:
                        coeffa, intercepta = np.polyfit(trend2.index.values, trend2.LapTime.values,1)
                        if coeffa>0 and coeffa< max_DAG:
                            self.DAG.append(coeffa)
                            got_DAG += 1
                            
                            #Which tyre
                            b = Counter(trend2['TyreType'].values)
                            self.tyre_tp.append(b.most_common()[0][0])
        
        if got_DAG >0:
            for st, tr in zip(self.tyre_tp, self.DAG):
                if st == 'P':
                    self.blacks_DAG.append(tr)
                if st == 'A':
                    self.reds_DAG.append(tr) 
     
            self.blacks_DAG = np.mean(np.array(self.blacks_DAG))
            self.reds_DAG = np.mean(np.array(self.reds_DAG))
        else:
            self.blacks_DAG = 0
            self.reds_DAG = 0
            
    
def fit(params, car, lap_times):
    
    model = Tyre_P(car, lap_times, params)
    model.apply_pits_w(params)  
    model.lap_filter(params)    
    model.diffs()
    
    for i in range(0, len(model.new_df)):
        model.tyre_age(i, params)
        model.fastest_lap(i, params)
        model.track_cont_descent(i)
        model.combine_probs(i)
        model.evaluate(i, params)
        
    
    #Evaluation complete, perform post-hoc
    if model.stints_num>0:
        model.post_hoc(car)
    
    model.get_DAG()
    return model




class ICReceiver:
    """
    It receives the stream through a TCP/IP socket and can output messages individually in a bytestring.
    
    https://docs.python.org/3/howto/sockets.html
    
    """

    def __init__(self, sock=None):
        """ Initializes a socket for the connection if none exists """
        if sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock

    def connect(self, host, port_score):
        """Opens a connection on the current socket according to a specified host and port"""

        self.sock.connect((host, port_score))
        
    def close(self):
        """Closes connection"""

        self.sock.close()

    def receiveAll(self, part=''):
        """ Returns a list of received messages, with the last index of the list being the remainder
        of the received data after the last SoR character. Any new line characters at the end of messages
        are stripped.
        """

        msgs = []
        # buff = part.encode('mbcs')
        buff = part

        searchpattern = '\$[^\n\r\$]+'

        while len(msgs) < 2:
            msgs = []
            buff += self.sock.recv(4096).decode('ISO-8859-1')
            # buff = buff.decode('mbcs')
            # buff = re.sub(b'TelemStream \xc2\xa92020, Basic Telemetry Feed', b'\n', buff)
            buff = re.sub('TelemStream.+Feed', '\n', buff)
            msgs = msgs + re.findall(searchpattern, buff)

        part = msgs[-1]
        msgs = msgs[0:-1]

        return msgs, part
    



#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
#------------------------------------------------------------------------------------------------------#
#......................                                                          ......................#
#......................                INDYCAR TIMING DATA SOCKET                ......................#
#......................                                                          ......................#
#------------------------------------------------------------------------------------------------------#
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#




class ReadDataSocket(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
        
    def run(self):

        global connection 

        time_diff_s = timeZone_offset(timeZone)
        time_diff_s_for_laptimes = 3600.0*4

        part = ' '
        msg = []

        while True:
            msg, part = connection.receiveAll(part)

            for line in msg:

                if line.startswith('$H'):

                    h_point = re.split('¦', line)

                    date_time = h_point[5]
                    date_time = int(date_time, 16)
                    adj_timestamp = date_time + time_diff_s_for_laptimes
                    t = datetime.fromtimestamp(adj_timestamp).time()
                    time_stamp = t.hour*3600.0 + t.minute*60.0+t.second
                    
                    #print('LAPTIMES!')
                    #print('Before adjustment: {}'.format(time_stamp-time_diff_s))
                    #print('After adjustment: {}'.format(t))
                    

                #Get timestamp first:
                if 'time_stamp' not in locals():
                    continue

                #LapTimes
                if line.startswith('$O'):

                    skip_entry = False

                    info = line.strip()
                    info = re.split('¦', info)
                    car = int(info[29])
                    lastname = info[31]
                
                    if lastname == 'Castroneves':
                        if special_entry:
                            car = 66
                        else:
                            car = 6

                    lap_num = int(info[13], 16)
                    lap_time = int(info[12],16)/10000

                    if lap_time != 0:
                        speed = track_distance/lap_time*3600
                        #if INDY500_filter:
                            #if speed>INDY500_filter:
                                #skip_entry = True
                    else:
                        speed = 0

                    t_type= info[-3]
                    if t_type == '0':
                        t_type = 'P'
                    if t_type == '1':
                        t_type = 'A'

                    status = info[24]
                    if status == '0':
                        status = 0
                    if status == '1':
                        status = 1
                    if status == 'P':
                        status = 0
                    if status == 'T':
                        status = 1
                    
                    #print('Car {} lap_times: {}'.format(car, time_stamp))
                
                    #Pits
                    pits_tot = int(info[25], 16)

                    #Last Pit Lap
                    last_pit_lap = int(info[26], 16)

                    if car not in list(lap_times.keys()):
                        lap_times[car] = []
                    if lap_num > 0 and lap_time>MIN_LAP_FILTER and not skip_entry:
                        if len(lap_times[car]) == 0:
                            lap_times[car].append([lap_num, lap_time, status, time_stamp, 
                                                   t_type, speed, pits_tot, last_pit_lap])
                        else:
                            if lap_times[car][-1][1] != lap_time:
                                lap_times[car].append([lap_num, lap_time, status, time_stamp, 
                                                       t_type, speed, pits_tot, last_pit_lap])

                if line.startswith('$P'):
                    
                    info = line.strip()
                    info = re.split('¦', info)
                    car = info[1]

                    if car == '06':
                        car = 6
                    else:
                        if car.isdigit():
                            car = int(car)
                        else:
                            continue

                    date_time = info[6]
                    t = datetime.strptime(date_time, time_format).time()
                    t = t.hour*3600.0 + t.minute*60.0+t.second
                    t_adj = t - time_diff_s
                    TOD = t_adj
                    
                    #print('Raw telem: {}'.format(t))
                    #print('Adjustment telem: {}'.format(time_diff_s))
                    
                    speed = float(info[8])
                    if speed>0:
                        lap_time = track_distance/speed*3600.0
                    else:
                        lap_time = 0

                    gear = int(info[10])
                    throttle = float(info[11])
                    brake = float(info[12])


                    if car not in list(telem_dict.keys()):
                        telem_dict[car] = []
                    if len(telem_dict[car]) == 0:
                        telem_dict[car].append([TOD, speed, throttle, gear, brake])
                    else:
                        if telem_dict[car][-1][0] != TOD:
                            telem_dict[car].append([TOD, speed, throttle, gear, brake])
   

   





########################################################################################################
#^^^^^^^^^^^^^^^^^^^^^^                                                          ^^^^^^^^^^^^^^^^^^^^^^#
#^^^^^^^^^^^^^^^^^^^^^^                    DASH APP STRUCTURE                    ^^^^^^^^^^^^^^^^^^^^^^#
#^^^^^^^^^^^^^^^^^^^^^^                                                          ^^^^^^^^^^^^^^^^^^^^^^#
########################################################################################################



class DashThread(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
        
    def run(self):
        global lap_times
        global app
        global DAG_dict
        global TYRE_STINTS
        global drivers_dict
        global drivers

     
        app.layout = html.Div([
            html.Center([
                html.H1('RLL Strategy Dash', style= {'text align':'center'}),
                html.H3('Field mean DEG:'),
                html.Div(id='glob_DAG_B', style=dict(color='black', fontWeight='bold', fontSize=30)),
                html.Div(id='glob_DAG_R', style=dict(color='red', fontWeight='bold', fontSize=30))
            ]),

            html.Div([
                dcc.Graph(id = 'live-graph-gpl', animate = False),
                dcc.Graph(id = 'live-graph-fuelused', animate = False),
                dcc.Interval(id = 'graph-update', interval = 30000, n_intervals=0)
            ]),

            html.Div([
                html.Div([
                    html.H3('DEG estimation:'),
                    html.Br(),
                    html.H4('Select your car:'),
                    dcc.RadioItems(id='select_car_DEG', 
                                   options= [15, 30, 45], value = 15, inline=False)
                ]),
                html.Div([
                    html.H4('Select car to compare:'),
                    dcc.Dropdown(id='select_car_DEG2', options = drivers_dict, value = None, multi=False)
                ],style={'display': 'inline-block', 'height':'10%', 'width':'30%'}),
                
                dcc.Graph(id = 'DEG-delta', animate = False)
            ]),

            html.Div([
                dash_table.DataTable(
                    id='DAG_table', data=[]),
                dcc.Interval(id = 'table-update', interval = 30000, n_intervals=0)
            ], style={'display': 'inline-block', 'height':'10%', 'width':'30%'}),
            
            html.Div([
                html.Div([
                    html.H4('Select car:'),
                    dcc.Dropdown(id='select_car', options = drivers_dict, value = 15, multi=False)
                ],style={'display': 'inline-block', 'height':'10%', 'width':'30%'}),
                dcc.Graph(id = 'DEG-graph', animate = False)
            ])
        ])







########################################################################################################
#^^^^^^^^^^^^^^^^^^^^^^                                                          ^^^^^^^^^^^^^^^^^^^^^^#
#^^^^^^^^^^^^^^^^^^^^^^                    CALLBACKS                             ^^^^^^^^^^^^^^^^^^^^^^#
#^^^^^^^^^^^^^^^^^^^^^^                                                          ^^^^^^^^^^^^^^^^^^^^^^#
########################################################################################################





        @app.callback(
            [Output('DAG_table', 'data'), Output('glob_DAG_B', 'children'), Output('glob_DAG_R', 'children')],
            [ Input('table-update', 'n_intervals') ]
        )
        def update_DAG_table(n_intervals):

            degs = np.zeros(len(drivers))
            stnts = np.ones(len(drivers))
            df = pd.DataFrame(drivers, columns=['Car'])
            df['Red DEG'] = degs
            df['Black DEG'] = degs
            df['Stints'] = stnts
            DAGs_R = []
            DAGs_B = []
            stints = []

            if lap_times:
                for car in drivers:
                    if car in list(lap_times.keys()):
                        if lap_times[car]:
                            model = fit(params, car, lap_times)
                            if model.blacks_DAG>0:
                                DAGs_B.append(model.blacks_DAG)
                                df.loc[df.Car==car, 'Black DEG'] = round(model.blacks_DAG,3)
                            if model.reds_DAG>0:
                                DAGs_R.append(model.reds_DAG)
                                df.loc[df.Car==car, 'Red DEG'] = round(model.reds_DAG,3)
                            if model.stints_num>0:
                                stints.append(model.stints_num)
                                df.loc[df.Car==car, 'Stints'] += model.stints_num
                if DAGs_R:
                    mean_DAG_R = np.mean(np.array(DAGs_R))
                else:
                    mean_DAG_R = 0
                if DAGs_B:
                    mean_DAG_B = np.mean(np.array(DAGs_B))
                else:
                    mean_DAG_B = 0
            else:
                mean_DAG_B = 0
                mean_DAG_R = 0
                
            return df.to_dict('records'), round(mean_DAG_B, 3), round(mean_DAG_R, 3)

        @app.callback(
            Output('DEG-graph', 'figure'),
            [ Input('select_car', 'value') ]
        )

        def update_DAG_graph(select_car):
            
            fig = go.Figure()  
            fig.update_layout(title='Car '+'<b>'+str(select_car)+'</b>'+' DEG', xaxis_title="Laps", 
                yaxis_title="Lap Time (s)", width=width_plot, height=height_plot) 

            if lap_times and select_car in list(lap_times.keys()):
                if lap_times[select_car]:
                    model = fit(params, select_car, lap_times)
                    fig = model.plot_stints(select_car)

            return fig


        @app.callback(
            [Output('live-graph-gpl', 'figure'), Output('live-graph-fuelused', 'figure')],
            [ Input('graph-update', 'n_intervals') ]
        )

        def get_fuel(n_intervals):
            
            columns_mpg = ['TOD', 'Speed', 'Throttle', 'Gear', 'Break']
            columns_lt = ['Lap', 'LapTime', 'Status', 'TOD', 'TyreType', 'Speed', 'PitsTotal', 'LastPitLap']
            
            #Load Fuel Map data
            #df_15 = pd.read_csv(r'C:\Users\FrantishekAkulich\OneDrive - Rahal Letterman Lanigan\Desktop/Fuel_Train_Data.csv')
            df_15 = pd.read_csv('Fuel_Train_Data.csv')
            X = df_15[['Lap time (sec)', 'Thr_Pos (%)']].values
            Y = df_15['FuelCount (USgal)'].values

            rfr = LinearRegression()
            rfr.fit(X, Y)

            ymin = 1000
            ymax = 0

            fig = go.Figure()
            fig2 = go.Figure()
            
            telem_dict_stat = copy.deepcopy(telem_dict)

            #Select car to iterate through
            if lap_times and telem_dict_stat:
                
                for car in list(lap_times.keys()):
                    if len(lap_times[car])<2 or len(telem_dict_stat[car])<2:
                        continue
                            
                    df_mpg = pd.DataFrame(telem_dict_stat[car], columns=columns_mpg)
                    if len(telem_dict_stat[car]) != df_mpg.shape[0]:
                        print('Car {} ERROR!'.format(car))
                    
                    df_mpg2 = pd.DataFrame(lap_times[car], columns=columns_lt)
                    model = fit(params, car, lap_times)
                    laps_when_pitted = model.stint_laps

                    thrott_arr = []

                    for lp in range(0, df_mpg2.shape[0]):

                        curr_lap = df_mpg2.Lap[lp]
                        lap_end = df_mpg2.TOD[lp]
                        
                        if lp == 0:
                            lap_start = lap_end - LAP_FILTER
                            if lap_start not in df_mpg['TOD'].values:
                                lap_start = lap_end
                                thrott_arr.append(np.array(Y).mean())
                                continue

                        idx = df_mpg['TOD'].between(lap_start, lap_end, inclusive=False)
                        avg_thr = df_mpg[idx].Throttle.values.mean()
                        thrott_arr.append(avg_thr)
                        lap_start = lap_end
                    
                    df_mpg2['ThrottleAvg'] = thrott_arr

                    df_mpg2 = df_mpg2.astype(object)
                    df_fuel_use = df_mpg2[['Lap', 'LapTime', 'ThrottleAvg']]
                    X_test = df_fuel_use.iloc[:,1:]

                    if X_test.isnull().values.any():
                        continue
                    
                    mpg_pred = rfr.predict(X_test.values)

                    for pred in range(0, len(mpg_pred)):
                        if X_test.iloc[pred, 0] > LAP_FILTER and mpg_pred[pred] > GPL_CUTOFF:
                            mpg_pred[pred] = np.array(Y).mean()

                    #pits_idx = df_mpg2[df_mpg2.PitsTotal.diff()>0].index.values
                    pits_idx = df_mpg2[df_mpg2.Lap.isin(laps_when_pitted)].index.values
                    cap_live = mpg_pred.copy()
                    fuel_use_arr = [capacity]

                    for i in range(0, df_mpg2.shape[0]):
                        if i in(pits_idx):
                            fuel_use_arr.append(capacity)
                            cap_live[:i] = 0
                            continue
                        else:
                            fuel_use_arr.append(capacity - cap_live.cumsum()[i])
                    
                    if car in list(scatter_color.keys()):
                        fig.add_trace(go.Scatter(y=mpg_pred, x = df_fuel_use.Lap.values, 
                                                 line=dict(color=scatter_color[car]), name=str(car)))
                        fig2.add_trace(go.Scatter(y=fuel_use_arr, x = df_fuel_use.Lap.values, 
                                                  line=dict(color=scatter_color[car]), name=str(car)))
                    else:
                        fig.add_trace(go.Scatter(y=mpg_pred, x = df_fuel_use.Lap.values, 
                                                 line=dict(color='lightgrey'), name=str(car)))
                        fig2.add_trace(go.Scatter(y=fuel_use_arr, x = df_fuel_use.Lap.values, 
                                                  line=dict(color='lightgrey'), name=str(car)))
                        
                    if min(mpg_pred) < ymin and min(mpg_pred)>0.2:
                        ymin = min(mpg_pred)
                    if max(mpg_pred) > ymax and max(mpg_pred)<1.6:
                        ymax = max(mpg_pred)
                                    
                    
                    
            fig.update_layout(title='Fuel Consumption', xaxis_title="Lap", yaxis_title='GPL', yaxis = dict(range=[ymin, ymax]), width=width_plot, height=height_plot)  
            fig2.update_layout(title='Net Fuel Usage', xaxis_title="Lap", yaxis_title='Fuel Remaining (Gal)', width=width_plot, height=height_plot)
            return fig, fig2





        @app.callback(
            Output('live-graph', 'figure'),
            [ Input('graph-update', 'n_intervals') ]
        )

        def update_graph_scatter(n_intervals):

            othercarscolor = 'yellow'
            minx = 1000
            maxx = 0
            fig = go.Figure()

            if road_course:
                title_main = 'Field Lap Times'
                title_y = 'Lap Time (s)'
            else:
                title_main = 'Field Speed'
                title_y = 'Speed (mph)'

            fig.update_layout(title=title_main, xaxis_title="Laps", yaxis_title=title_y, 
                    width=width_plot, height=height_plot)

            
            if lap_times:
                #print('Laptimes present')
                for car in list(lap_times.keys()):
                    #print('Laptime keys: {}'.format(lap_times.keys()))
                    if len(lap_times[car]) == 0:
                        #rint('For car {} no laptimes, skipping'.format(car))
                        continue

                    X = []
                    Y = []

                    for i in range(0, len(lap_times[car])):
                        if lap_times[car][i][1] < LAP_FILTER:
                            X.append(lap_times[car][i][0])
                            if road_course:
                                Y.append(lap_times[car][i][1])
                            else:
                                Y.append(lap_times[car][i][5])

                            if lap_times[car][i][0] < minx:
                                minx = lap_times[car][i][0]
                            if lap_times[car][i][0] > maxx:
                                maxx = lap_times[car][i][0]

                    if car in list(scatter_color.keys()):
                        plot = go.Scatter(x=X, y=Y, name=str(car), mode= 'lines+text', line=dict(color=scatter_color[car]))
                    else:
                        plot = go.Scatter(x=X, y=Y, name=str(car), mode= 'lines+text', line=dict(color=othercarscolor))
                    fig.add_trace(plot)


                    #plot = go.Scatter(x=X, y=Y, name=str(car), mode= 'lines+text')
                    #fig.add_trace(plot)
                    
                fig.update_layout(title=title_main, xaxis_title="Laps", yaxis_title=title_y, 
                                  xaxis = dict(range=[minx, maxx]), xaxis_rangeslider_visible=True, 
                                  width=width_plot, height=height_plot)    

            return fig



        @app.callback(
            Output('DEG-delta', 'figure'),
            [Input('select_car_DEG', 'value'), Input('select_car_DEG2', 'value')])

        def update_DAG_graph(select_car_DEG, select_car_DEG2):
            
            car1 = select_car_DEG
            car2 = select_car_DEG2

            laps = np.arange(1,MAX_LAPS,1)

            red_prog = []
            blk_prog = []
            blk_prog_field = []
            red_prog_field = []
            red_prog_car2 = []
            blk_prog_car2 = []

            #Field DEG
            DAGs_R = []
            DAGs_B = []
            mean_field_Lapt = []

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=laps, y=np.repeat(0, MAX_LAPS), name='Field DEG', 
                line = dict(color='grey', dash='dot')))

            if lap_times:
                if lap_times and car1 in list(lap_times.keys()):
                    if lap_times[car1]:
                        model = fit(params, car1, lap_times)


                        if model.stint_on_red:
                            keys = list(model.stint_on_red.keys())
                            last_stnt = len(keys)
                            red_arr = model.stint_on_red[keys[last_stnt-1]]
                            min_redtime = red_arr.LapTime.values.mean()
                            #red_laps = np.arange(1,len(red_arr),1)
                            last_stnt_idx = last_stnt-1
                            my_red_deg = model.DAG[last_stnt_idx]

                            #MY Red DEG
                            time = min_redtime
                            for i in laps:
                                red_prog.append(time)
                                time += my_red_deg


                        if model.stint_on_blk:
                            keys = list(model.stint_on_blk.keys())
                            last_stnt = len(keys)
                            blk_arr = model.stint_on_blk[keys[last_stnt-1]]
                            min_blktime = blk_arr.LapTime.values.mean()
                            #blk_laps = np.arange(1,len(blk_arr),1)
                            last_stnt_idx = last_stnt-1
                            my_blk_deg = model.DAG[last_stnt_idx]

                            #MY Black DEG
                            time = min_blktime    
                            for i in laps:
                                blk_prog.append(time)
                                time += my_blk_deg


                        if car2:
                            if lap_times and car2 in list(lap_times.keys()):
                                if lap_times[car2]:

                                    model = fit(params, car2, lap_times)

                                    if model.stint_on_red:
                                        keys = list(model.stint_on_red.keys())
                                        last_stnt = len(keys)
                                        red_arr = model.stint_on_red[keys[last_stnt-1]]
                                        min_redtime = red_arr.LapTime.values.mean()
                                        #red_laps = np.arange(1,len(red_arr),1)
                                        last_stnt_idx = last_stnt-1
                                        car2_red_deg = model.DAG[last_stnt_idx]

                                        #Car 2 Red DEG
                                        time = min_redtime
                                        for i in laps:
                                            red_prog_car2.append(time)
                                            time += car2_red_deg

                                    if model.stint_on_blk:
                                        keys = list(model.stint_on_blk.keys())
                                        last_stnt = len(keys)
                                        blk_arr = model.stint_on_blk[keys[last_stnt-1]]
                                        min_blktime = blk_arr.LapTime.values.mean()
                                        #blk_laps = np.arange(1,len(blk_arr),1)
                                        last_stnt_idx = last_stnt-1
                                        car2_blk_deg = model.DAG[last_stnt_idx]

                                        #MY Black DEG
                                        time = min_blktime    
                                        for i in laps:
                                            blk_prog_car2.append(time)
                                            time += car2_blk_deg


                        for car in drivers:
                            if car in list(lap_times.keys()):
                                if lap_times[car]:
                                    model = fit(params, car, lap_times)
                                    mean_field_Lapt.append(model.new_df.LapTime.values.mean())
                                    if model.blacks_DAG>0:
                                        DAGs_B.append(model.blacks_DAG)
                                    if model.reds_DAG>0:
                                        DAGs_R.append(model.reds_DAG)

                        if DAGs_R:
                            mean_DAG_R = np.mean(np.array(DAGs_R))
                        else:
                            mean_DAG_R = 0
                        if DAGs_B:
                            mean_DAG_B = np.mean(np.array(DAGs_B))
                        else:
                            mean_DAG_B = 0

                        notnan_arr = np.array(mean_field_Lapt)
                        notnan_arr = notnan_arr[~np.isnan(notnan_arr)]
                        field_time = notnan_arr.mean()


                        if mean_DAG_R != 0: 
                            #FIELD Red DEG
                            time = field_time
                            for i in laps:
                                red_prog_field.append(time)
                                time += mean_DAG_R

                        if mean_DAG_B != 0: 
                            #FIELD Black DEG
                            time = field_time    
                            for i in laps:
                                blk_prog_field.append(time)
                                time += mean_DAG_B   

                        if red_prog_field and red_prog:    
                            cumdiff_red = np.array(red_prog_field) - np.array(red_prog)
                            fig.add_trace(go.Scatter(x=laps, y=cumdiff_red, name='My A', line = dict(color='red')))

                        if blk_prog_field and blk_prog:   
                            cumdiff_blk = np.array(blk_prog_field) - np.array(blk_prog)
                            fig.add_trace(go.Scatter(x=laps, y=cumdiff_blk, name='My P', line = dict(color='black')))

                        if red_prog_field and red_prog_car2:    
                            cumdiff_red_2 = np.array(red_prog_field) - np.array(red_prog_car2)
                            fig.add_trace(go.Scatter(x=laps, y=cumdiff_red_2, 
                                                     name='A - Car #'+str(car2), line = dict(color='red', dash='dash')))

                        if blk_prog_field and blk_prog_car2:   
                            cumdiff_blk_2 = np.array(blk_prog_field) - np.array(blk_prog_car2)
                            fig.add_trace(go.Scatter(x=laps, y=cumdiff_blk_2, 
                                                     name='P - Car #'+str(car2), line = dict(color='black', dash='dash')))    

                        if car2:
                            fig.update_layout(title='DEG Compare Car '+'<b>'+ str(car1)+'</b>' +' & Car ' +'<b>'+str(car2)+'</b>', 
                                              xaxis_title='Laps', yaxis_title='Delta to Field (s)',
                                             width=width_plot, height=height_plot) 
                        else:
                            fig.update_layout(title='DEG Compare Car ' + str(car1), xaxis_title='Laps', 
                                              yaxis_title='Delta to Field (s)',
                                             width=width_plot, height=height_plot) 
            else:
                fig.update_layout(title='DEG Compare Car ' + str(car1), xaxis_title='Laps', 
                                          yaxis_title='Delta to Field (s)',
                                         width=width_plot, height=height_plot) 
                
            return fig
        
        #Run server
        #app.run_server(debug=False)
    


a = DashThread("The Dash Application")
b = ReadDataSocket("Time and Scoring Thread")
a.start()

connection = ICReceiver()
try:
    connection.connect(ip, port_score)
    connected = True
    print( "connected to server" ) 
    b.start()
except:
    try:
        print('Trying again to connect')
        connection.connect(ip, port_score)
        connected = True
        print( "connected to server" ) 
        b.start()
    except:
        print('Tried twice, exiting')
        connection.close()