import re
import socket

import pandas as pd
import numpy as np
import math
from datetime import datetime
import time

import dash
from dash.dependencies import Output, Input, State
from dash import dcc, html, dash_table
import plotly
import plotly.express as px
import plotly.graph_objects as go

import threading

import logging 
logging.getLogger('werkzeug').setLevel(logging.ERROR)

#Main Dash app init
app = dash.Dash(__name__)
server = app.server

#Main dictionary to store laptimes data
lap_times = {}

#Get current time stamp offset and format
default_time = datetime(1970, 1, 1, 0, 0, 0) 
dateFromCurrentTime = datetime.fromtimestamp(0)
time_diff = default_time - dateFromCurrentTime
time_diff_s = time_diff.total_seconds()
time_format = "%I:%M:%S"

#HYPERPARAMS
LAP_FILTER = 75 #seconds 30
MAX_LAPS = 30 #information from Engineering meeting 50
CUTOFF_PROB = 0.8
###

DAG_dict = {}
TYRE_STINTS = {}

params = {'w_Lap1out': 1.5, 'w_Lap2out': 1.25, 'minLapTime':LAP_FILTER, 'maxL':MAX_LAPS, 'prob':CUTOFF_PROB,
         'adjAfterLaps':10, 'minLapW':2.0, 'firstMinLap':12}

#load_old_df = False
#if load_old_df:    
    #file = '210502TEXR P1 Time Card.csv'
    #drivers, lap_times, practice_df = load_filter_TimeCard(file)

#TEXAS 2022 ENTRY DRIVERS
drivers = np.array([2,3,4,5,6,7,8,9,10,11,12,14,15,18,20,21,26,27,28,29,30,33,45,48,51,60,77])

drivers_dict = []
for car in drivers:
    car_dict = {'label': 'Car '+str(car), 'value': car}
    drivers_dict.append(car_dict)

#TIME & SCORE IP
ip = "indycar.livetiming.net"
port = 50005


#Set time stamps

def get_time_diff():
    default_time = datetime(1970, 1, 1, 0, 0, 0) 
    dateFromCurrentTime = datetime.fromtimestamp(0)
    time_diff = default_time - dateFromCurrentTime
    time_diff_s = time_diff.total_seconds()
    time_format = "%I:%M:%S"
    return time_diff_s, time_format


def load_filter_TimeCard(file):

    global lap_times

    df1 = pd.read_csv(file)  
    drivers = np.unique(df1['Car'])

    practice_df = df1[['Car', 'Status', 'LapTime', 'Lap', 'TOD']]
    practice_df.Status.values[practice_df.Status =='P'] = 1
    practice_df.Status.values[practice_df.Status !=1] = 0
    practice_df = practice_df[practice_df.LapTime < LAP_FILTER]
    
    max_laps = max(practice_df.Lap)

    for car in drivers:
        lap_times[car] = []
        dr_df = practice_df[practice_df.Car == car]
        lap_t = dr_df.LapTime.values[0]
        lap_l = dr_df.Lap.values[0]
        lap_tod = dr_df.TOD.values[0]
        lap_pit = dr_df.Status.values[0]
        lap_times[car].append([lap_l, lap_t, lap_pit, lap_tod])
        for i in range(1, len(dr_df)):
            lap_t = dr_df.LapTime.values[i]
            lap_l = dr_df.Lap.values[i]
            lap_tod = dr_df.TOD.values[i]
            lap_pit = dr_df.Status.values[i]
            lap_times[car].append([lap_l, lap_t, lap_pit, lap_tod])
    return drivers, lap_times, practice_df



class Tyre_P:
    
    def __init__(self, car, lap_times, params):
        
        self.laps_pitted = []
        columns = ['Lap', 'LapTime', 'Status', 'TimeOfDay', 'TyreType']
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
        #Last Stint Lap (LPL)
        self.LSL = 0
        self.cont_diff = 0
        self.extended_tyreage = 0
        self.LSL_idx = []
        self.stints_num = 0
        self.stint_laps = []
        self.stint_info = []
        self.trend_comp = {}
        self.maxLTime = max(self.new_df.LapTime[self.new_df.LapTime.values < self.LapFilter])
        self.adj_breakline = []
        self.DAG = []
        self.med_laptimes = []
        
    def apply_pits_w(self, params):
        
        for iter_num in range(0, len(self.new_df)):
            if self.new_df.Status.values[iter_num] == 0 and iter_num>2:
                if self.new_df['Status W'].values[iter_num-1] == params['w_Lap1out']:
                    self.new_df['Status W'].values[iter_num] = params['w_Lap2out']

                if self.new_df.Status.values[iter_num-2] == 1:
                    self.laps_pitted.append(self.new_df.Lap.values[iter_num])
                    self.new_df['Status W'].values[iter_num] = params['w_Lap1out']
                
    def lap_filter(self, params):
        self.new_df = self.new_df[self.new_df.LapTime < self.LapFilter] 
    
    def diffs(self):
        
        self.dips = self.new_df['LapTime'].diff(periods=1).fillna(0)
        for i in range(0,len(self.dips)):
            self.dips.values[i] = -math.tanh(self.dips.values[i])
            
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
        
        p_tyre = self.new_df['Tyre W'].values[i]
        p_diff = self.new_df.LapDiff.values[i]
        p_pit = self.new_df['Status W'].values[i]
        
        #Combined probabilities
        if self.new_df.LapDiff.values[i] > 0:
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
                self.new_df.Proba.values[i] = self.new_df.Proba.values[i-1]+0.05
                
        #If no stint although tyres are pass maximum stint
        if self.extended_tyreage>0:
            self.new_df.Proba.values[i] = self.new_df.Proba.values[i] + (self.extended_tyreage*0.025)
                
    def evaluate(self, i, params):
        #If clear A->P or P->A tyre change
        if i>0:
            prev_tyre = self.new_df['TyreType'].values[i-1]
            curr_tyre = self.new_df['TyreType'].values[i]
            if prev_tyre != curr_tyre:
                self.new_df.Proba.values[i] = 1
                
        #Probability treshold for assigning tyre change
        if self.new_df.Proba.values[i] >= params['prob']:
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
        
        observations_id = np.arange(-5,max_obs)
        
        for i in range(-5,max_obs):

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
        
        #Plot DAG        
        timing_df = self.new_df[['Lap', 'LapTime']]
        fig1 = px.line(timing_df, x="Lap", y="LapTime") 
        fig_data = fig1.data
        
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

                trend_b = px.scatter(trend1, x="Lap", y="LapTime", trendline="ols")
                fig_data += trend_b.data
                
                if (i+1) == self.stints_num:
                    trend_a = px.scatter(trend2, x="Lap", y="LapTime", trendline="ols")
                    fig_data += trend_a.data
            
        fig_f = go.Figure(data=(fig_data))
        
        if self.stints_num>0:
            for i in range(0, self.stints_num):
                
                x1b = self.new_df.Lap.values[self.LSL_idx[i]]
                fig_f.add_shape(type='line', x0=x1b, y0=self.minLapTime,
                                x1=x1b, y1=self.maxLTime, 
                                line=dict(color='Red',), xref='x', yref='y')
                
                #print('Number of stints: {} and numner of breaklines: {}'.format(self.stints_num, len(self.adj_breakline)))
                
                if len(self.adj_breakline) != self.stints_num and (i+1) ==  len(self.adj_breakline):
                    break
                else:
                    x0b = self.new_df.Lap.values[self.adj_breakline[i]]
                    if x0b!=x1b:
                        fig_f.add_shape(type='line', x0=x0b, y0=self.minLapTime,
                                    x1=x0b, y1=self.maxLTime, 
                                    line=dict(color='Orange',), xref='x', yref='y')

                        fig_f.add_shape(type="rect", x0=x0b, y0=self.minLapTime,
                                        x1=x1b, y1=self.maxLTime,
                                        line=dict(color='Yellow',),
                                        fillcolor='Yellow', opacity = 0.5) 
            
        fig_f.update_layout(title='Car#'+str(car)+' DAG', xaxis_title="Laps", yaxis_title="Lap Time (s)")        
        return fig_f
    
    
    def get_DAG(self):
        #print('Model so far:')
        #print('Adj Breaklines')
        #print(self.adj_breakline)
        #print('Proba breaklines')
        #print(self.LSL_idx)
        
        if self.stints_num<1:
            if len(self.new_df.LapTime.values) > 2:
                coeff, intercept = np.polyfit(self.new_df.index.values, self.new_df.LapTime.values,1)
                if coeff>0:
                    self.DAG.append(coeff)

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
                    if coeffb>0:
                        self.DAG.append(coeffb)
                
                if (i+1) == self.stints_num:
                    if len(trend2)>2:
                        coeffa, intercepta = np.polyfit(trend2.index.values, trend2.LapTime.values,1)
                        if coeffa>0:
                            self.DAG.append(coeffa)
                
        if self.DAG:
            self.DAG = np.mean(np.array(self.DAG))
        else:
            self.DAG = 0
            
    
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

    def connect(self, host, port):
        """Opens a connection on the current socket according to a specified host and port"""

        self.sock.connect((host, port))
        
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
            buff += self.sock.recv(4096).decode('mbcs')
            # buff = buff.decode('mbcs')
            # buff = re.sub(b'TelemStream \xc2\xa92020, Basic Telemetry Feed', b'\n', buff)
            buff = re.sub('TelemStream.+Feed', '\n', buff)
            msgs = msgs + re.findall(searchpattern, buff)

        part = msgs[-1]
        msgs = msgs[0:-1]

        return msgs, part
    

#DASH APP STRUCTURE

class ReadDataSocket(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
        
    def run(self):

        global lap_times

        #path_log = r'C:\Users\FrantishekAkulich\OneDrive - Rahal Letterman Lanigan\Desktop/'
        path = ''
        file_name = 'Firestone Grand Prix of St. Petersburg-Race_R.I.log'
        #file1 = open(path_log+file_name, 'r+')
        file1 = open(file_name, 'r+')

        #for line in Lines:
        while True:
            line = file1.readline()
            line = line.strip()
            if not line:
                print('The Data Feed has ended')
                break

            if line.startswith('$H'):
                heartbeat = line.strip()
                h_point = re.split('¦', heartbeat)

                date_time = h_point[5]
                date_time = int(date_time, 16)
                adj_timestamp = date_time + time_diff_s
                time_stamp = datetime.fromtimestamp(adj_timestamp).strftime(time_format)

            #Get timestamp first:
            if 'time_stamp' not in locals():
                continue

            #LapTimes
            if line.startswith('$O'):
                #lap = line.strip()
                #lap = re.split('¦', lap)
                #car = int(lap[5])
                #lap_time = int(lap[9],16)/10000 
                #lap_num = int(lap[7],16)
                #Tack status
                #status = lap[10]
                
                info = line.strip()
                info = re.split('¦', info)
                car = int(info[29])
                lap_num = int(info[13], 16)
                lap_time = int(info[12],16)/10000
                t_type= info[-3]
                if t_type == '0':
                    t_type = 'P'
                if t_type == '1':
                    t_type = 'A'

                status = info[24]
                if status == '0':
                    status = 'P'
                if status == '1':
                    status = 'T'

                if car not in list(lap_times.keys()):
                    lap_times[car] = []
                if len(lap_times[car]) == 0:
                    if lap_time > 0 and lap_time < LAP_FILTER:
                        lap_times[car].append([lap_num, lap_time, status, time_stamp, t_type])
                else:
                    if lap_times[car][-1][1] != lap_time and lap_time > 0 and lap_time < LAP_FILTER:
                        lap_times[car].append([lap_num, lap_time, status, time_stamp, t_type])

            #time.sleep(0.001)        

    
class DashThread(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
        
    def run(self):
        global lap_times
        global app
        global server
        global DAG_dict
        global TYRE_STINTS
        global drivers_dict
        global drivers
        
        server = app.server        
        app.layout = html.Div([
            html.Center([
                html.H1('RLL Tyre Stint & DAG Estimator', style= {'text align':'center'}),
                html.H3('Field mean DAG:'),
                html.Div(id='glob_DAG', style=dict(color='red', fontWeight='bold', fontSize=30))
            ]),

            #Add button to enter lap time cutoff
            #Add section to select Session (P1-P3, Quali, Race)
            #Session determines if LSP should be used (or build continency around it)
            html.Div([
                dcc.Graph(id = 'live-graph', animate = False),
                dcc.Interval(id = 'graph-update', interval = 30000, n_intervals=0)
            ]),
            html.Div([
                html.Div([
                    html.H3('Select car:'),
                    dcc.Dropdown(id='select_car', options = drivers_dict, value = 2, multi=False)
                ],style={'display': 'inline-block', 'height':'10%', 'width':'30%'}),
                dcc.Graph(id = 'DAG-graph', animate = False)
            ]),
            html.Div([
                html.H3('DAG estimation:'),
                dash_table.DataTable(
                    id='DAG_table', data=[]),
                dcc.Interval(id = 'table-update', interval = 30000, n_intervals=0)
            ], style={'display': 'inline-block', 'height':'10%', 'width':'30%'})
        ])

        @app.callback(
            [Output('DAG_table', 'data'), Output('glob_DAG', 'children')],
            [ Input('table-update', 'n_intervals') ]
        )
        def update_DAG_table(n_intervals):

            degs = np.zeros(len(drivers))
            df = pd.DataFrame(drivers, columns=['Cars'])
            df['DAG'] = degs
            DAGs = []

            if lap_times:
                for car in drivers:
                    if car in list(lap_times.keys()):
                        if lap_times[car]:
                            model = fit(params, car, lap_times)
                            if model.DAG>0:
                                DAGs.append(model.DAG)
                                df.loc[df.Cars==car, 'DAG'] = round(model.DAG,3)
                if DAGs:
                    mean_DAG = np.mean(np.array(DAGs))
                else:
                    mean_DAG = 0
            else:
                mean_DAG = 0
            return df.to_dict('records'), round(mean_DAG, 3)

        @app.callback(
            Output('DAG-graph', 'figure'),
            [ Input('select_car', 'value') ]
        )

        def update_DAG_graph(select_car):
            #print('Update DAG graph')
            #print(lap_times)
            if lap_times and select_car in list(lap_times.keys()) and lap_times[select_car]:
                model = fit(params, select_car, lap_times)
                fig = model.plot_stints(select_car)
            else:
                fig = go.Figure()  
                fig.update_layout(title='Car#'+str(select_car)+' DAG', xaxis_title="Laps", yaxis_title="Lap Time (s)") 
                
            return fig


        @app.callback(
            Output('live-graph', 'figure'),
            [ Input('graph-update', 'n_intervals') ]
        )

        def update_graph_scatter(n_intervals):

            minx = 1000
            maxx = 0
            fig = go.Figure()
            
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
                        X.append(lap_times[car][i][0])
                        Y.append(lap_times[car][i][1])
                        #print('Here, car {} lap {} laptime {}'.format(car, lap_times[car][i][0], lap_times[car][i][1]))
                        if lap_times[car][i][0] < minx:
                            minx = lap_times[car][i][0]
                        if lap_times[car][i][0] > maxx:
                            maxx = lap_times[car][i][0]

                    plot = go.Scatter(x=X, y=Y, name=str(car), mode= 'lines+text')
                    fig.add_trace(plot)
                    
                fig.update_layout(title='Field Lap Times', xaxis_title="Laps", yaxis_title="Lap Time (s)", 
                                  xaxis = dict(range=[minx, maxx]), xaxis_rangeslider_visible=True)    
            else:
                fig.update_layout(title='Field Lap Times', xaxis_title="Laps", yaxis_title="Lap Time (s)")
                
            return fig
        
        #Run server
        app.run_server(debug=False)
    


if __name__ == '__main__': 
    
    a = DashThread("The Dash Application")
    b = ReadDataSocket("Time and Scoring Thread")
    
    a.start()
    b.start()
