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



class Tyre_P:
    
    def __init__(self, car, lap_times, params):
        
        self.laps_pitted = []
        columns = ['Lap', 'LapTime', 'Status', 'TimeOfDay']
        self.new_df = pd.DataFrame(lap_times[car], columns=columns)
        self.new_df.Status.values[self.new_df.Status =='P'] = 1
        self.new_df.Status.values[self.new_df.Status !=1] = 0
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
        self.LSL_idx = []
        
        self.stint_laps = []
        self.stint_info = []
        self.trend_comp = {}
        self.maxLTime = max(self.new_df.LapTime[self.new_df.LapTime.values < self.LapFilter])
        self.adj_breakline = []
        self.DAG = []
        
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
                
    def evaluate(self, i, params):
        #Probability treshold for assigning tyre change
        if self.new_df.Proba.values[i] >= params['prob']:
            self.LSL = self.new_df.Lap.values[i]
            self.LSL_idx.append(i)
            time_stamp = self.new_df.TimeOfDay.values[i]
            self.stint_info.append([self.LSL, time_stamp])
            self.stint_laps.append(self.LSL)
    
    def break_trends(self, break_point):
        trend1 = self.new_df[:break_point]
        trend2 = self.new_df[break_point:]
        return trend1, trend2
    
    def trend_decomp(self, stint_idx):

        scores = []
        offset = []
        max_idx = len(self.new_df) - self.LSL_idx[stint_idx]
        max_obs = max_idx-5
        if max_obs < 10:
            max_obs -= 1
        else:
            max_obs = 9

        for i in range(-5,max_obs):

            break_point = self.LSL_idx[stint_idx] + i
            trend1, trend2 = self.break_trends(break_point)
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
        if len(scores) == 0:
            self.adj_breakline.append(self.LSL_idx[stint_idx])
        else:
            obs = np.argmax(scores)
            self.adj_breakline.append(self.LSL_idx[stint_idx] + offset[obs])

    
    def post_hoc(self, car):
        self.stints_num = len(self.stint_laps)
        if self.stints_num ==1:
            self.trend_decomp(stint_idx=0)
        if self.stints_num ==2:
            self.trend_decomp(stint_idx=1)
        if self.stints_num ==3:
            self.trend_decomp(stint_idx=2)
        if self.stints_num ==4:
            self.trend_decomp(stint_idx=3)
        if self.stints_num ==5:
            self.trend_decomp(stint_idx=4)
        if self.stints_num ==6:
            self.trend_decomp(stint_idx=5)
            
            
    def plot_stints(self, car):  
        stints_num = len(self.stint_laps)
        #Plot DAG        
        timing_df = self.new_df[['Lap', 'LapTime']]
        fig1 = px.line(timing_df, x="Lap", y="LapTime") 
        fig_data = fig1.data
        
        if stints_num>0:
            for i in range(0, self.stints_num):
                trend1, trend2 = self.break_trends(self.adj_breakline[i])
                fig1 = px.scatter(trend1, x="Lap", y="LapTime", trendline="ols")
                fig2 = px.scatter(trend2, x="Lap", y="LapTime", trendline="ols")
                fig_data += fig1.data
                fig_data += fig2.data
            
        fig_f = go.Figure(data=(fig_data))
        
        if stints_num>0:
            for i in range(0, self.stints_num):
                x0b = self.new_df.Lap.values[self.adj_breakline[i]]
                x1b = self.new_df.Lap.values[self.LSL_idx[i]]

                if self.adj_breakline:
                    fig_f.add_shape(type='line', x0=x0b, y0=self.minLapTime,
                                x1=x0b, y1=self.maxLTime, 
                                line=dict(color='Red',), xref='x', yref='y')
                if self.LSL_idx[0]:
                    fig_f.add_shape(type='line', x0=x1b, y0=self.minLapTime,
                                    x1=x1b, y1=self.maxLTime, 
                                    line=dict(color='Red',), xref='x', yref='y')

                if self.adj_breakline and self.LSL_idx[0]:
                    fig_f.add_shape(type="rect", x0=x0b, y0=self.minLapTime,
                                    x1=x1b, y1=self.maxLTime,
                                    line=dict(color='Yellow',),
                                    fillcolor='Yellow', opacity = 0.5) 
            
        fig_f.update_layout(title='Car#'+str(car)+' DAG', xaxis_title="Laps", yaxis_title="Lap Time (s)")        
        return fig_f
    
    
    def get_DAG(self):
        stints_num = len(self.stint_laps)
        
        if stints_num==0:
            coeff, intercept = np.polyfit(self.new_df.index.values, self.new_df.LapTime.values,1)
            if coeff>0:
                self.DAG.append(coeff)
            
        if stints_num>0:
            for i in range(0, stints_num):
                trend1, trend2 = self.break_trends(self.adj_breakline[i])
                coeffb, interceptb = np.polyfit(trend1.index.values, trend1.LapTime.values,1)
                coeffa, intercepta = np.polyfit(trend2.index.values, trend2.LapTime.values,1)

                if coeffb>0 and i<1:
                    self.DAG.append(coeffb)
                if coeffa>0:
                    self.DAG.append(coeffa)
            
    
def fit(params, car, lap_times):
    
    model = Tyre_P(car, lap_times, params)
    model.apply_pits_w(params)  
    model.lap_filter(params)    
    model.diffs()
    
    for i in range(0, len(model.new_df)):
        model.tyre_age(i, params)
        model.fastest_lap(i, params)
        model.combine_probs(i)
        model.evaluate(i, params)
    
    #Evaluation complete, perform post-hoc
    if len(model.stint_laps)>0:
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


def read_data_socket():
    
    connection = ICReceiver()
    connection.connect(ip, port)
    connected = True
    print( "connected to server" )  

    part = ' '
    msg = []
    
    while True:
        try:
            msg, part = connection.receiveAll(part)

            for line in msg:
                #print(line)
                #Get time stamp
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
                if line.startswith('$C'):
                    lap = line.strip()
                    lap = re.split('¦', lap)
                    car = int(lap[5])
                    lap_time = int(lap[9],16)/10000 
                    lap_num = int(lap[7],16)
                    #Tack status
                    status = lap[20]

                    if car not in list(lap_times.keys()):
                        lap_times[car] = []
                    if len(lap_times[car]) == 0:
                        if lap_time > 0 and lap_time < LAP_FILTER:
                            lap_times[car].append([lap_num, lap_time, status, time_stamp])
                    else:
                        if lap_times[car][-1][1] != lap_time and lap_time > 0 and lap_time < LAP_FILTER:
                            lap_times[car].append([lap_num, lap_time, status, time_stamp])


        except socket.error:  
            # set connection status and recreate socket  
            connected = False  
            connection = ICReceiver()
            print("connection lost... reconnecting")  
            while not connected:  
                # attempt to reconnect, otherwise sleep for 2 seconds  
                try:  
                    connection.connect(ip, port) 
                    connected = True  
                    print("re-connection successful")  
                except socket.error:  
                    time.sleep(2) 

    connection.close();       
    
    
def run_tyre_dag():
    while True:
        for car in drivers:
            model = fit(params, car, lap_times)

            TYRE_STINTS[car] = model.stint_info
            if model.DAG:
                car_DAG = []
                for i in model.DAG:
                    DAG.append(i)
                    car_DAG.append(i)
                DAG_dict[car] = np.array(car_DAG).mean()
            else:
                DAG_dict[car] = 0
        time.sleep(20)
            

#HYPERPARAMS
LAP_FILTER = 30 #seconds
MAX_LAPS = 50 #information from Engineering meeting
CUTOFF_PROB = 0.8
###
DAG = []
DAG_dict = {}
TYRE_STINTS = {}

params = {'w_Lap1out': 1.5, 'w_Lap2out': 1.25, 'minLapTime':LAP_FILTER, 'maxL':MAX_LAPS, 'prob':CUTOFF_PROB,
         'adjAfterLaps':10, 'minLapW':2.0, 'firstMinLap':12}
  
load_old_df = True
if load_old_df:    
    file = '210502TEXR P1 Time Card.csv'
    drivers, lap_times, practice_df = load_filter_TimeCard(file)
else:
    lap_times = {}
    
drivers_dict = []
for car in drivers:
    car_dict = {'label': 'Car '+str(car), 'value': car}
    drivers_dict.append(car_dict)

#TIME & SCORE IP
ip = "indycar.livetiming.net"
port = 50005

    
#DASH INIT & LAYOUT    
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.Center([
        html.H1('RLL Tyre Stint & DAG Estimator', style= {'text align':'center'}),
        html.H3('Field mean DAG:'),
        html.Div(id='glob_DAG', style=dict(color='red', fontWeight='bold', fontSize=30))
    ]),
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
    if DAG_dict:
        for i in DAG_dict:
            if DAG_dict[i]>0:
                df.loc[df.Cars==i, 'DAG'] = round(DAG_dict[i],3)
                
    mean_DAG = np.array(DAG).mean()
    mean_DAG = round(mean_DAG, 3)

    return df.to_dict('records'), mean_DAG

@app.callback(
    Output('DAG-graph', 'figure'),
    [ Input('select_car', 'value') ]
)

def update_DAG_graph(select_car):
    model = fit(params, select_car, lap_times)
    fig = model.plot_stints(select_car)
    return fig
    

@app.callback(
    Output('live-graph', 'figure'),
    [ Input('graph-update', 'n_intervals') ]
)

def update_graph_scatter(n_intervals):

    minx = 1000
    maxx = 0
    fig = plotly.graph_objs.Figure()

    for car in list(lap_times.keys()):

        if len(lap_times[car]) == 0:
            continue

        #Binarize array

        arr_size = len(lap_times[car])
        bin_laps = [' '] * arr_size

        if car in list(TYRE_STINTS.keys()):
            if len(TYRE_STINTS[car]) > 0:
                for z in TYRE_STINTS[car]:
                    for i in range(0, arr_size):
                        if z == lap_times[car][i][0]:
                            bin_laps[i] = str(car) + lap_times[car][i][3]

        X = []
        Y = []

        for i in range(0, len(lap_times[car])):
            X.append(lap_times[car][i][0])
            Y.append(lap_times[car][i][1])
            if lap_times[car][i][0] < minx:
                minx = lap_times[car][i][0]
            if lap_times[car][i][0] > maxx:
                maxx = lap_times[car][i][0]

        plot = plotly.graph_objs.Scatter(x=X, y=Y, name=str(car), 
                                         mode= 'lines+text', text=bin_laps, textposition="bottom center")
        fig.add_trace(plot)
    fig.update_layout(title='Field Lap Times', xaxis_title="Laps", yaxis_title="Lap Time (s)",
                      xaxis = dict(range=[minx, maxx]), xaxis_rangeslider_visible=True)    
    return fig


    

def execute_socket():
    threading.Thread(target=read_data_socket).start()
    
def rus_estimation():
    threading.Thread(target=run_tyre_dag).start()

def start_app():
    threading.Thread(target=app.run_server(debug=False)).start()

if __name__ == '__main__': 
    execute_socket()
    rus_estimation()
    start_app()
