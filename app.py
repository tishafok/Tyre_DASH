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
import threading

import logging 
logging.getLogger('werkzeug').setLevel(logging.ERROR)


#HYPERPARAMS
CUTOFF_PROB = 0.79
LAP_FILTER = 65 #seconds
###


### Session 2? Add session1
add_sess1 = True
#wd = '../'
path = 'P2 STGP.csv'


#TIME & SCORE IP
ip = "indycar.livetiming.net"
port = 50005


#Set time stamps

default_time = datetime(1970, 1, 1, 0, 0, 0) 
dateFromCurrentTime = datetime.fromtimestamp(0)
time_diff = default_time - dateFromCurrentTime
time_diff_s = time_diff.total_seconds()
#time_format = "%A, %B %d, %Y %I:%M:%S"
time_format = "%I:%M:%S"

#Variables to save
columns = ['Lap', 'LapTime', 'TimeOfDay']
lap_stint = []  
TYRE_DEG = {}
TYRE_DEG_DFS = {} 




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
    
    
### Session 2? Add session1

def load_sess1(file, LAP_FILTER):

    lap_times = {}

    df1 = pd.read_csv(file)  
    drivers = np.unique(df1['Car'])

    practice_df = df1[['Car', 'LapTime', 'Lap', 'TOD']]
    practice_df = practice_df[practice_df.LapTime < LAP_FILTER]
    
    #8AM start
    start_hr = 8
    new_stamps = []
    for i in range(0, len(practice_df)):
        
        h_point = re.split(':', practice_df.TOD.values[i])
        if len(h_point) == 3:
            hrs = int(h_point[0])
            if hrs < 12:
                hrs = int(h_point[0])+start_hr
                time_str_stamp = str(hrs)+':'+h_point[1]+':'+str(round(float(h_point[2])))
            if hrs >= 12:
                hrs = int(h_point[0])-12+start_hr
                time_str_stamp = str(hrs)+':'+h_point[1]+':'+str(round(float(h_point[2])))

        else:
            hrs = start_hr
            time_str_stamp = str(hrs)+':'+h_point[0]+':'+str(round(float(h_point[1])))
            
        new_stamps.append(time_str_stamp)
    practice_df['TOD'] = new_stamps

    for car in drivers:
        lap_times[car] = []
        dr_df = practice_df[practice_df.Car == car]
        lap_t = dr_df.LapTime.values[0]
        lap_l = dr_df.Lap.values[0]
        lap_tod = dr_df.TOD.values[0]
        lap_times[car].append([lap_l, lap_t, lap_tod])
        for i in range(1, len(dr_df)):
            lap_t = dr_df.LapTime.values[i]
            lap_l = dr_df.Lap.values[i]
            lap_tod = dr_df.TOD.values[i]
            lap_times[car].append([lap_l, lap_t, lap_tod])

    last_laps_am_sess = {}
    for car in lap_times:
        last_lap = lap_times[car][-1][0]
        last_laps_am_sess[car] = last_lap
    
    return lap_times


def run_estimation(new_df, last_lap_stint_happ, minLapTime, stint_adjusted, last_stint_idx, max_observe):
    i=0
    observe = 0
    found_lower_laps=0
    while i<len(new_df):

        #If no tyre change was assigned, extend high tyre age probability to next lap
        tyre_w_val = new_df.Lap.values[i]-last_lap_stint_happ
        if tyre_w_val > 30:
            tyre_w_val = 30

        #Assign tyre age probabilities
        new_df['Tyre W'].values[i] = stint_length_w[tyre_w_val]
        #Assign normalized differentials
        new_df.LapDiff.values[i] = dips.values[i]

        #All-time lowest laptime condition                                #THIS IS SKETCH
        if new_df.LapTime.values[i] < minLapTime:
            minLapTime = new_df.LapTime.values[i]
            minLapTimeL = new_df.Lap.values[i]
            #IDs
            conseq_mins_id.append(i)
            #Min lap to start cosidering for probability adjustment is 20
            #print('Found new min lap {}'.format(minLapTimeL))
            if (new_df.Lap.values[i]-last_lap_stint_happ)>10:
                if new_df.LapDiff.values[i] < new_df.LapDiff.values[i-1]:
                    new_df.LapDiff.values[i-1] = new_df.LapDiff.values[i-1]*all_time_weight
                    i-=1
                else:
                    new_df.LapDiff.values[i] = new_df.LapDiff.values[i]*all_time_weight


        #Combined probabilities
        if new_df.LapDiff.values[i] > 0:
            new_df.Proba.values[i] = round(new_df['Tyre W'].values[i] * new_df.LapDiff.values[i],2)
        else:
            reduced_prob = round(new_df.Proba.values[i-1] + (new_df['Tyre W'].values[i]*new_df.LapDiff.values[i]),2)
            if reduced_prob >= 0 and new_df.Lap.values[i-1] not in just_laps:
                new_df.Proba.values[i] = reduced_prob
            else:
                new_df.Proba.values[i] = 0 

        #Probability treshold for assigning tyre change
        if new_df.Proba.values[i] >= CUTOFF_PROB:
            last_lap_stint_happ = new_df.Lap.values[i]
            lap_time_stamp = new_df_full.TimeOfDay.values[i]
            lap_stint.append(last_lap_stint_happ)
            just_laps.append(last_lap_stint_happ)
            last_stint_idx = i
            observe = 1
            #print('New tyre stint lap added {}'.format(last_lap_stint_happ))


        #Reset minimum lap time for second stint analysis
        if len(just_laps) > 0 and i==(last_stint_idx+max_observe):
            #print('Min lap time resetted!')
            minLapTime = minLapTime+0.1
            minLapTimeL = minLapTimeL+1

        obs_id = last_stint_idx+observe
        if len(new_df) <= obs_id:
            obs_id = len(new_df)-1
        obs_lap = new_df.Lap[obs_id]

        #Post-hoc analysis
        if len(just_laps) > 0 and i == obs_id and observe<max_observe and obs_lap < MAX_STNT:

            last_stint_a = new_df.LapTime[(last_stint_idx+1):(obs_id+1)]
            pm = np.array([-1, +observe])
            pl, mn = last_lap_stint_happ + pm
            min_lap_range = np.arange(pl, mn+1)
            observe +=1

            #print('Min lap {} in range {}'.format(minLapTimeL, min_lap_range))
            if new_df.Lap[conseq_mins_id[-1]] in min_lap_range or new_df.Lap[conseq_mins_id[-2]] in min_lap_range:
                if minLapTimeL!= last_lap_stint_happ:
                    #print('Found!')
                    #i+=1
                    #continue
                    #Check if several last laps were recorded as min
                    lap_dev = last_lap_stint_happ - new_df.Lap[conseq_mins_id[-1]]
                    if len(conseq_mins_id)>1 and lap_dev<observe:
                        min_diff = conseq_mins_id[-1] - conseq_mins_id[-2]
                        if min_diff == 1 or min_diff == 2:
                            if last_stint_idx == conseq_mins_id[-1] or last_stint_idx == conseq_mins_id[-2]:
                                #print('Last two laps were min')
                                #conseq_mins_id.pop()
                                i+=1
                                continue

                if observe == max_observe:

                    found_lower_laps = 0
                    lap_stint.pop()
                    just_laps.pop()
                    new_df.Proba.values[last_stint_idx] = CUTOFF_PROB 
                    i = last_stint_idx
                    last_stint_idx = 0
                    last_lap_stint_happ = 0

        i+=1
        
        
        
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

                    #If session 2, lap numbers are adjusted
                    if add_sess1:
                        lap_num = last_laps_am_sess[car] + lap_num


                    if car not in list(lap_times.keys()):
                        lap_times[car] = []
                    if len(lap_times[car]) == 0:
                        if lap_time > 0 and lap_time < LAP_FILTER:
                            lap_times[car].append([lap_num, lap_time, time_stamp])
                    else:
                        if lap_times[car][-1][1] != lap_time and lap_time > 0 and lap_time < LAP_FILTER:
                            lap_times[car].append([lap_num, lap_time, time_stamp])

                        #Initialize dataframe
                        new_df_full = pd.DataFrame(lap_times[car], columns=columns)
                        new_df = new_df_full[['Lap', 'LapTime']]
                        laps = np.zeros(len(new_df))
                        new_df['Tyre W'] = laps
                        new_df['LapDiff'] = laps
                        new_df['Proba'] = laps
                        new_df.Proba.values[0] = 0.05

                        #Generate differentials
                        dips = new_df['LapTime'].diff(periods=1).fillna(0)
                        for i in range(0,len(dips)):
                            dips.values[i] = -math.tanh(dips.values[i])

                        #Assign tyre age probabilities
                        stint_length_w = np.arange(0.001,1, 0.033)

                        #Variables to fill
                        lap_stint = []
                        just_laps = []

                        #Hyperparams
                        last_lap_stint_happ = 0
                        max_observe = 10
                        minLapTime = LAP_FILTER
                        all_time_weight = 2.0
                        stint_adjusted = False
                        last_stint_idx = 0
                        MAX_STNT = 43
                        conseq_mins_id = []

                        run_estimation(new_df, last_lap_stint_happ, minLapTime, stint_adjusted, last_stint_idx, max_observe)

                        TYRE_DEG[car] = lap_stint
                        TYRE_DEG_DFS[car] = new_df

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
            

def dashboard(app):
    #print('RUNNING DASH')
    
    app.layout = html.Div([
        dcc.Graph(id = 'live-graph', animate = False),
        dcc.Interval(id = 'graph-update', interval = 30000, n_intervals=0)
    ])

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

            if car in list(TYRE_DEG.keys()):
                if len(TYRE_DEG[car]) > 0:
                    for z in TYRE_DEG[car]:
                        for i in range(0, arr_size):
                            if z == lap_times[car][i][0]:
                                bin_laps[i] = str(car) + lap_times[car][i][2]

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

        fig.update_layout(xaxis = dict(range=[minx, maxx]), xaxis_rangeslider_visible=True)    
        return fig
    
    
    
#INITIATE CONNECTION

#t1 = threading.Thread(target=read_data_socket)
#t2 = threading.Thread(target=dashboard)
#t1.start()
#print('Read data thread started')
#t2.start()

if add_sess1:
    lap_times = load_sess1(path, LAP_FILTER)
else:
    lap_times = {}
    
    
app = dash.Dash(__name__)
server = app.server

def execute_this():
    threading.Thread(target=read_data_socket).start()

def start_app(app):
    threading.Thread(target=dashboard(app)).start()

if __name__ == '__main__': 
    execute_this()
    start_app(app)
    app.run_server(debug=False) 
