from tkinter import N
import numpy as np    
import socket
import threading
import time
import joblib
import json
import os
from device import Device
from device1 import Device1
from device2 import Device2
from device3 import Device3
from device4 import Device4
import pandas as pd

from streaming_server import Server
from streaming_server1 import Server1
from streaming_server2 import Server2
from streaming_server3 import Server3
from streaming_server4 import Server4

from dsp_utils import DSPUtils
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters


DATA_COLLECTION_FOLDER = "./activity_data/"
activity_list = ["Writing",  "Clapping" , "Grating",   "Drinking", "Chewing","Coughing","Snoring", "Speaking", "Walking",  "Cleaning", "Crunch", "Using toothbrush",  "Noise"]


## Is it rewriting its data? Is it the last 6 sec or the first part? Is it running for 6 sec?How long?

def collect_data(devices, streaming_servers):
    thing_name = input("Item Name :\n")
    if not os.path.exists(DATA_COLLECTION_FOLDER + thing_name + "/"):
        os.makedirs(DATA_COLLECTION_FOLDER + thing_name + "/")

    for i, (device, streaming_server) in enumerate(zip(devices, streaming_servers)):
            user_name = input(f"Participant Name for device {i + 1}:\n")
            device_folder = DATA_COLLECTION_FOLDER + thing_name + f'device{i + 1}' + "/"
            user_device_folder = device_folder + user_name + f'device{i + 1}' + "/"

            if not os.path.exists(user_device_folder):
                os.makedirs(user_device_folder)

                

    record_data = [[] for _ in range(5)]
    is_recording = False
    overwrite = False
    calibrated = False
    activity_index = 0

    background_fft_profiles = [device.calculate_background_fft_profile() for device in devices]

    while True:
        #print("it's true:)")
        cmd = streaming_servers[0].read_client_command()
        cmd1 = streaming_servers[1].read_client_command()
        cmd2 = streaming_servers[2].read_client_command()
        cmd3 = streaming_servers[3].read_client_command()
        cmd4 = streaming_servers[4].read_client_command()
        
        #cmd=cmd[0] or cmd [1] or cmd[2] or cmd[3] or cmd[4]

        if cmd or cmd1 or cmd2 or cmd3 or cmd4:
            if cmd == 'Z' or cmd1 == 'Z' or cmd2 == 'Z' or cmd3 == 'Z' or cmd4 == 'Z':
                is_recording = True
                record_data[i] = []
                start_time = time.time()
                print(f"Recording data for device {i + 1}...")
            elif cmd == 'Y' or cmd1 == 'Y' or cmd2 == 'Y' or cmd3 == 'Y' or cmd4 == 'Y':
                is_recording = False
            elif cmd == 'X' or cmd1 == 'X' or cmd2 == 'X' or cmd3 == 'X' or cmd4 == 'X':
                print(f"Overwriting the last one for device {i + 1}")
                overwrite = True
            elif cmd == 'W' or cmd1 == 'W' or cmd2 == 'W' or cmd3 == 'W' or cmd4 == 'W':
                for i,device in devices:
                    background_fft_profiles[i] = device.calculate_background_fft_profile()
                    print(f"Update background profile for device {i + 1}")
            else:
                o_index = ord(cmd) - ord('A')
                if 0 <= o_index < len(activity_list):
                    activity_index = o_index
                    for i,device in enumerate(zip(devices)):
                        print(f"Record activity {activity_list[activity_index]} for device {i + 1}")

        for i, (device, streaming_server) in enumerate(zip(devices, streaming_servers)):
            signal_in_one_window = device.sample()

            if len(signal_in_one_window) > 0:
                streaming_server.streaming_signal_in_FFT(signal_in_one_window, background_fft_profiles[i])
                if is_recording:
                    record_data[i].append(signal_in_one_window.tolist())
                    print(f"Elapsed time for recording {i + 1}: {time.time() - start_time}")
                    if time.time() - start_time > 6:
                        is_recording = False
        for i, (device) in enumerate(zip(devices)):

            if len(record_data[i]) > 0 and not is_recording:
                try:
                    with open(user_device_folder + activity_list[activity_index] + f'device{i + 1}' + '.json', "r") as file:
                        listObj = json.load(file)
                except:
                    listObj = []
                    print("New file")
#        for i, (device) in enumerate(zip(devices)):
                with open(user_device_folder + activity_list[activity_index] + f'device{i + 1}' + '.json', "w+") as file:
                    print(activity_list[activity_index])
                    print(len(listObj))
                    if overwrite and len(listObj) > 0:
                        listObj[-1] = {"background": background_fft_profiles[i].tolist(), "record_data": record_data[i]}
                        overwrite = False
                    else:
                        listObj.append({"background": background_fft_profiles[i].tolist(), "record_data": record_data[i]})
                    json.dump(listObj, file, allow_nan=True)
                    print(f"Record for device {i + 1} finished")
                record_data[i] = []



def demo(devices, streaming_servers):
        POLL_SIZE = 10
        PREDICTION_WINDOW_SIZE = 12
        poll = []

        model_file = './model/'+'ex32_RF_modelCoughing_speaking'
        loaded_model = joblib.load(model_file)


        # Define common data lists and background FFT profiles
        windows = [[] for _ in range(5)]
        
        
        background_fft_profiles = [device.calculate_background_fft_profile() for device in devices]

        while True:
            cmds = [server.read_client_command() for server in streaming_servers]
    
            if any(cmd == 'Z' for cmd in cmds):
                background_fft_profiles = [device.calculate_background_fft_profile() for device in devices]
                print("update background profile")

            data_list = [device.sample().tolist() for device in devices]

            for i, streaming_server in enumerate(streaming_servers):
                streaming_server.streaming_signal_in_FFT(data_list[i], background_fft_profiles[i])

            print("Receiving1")
            data_lengths = [len(data) for data in data_list]
            print("length data", data_lengths)

            if all(data and len(data) > 0 for data in data_list):
                windows = [w + d for w, d in zip(windows, data_list)]
                print("Receiving2")

            window_lengths = [len(win) for win in windows]
            print("length windows:", window_lengths)

            if window_lengths[0] > PREDICTION_WINDOW_SIZE:
                windows = [w[1:] for w in windows]
                print("Receiving3")
                print("length windows:", window_lengths)

            if all(len(win) >= PREDICTION_WINDOW_SIZE for win in windows):
                record_sigs = [element for sublist in windows for element in sublist]
                print("Receiving4")

                fft_windows_list = [DSPUtils.segment_along_windows(win, background_fft_profile, Device.BUFFER_SIZE, Device.SHIFT_SIZE) for win, background_fft_profile in zip(windows, background_fft_profiles)]
        
                features_list = [DSPUtils.extract_feature(fft_win) for fft_win in fft_windows_list]

                predictions = [loaded_model.predict(features.reshape(1, -1)) for features in features_list]

                poll.append(predictions[0])
                print(predictions[0])

            if len(poll) > POLL_SIZE:
                poll.pop(0)
                print("Polling")

            if len(poll) >= POLL_SIZE:
                max_occur = max(poll, key=poll.count)
                if poll.count(max_occur) >= POLL_SIZE / 2:
                    data_string = "result," + max_occur + '\n'
                    for streaming_server in streaming_servers:
                        streaming_server.enqueue(data_string)
                    print(max_occur)
            else:
                data_string = "result" + "Noise" + '\n'
                for streaming_server in streaming_servers:
                    streaming_server.enqueue(data_string)


def main():
    device = Device(Device.SAMPLE_DEVICE_ANALOG_DISCOVERY)
    device1 = Device1(Device1.SAMPLE_DEVICE_ANALOG_DISCOVERY)
    device2 = Device2(Device2.SAMPLE_DEVICE_ANALOG_DISCOVERY)
    device3 = Device3(Device3.SAMPLE_DEVICE_ANALOG_DISCOVERY)
    device4 = Device4(Device4.SAMPLE_DEVICE_ANALOG_DISCOVERY)

    print('start server')

    streaming_server = Server('0.0.0.0', 8080)
    streaming_server1 = Server1('0.0.0.0', 8081)
    streaming_server2 = Server2('0.0.0.0', 8082)
    streaming_server3 = Server3('0.0.0.0', 8083)
    streaming_server4 = Server4('0.0.0.0', 8084)


    streaming_server.start_server()
    time.sleep(1)
    streaming_server1.start_server()
    time.sleep(1)
    streaming_server2.start_server()
    time.sleep(1)
    streaming_server3.start_server()
    time.sleep(1)
    streaming_server4.start_server()
    time.sleep(1)

    devices=[device,device1,device2,device3,device4]
    streaming_servers=[streaming_server,streaming_server1,streaming_server2,streaming_server3,streaming_server4]

    try:
        server_use = input("Please enter what you are going to do? (0: data), (1:demo) :\n")
        if server_use == '0':
            collect_data(devices,streaming_servers)
        elif server_use == '1':
            demo(devices,streaming_servers) 
          
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating server")


if __name__ == '__main__':
    main()
    # print(load_bitmasks())