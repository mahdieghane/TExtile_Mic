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

def collect_data(device,device1,device2,device3,device4, streaming_server,streaming_server1,streaming_server2,streaming_server3,streaming_server4):
    thing_name = input("Item Name :\n")
    #thing_name = "cuttingboard"
    if not os.path.exists(DATA_COLLECTION_FOLDER+thing_name + "/"):
        os.makedirs(DATA_COLLECTION_FOLDER+thing_name + "/")
    user_name = input("Participant Name :\n")
    #user_name = "p0"
    if not os.path.exists(DATA_COLLECTION_FOLDER+thing_name + "/" +user_name + "/"):
        os.makedirs(DATA_COLLECTION_FOLDER+thing_name + "/" + user_name + "/")

    if not os.path.exists(DATA_COLLECTION_FOLDER+thing_name + 'device1'+ "/"):
        os.makedirs(DATA_COLLECTION_FOLDER+thing_name + 'device1'+"/")
    user_name = input("Participant Name :\n")
    #user_name = "p0"
    if not os.path.exists(DATA_COLLECTION_FOLDER+thing_name + 'device1'+ "/" +user_name + 'device1'+"/"):
        os.makedirs(DATA_COLLECTION_FOLDER+thing_name + 'device1'+ "/" + user_name + 'device1'+"/")


    if not os.path.exists(DATA_COLLECTION_FOLDER+thing_name + 'device2'+ "/"):
        os.makedirs(DATA_COLLECTION_FOLDER+thing_name + 'device2'+"/")
    user_name = input("Participant Name :\n")
    #user_name = "p0"
    if not os.path.exists(DATA_COLLECTION_FOLDER+thing_name+ 'device2' + "/" +user_name + 'device2'+"/"):
        os.makedirs(DATA_COLLECTION_FOLDER+thing_name+ 'device2' + "/" + user_name + 'device2'+"/")

    if not os.path.exists(DATA_COLLECTION_FOLDER+thing_name + 'device3'+ "/"):
        os.makedirs(DATA_COLLECTION_FOLDER+thing_name + 'device3'+"/")
    user_name = input("Participant Name :\n")
    #user_name = "p0"
    if not os.path.exists(DATA_COLLECTION_FOLDER+thing_name+ 'device3' + "/" +user_name + 'device3'+"/"):
        os.makedirs(DATA_COLLECTION_FOLDER+thing_name+ 'device3' + "/" + user_name + 'device3'+"/")


    if not os.path.exists(DATA_COLLECTION_FOLDER+thing_name + 'device4'+ "/"):
        os.makedirs(DATA_COLLECTION_FOLDER+thing_name + 'device4'+"/")
    user_name = input("Participant Name :\n")
    #user_name = "p0"
    if not os.path.exists(DATA_COLLECTION_FOLDER+thing_name+ 'device4' + "/" +user_name + 'device4'+"/"):
        os.makedirs(DATA_COLLECTION_FOLDER+thing_name+ 'device4' + "/" + user_name + 'device4'+"/")



    record_data = []
    record_data1 = []
    record_data2 = []
    record_data3 = []
    record_data4 = []

    is_recording = False
    record_starting_time = 0,

    overwrite = False
    calibrated = False
    activity_index = 0
    
    background_fft_profile = device.calculate_background_fft_profile()
    background_fft_profile1 = device1.calculate_background_fft_profile()
    background_fft_profile2 = device2.calculate_background_fft_profile()
    background_fft_profile3 = device3.calculate_background_fft_profile()
    background_fft_profile4 = device4.calculate_background_fft_profile()

    #### calibrate

    while True:

        ### get and interpret the command from the client
        cmd = streaming_server.read_client_command()
        cmd1 = streaming_server1.read_client_command()
        cmd2 = streaming_server2.read_client_command()
        cmd3 = streaming_server3.read_client_command()
        cmd4 = streaming_server4.read_client_command()

        if cmd or cmd1 or cmd2 or cmd3 or cmd4:
            if cmd == 'Z' or cmd1 == 'Z' or cmd2 == 'Z' or cmd3 == 'Z' or cmd4 == 'Z':
                is_recording = True
                record_data = []
                start_time = time.time()
                print("recording data...")
            elif cmd == 'Y' or cmd1 == 'Y' or cmd2 == 'Y' or cmd3 == 'Y' or cmd4 == 'Y':
                is_recording = False
            elif cmd == 'X' or cmd1 == 'X' or cmd2 == 'X' or cmd3 == 'X' or cmd4 == 'X':
                print("overwrite the last one")
                overwrite = True
            elif cmd == 'W' or cmd1 == 'W' or cmd2 == 'W' or cmd3 == 'W' or cmd4 == 'W':
                background_fft_profile = device.calculate_background_fft_profile()
                background_fft_profile1 = device1.calculate_background_fft_profile()
                background_fft_profile2 = device2.calculate_background_fft_profile()
                background_fft_profile3 = device3.calculate_background_fft_profile()
                background_fft_profile4 = device4.calculate_background_fft_profile()
                print("update background profile")
            else:
                print("cmd is", cmd)
                print("commend type=",type(cmd))
                o_index = ord(cmd)-ord('A')
                print(o_index)
                if 0 <= o_index < len(activity_list):
                    activity_index = o_index
                    print("record activity " + activity_list[activity_index]) 

        signal_in_one_window = device.sample()
        #print("recording 1",time.time() )
        signal_in_one_window1 = device1.sample()
        #print("recording 2",time.time() )
        signal_in_one_window2 = device2.sample()
        #print("recording 3",time.time() )
        signal_in_one_window3 = device3.sample()
        #print("recording 4",time.time() )
        signal_in_one_window4 = device4.sample()
        #print("recording 5",time.time() )

        if len(signal_in_one_window) > 0 :
            
            streaming_server.streaming_signal_in_FFT(signal_in_one_window, background_fft_profile)
            if is_recording:
                record_data.append(signal_in_one_window.tolist())
                print("elapsed time for recording 0",time.time() - start_time)
                if time.time() - start_time > 6:
                     is_recording = False

        if len(signal_in_one_window1) > 0 :
            streaming_server1.streaming_signal_in_FFT(signal_in_one_window1, background_fft_profile1)
            if is_recording:
                record_data1.append(signal_in_one_window1.tolist())
                print("elapsed time for recording 1",time.time() - start_time)
                if time.time() - start_time > 6:
                     is_recording = False


        if len(signal_in_one_window2) > 0 :
            streaming_server2.streaming_signal_in_FFT(signal_in_one_window2, background_fft_profile2)
            if is_recording:
                record_data2.append(signal_in_one_window2.tolist())
                print("elapsed time for recording 2",time.time() - start_time)
                if time.time() - start_time > 6:
                     is_recording = False


        if len(signal_in_one_window3) > 0 :
            streaming_server3.streaming_signal_in_FFT(signal_in_one_window3, background_fft_profile3)
            if is_recording:
                record_data3.append(signal_in_one_window3.tolist())
                print("elapsed time for recording 3",time.time() - start_time)
                if time.time() - start_time > 6:
                     is_recording = False


        if len(signal_in_one_window4) > 0 :
            streaming_server4.streaming_signal_in_FFT(signal_in_one_window4, background_fft_profile4)
            if is_recording:
                record_data4.append(signal_in_one_window4.tolist())
                print("elapsed time for recording 4",time.time() - start_time)
                if time.time() - start_time > 6:
                     is_recording = False

                
        if len(record_data) > 0 and not is_recording:
            
            print("Record 1 started")
            ## create file if it doesn't exist or 
            ## append data to the file if it exists
            try:
                with open(DATA_COLLECTION_FOLDER + thing_name + "/" + user_name + "/" + activity_list[activity_index]+'.json', "r") as file:
                    listObj = json.load(file)
            except:
                listObj = []
                print("new file")

            ### store the data into the list
            with open(DATA_COLLECTION_FOLDER+thing_name + "/" + user_name + "/"+ activity_list[activity_index]+'.json', "w+") as file:
                print("Storing Data 1")
                print(activity_list[activity_index])
                print(len(listObj))
                if overwrite and len(listObj) > 0:
                    listObj[-1] = {"background": background_fft_profile.tolist(), "record_data":record_data}
                    overwrite = False
                else:
                    listObj.append({"background": background_fft_profile.tolist(), "record_data":record_data})
                json.dump(listObj, file, allow_nan = True)
                print("Record 1 finished")
            record_data = []


        if len(record_data1) > 0 and not is_recording:
            
            ## create file if it doesn't exist or 
            ## append data to the file if it exists
            try:
                with open(DATA_COLLECTION_FOLDER + thing_name+'device1' + "/" + user_name +'device1'+ "/" + activity_list[activity_index]+'device1'+'.json', "r") as file:
                    listObj = json.load(file)
            except:
                listObj = []
                print("new file")

            ### store the data into the list
            with open(DATA_COLLECTION_FOLDER+thing_name +'device1'+ "/" + user_name +'device1'+ "/"+ activity_list[activity_index]+'device1'+'.json', "w+") as file:
                print(activity_list[activity_index])
                print(len(listObj))
                if overwrite and len(listObj) > 0:
                    listObj[-1] = {"background": background_fft_profile1.tolist(), "record_data":record_data1}
                    overwrite = False
                else:
                    listObj.append({"background": background_fft_profile1.tolist(), "record_data":record_data1})
                json.dump(listObj, file, allow_nan = True)
                print("Record Finish")
            record_data1 = []


        if len(record_data2) > 0 and not is_recording:
            
            ## create file if it doesn't exist or 
            ## append data to the file if it exists
            try:
                with open(DATA_COLLECTION_FOLDER + thing_name+'device2' + "/" + user_name +'device2'+ "/" + activity_list[activity_index]+'device2'+'.json', "r") as file:
                    listObj = json.load(file)
            except:
                listObj = []
                print("new file")

            ### store the data into the list
            with open(DATA_COLLECTION_FOLDER+thing_name+'device2' + "/" + user_name +'device2'+ "/"+ activity_list[activity_index]+'device2'+'.json', "w+") as file:
                print(activity_list[activity_index])
                print(len(listObj))
                if overwrite and len(listObj) > 0:
                    listObj[-1] = {"background": background_fft_profile2.tolist(), "record_data":record_data2}
                    overwrite = False
                else:
                    listObj.append({"background": background_fft_profile2.tolist(), "record_data":record_data2})
                json.dump(listObj, file, allow_nan = True)
                print("Record Finish")
            record_data2 = []


        if len(record_data3) > 0 and not is_recording:
            
            ## create file if it doesn't exist or 
            ## append data to the file if it exists
            try:
                with open(DATA_COLLECTION_FOLDER + thing_name+'device3' + "/" + user_name +'device3'+ "/" + activity_list[activity_index]+'device3'+'.json', "r") as file:
                    listObj = json.load(file)
            except:
                listObj = []
                print("new file")

            ### store the data into the list
            with open(DATA_COLLECTION_FOLDER+thing_name+'device3' + "/" + user_name +'device3'+ "/"+ activity_list[activity_index]+'device3'+'.json', "w+") as file:
                print(activity_list[activity_index])
                print(len(listObj))
                if overwrite and len(listObj) > 0:
                    listObj[-1] = {"background": background_fft_profile3.tolist(), "record_data":record_data3}
                    overwrite = False
                else:
                    listObj.append({"background": background_fft_profile3.tolist(), "record_data":record_data3})
                json.dump(listObj, file, allow_nan = True)
                print("Record Finish")
            record_data3 = []


        if len(record_data4) > 0 and not is_recording:
            
            ## create file if it doesn't exist or 
            ## append data to the file if it exists
            try:
                with open(DATA_COLLECTION_FOLDER + thing_name+'device4' + "/" + user_name +'device4'+ "/" + activity_list[activity_index]+'device4'+'.json', "r") as file:
                    listObj = json.load(file)
            except:
                listObj = []
                print("new file")

            ### store the data into the list
            with open(DATA_COLLECTION_FOLDER+thing_name+'device4' + "/" + user_name +'device4'+ "/"+ activity_list[activity_index]+'device4'+'.json', "w+") as file:
                print(activity_list[activity_index])
                print(len(listObj))
                if overwrite and len(listObj) > 0:
                    listObj[-1] = {"background": background_fft_profile4.tolist(), "record_data":record_data4}
                    overwrite = False
                else:
                    listObj.append({"background": background_fft_profile4.tolist(), "record_data":record_data4})
                json.dump(listObj, file, allow_nan = True)
                print("Record Finish")
            record_data4 = []     


def demo(device,device1,device2,device3,device4, streaming_server,streaming_server1,streaming_server2,streaming_server3,streaming_server4):

        model_file = './model/'+'ex32_RF_modelCoughing_speaking'
        loaded_model = joblib.load(model_file)

        windows = []
        windows1 = []
        windows2 = []
        windows3 = []
        windows4 = []

        all_data=[]

        POLL_SIZE = 20
        PREDICTION_WINDOW_SIZE = 30

        poll = []

        background_fft_profile = device.calculate_background_fft_profile()
        background_fft_profile1 = device1.calculate_background_fft_profile()
        background_fft_profile2 = device2.calculate_background_fft_profile()
        background_fft_profile3 = device3.calculate_background_fft_profile()
        background_fft_profile4 = device4.calculate_background_fft_profile()

        while True:
            cmd = streaming_server.read_client_command()
            cmd1 = streaming_server.read_client_command()
            cmd2 = streaming_server.read_client_command()
            cmd3 = streaming_server.read_client_command()
            cmd4 = streaming_server.read_client_command()
            #print("Receiving")
            if cmd and cmd == 'Z' or cmd1 and cmd1 == 'Z'or cmd2 and cmd2 == 'Z'or cmd3 and cmd3 == 'Z'or cmd4 and cmd4 == 'Z' :
                background_fft_profile = device.calculate_background_fft_profile()
                print("update background profile")

            data = device.sample().tolist()
            data1 = device1.sample().tolist()
            data2 = device2.sample().tolist()
            data3 = device3.sample().tolist()
            data4 = device4.sample().tolist()
            if len(data) > 0 or len(data1) > 0 or len(data2) > 0 or len(data3) > 0:
                streaming_server.streaming_signal_in_FFT(data, background_fft_profile)
                streaming_server1.streaming_signal_in_FFT(data1, background_fft_profile1)
                streaming_server2.streaming_signal_in_FFT(data2, background_fft_profile2)
                streaming_server3.streaming_signal_in_FFT(data3, background_fft_profile3)
                streaming_server4.streaming_signal_in_FFT(data4, background_fft_profile4)
                print("Receiving1")
                print("length data",len(data),len(data1),len(data2),len(data3),len(data4))
                if data != None and len(data) > 0 and data1 != None and len(data1) > 0 and data2 != None and len(data2) > 0 and data3 != None and len(data4) > 0 and data4 != None and len(data4) > 0:
                    windows.append(data)
                    windows1.append(data1)
                    windows2.append(data2)
                    windows3.append(data3)
                    windows4.append(data4)
                    print("Receiving2")

                print("length windows:",len(windows))
                print("length windows1:",len(windows1))
                print("length windows2:",len(windows2))
                print("length windows3:",len(windows3))
                print("length windows4:",len(windows4))

                if len(windows) > PREDICTION_WINDOW_SIZE:
                    windows.pop(0)
                    windows1.pop(0)
                    windows2.pop(0)
                    windows3.pop(0)
                    windows4.pop(0)
                    print("Receiving3")
                    print("length windows:",len(windows))
                    print("length windows1:",len(windows1))
                    print("length windows2:",len(windows2))
                    print("length windows3:",len(windows3))
                    print("length windows4:",len(windows4))
                    #all_data.append(windows)


                if len(windows) >= PREDICTION_WINDOW_SIZE :
                    #data_all=windows+windows1+windows2+windows3
                    record_sig=[element for sublist in windows for element in sublist]
                    record_sig1=[element for sublist in windows1 for element in sublist]
                    record_sig2=[element for sublist in windows2 for element in sublist]
                    record_sig3=[element for sublist in windows3 for element in sublist]
                    record_sig4=[element for sublist in windows4 for element in sublist]
                    print("Receiving4")

                    #if not DSPUtils.is_noisy(all_data):
                    #df = pd.DataFrame({'time': range(len(all_data)), 'value': all_data})
                    #signal, fft_windows = DSPUtils.segment_along_windows(windows, background_fft_profile, Device.BUFFER_SIZE, Device.SHIFT_SIZE)
                    fft_windows,fft_windows1,fft_windows2,fft_windows3,fft_windows4= DSPUtils.segment_along_windows(windows,windows1,windows2,windows3,windows4, Device.BUFFER_SIZE, Device.SHIFT_SIZE)
                    prediction = loaded_model.predict((DSPUtils.extract_feature(fft_windows,fft_windows1,fft_windows2,fft_windows3,fft_windows4)).reshape(1,-1 ))
                    #features = extract_features(df, column_id='time', column_sort='time', default_fc_parameters=ComprehensiveFCParameters())

                    #prediction = loaded_model.predict(features)
                    poll.append(prediction[0])
                    print(prediction[0])
                    #else:
                    #    poll.append("Noisy")
                if len(poll) > POLL_SIZE:
                    poll.pop(0)
                    print("Polling")
                if len(poll) >= POLL_SIZE:
                    max_occur = max(poll,key=poll.count)
                    if poll.count(max_occur) >= POLL_SIZE/2:
                        data_string = "result,"+ max_occur + '\n'
                        streaming_server.enqueue(data_string)
                        streaming_server1.enqueue(data_string)
                        streaming_server2.enqueue(data_string)
                        streaming_server3.enqueue(data_string)
                        streaming_server4.enqueue(data_string)
                        print(max_occur)
                else:
                    data_string= "result"+"Noise"+'\n'
                    streaming_server.enqueue(data_string) 
                    streaming_server1.enqueue(data_string)
                    streaming_server2.enqueue(data_string) 
                    streaming_server3.enqueue(data_string) 
                    streaming_server4.enqueue(data_string)                     


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

    try:
        server_use = input("Please enter what you are going to do? (0: data), (1:demo) :\n")
        if server_use == '0':
            collect_data(device, device1,device2,device3,device4,streaming_server,streaming_server1,streaming_server2,streaming_server3,streaming_server4)
        elif server_use == '1':
            demo(device,device1, device2,device3,device4, streaming_server, streaming_server1,streaming_server2,streaming_server3,streaming_server4) 
          
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating server")


if __name__ == '__main__':
    main()
    # print(load_bitmasks())