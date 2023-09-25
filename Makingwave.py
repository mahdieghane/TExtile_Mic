import numpy as np
from scipy.io import wavfile
import json
import matplotlib.pyplot as plt
import numpy as np
from dsp_utils import *
import pandas as pd

sample_rate =1024  # Sample rate (1024 Hz)
duration = 6  # Duration in seconds



activity_list = ["Crunch", "Squat"]

thing_to_activity = {"e6":["Crunch", "Squat"]}
things = ["e6"]
participants = ["2"]


for thing in things:
        for participant in participants:
            for a in thing_to_activity[thing]:
                file_name ='./activity_data/'+thing + "/" + participant + "/"+ a +'.json'
                file_name1 ='./activity_data/'+thing +'device1'+ "/" + participant +'device1'+ "/"+ a +'device1'+'.json'
                file_name2 ='./activity_data/'+thing +'device2'+ "/" + participant +'device2'+ "/"+ a +'device2'+'.json'
                file_name3 ='./activity_data/'+thing +'device3'+ "/" + participant +'device3'+ "/"+ a +'device3'+'.json'
                file_name4 ='./activity_data/'+thing +'device4'+ "/" + participant +'device4'+ "/"+ a +'device4'+'.json'
                N='C:/Users/ASUS-PLUS/Desktop/pics/New folder/'+thing+participant+a+'.wav'
                N1='C:/Users/ASUS-PLUS/Desktop/pics/New folder/'+thing+participant+a+'1'+'.wav'
                N2='C:/Users/ASUS-PLUS/Desktop/pics/New folder/'+thing+participant+a+'2'+'.wav'
                N3='C:/Users/ASUS-PLUS/Desktop/pics/New folder/'+thing+participant+a+'3'+'.wav'
                N4='C:/Users/ASUS-PLUS/Desktop/pics/New folder/'+thing+participant+a+'4'+'.wav'



                # Open the JSON file
                with open(file_name, 'r') as file:
                    # Load the JSON data
                    data = json.load(file)
                    for d in data:
                        new_data = []                        
                        record_sig = np.array(d["record_data"])
                        signal0= [element for sublist in record_sig for element in sublist]
                        time = np.linspace(0, duration, int(sample_rate * duration))
                        record_sig_normalized = signal0 / np.max(np.abs(signal0))
                        audio_data_int16 = np.int16(record_sig_normalized * 32767)
                        wavfile.write(N, sample_rate, audio_data_int16)
                        print("Audio saved successfully.")

                with open(file_name1, 'r') as file:
                    # Load the JSON data
                    data = json.load(file)
                    for d in data:
                        new_data = []                        
                        record_sig = np.array(d["record_data"])
                        signal0= [element for sublist in record_sig for element in sublist]
                        time = np.linspace(0, duration, int(sample_rate * duration))
                        record_sig_normalized = signal0 / np.max(np.abs(signal0))
                        audio_data_int16 = np.int16(record_sig_normalized * 32767)
                        wavfile.write(N1, sample_rate, audio_data_int16)
                        print("Audio saved successfully.")

                with open(file_name2, 'r') as file:
                    # Load the JSON data
                    data = json.load(file)
                    for d in data:
                        new_data = []                        
                        signal0= [element for sublist in record_sig for element in sublist]
                        time = np.linspace(0, duration, int(sample_rate * duration))
                        record_sig_normalized = signal0 / np.max(np.abs(signal0))
                        audio_data_int16 = np.int16(record_sig_normalized * 32767)
                        wavfile.write(N2, sample_rate, audio_data_int16)
                        print("Audio saved successfully.")


                with open(file_name3, 'r') as file:
                    # Load the JSON data
                    data = json.load(file)
                    for d in data:
                        new_data = []                        
                        signal0= [element for sublist in record_sig for element in sublist]
                        time = np.linspace(0, duration, int(sample_rate * duration))
                        record_sig_normalized = signal0 / np.max(np.abs(signal0))
                        audio_data_int16 = np.int16(record_sig_normalized* 32767)
                        wavfile.write(N3, sample_rate, audio_data_int16)
                        print("Audio saved successfully.")


                with open(file_name4, 'r') as file:
                    # Load the JSON data
                    data = json.load(file)
                    for d in data:
                        new_data = []                        
                        signal0= [element for sublist in record_sig for element in sublist]
                        time = np.linspace(0, duration, int(sample_rate * duration))
                        record_sig_normalized = signal0 / np.max(np.abs(signal0))
                        audio_data_int16 = np.int16(record_sig_normalized * 32767)
                        wavfile.write(N4, sample_rate, audio_data_int16)
                        print("Audio saved successfully.")


