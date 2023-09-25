import json
import matplotlib.pyplot as plt
import numpy as np
from dsp_utils import *
import pandas as pd
from scipy.signal import spectrogram




#from dsp_utils import  segment_along_windows, extract_feature

        




BUFFER_SIZE = 512
SHIFT_SIZE = 1 
sample_rate=1024
window = 'hann'
tolerance=10




activity_list = ["Coughing","Speaking","Using toothbrush","Cleaning",  "Noise"]

thing_to_activity = {"ex32":["Coughing","Speaking","Using toothbrush","Cleaning",  "Noise"]}
things = ["ex32"]
participants = ["1","2","3","4","5","6","7","8","9","10"]

all_data = []
for thing in things:
        for participant in participants:
            for a in thing_to_activity[thing]:
                if a=="Using toothbrush":

                    file_name ='./activity_data/'+thing + "/"+thing + "/" + a+participant  +'.json'
                    file_name1 ='./activity_data/'+thing +'device1'+ "/"+thing +'device1'+ "/" + a+participant +'.json'
                    file_name2 ='./activity_data/'+thing +'device2'+ "/"+thing +'device2'+ "/" + a+participant +'.json'
                    file_name3 ='./activity_data/'+thing +'device3'+ "/"+thing +'device3'+ "/" + a+participant +'.json'
                    file_name4 ='./activity_data/'+thing +'device4'+ "/"+thing +'device4'+ "/" + a+participant +'.json'
                else:
                    file_name ='./activity_data/'+thing + "/" +thing + "/"+ participant + "/"+ a +'.json'
                    file_name1 ='./activity_data/'+thing +'device1'+ "/"+thing +'device1'+ "/" + participant +'device1'+ "/"+ a +'device1'+'.json'
                    file_name2 ='./activity_data/'+thing +'device2'+ "/"+thing +'device2'+ "/" + participant +'device2'+ "/"+ a +'device2'+'.json'
                    file_name3 ='./activity_data/'+thing +'device3'+ "/"+thing +'device3'+ "/" + participant +'device3'+ "/"+ a +'device3'+'.json'
                    file_name4 ='./activity_data/'+thing +'device4'+ "/"+thing +'device4'+ "/" + participant +'device4'+ "/"+ a +'device4'+'.json'
                N='C:/Users/ASUS-PLUS/Desktop/paper pics/pics/New folder/'+thing+participant+a+'for_demo'+'.png'


                # Open the JSON file
                with open(file_name, 'r') as file:
                    # Load the JSON data
                    data = json.load(file)
                    for d in data:
                        new_data = []
                        record_sig=[]                        
                        record_sig = np.array(d["record_data"])
                        #s,fft_windows = DSPUtils.segment_along_windows(record_sig, d["background"], BUFFER_SIZE, SHIFT_SIZE)
                        
                        #data =  DSPUtils.extract_feature(signal, fft_windows)
                        #print(data)
                        #new_data= [element for sublist in fft_windows for element in sublist]
                        #new_data=np.array(fft_windows).flatten()
                        #m_fft_windows=np.array(fft_windows)
                        #all_data.extend(m_fft_windows)
                        signal0= [element for sublist in record_sig for element in sublist]
                        



                with open(file_name1, 'r') as file:
                    # Load the JSON data
                    data1 = json.load(file)
                    for d in data1:
                        new_data1 = []                       
                        record_sig = np.array(d["record_data"])
                        #s,fft_windows = DSPUtils.segment_along_windows(record_sig, d["background"], BUFFER_SIZE, SHIFT_SIZE)
                        #print(fft_windows)
                        #data1 =  DSPUtils.extract_feature(signal, fft_windows)
                        #print(data1)
                        #new_data1= [element for sublist in fft_windows for element in sublist]
                        #new_data1=np.array(fft_windows).flatten()
                        #=np.array(fft_windows)
                        #all_data.extend(m_fft_windows1)
                        signal1= [element for sublist in record_sig for element in sublist]
                        


                with open(file_name2, 'r') as file:
                    # Load the JSON data
                    data2 = json.load(file)
                    for d in data2:
                        new_data2 =[]                        
                        record_sig = np.array(d["record_data"])
                        #s,fft_windows = DSPUtils.segment_along_windows(record_sig, d["background"], BUFFER_SIZE, SHIFT_SIZE)
                        #data2 =  DSPUtils.extract_feature(signal, fft_windows)
                        #print(data2)
                        #new_data2= [element for sublist in fft_windows for element in sublist]
                        #new_data2=np.array(fft_windows).flatten()
                        #m_fft_windows2=np.array(fft_windows)
                        #all_data.extend(m_fft_windows2)
                        signal2= [element for sublist in record_sig for element in sublist]


                with open(file_name3, 'r') as file:
        # Load the JSON data
                    data3 = json.load(file)
                    for d in data3:
                        new_data3 = []                       
                        record_sig = np.array(d["record_data"])
                        #s,fft_windows = DSPUtils.segment_along_windows(record_sig, d["background"], BUFFER_SIZE, SHIFT_SIZE)
                        #data3 =  DSPUtils.extract_feature(signal, fft_windows)
                        #print(data3)
                        #new_data3= [element for sublist in fft_windows for element in sublist]
                        #new_data3=np.array(fft_windows).flatten()
                        #m_fft_windows3=np.array(fft_windows)
                        #all_data.extend(m_fft_windows3)
                        signal3= [element for sublist in record_sig for element in sublist]
                        

                with open(file_name4, 'r') as file:
        # Load the JSON data
                    data4 = json.load(file)
                    for d in data4:
                        new_data4 = []                       
                        record_sig = np.array(d["record_data"])
                        #s,fft_windows = DSPUtils.segment_along_windows(record_sig, d["background"], BUFFER_SIZE, SHIFT_SIZE)
                        #data4 =  DSPUtils.extract_feature(signal, fft_windows)
                        #print(data4)
                        #new_data4= [element for sublist in fft_windows for element in sublist]
                        #new_data4=np.array(fft_windows).flatten()
                        #m_fft_windows4=np.array(fft_windows)
                        #all_data.extend(m_fft_windows4)
                        signal4= [element for sublist in record_sig for element in sublist]
                        

                all_data1= [element for sublist in all_data for element in sublist]

                signal0=np.array(signal0)
                signal1=np.array(signal1)
                signal2=np.array(signal2)
                signal3=np.array(signal3)
                signal4=np.array(signal4)

                frequencies, times, Sxx = spectrogram(signal0, fs=sample_rate, nperseg=BUFFER_SIZE, noverlap=384, nfft=BUFFER_SIZE)
                frequencies1, times1, Sxx1 = spectrogram(signal1, fs=sample_rate, nperseg=BUFFER_SIZE, noverlap=384, nfft=BUFFER_SIZE)
                frequencies2, times2, Sxx2 = spectrogram(signal2, fs=sample_rate, nperseg=BUFFER_SIZE, noverlap=384, nfft=BUFFER_SIZE)
                frequencies3, times3, Sxx3 = spectrogram(signal3, fs=sample_rate, nperseg=BUFFER_SIZE, noverlap=384, nfft=BUFFER_SIZE)
                frequencies4, times4, Sxx4 = spectrogram(signal4, fs=sample_rate, nperseg=BUFFER_SIZE, noverlap=384, nfft=BUFFER_SIZE)

                common_frequencies = np.intersect1d(frequencies, np.intersect1d(frequencies1, np.intersect1d(frequencies2, np.intersect1d(frequencies3, frequencies4))))
                minimum_len_time=min(len(times),len(times1),len(times2),len(times3),len(times4))
                concatenated_sxx = np.vstack((10 * np.log10(Sxx[:,0:minimum_len_time]),10 * np.log10(Sxx1[:,0:minimum_len_time]),10 * np.log10(Sxx2[:,0:minimum_len_time]),10 * np.log10(Sxx3[:,0:minimum_len_time]),10 * np.log10(Sxx4[:,0:minimum_len_time])))
                fre=np.hstack((frequencies,frequencies+500,frequencies+1000,frequencies+1500,frequencies+2000))
                
                fig1=plt.figure(figsize=(8, 6))

                plt.pcolormesh(times2[0:minimum_len_time], fre, concatenated_sxx)
                plt.xticks([])
                plt.yticks([])
                plt.savefig(N)
                """
                #plt.xlabel('Times')
                #plt.ylabel('Frequency (Hz)')
                #plt.title('first device')


                #print(new_data.shape[0])
                # Plot the first subplot
                
                plt.subplot(5, 1, 1)  # (rows, columns, plot_number)
                #plt.plot(list(range(len(new_data))), new_data)
                plt.pcolormesh(times[0:len(times)], frequencies, 10 * np.log10(Sxx[:,0:len(times)]))
                plt.xlabel('Times')
                plt.ylabel('Frequency (Hz)')
                plt.title('first device')
                #plt.colorbar(label='Power Spectral Density (dB)')
                #plt.legend()

                # Plot the second subplot
                #print(new_data1)
                plt.subplot(5, 1, 2)  # (rows, columns, plot_number)
                #plt.plot(list(range(len(new_data1))), new_data1)




                plt.pcolormesh(times1[0:len(times)], frequencies1, 10 * np.log10(Sxx1[:,0:len(times)]))
                plt.xlabel('Times')
                plt.ylabel('F(Hz)')
                plt.title('Second device')
                #plt.colorbar(label='Power Spectral Density (dB)')
                #plt.legend()

                #print(new_data2.shape[0])
                plt.subplot(5, 1, 3)  # (rows, columns, plot_number)
                #plt.plot(list(range(len(new_data2))), new_data2)
                plt.pcolormesh(times2[0:len(times)], frequencies2, 10 * np.log10(Sxx2[:,0:len(times)]))
                plt.xlabel('Times')
                plt.ylabel('F(Hz)')
                plt.title('Third device')
                #plt.colorbar(label='Power Spectral Density (dB)')
                #plt.legend()

                #print(new_data3.shape[0])
                plt.subplot(5, 1, 4)  # (rows, columns, plot_number)
                #plt.plot(list(range(len(new_data3))), new_data3)
                plt.pcolormesh(times3[0:len(times)], frequencies3, 10 * np.log10(Sxx3[:,0:len(times)]))
                plt.xlabel('Times')
                plt.ylabel('F(Hz)')
                plt.title('fourth device')
                #plt.colorbar(label='Power Spectral Density (dB)')
                #plt.legend()

                #print(new_data4.shape[0])
                plt.subplot(5, 1, 5)  # (rows, columns, plot_number)
                #plt.plot(list(range(len(new_data4))), new_data4)
                plt.pcolormesh(times4[0:len(times)], frequencies4, 10 * np.log10(Sxx4[:,0:len(times)]))
                plt.xlabel('Times')
                plt.ylabel('F(Hz)')
                plt.title('fifth device')
                #plt.colorbar(label='Power Spectral Density (dB)')
                #plt.legend()

                # Adjust spacing between subplots
                plt.tight_layout()
                
                """
                #fig2=plt.figure(figsize=(8, 6))
                #plt.plot(list(range(len(all_data1))), all_data1)
                #plt.xlabel('x')
                #plt.ylabel('y')
                #plt.title('features')
                #plt.legend()
                """
                # Show the figure
                #plt.show()
                plt.savefig(N)

#print(len(common_frequencies))
#print(frequencies)
"""
"""
for common_freq in common_frequencies:
    idx = np.argmin(np.abs(frequencies - common_freq))
    idx1 = np.argmin(np.abs(frequencies1 - common_freq))
    idx2 = np.argmin(np.abs(frequencies2 - common_freq))
    idx3 = np.argmin(np.abs(frequencies3 - common_freq))
    idx4 = np.argmin(np.abs(frequencies4 - common_freq))
    all_spectrograms = np.stack([Sxx3[idx3], Sxx2[idx2],  Sxx1[idx1]], axis=0)
    #print(len(Sxx1[idx1]))
    #print(len(Sxx2[idx2]))
    #print(len(Sxx3[idx3]))
    #print(len(Sxx4[idx4]))
    #print(len(Sxx[idx]))
     
    min_values = np.min(all_spectrograms, axis=0)


    Sxx1[idx1, :] -= min_values
    Sxx2[idx2, :] -= min_values
    Sxx3[idx3, :] -= min_values
    #Sxx4[idx4, :] -= min_values
    Sxx[idx, :] -= min_values
    """

#    Sxx[idx,:] = 0
#    Sxx1[idx1,:] = 0
#    Sxx2[idx2,:] = 0
#    Sxx3[idx3,:] = 0
#    Sxx4[idx4,:] = 0

#print(all_data1)
#   df=pd.DataFrame(all_data)
# Plot the data
