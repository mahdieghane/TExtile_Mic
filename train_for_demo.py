from os import write
import joblib
import numpy as np
from numpy import loadtxt
from model1 import generate_models
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from dsp_utils import DSPUtils
from statistics import mean, stdev
import pandas as pd
import json
import matplotlib.pyplot as plt


DATA_COLLECTION_FOLDER = "./demo_data/"
THRESHOLD = 3
SAMPLE_RATE = 1024
BUFFER_SIZE = 512
SHIFT_SIZE = 128 
participants = ["1","2","3","4","5","6","7","8","9","10"]
#participants = ["p0"]
# items = ["table", "drawer", 'cuttingboard'] ### what items are the model trained for?
items = ["ex32"] ### what items are the model trained for?
### mapping from items to their corresponding activities
item_to_activities = {"ex32":[ "Coughing",  "Speaking"] } 

def printOverallAccuracy(cm):
    acc = []
    for i in range(len(cm)):
        if sum(cm[i]) > 0:
            acc.append(cm[i][i]/ sum(cm[i]))
    print('acc')
    print(mean(acc))
    print("std")
    print(stdev(acc))

def train(): 
    all_data = []
    for item in items:
        for participant in participants:
            for a in item_to_activities[item]:
                if a=="Using toothbrush":

                    file_name ='./activity_data/'+item + "/"+item + "/" + a+participant  +'.json'
                    file_name1 ='./activity_data/'+item +'device1'+ "/"+item +'device1'+ "/" + a+participant +'.json'
                    file_name2 ='./activity_data/'+item +'device2'+ "/"+item +'device2'+ "/" + a+participant +'.json'
                    file_name3 ='./activity_data/'+item +'device3'+ "/"+item +'device3'+ "/" + a+participant +'.json'
                    file_name4 ='./activity_data/'+item +'device4'+ "/"+item +'device4'+ "/" + a+participant +'.json'
                else:
                    file_name ='./activity_data/'+item + "/" +item + "/"+ participant + "/"+ a +'.json'
                    file_name1 ='./activity_data/'+item +'device1'+ "/"+item +'device1'+ "/" + participant +'device1'+ "/"+ a +'device1'+'.json'
                    file_name2 ='./activity_data/'+item +'device2'+ "/"+item +'device2'+ "/" + participant +'device2'+ "/"+ a +'device2'+'.json'
                    file_name3 ='./activity_data/'+item +'device3'+ "/"+item +'device3'+ "/" + participant +'device3'+ "/"+ a +'device3'+'.json'
                    file_name4 ='./activity_data/'+item +'device4'+ "/"+item +'device4'+ "/" + participant +'device4'+ "/"+ a +'device4'+'.json'
                with open(file_name, 'r+') as file:
                     with open(file_name1, 'r+') as file1:
                          with open(file_name2, 'r+') as file2:
                               with open(file_name3, 'r+') as file3:
                                    with open(file_name4, 'r+') as file4:
                    #print(file)
                                        data = json.load(file)
                                        data1 = json.load(file1)
                                        data2 = json.load(file2)
                                        data3 = json.load(file3)
                                        data4 = json.load(file4)
                                        for d in data[:]:
                                            new_data = {}
                                            new_data["participant"] = participant
                                            new_data["item"] = item
                                            new_data["activity"] = a
                                            #record_sig = np.array(d["record_data"])
                                            #signal, fft_windows = DSPUtils.segment_along_windows(record_sig, d["background"], BUFFER_SIZE, SHIFT_SIZE)
                                            #new_data["feature"] =  DSPUtils.extract_feature(signal, fft_windows)

                                            record_sig = np.array(d["record_data"])
                                        #new_data["feature"]= [element for sublist in record_sig for element in sublist]
                                        for d in data1:    
                                            record_sig1 = np.array(d["record_data"])
                                        #new_data["feature"]= new_data["feature"]+[element for sublist in record_sig1 for element in sublist]
                                        for d in data2:
                                            record_sig2 = np.array(d["record_data"])
                                        #new_data["feature"]= new_data["feature"]+[element for sublist in record_sig2 for element in sublist]
                                        for d in data3:
                                            record_sig3 = np.array(d["record_data"])
                                        #new_data["feature"]= new_data["feature"]+[element for sublist in record_sig3 for element in sublist]
                                        for d in data4:
                                            record_sig4 = np.array(d["record_data"])

                                        fft_windows,fft_windows1,fft_windows2,fft_windows3,fft_windows4= DSPUtils.segment_along_windows(record_sig,record_sig1,record_sig2,record_sig3,record_sig4, BUFFER_SIZE, SHIFT_SIZE)

                                        new_data["feature"] =  DSPUtils.extract_feature(fft_windows,fft_windows1,fft_windows2,fft_windows3,fft_windows4)
                                        if min(len(fft_windows),len(fft_windows1),len(fft_windows2),len(fft_windows3),len(fft_windows4))>0:
                                            all_data.append(new_data)






    df = pd.DataFrame(all_data)
    #df.to_excel('my_data.xlsx', index=False)
    models, model_names = generate_models()
    saved_models = []
    i = 0
    
    for m in models:
        for item in items:
            item_data = df.loc[(df['item'] == item)]
            scaler = MinMaxScaler()
            features = item_data["feature"].to_list()
            print(features)
           
            strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
            model = clone(m)
            y_pred = cross_val_predict(model, features, item_data["activity"].to_list(), cv=strat_k_fold)
            cm = confusion_matrix(item_data["activity"].to_list(), y_pred, labels=item_to_activities[item])
            print(item)
            print(model_names[i])
            print("Two Fold Acc:")
            print(cm)
            printOverallAccuracy(cm)

            model = clone(m)
            model.fit(features, item_data["activity"].to_list())
            model_file_name = model_names[i] + '_model'
            joblib.dump(model, './model/'+item+ '_' + model_file_name+'Coughing_speaking')
        i+=1

if __name__ == '__main__':
    train()
    
