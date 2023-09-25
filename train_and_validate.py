from os import write
import joblib
import numpy as np
from sklearn.utils import shuffle
from numpy import loadtxt
from model import generate_models
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
#from micromlgen import port
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction.settings import EfficientFCParameters

from sklearn.preprocessing import normalize
from dsp_utils import DSPUtils
# from dsp import compute_relevant_features
from statistics import mean, stdev
import pandas as pd
import json

import matplotlib.pyplot as plt
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute


SAMPLE_RATE = 1024
BUFFER_SIZE = 512
SHIFT_SIZE = 512
participants = ["5","3","4"]

activity_list = [  "Grating",   "Snoring",  "Walking", "Standing Up", "Fall detection", "Crunch", "Squat",  "Noise"]

thing_to_activity = {"ex2":[   "Grating",   "Snoring",  "Walking", "Standing Up", "Fall detection", "Crunch", "Squat",  "Noise"]}
things = ["ex2"]

def printOverallAccuracy(cm):
    acc = []
    for i in range(len(cm)):
        if sum(cm[i]) > 0:
            acc.append(cm[i][i]/ sum(cm[i]))
    print('acc')
    print(mean(acc))
    print("std")
    print(stdev(acc))

def plot_and_save_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    if len(target_names) > 10:
        plt.rcParams.update({'font.size': 20})
    else:
        plt.rcParams.update({'font.size': 30})

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(40, 40), dpi = 120)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = cm * 100

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                if cm[i, j] == 100:
                    plt.text(j, i, "{:,}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
                else:
                    plt.text(j, i, "{:0.1f}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./cm/confusion_matrix_'+str(title)+'.jpg')
    plt.clf()
    # plt.show()

def train(): 
    all_data = []
    for thing in things:
        for participant in participants:
            for a in thing_to_activity[thing]:
                file_name ='./activity_data/'+thing + "/" + participant + "/"+ a +'.json'
                file_name1 ='./activity_data/'+thing +'device1'+ "/" + participant +'device1'+ "/"+ a +'device1'+'.json'
                file_name2 ='./activity_data/'+thing +'device2'+ "/" + participant +'device2'+ "/"+ a +'device2'+'.json'
                file_name3 ='./activity_data/'+thing +'device3'+ "/" + participant +'device3'+ "/"+ a +'device3'+'.json'
                file_name4 ='./activity_data/'+thing +'device4'+ "/" + participant +'device4'+ "/"+ a +'device4'+'.json'

                with open(file_name, 'r+') as file:
                    with open (file_name1, 'r+') as file1:
                        with open (file_name2, 'r+') as file2:
                            with open (file_name3, 'r+') as file3:
                                with open (file_name4, 'r+') as file4:

                    #print(file)
                                    data = json.load(file)
                                    data1 = json.load(file1)
                                    data2 = json.load(file2)
                                    data3 = json.load(file3)
                                    data4 = json.load(file4)
                                    for d in data:
                                        new_data = {}
                                        new_data["participant"] = participant
                                        new_data["thing"] = thing
                                        new_data["activity"] = a
                                        new_data1 = {}
                                        new_data1["participant"] = participant
                                        new_data1["thing"] = thing
                                        new_data1["activity"] = 'Noise'
                                        # record_sig = down_sample(np.array(d["record_data"]))                        
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
                                        #new_data["feature"]= new_data["feature"]+[element for sublist in record_sig4 for element in sublist]
                                        
                                        
                #all_data.append(new_data)
                                    
                                    fft_windows,fft_windows1,fft_windows2,fft_windows3,fft_windows4,fft_zero_windows,fft_zero_windows1,fft_zero_windows2,fft_zero_windows3,fft_zero_windows4 = DSPUtils.segment_along_windows(record_sig,record_sig1,record_sig2,record_sig3,record_sig4, BUFFER_SIZE, SHIFT_SIZE)
                                    
                                    midpoint = len(fft_windows) // 2
                                    list1_fft = fft_windows[:midpoint]
                                    list2_fft = fft_windows[midpoint:]

                                    midpoint = len(fft_windows1) // 2
                                    list1_fft1 = fft_windows[:midpoint]
                                    list2_fft1 = fft_windows[midpoint:]


                                    midpoint = len(fft_windows2) // 2
                                    list1_fft2 = fft_windows[:midpoint]
                                    list2_fft2 = fft_windows[midpoint:]


                                    midpoint = len(fft_windows3) // 2
                                    list1_fft3 = fft_windows[:midpoint]
                                    list2_fft3 = fft_windows[midpoint:]


                                    midpoint = len(fft_windows4) // 2
                                    list1_fft4 = fft_windows[:midpoint]
                                    list2_fft4 = fft_windows[midpoint:]

                                    new_data["feature"] =  DSPUtils.extract_feature(list1_fft,list1_fft1,list1_fft2,list1_fft3,list1_fft4)
                                    if min(len(list1_fft),len(list1_fft1),len(list1_fft2),len(list1_fft3),len(list1_fft4))>0:
                                        all_data.append(new_data)

                                    new_data["feature"] =  DSPUtils.extract_feature(list2_fft,list2_fft1,list2_fft2,list2_fft3,list2_fft4)
                                    if min(len(list2_fft),len(list2_fft1),len(list2_fft2),len(list2_fft3),len(list2_fft4))>0:
                                        all_data.append(new_data)

                                    new_data1["feature"] =  DSPUtils.extract_feature(fft_zero_windows,fft_zero_windows1,fft_zero_windows2,fft_zero_windows3,fft_zero_windows4)
                                    if min(len(fft_zero_windows),len(fft_zero_windows1),len(fft_zero_windows2),len(fft_zero_windows3),len(fft_zero_windows4))>0:
                                        all_data.append(new_data1)
                                    


    df = pd.DataFrame(all_data)


    # Load your tabular data (replace this with your actual data loading code)
# df contains your tabular data

# Calculate the class counts
    class_counts = df["activity"].value_counts()

# Find the class with the highest count
    max_count = class_counts.max()

# Create a list to hold the augmented data
    augmented_data = []

# Loop through each class
    for class_label, count in class_counts.items():
    # Calculate the number of augmentations needed for this class
        num_augmentations = max_count - count

    # If the class needs augmentation
        if num_augmentations > 0:
        # Extract the rows for the current class
            class_data = df[df["activity"] == class_label]

        # Randomly select existing rows for augmentation
            selected_rows = class_data.sample(n=num_augmentations, replace=True)

        # Apply augmentation (for demonstration purposes, we're just adding random noise)
            for i in range(num_augmentations):
                augmented_instance = selected_rows.iloc[i].copy()

            # Adding random noise to some columns
                columns_to_augment = ["feature"]
                for col in columns_to_augment:
                    noise = np.random.normal(scale=0.005)
                    augmented_instance[col] += noise

                augmented_data.append(augmented_instance)

    #print(augmented_data)
    #augmented_data = augmented_data.reset_index(drop=True)
    dff=pd.DataFrame(augmented_data)
    #print(dff['activity'].value_counts())
# Concatenate original and augmented data
    augmented_df = pd.concat([df,dff] , ignore_index=True)

    print(augmented_df['activity'].value_counts())

    df=shuffle(augmented_df)
    #print(df)
    df.to_json('DF_saved.json', orient='records')


    models, model_names = generate_models()
    saved_models = []
    i = 0
    
    for m in models:
        cross_user_on_everything_done = False
        within_user_on_everything_done = False
        cross_thing = {"Actual": [], "Predict": []}
        for thing in things:
            print(thing)
            cross_user_on_thing = {"Actual": [], "Predict": []}
            cross_user_on_everything = {"Actual": [], "Predict": []}
            within_user_on_thing = {"Actual": [], "Predict": []}
            within_user_on_everything = {"Actual": [], "Predict": []}
            for participant in participants:



                ### cross-user accuracy on specific thing
                other_user = df.loc[(df['participant'] != participant) & (df['thing'] == thing)]
                target_user = df.loc[(df['participant'] == participant) & (df['thing'] == thing)]
                model = clone(m)
                #other_user["feature"].fillna(0.001, inplace=True)
                #print(other_user["feature"].to_list())
                #print(other_user["activity"].to_list())
                pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),  # Choose appropriate imputation strategy
                ('svc', model)
                ])

                pipeline.fit(other_user["feature"].to_list(), other_user["activity"].to_list())
                y = pipeline.predict(target_user['feature'].to_list())
                cm = confusion_matrix(target_user['activity'].to_list(), list(y), labels=thing_to_activity[thing])
                cross_user_on_thing["Actual"] += target_user['activity'].to_list()
                cross_user_on_thing["Predict"] += list(y)
                print(participant)
                print("Cross user:")
                print(cm)
                printOverallAccuracy(cm)
                #target_user.to_json('Target_user_'+participant+'saved.json', orient='records')
                with open('labels'+participant+'.json', 'w') as json_file:
                    json.dump(cross_user_on_thing, json_file)
                # saveConfusionMatrix(cm, participant + '_' + thing)
                """
                ### within-user accuracy on thing 
                model = clone(m)
                strat_k_fold = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
                y_pred = cross_val_predict(model, target_user["feature"].to_list(), target_user["activity"].to_list())
                cm = confusion_matrix(target_user["activity"].to_list(), y_pred, labels=thing_to_activity[thing])
                within_user_on_thing["Actual"] += target_user['activity'].to_list()
                within_user_on_thing["Predict"] += list(y_pred)
                # print(participant)
                # print("Two Fold Acc:")
                # print(cm)
                # printOverallAccuracy(cm)
                """



                ### cross-user accuracy on everything 
                if not cross_user_on_everything_done:
                    # print("cross_user_on_everything")
                    other_user = df.loc[(df['participant'] != participant)]
                    target_user = df.loc[(df['participant'] == participant)]
                    model = clone(m)
                    pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),  # Choose appropriate imputation strategy
                    ('svc', model)
                    ])
                    pipeline.fit(other_user["feature"].to_list(), other_user["activity"].to_list())
                    y = pipeline.predict(target_user['feature'].to_list())
                    cross_user_on_everything["Actual"] += target_user['activity'].to_list()
                    cross_user_on_everything["Predict"] += list(y)
                    target_user.to_json('Target_user_'+participant+'1_saved.json', orient='records')
 
                ### within-user accuracy on everything
                #if not within_user_on_everything_done:
                    # print("cross_user_on_everything")
                    #target_user = df.loc[(df['participant'] == participant)]
                    #model = clone(m)
                    #strat_k_fold = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
                    #y_pred = cross_val_predict(model, target_user["feature"].to_list(), target_user["activity"].to_list(), cv=strat_k_fold)
                    #cm = confusion_matrix(target_user["activity"].to_list(), y_pred, labels=thing_to_activity[thing])
                    #within_user_on_everything["Actual"] += target_user['activity'].to_list()
                    #within_user_on_everything["Predict"] += list(y_pred)

            # ### plot confusion matrix for cross-user accuracies on specific thing
            # cm = confusion_matrix(cross_user_on_thing["Actual"], cross_user_on_thing["Predict"], labels=thing_to_activity[thing])
            # plot_and_save_confusion_matrix(cm, thing_to_activity[thing], thing + "-cross-user")
            # ### print cross-user accuracy on specific thing
            # print(thing + " cross_user acc:")
            # printOverallAccuracy(cm)
            # print(cm)

            #  ### plot confusion matrix for within-user accuracies on specific thing
            # cm = confusion_matrix(within_user_on_thing["Actual"], within_user_on_thing["Predict"], labels=thing_to_activity[thing])
            # ### print cross-user accuracy on specific thing
            # plot_and_save_confusion_matrix(cm, thing_to_activity[thing], thing + "-within-user")
            # print(thing + " within_user acc:")
            # printOverallAccuracy(cm)
            # print(cm)

            if not cross_user_on_everything_done:

                ### plot confusion matrix for cross-user accuracies
                cm = confusion_matrix(cross_user_on_everything["Actual"], cross_user_on_everything["Predict"], labels=activity_list)
                ### print cross-user accuracies on everything 
                print("cross_user_on_everything acc:")
                plot_and_save_confusion_matrix(cm, activity_list, "cross_user_on_everything")
                printOverallAccuracy(cm)
                print(cm)
                
                cross_user_on_everything_done = True

            if not within_user_on_everything_done:
                ### plot confusion matrix for cross-user accuracies
                cm = confusion_matrix(within_user_on_everything["Actual"], within_user_on_everything["Predict"], labels=activity_list)
                ### print cross-user accuracies on everything 
                print("within_user_on_everything acc:")
                plot_and_save_confusion_matrix(cm, activity_list, "within_user_on_everything")
                printOverallAccuracy(cm)
                print(cm)
                within_user_on_everything_done = True
            
            """"

            if thing != "table":
                ### cross-thing accuracy
                other_thing = df.loc[(df['thing'] != thing)  & (df['activity'].isin(thing_to_activity[thing]))]
                target_thing = df.loc[(df['thing'] == thing) & (df['activity'].isin(thing_to_activity[thing]))]
                # strat_k_fold = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
                # acc = cross_val_score(model, features, labels, cv=strat_k_fold, scoring='recall_macro')
                # m = model.fit(other_thing["feature"].to_list(), other_thing["activity"].to_list())
                # y = list(model.predict(target_thing['feature'].to_list()))
                # # print(y)
                # cross_thing["Actual"] += target_thing['activity'].to_list()
                # cross_thing["Predict"] += y
                model = clone(m)
                model.fit(other_thing["feature"].to_list(), other_thing["activity"].to_list())
                y = list(model.predict(target_thing['feature'].to_list()))
                # print(y)
                cross_thing["Actual"] += target_thing['activity'].to_list()
                cross_thing["Predict"] += y
            """
               

            thing_data = df.loc[(df['thing'] == thing)]
            
            # strat_k_fold = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
            # model = clone(m)
            # y_pred = cross_val_predict(model, thing_data["feature"].to_list(), thing_data["activity"].to_list(), cv=strat_k_fold)
            # cm = confusion_matrix(thing_data["activity"].to_list(), y_pred, labels=thing_to_activity[thing])
            # print(thing)
            # print(model_names[i])
            # print("Two Fold Acc:")
            # print(cm)
            # printOverallAccuracy(cm)

         ### plot confusion matrix for cross-thing accuracies
        cm = confusion_matrix(cross_thing["Actual"], cross_thing["Predict"], labels=activity_list)
        plot_and_save_confusion_matrix(cm, activity_list, "cross_thing")
        print(model_names[i])
        print("crossthing acc:")
        print(cm)
        #printOverallAccuracy(cm)
        ## print cross-thing accuracies on everything 
        

        # model = clone(m)
        # strat_k_fold = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
        # y_pred = cross_val_predict(model, df["feature"].to_list(), df["activity"].to_list(), cv=strat_k_fold)
        # cm = confusion_matrix(df["activity"].to_list(), y_pred, labels=activity_list)

        # print(model_names[i])
        # print("overall model")
        # print(cm)
        # printOverallAccuracy(cm)
        

        # model = clone(m)
        # strat_k_fold = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
        # y_pred = cross_val_predict(model, df["feature"].to_list(), df["thing"].to_list(), cv=strat_k_fold)
        # cm = confusion_matrix(df["thing"].to_list(), y_pred, labels=things)
        # print("thing detection")
        # print(cm)
        # printOverallAccuracy(cm)
        i += 1

    # for model_name, model in saved_models:
    #     model_file_name =  model_name + '_model'
    #     joblib.dump(model, './model/'+model_file_name)
    #     text = port(model)
    #     with open('./model_script/'+model_file_name+'.c','w') as model_script:
    #         model_script.write(text)
    #         model_script.close()

if __name__ == '__main__':
    train()
    # matrix = [
    # [83.5, 0, 0, 6.5, 0, 9, 0, 0.5, 0, 0, 0, 0.5],
    # [0, 79, 1.5, 2.5, 0.5, 4, 0, 0, 0, 2.5, 3.5, 6.5],
    # [0, 0, 94.5, 0, 3, 1.5, 0, 0, 0, 0, 1, 0],
    # [5, 2.5, 0, 83, 0.5, 5, 0, 0.5, 0, 0, 0.5, 3],
    # [3, 1.5, 4, 2.5, 72.5, 16, 0, 1, 0, 0, 0, 0.5],
    # [7, 1, 1, 2, 5.5, 83, 0, 0, 0, 0, 0, 0.5],
    # [0, 0.5, 0, 0, 0, 0, 97.5, 1.5, 0, 0, 0.5, 0],
    # [1, 0, 0.5, 0.5, 0, 0.5, 4.5, 93, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 1, 0, 97.5, 0.5, 1, 0],
    # [0, 5.5, 0, 0, 0, 0, 0, 0, 2.5, 81.5, 9, 1.5],
    # [0.5, 2.5, 1, 0, 0, 0.5, 2.5, 0.5, 4.0, 7, 73.6, 8],
    # [1, 5, 0, 4, 0, 1, 0, 0, 0.5, 0, 2.5, 86],
    # ]
    # matrix = np.array(matrix)
    # printOverallAccuracy(matrix)
    # plot_and_save_confusion_matrix(matrix, activity_list, "within-user_mixed-item")

    