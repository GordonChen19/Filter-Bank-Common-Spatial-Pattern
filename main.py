import MLEngine
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import mne

channel_names = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 
                 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 
                 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 
                 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 
                 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 
                 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 
                 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 
                 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 
                 'O1', 'Oz', 'O2', 'Iz']



if __name__ == "__main__":

    
    kfold=[5]
    m_filters=[2]
    
    dataset_details={'file_path':'/Users/gordonchen/Documents/MACS/URECA/Dataset/',
                'window_details':{'tmin':0,'tmax':4.1},
                'task_number':4,
                'ntimes':1
                }

testing=[]
subject_number=10
testing_accuracy=0
parameters=[-1,-1]

channel_variance=defaultdict(float)

for k in kfold:
    for m in m_filters:
        for i in range (subject_number):
            print(f'subject {i+1}')
            ML=MLEngine.MLEngine(**dataset_details,subject_number=i+1,kfold=k,m_filters=m)
            trainingMean,testingMean,channel_variance=ML.channel_reduce(channel_variance)
            testing.append(testingMean)
        
        
        sorted_channel_variance=sorted(channel_variance.items(),key=lambda x: x[1],reverse=True)
        print(f'channel variance is {sorted_channel_variance}')
        
        scalp_topography_data=[[0] for x in range(64)]
        for i in range(64): 
            if i in channel_variance:
                scalp_topography_data[i][0]=channel_variance[i]/float(subject_number)
        
        scalp_topography_data[0][0]=0
        print(scalp_topography_data)
        
        
        # Try to set the BCI2000 system montage on the info structure
        info=mne.create_info(ch_names=channel_names,sfreq=160,ch_types='eeg')
        montage = mne.channels.make_standard_montage('standard_1005')
        info.set_montage(montage)
        
        
        evoked = mne.EvokedArray(data=scalp_topography_data, info=info)
        evoked.plot_topomap(times=[0], show_names=True, show=False)
        plt.title('Head Topography - Top View')
        plt.show()

                
        
        meanTestingAccuracy = sum(testing)/subject_number
        testing.clear()
        print(f'mean testing accuracy: {meanTestingAccuracy}')
        if meanTestingAccuracy>testing_accuracy:
                testing_accuracy=meanTestingAccuracy
                parameters[0]=k
                parameters[1]=m
                    
print (f'optimized hyperparameters are k:{parameters[0]},m:{parameters[1]} with accuracy {testing_accuracy}')