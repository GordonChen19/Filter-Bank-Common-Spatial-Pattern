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

# ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.',
#  'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 
#  'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 
#  'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 
#  'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 
#  'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 
#  'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 
#  'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 
#  'O1..', 'Oz..', 'O2..', 'Iz..']


if __name__ == "__main__":
    
    dataset_details={'file_path':'/Users/gordonchen/Documents/MACS/URECA/Dataset/',
                'window_details':{'tmin':0,'tmax':4.1},
                'task_number':4,
                'ntimes':1,
                'kfold':5,
                'm_filters':2
                }


subject_number=1
testing_accuracy=0

channel_variance=defaultdict(float)

accuracies=[]
channel_usage=[]
for i in range (subject_number):
    print(f'subject {i+1}')
    ML=MLEngine.MLEngine(**dataset_details,subject_number=i+1)
    _,testingMean,number_channels=ML.channel_reduce()#[8,10,12,3,17]
    accuracies.append(testingMean)
    channel_usage.append(number_channels)
    
    
print(f'accuracies are {accuracies}')
print(f'number of channels used are {channel_usage}')



# #Displaying scalp topography
# scalp_topography_data=[[0] for x in range(64)]
# for i in range(64): 
#     if i in channel_variance:
#         scalp_topography_data[i][0]=channel_variance[i]/float(subject_number)

# scalp_topography_data[0][0]=0
# print(scalp_topography_data)


# # Try to set the BCI2000 system montage on the info structure
# info=mne.create_info(ch_names=channel_names,sfreq=160,ch_types='eeg')
# montage = mne.channels.make_standard_montage('standard_1005')
# info.set_montage(montage)


# evoked = mne.EvokedArray(data=scalp_topography_data, info=info)
# evoked.plot_topomap(times=[0], show_names=True, show=False)
# plt.title('Head Topography - Top View')
# plt.show()

        


