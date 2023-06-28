import mne
import numpy as np

class LoadData:
    def __init__(self,eeg_file_path: str):
        self.eeg_file_path = eeg_file_path
        self.raw_eeg_data=[]
        self.raw_eeg_baseline_data=[]
        
    def load_raw_data_edf(self,file_to_load,subject_number):
        
        raw_data=mne.io.read_raw_edf(self.eeg_file_path  + 'S' + str(subject_number).zfill(3)
                                                    + '/' + 'S' + str(subject_number).zfill(3) 
                                                    + file_to_load)
        
        # print(raw_data.info['ch_names'])
        
        # for channel_name in channel_names:
        #     full_channel_name = channel_name.rstrip('.')
        #     print(full_channel_name)
        
        self.raw_eeg_data.append(raw_data)
        return self
    def load_baseline_edf(self,subject_number):
        raw_baseline= mne.io.read_raw_edf(self.eeg_file_path+'S'+str(subject_number).zfill(3)+
                                                              '/S'+ str(subject_number).zfill(3) + 
                                                              'R01.edf')
        
        self.raw_eeg_baseline_data.append(raw_baseline) #eyes open
class LoadMyData(LoadData):
    '''Subclass of LoadData for loading Physionet data'''
    
    def __init__(self,file_path,subject_number,task_number):
        self.task_number=task_number
        self.subject_number=subject_number
        self.fs=160 #Hz
        super(LoadMyData,self).__init__(file_path)
        
    def get_epochs(self):
        # for i in range(30): #number of participants (to change back to 109)
        self.load_baseline_edf(self.subject_number) #change back to i+1
        for j in range(3): #3 repeated trials
            run_number=self.task_number+2+(4*j)
            self.load_raw_data_edf('R'+str(run_number).zfill(2)+'.edf',self.subject_number)#change back to i+1
            
        
        #[s1r1,s1r2,s1r3,s2r1,s2r2,s2r3...] raw_eeg_data
        #[s1,s2,s3...] raw_eeg_baseline_data
        
        #calculate mean and std of each baseline run
        
        z_score={}
        for i in range(len(self.raw_eeg_baseline_data)):
            raw_baseline=self.raw_eeg_baseline_data[i].get_data()
            
            
            mean=raw_baseline.mean(axis=1) #baseline mean wrt each channel
            std=raw_baseline.std(axis=1) #baseline std wrt each channel
            
            z_score[f'{i+1}']=[mean,std]
                

        x_datas=[]
        y_labels=[]
        for i in range(len(self.raw_eeg_data)):
            events, event_ids = mne.events_from_annotations(self.raw_eeg_data[i])
            # print(events)
            
            del event_ids['T0']
            epoch=mne.Epochs(self.raw_eeg_data[i],events,event_ids,tmin=0,tmax=4.1,
                                         event_repeated='drop',baseline=None,
                                         preload=True,proj=False,reject_by_annotation=False)
            
            # print(epoch.info['ch_names'])
            
            if(epoch.info['sfreq']!=160):
                continue
            
            x_data=np.array(epoch)*1e6
            subject_number=(int)(i/3+1)
            baseline_mean=z_score[f'{subject_number}'][0].reshape(1,64,1)*1e6
            baseline_std=z_score[f'{subject_number}'][1].reshape(1,64,1)*1e6
            x_data_normal=(x_data-baseline_mean)/baseline_std  #normalising data wrt z scores
            y_label=epoch.events[:,-1]
            
            x_datas.append(x_data_normal)
            y_labels.append(y_label)
            
    
        self.x_data=np.concatenate(x_datas,axis=0)
        # print("printing x_data shape")
        # print(self.x_data.shape)
        # print("printing x_data")
        # print(self.x_data)
        self.y_labels=np.concatenate(y_labels)
        
      
        
        return {'x_data':self.x_data, 'y_labels':self.y_labels,'fs':self.fs}
        
# subject1=LoadMyData('/Users/gordonchen/Documents/MACS/URECA/Dataset/',
#                     1,
#                     1)

# subject1.get_epochs()