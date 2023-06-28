import numpy as np
import scipy.signal as signal
from scipy.signal import cheb2ord
import FBCSP
import CSP
import Classifier
import LoadData
from sklearn.svm import SVR
from sklearn.decomposition import PCA
import mne
from collections import defaultdict
import matplotlib.pyplot as plt
class FilterBank:
    def __init__(self,fs):
        self.fs = fs
        self.f_trans = 2
        self.f_pass = np.arange(4,40,4)
        self.f_width = 4
        self.gpass = 3
        self.gstop = 30
        self.filter_coeff={}

    def get_filter_coeff(self):
        Nyquist_freq = self.fs/2

        for i, f_low_pass in enumerate(self.f_pass):
            f_pass = np.asarray([f_low_pass, f_low_pass+self.f_width])
            f_stop = np.asarray([f_pass[0]-self.f_trans, f_pass[1]+self.f_trans])
            wp = f_pass/Nyquist_freq
            ws = f_stop/Nyquist_freq
            order, wn = cheb2ord(wp, ws, self.gpass, self.gstop)
            b, a = signal.cheby2(order, self.gstop, ws, btype='bandpass')
            self.filter_coeff.update({i:{'b':b,'a':a}})

        return self.filter_coeff
    
    def filter_data(self,eeg_data,window_details={}):
        n_trials, n_channels, n_samples = eeg_data.shape
        if window_details:
            n_samples = int(self.fs*(window_details.get('tmax')-window_details.get('tmin')))+1
        filtered_data=np.zeros((len(self.filter_coeff),n_trials,n_channels,n_samples))
        for i, fb in self.filter_coeff.items():
            b = fb.get('b')
            a = fb.get('a')
            eeg_data_filtered = np.asarray([signal.lfilter(b,a,eeg_data[j,:,:]) for j in range(n_trials)])
            if window_details:
                eeg_data_filtered = eeg_data_filtered[:,:,int((window_details.get('tmin'))*self.fs):int((window_details.get('tmax'))*self.fs)+1]
            filtered_data[i,:,:,:]=eeg_data_filtered

        return filtered_data



class MLEngine:
    def __init__(self,task_number,subject_number,file_path,ntimes=1,kfold=2,m_filters=2,window_details={}):
        self.file_path = file_path
        self.kfold = kfold
        self.ntimes=ntimes
        self.window_details = window_details
        self.m_filters = m_filters
        self.task_number=task_number
        self.subject_number=subject_number
        
    def correlation(self,signal1,signal2):
        
        # Normalize the signals
        signal1_norm = (signal1 - np.mean(signal1))/np.std(signal1)
        signal2_norm = (signal2 - np.mean(signal2))/np.std(signal2)
        # Compute the correlation of determination between the two normalized signals
        corr_coef = np.corrcoef(signal1_norm, signal2_norm)[0, 1]
        # print(f'correlation coefficient is {corr_coef}')
        return corr_coef
    
    def compute_matrix(self,frequency_specific_data,y_labels,draw):
        
        #Sperating data into individual classes -> class1_data and class2_data
        class_filtered_data=defaultdict(list)
        for i,label in enumerate(y_labels):
            class_filtered_data[label].append(frequency_specific_data[i])
            
        class1_data=np.stack(class_filtered_data[2],axis=0)
        class2_data=np.stack(class_filtered_data[3],axis=0)
        class_data=[class1_data,class2_data]
        
        
        #(21,64,657)
        #(24,65,657)
        
        ###########################################################################

        # Ensemble averaging involves averaging the EEG signals across trials to reduce 
        # the effects of noise and identify common features -> channel_merged_data

        channel_merged_data=[defaultdict(list),defaultdict(list)]

        for classes in range(2): 
            for channel in range(frequency_specific_data.shape[1]):
                channel_data=class_data[classes][:,channel,:]
                evoked=np.mean(channel_data,axis=0)
                # print("printing evoked shape")
                # print(evoked.shape)

                channel_merged_data[classes][channel+1]=evoked
                # time = np.arange(len(evoked))
                # plt.plot(time, evoked)
                # plt.xlabel('Time (ms)')
                # plt.ylabel('EEG Amplitude')
                # plt.title('Single EEG Signal')
                # plt.show()
        
        
        ###########################################################################
        
        #Computing mean of signals for each class across all channels -> channel_mean
        
        channel_mean=[defaultdict(list),defaultdict(list)]
        for classes in range(2):
            for key,items in channel_merged_data[classes].items():
                channel_mean[classes][key].append(np.mean(items))
                
        
        ###########################################################################
        
        #Set up NxNx2 Determination Matrix
        
        channel_names = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 
                        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 
                        'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 
                        'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 
                        'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 
                        'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 
                        'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 
                        'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 
                        'O1', 'Oz', 'O2', 'Iz']

        
        determination_Matrix=[[[0]*64 for _ in range(64)]for _ in range(2)]
        
    
        ###########################################################################
        
        #Compute NxNx2 Determination Matrix
        #Find brain networks with union find
        
        
        determination_threshold=0.9
        
        correlation_score_2D=[] #highly correlated channels from each brain network are selected
        correlated_number=[]
        
        networks_by_root=defaultdict(list)
    
        
        uf=[i for i in range(64)]
        size=[1 for _ in range(64)]
        
        def union(a,b):
            nonlocal uf
            nonlocal size
            rootA=find(a)
            rootB=find(b)
            if(rootA==rootB):
                return
            elif(size[rootA]>size[rootB]):
                uf[rootB]=rootA
                size[rootA]+=size[rootB]
            else:
                uf[rootA]=rootB
                size[rootB]+=size[rootA]
            
        def find(a):
            nonlocal uf
            while(uf[a]!=a):
                uf[a]=uf[uf[a]]
                a=uf[uf[a]]
            return a
            
        
        for classes in range(2):
            for row in range(64):
                for column in range(row+1,64):
                    determination_Matrix[classes][row][column]=(self.correlation(channel_merged_data[classes][row+1],
                                                                                channel_merged_data[classes][column+1]))**2
                    
                    if(determination_Matrix[classes][row][column]>determination_threshold):
                        union(row,column)
                        
                correlated_channels=[i for i in determination_Matrix[classes][row] if i > determination_threshold]
                if(len(correlated_channels)==0):
                    correlation_score_2D.append(0)
                    correlated_number.append(0)
                else:
                    correlation_score_2D.append(sum(correlated_channels)/len(correlated_channels))
                    correlated_number.append(len(correlated_channels))
                    
        ###########################################################################
        # # Plot brain network graph
        
        import networkx as nx
        if(draw==True):
            G=nx.Graph()
            #Class 1 
            
            for row in range(64):
                for column in range(row,64):
                    if(determination_Matrix[1][row][column]>determination_threshold):
                        G.add_edge(channel_names[row],channel_names[column])
            
            montage = mne.channels.make_standard_montage('standard_1005')
            coords = montage.get_positions()
            
            pos = {channel_names[i]: (coords['ch_pos'][channel_names[i]][0],coords['ch_pos'][channel_names[i]][1]) for i in range(len(channel_names))}

            nx.draw(G,pos=pos,with_labels=True)
            plt.show()
                
        
        ###########################################################################
        
        correlation_score=[]
        correlated_number_score=[]
        for i in range(64): 
            avg_score = (correlation_score_2D[i] + correlation_score_2D[i+64]) / 2
            avg_number = (correlated_number[i] + correlated_number[i+64]) / 2
            correlation_score.append(avg_score)
            correlated_number_score.append(avg_number)
            

        

        for i,root in enumerate(uf):
            networks_by_root[root].append(i) 
            
        print("printing networks by root")
        print(networks_by_root)
            

        selected_channels=[]
        #select most informative channel from each brain network
        for values in networks_by_root.values():
            network_channel_scores=[correlation_score[i] for i in values]
            mean_correlation=sum(network_channel_scores)/len(network_channel_scores)
            selected_channels.extend([channels for channels in values if correlation_score[channels]>mean_correlation])
        
       
        return selected_channels
        
    

        
    
        
        
    def coefficient_analysis(self):
        
        
        my_data=LoadData.LoadMyData(self.file_path,self.subject_number,self.task_number)
        eeg_data=my_data.get_epochs()

    
        
        fbank=FilterBank(160)
        fbank_coeff=fbank.get_filter_coeff()
        filtered_data=fbank.filter_data(eeg_data.get('x_data'),self.window_details)
        
        channels_by_frequency=[]
        for i in range(filtered_data.shape[0]): #for each frequency band
            curr_channels=self.compute_matrix(filtered_data[i],eeg_data.get('y_labels'),i==1)
            print("selected channels are")
            print(curr_channels)
            channels_by_frequency.append(curr_channels)
            
        
        
        ###########################################################################

        # from collections import Counter
        # one_d_list=[]
        # for sublist in channels_by_frequency:
        #     for element in sublist:
        #         one_d_list.append(element)
        # freq=Counter(one_d_list)
        # most_common=freq.most_common(15)
        # common_nums=[x[0] for x in most_common]
        
        ###########################################################################
        
        intersections = []
        for array in channels_by_frequency:
            intersections.append(set(array))

        common_nums = set.intersection(*intersections)

        # Convert the set to a list
        common_nums = list(common_nums)

        ###########################################################################
        
        return common_nums
    
    
    def channel_reduce(self):
        selected_channels=self.coefficient_analysis()
        
        manual_channels=['C3','C4','Cz'] #,'FCz','CPz'
        
        
        channel_names = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 
                        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 
                        'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 
                        'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 
                        'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 
                        'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 
                        'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 
                        'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 
                        'O1', 'Oz', 'O2', 'Iz']
        
        for i in manual_channels:
            if(channel_names.index(i) not in selected_channels):
                selected_channels.append(channel_names.index(i))
                
        trainingMean,testingMean,number_channels=self.experiment(selected_channels)
        return trainingMean,testingMean,number_channels

    def experiment(self,selected_channels=[i for i in range(64)]):
        
        channel_names = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 
                    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 
                    'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 
                    'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 
                    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 
                    'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 
                    'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 
                    'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 
                    'O1', 'Oz', 'O2', 'Iz']
        print(f'Selected {len(selected_channels)} channels specifically {[channel_names[i] for i in selected_channels]}')
        
        '''Physionet Data Loading'''
        my_data = LoadData.LoadMyData(self.file_path,self.subject_number,self.task_number) 
        eeg_data = my_data.get_epochs()
        
        
        
        fbank = FilterBank(160)
        fbank_coeff = fbank.get_filter_coeff()
        filtered_data = fbank.filter_data(eeg_data.get('x_data')[:,selected_channels,:],self.window_details)
        
        print("printing shape of filtered data")
        print(filtered_data.shape)
        y_labels = eeg_data.get('y_labels')

        training_accuracy = []
        testing_accuracy = []
        for k in range(self.ntimes):
            # '''for N times x K fold CV'''
            # train_indices, test_indices = self.cross_validate_Ntimes_Kfold(y_labels,ifold=k)
            '''for K fold CV by sequential splitting'''
            train_indices, test_indices = self.cross_validate_sequential_split(y_labels)
            
            for i in range(self.kfold):
                train_idx = train_indices.get(i)
                test_idx = test_indices.get(i)
                print(f'Times {str(k)}, Fold {str(i)}\n')
                y_train, y_test = self.split_ydata(y_labels, train_idx, test_idx)
                x_train_fb, x_test_fb = self.split_xdata(filtered_data, train_idx, test_idx)

                y_classes_unique = np.unique(y_train)
                n_classes = len(np.unique(y_train))
                
                
  
                fbcsp = FBCSP.FBCSP(self.m_filters)

    
                
                fbcsp.fit(x_train_fb,y_train)
                
                y_train_predicted = np.zeros((y_train.shape[0], n_classes), dtype=np.float64)
                y_test_predicted = np.zeros((y_test.shape[0], n_classes), dtype=np.float64)

                for j in range(n_classes):
                    cls_of_interest = y_classes_unique[j]
                    select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]

                    y_train_cls = np.asarray(select_class_labels(cls_of_interest, y_train))
                    y_test_cls = np.asarray(select_class_labels(cls_of_interest, y_test))

                    x_features_train = fbcsp.transform(x_train_fb,class_idx=cls_of_interest-2)
                    x_features_test = fbcsp.transform(x_test_fb,class_idx=cls_of_interest-2)

                    classifier_type = SVR(gamma='auto')
                    classifier = Classifier.Classifier(classifier_type)
                    y_train_predicted[:,j] = classifier.fit(x_features_train,np.asarray(y_train_cls,dtype=np.float64))
                    y_test_predicted[:,j] = classifier.predict(x_features_test)


                y_train_predicted_multi = self.get_multi_class_regressed(y_train_predicted)

                y_test_predicted_multi = self.get_multi_class_regressed(y_test_predicted)
                
                
                

                tr_acc =np.sum(y_train_predicted_multi+2 == y_train, dtype=np.float64) / len(y_train)
                te_acc =np.sum(y_test_predicted_multi+2 == y_test, dtype=np.float64) / len(y_test)


                print(f'Training Accuracy = {str(tr_acc)}\n')
                print(f'Testing Accuracy = {str(te_acc)}\n \n')

                training_accuracy.append(tr_acc)
                testing_accuracy.append(te_acc)

        mean_training_accuracy = np.mean(np.asarray(training_accuracy))
        mean_testing_accuracy = np.mean(np.asarray(testing_accuracy))

        print('*'*10,'\n')
        print(f'Mean Training Accuracy = {str(mean_training_accuracy)}\n')
        print(f'Mean Testing Accuracy = {str(mean_testing_accuracy)}')
        print('*' * 10, '\n')
        return mean_training_accuracy,mean_testing_accuracy,len(selected_channels)

    def cross_validate_Ntimes_Kfold(self, y_labels, ifold=0):
        from sklearn.model_selection import StratifiedKFold
        train_indices = {}
        test_indices = {}
        random_seed = ifold
        skf_model = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=random_seed)
        i = 0
        for train_idx, test_idx in skf_model.split(np.zeros(len(y_labels)), y_labels):
            train_indices.update({i: train_idx})
            test_indices.update({i: test_idx})
            i += 1
        return train_indices, test_indices

    def cross_validate_sequential_split(self, y_labels):
        from sklearn.model_selection import StratifiedKFold
        train_indices = {}
        test_indices = {}
        skf_model = StratifiedKFold(n_splits=self.kfold, shuffle=False)
        i = 0
        for train_idx, test_idx in skf_model.split(np.zeros(len(y_labels)), y_labels):
            train_indices.update({i: train_idx})
            test_indices.update({i: test_idx})
            i += 1
        return train_indices, test_indices

    def split_xdata(self,eeg_data, train_idx, test_idx):
        x_train_fb=np.copy(eeg_data[:,train_idx,:,:])
        x_test_fb=np.copy(eeg_data[:,test_idx,:,:])
        return x_train_fb, x_test_fb

    def split_ydata(self,y_true, train_idx, test_idx):
        y_train = np.copy(y_true[train_idx])
        y_test = np.copy(y_true[test_idx])

        return y_train, y_test

    def get_multi_class_label(self,y_predicted, cls_interest=0):
        y_predict_multi = np.zeros((y_predicted.shape[0]))
        for i in range(y_predicted.shape[0]):
            y_lab = y_predicted[i, :]
            lab_pos = np.where(y_lab == cls_interest)[0]
            if len(lab_pos) == 1:
                y_predict_multi[i] = lab_pos
            elif len(lab_pos > 1):
                y_predict_multi[i] = lab_pos[0]
        return y_predict_multi

    def get_multi_class_regressed(self, y_predicted):
        y_predict_multi = np.asarray([np.argmin(y_predicted[i,:]) for i in range(y_predicted.shape[0])])
        return y_predict_multi



