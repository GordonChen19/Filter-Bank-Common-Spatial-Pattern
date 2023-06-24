import numpy as np
import CSP
from collections import defaultdict

class FBCSP:
    def __init__(self,m_filters):
        self.m_filters = m_filters
        self.fbcsp_filters_multi=[]
        self.selected_channels=[]
        
    def channel_reduce(self,x_train_fb,y_train,channel_variance):
        y_classes_unique = np.unique(y_train)
        n_classes = len(y_classes_unique)
        self.csp = CSP.CSP(self.m_filters)

        def get_csp(x_train_fb, y_train_cls):
            fbcsp_filters = {}
            for j in range(x_train_fb.shape[0]):
                x_train = x_train_fb[j, :, :, :]
                eig_values, u_mat = self.csp.fit(x_train, y_train_cls)
                fbcsp_filters.update({j: {'eig_val': eig_values, 'u_mat': u_mat}})
            return fbcsp_filters

        for i in range(n_classes):
            cls_of_interest = y_classes_unique[i]

            select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]
            y_train_cls = np.asarray(select_class_labels(cls_of_interest, y_train))

            fbcsp_filters=get_csp(x_train_fb,y_train_cls)
            self.fbcsp_filters_multi.append(fbcsp_filters)
            
         # Channel Selection based on variance difference
        
        
        for channel_idx in range(x_train_fb.shape[1]):
            variance_diff = np.abs(np.mean([self.fbcsp_filters_multi[0][fbank_idx]['eig_val'][channel_idx] - self.fbcsp_filters_multi[1][fbank_idx]['eig_val'][channel_idx] for fbank_idx in range(x_train_fb.shape[0])]))
            if variance_diff > 0.0:
                self.selected_channels.append(channel_idx)
                # data.append([channel_idx,variance_diff])
                channel_variance[channel_idx]+=variance_diff
        print("number of channels selected")
        print(len(self.selected_channels))
        print("selected channels are")
        print(channel_variance)
        # print("channels selected are")
        # data.sort(key=lambda x: x[1])
        # print(data)
        return self.selected_channels,channel_variance

    def fit(self,x_train_fb,y_train):
        y_classes_unique = np.unique(y_train)
        n_classes = len(y_classes_unique)
        self.csp = CSP.CSP(self.m_filters)

        def get_csp(x_train_fb, y_train_cls):
            fbcsp_filters = {}
            for j in range(x_train_fb.shape[0]):
                x_train = x_train_fb[j, :, :, :]
                eig_values, u_mat = self.csp.fit(x_train, y_train_cls)
                fbcsp_filters.update({j: {'eig_val': eig_values, 'u_mat': u_mat}})
            return fbcsp_filters

        for i in range(n_classes):
            cls_of_interest = y_classes_unique[i]

            select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]
            y_train_cls = np.asarray(select_class_labels(cls_of_interest, y_train))

            fbcsp_filters=get_csp(x_train_fb,y_train_cls)
            self.fbcsp_filters_multi.append(fbcsp_filters)
        

    def transform(self,x_data,class_idx=0):
        x_data=x_data[:,:,:,:]
        n_fbanks, n_trials, n_channels, n_samples = x_data.shape
        x_features = np.zeros((n_trials,self.m_filters*2*len(x_data)),dtype=np.float64)
        for i in range(n_fbanks):
            eig_vectors = self.fbcsp_filters_multi[class_idx].get(i).get('u_mat')
            eig_values = self.fbcsp_filters_multi[class_idx].get(i).get('eig_val')
            for k in range(n_trials):
                x_trial = np.copy(x_data[i,k,:,:])
                csp_feat = self.csp.transform(x_trial,eig_vectors)
                for j in range(self.m_filters):
                    x_features[k, i * self.m_filters * 2 + (j+1) * 2 - 2]  = csp_feat[j]
                    x_features[k, i * self.m_filters * 2 + (j+1) * 2 - 1]= csp_feat[-j-1]

        return x_features

