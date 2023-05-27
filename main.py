import MLEngine
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":

    
    kfold=[5]
    m_filters=[2]
    
    dataset_details={'file_path':'/Users/gordonchen/Documents/MACS/URECA/URECA Codebase/URECA-FBCSP-vs-FBCNet-Architectures/FBCSP/bin/Physionet Dataset/',
                'window_details':{'tmin':0,'tmax':4.1},
                'task_number':3,
                'ntimes':1
                }

testing=[]
subject_number=10 
testing_accuracy=0
parameters=[-1,-1]

for k in kfold:
    for m in m_filters:
        for i in range (subject_number):
            print(f'subject {i+1}')
            ML=MLEngine.MLEngine(**dataset_details,subject_number=i+1,kfold=k,m_filters=m)
            trainingMean,testingMean=ML.experiment()
            testing.append(testingMean)
            
        meanTestingAccuracy = sum(testing)/subject_number
        testing.clear()
        print(f'mean testing accuracy: {meanTestingAccuracy}')
        if meanTestingAccuracy>testing_accuracy:
                testing_accuracy=meanTestingAccuracy
                parameters[0]=k
                parameters[1]=m
                    
print (f'optimized hyperparameters are k:{parameters[0]},m:{parameters[1]} with accuracy {testing_accuracy}')