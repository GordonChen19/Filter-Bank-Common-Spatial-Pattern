# Absract

Motor-imagery-based Brain Computer Interface (BCI) provides a promising channel of communication for users with motor disabilities or even severe paralysis. However, a lack of adequate training samples, high dimensional signals and large inter-subject variability impose key challenges in decoding electroencephalogram (EEG) signals from BCI. To address these issues, this paper implements a novel Filter Bank Common Spatial Pattern (FBCSP) and assesses its efficacy in classifying subject-specific motor imagery data as a reflection of its performance on limited training data as well as subject variability. The EEG signal is bandpass filtered into frequency bands, following which Common Spatial Patterns (CSP) are extracted from each band. Parameters and hyperparameters are then optimized through autonomous temporal spatial-pattern feature selection and grid search cross validation respectively, prior to support vector machine (SVM) classification. FBCSP demonstrates its effectiveness in handling limited training data by achieving a mean binary classification accuracy of 80.89% on Physionet’s motor imagery dataset. This performance is just 5.47% lower than the current state-of-the-art (SOA) performance, despite being trained on less than 1% of the data utilized by the SOA, making it a highly practical machine learning model. 

Index Terms – Brain Computer Interface, Motor Imagery Classification, EEG signal classification, Machine Learning.

We will be focusing the decoding of the opening and closing of the hands and feet (i.e. Task 3) using Task1 (Eyes open) as a baseline


T0 corresponds to rest

T1 corresponds to onset of motion (real or imagined) of
    # the left fist (in runs 3, 4, 7, 8, 11, and 12)
    # both fists (in runs 5, 6, 9, 10, 13, and 14)
    
T2 corresponds to onset of motion (real or imagined) of
    # the right fist (in runs 3, 4, 7, 8, 11, and 12)
    # both feet (in runs 5, 6, 9, 10, 13, and 14)