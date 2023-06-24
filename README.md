# Absract

Abstract - Motor-imagery-based Brain Computer Interface (BCI) provides a promising channel of communication for users with motor disabilities. However, high dimensional signals and inter-subject variability impose key challenges in decoding electroencephalogram (EEG) signals from BCI. An important aspect is determining the optimal number and selection of scalp electrodes (channels) to effectively discriminate different motor tasks while minimizing noise. This paper addresses these challenges by introducing a novel feature extraction method based on the Filter Bank Common Spatial Pattern (FBCSP) algorithm for channel selection. In this method, the EEG signal is bandpass filtered into frequency bands, following which Common Spatial Patterns (CSP) are extracted from each band. Channels are then ranked and selected by the discriminative power of their spectro-spatial synthesized features. The performance improvements achieved using optimized channels, compared to using all channels, or the typical channels for motor imagery (C3,C4,Cz) are evaluated using one-way ANOVA. The proposed algorithm is assessed on Physionet’s motor imagery dataset (N=109). The results show that the proposed algorithm improved subject-specific classification accuracies of all 4 motor imagery paradigms while reducing the number of channels by at least 44%. The selection results also reveal that the optimized channels differ across different paradigms and exhibit varying discriminative power, indicating a robust ability for feature selection. The source code can be found at https://github.com/GordonChen19/Filter-Bank-Common-Spatial-Pattern-Channel-Selection.
Index Terms – Brain Computer Interface, Motor Imagery Classification, Machine Learning.

T0 corresponds to rest

T1 corresponds to onset of motion (real or imagined) of
    # the left fist (in runs 3, 4, 7, 8, 11, and 12)
    # both fists (in runs 5, 6, 9, 10, 13, and 14)
    
T2 corresponds to onset of motion (real or imagined) of
    # the right fist (in runs 3, 4, 7, 8, 11, and 12)
    # both feet (in runs 5, 6, 9, 10, 13, and 14)