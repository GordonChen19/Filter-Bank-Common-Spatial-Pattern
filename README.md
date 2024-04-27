# Absract

In this paper, we present a channel selection method to improve common spatial patterns (CSP) for motor imagery (MI) classification. While traditional channel selection methods extract spectral power, coherence, or event-related potentials from the signals, this paper introduces a novel correlation-based channel selection method to identify channels that reflect the brain’s functional connectivity during MI tasks. It is known that regions of the brain related to MI share a temporal coherence in their neural activity. Hence, the proposed method aims to leverage on the idea of functional connectivity by selecting channels that contain more correlated and discriminative information over others. Each channel is assigned a connectivity score and grouped with channels that it is strongly correlated with. For each group of channels, those that are most representative of the group are selected. Channels are further distilled by computing the Fischer score from the spectro-spatial features of the Filter Bank Common Spatial Pattern (FBCSP) to identify the most discriminative channels for classification. When evaluated against PhysioNet’s motor imagery dataset (N=109), results showed that the proposed algorithm obtained superior classification accuracies across all 4 MI paradigms all while reducing the average number of channels by 83%. Correlation analysis also reveal interesting results that are congruent with neurophysiological principles, indicating a robust ability for feature selection, enabling the design of more efficient and accurate BCI systems.
<br>
Index Terms – Brain Computer Interface, Motor Imagery Classification, Machine Learning.
<br>
T0 corresponds to rest

T1 corresponds to onset of motion (real or imagined) of
    # the left fist (in runs 3, 4, 7, 8, 11, and 12)
    # both fists (in runs 5, 6, 9, 10, 13, and 14)
    
T2 corresponds to onset of motion (real or imagined) of
    # the right fist (in runs 3, 4, 7, 8, 11, and 12)
    # both feet (in runs 5, 6, 9, 10, 13, and 14)