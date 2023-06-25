================
Changelog
================
2022-Dec-04	updated image data due to some corrupted files 


================
Overview
================

This repository is associated with the manuscript "Fully automatic segmentation of glottis and vocal folds in endoscopic laryngeal 
high-speed videos using a deep Convolutional LSTM Network" [1], where we used a deep Convolutional Neural Network (CNN) approach for 
the first time to fully automatically segment not only the time-varying glottal area but also the vocal fold tissue directly from 
laryngeal HS video. The approach was developed and intensely evaluated on a dataset comprising 130 HS-sequences 
(13,000 HS video frames in total) obtained from healthy as well as pathologic subjects.

Here, we provide the used dataset, the code along with the best performing Neural Network, and scripts to evaluate the segmentation 
performance.

This work was supported by the German Research Foundation (DFG), LO-1413/2-2. Computational resources were provided by the High 
Performance Compute Cluster 'Elwetritsch' at the University of Kaiserslautern, which is part of the 'Alliance of High Performance 
Computing Rheinland-Pfalz' (AHRP). We kindly acknowledge the support.


=============================
Repository
=============================

----------------------
Folder "dataset\"
----------------------
The full dataset contains 13,000 frames from 130 HSV-sequences of healthy as well as pathologic (functional dysphonia, paresis, polyp, 
carcinoma) subjects. Each sequence comprises 100 consecutive frames that were captured with a spatial resolution of 256 x 256 pixels  
at a framerate of 4,000 fps using the rigid endoscopy system HRES ENDOCAM 5562 from Richard Wolf GmbH (Knittlingen, Germany). The 
dataset is split into subsets for training, validation and testing. Further information on the dataset can be found in "README_dataset.txt".

----------------------
Folder "code\"
----------------------
This folder contains the sourcecode to train the Neural Network and to generate predictions accordingly. Details on the trained networks 
can be found in Fehling et al. [1]. Tensorflow 1.8.0 or higher, Python 3.x and the following libraries are required: 
- skimage (from scikit-image)
- scipy (from scikit-image)
- numpy 
- pillow
Additionally, to train a U-LSTM, i.e. the provided U-LSTM_5_CE, two GPUs with computing capability of CUDA 5.1 or higher are mandatory. 

----------------------
Folder "matlab\"
----------------------
This folder contains the matlab-scipts we used to determine the points P_1 to P_4 in the Ground Truth as well as in the generated predictions. 
The points Pi can be calculated using the function "calc_points_Pi()". As input it needs coordinates and vectors stored in the file "<ID>.txt" 
(subfolder: dataset\test\coord\), further information can be found in the datasets "README.txt". 
Afterwards, the distances Di can be calculated using the function "calc_distanstances_Di()". 
Our implementation of the Dice Coefficient is based on the matlab function "dice()" and is provided as "own_dice()". 

----------------------
Folder "U-LSTM_5_CE\"
----------------------
This folder contains the U-LSTM_5^CE, which has shown the best overall segmentation performance. The trained Neural Network model can be found in the subfolder 
"LSTM_SA_L5_F64_Dim3_Seq10_LFsoftmax_LR1.00E-04_C4_CWNone_DeRNone_BNTrue_RegNone_RegS1.00E-05_StdFalse_CCTrue_constFFalse_GRUFalse_LSTMfull_rgb_nat_BS1_E40\". 
The used parameters are stored in the file "parameters_U-LSTM5CE.out" and predictions generated after the 30th epoch can be found in the subfolder "model30000\". 


================
License
================
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License. 


==============================
Further Information & Contact
==============================
https://www.hochschule-trier.de/go/quantitative-laryngoscopy
quantitative-laryngoscopy@hochschule-trier.de


================
References
================
[1] Fehling, M.K., Grosch, F., Schuster, M.E., Schick, B., Lohscheller, J., 2020.
    "Fully automatic segmentation of glottis and vocal folds in endoscopic laryngealhigh-speed videos using a deep Convolutional LSTM Network." 
    PloS one. DOI: 10.1371/journal.pone.0227791

