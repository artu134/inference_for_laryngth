================
Changelog
================
2022-Dec-04	updated image data due to some corrupted files 


================
Overview
================

The dataset is designed to develop new approaches to identify anatomical structures in endoscopic high-speed videos (HSV) on the one hand 
and to provide a reference for comparing different approaches. This dataset contains 4 classes in total, which are namely "glottis", 
"vocal fold right", "vocal fold left", and "background". It was initially designed to train different configurations of a Neural Network 
on automatic segmentation of glottis and vocal folds in individual HSV frames solving a four-class problem by assigning each pixel to one 
of the classes. Further information can be found in Fehling et al. [1].


================
Data Description
================

The full dataset contains 13,000 frames from 130 HSV-sequences of healthy as well as pathologic (functional dysphonia, paresis, polyp, 
carcinoma) subjects. Each sequence comprises 100 consecutive frames that were captured with a spatial resolution of 256 x 256 pixels  
at a framerate of 4,000 fps using the rigid endoscopy system HRES ENDOCAM 5562 from Richard Wolf GmbH (Knittlingen, Germany). The 
dataset is split into subsets for training, validation and testing.


Subjects
================
The file "INFO_subjects.xlsx" contains information on diagnosis, sex and age of the particular subject. Start and end of the sequence is 
indicated by <ID_from> and <ID_end>.


Training Files
================
Training of the Neural Network was done on the training dataset comprising 100 HSV-sequences in total whereof 50 are from healthy 
subjects and 50 from pathologic subjects (25 functional, 25 organic). All frames and corresponding masks used for training can be 
found in the folder "train\". 


Validation Files
================
Validation of the trained model was done after each epoch using the validation dataset. This dataset comprises 15 sequences containing 
3 sequences per diagnosis. All frames and corresponding masks used for training can be found in the folder "val\". 


Test Files
================
The performance of different network configurations was measured on the test dataset. This dataset comprises 15 sequences containing 
3 sequences per diagnosis. All frames and corresponding masks used for training can be found in the folder "test\". 


image data, classes and corresponding masks
============================================
The individual HSV-frames (<ID>) are provided as consecutively numbered RGB images ("<ID>_rgb.png") with corresponding masks. 
A corresponding mask ("<ID>_mask.png") defines the ground truth. The class "glottis" is labled with 1, the class "vocal fold right" 
with 2, and the class "vocal fold left" with 3. Class label 0 indicates the "background".   


M, gA, gAs, and points P1 to P4
================================
The file "<ID>.txt" (subfolder: test\coord\) contains the coordinates for the glottis center ("M"), as well as the vectors defining the 
glottal axis ("gA", starting from M) and the vector orthogonal to the glottal axis ("gAs", also starting from M). These points were 
initially determined on the complete video sequence on behalf of the "PvgAnalyzer" to enable for a comparison with with the work of 
Lohscheller et al. [2]. Based on these both axes, the points P1 to P4 were defined. 
For the test-dataset for each mask <ID> the glottal center "M", the vectors "gA" and "gAs" as well as the points P1 to P4 are given  
in the file "<ID>.txt", where [0;0] indicates that this point is not defined for the current frame. These points define the dorsal (P1) 
and the ventral end (P2), as well as medial positions of the right (P3) and the left vocal fold (P4). For further information on the 
concrete definition of these points please refer to the works of Lohscheller et al. [2] and Fehling et al. [1].  
This file also contains the coordinates for the point M, which is defined as "glottis-center" from the points P and A. (M +/- gA) defines
the glottal axis, along which the points P1 and P2 were determined. P3 and P4 were determined on a axis orthogonal to the glottal axis 
positioned medial the vocal folds (M), which is defined as: (M +/- gAs). The points P1 to P4 each were determined as intersection of the 
"glottis edge" with the particular axis.  


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
Data Citation
================
[1] Fehling, M.K., Grosch, F., Schuster, M.E., Schick, B., Lohscheller, J., 2020.
    "Fully automatic segmentation of glottis and vocal folds in endoscopic laryngealhigh-speed videos using a deep Convolutional LSTM Network." 
    PloS one. DOI: 10.1371/journal.pone.0227791

[2] Lohscheller, J., Toy, H., Rosanowski, F., Eysholdt, U. and Döllinger, M., 2007. 
    Clinically evaluated procedure for the reconstruction of vocal fold vibrations from endoscopic digital high-speed videos. 
    Medical image analysis, 11(4), pp.400-413. DOI: 10.1016/j.media.2007.04.005
