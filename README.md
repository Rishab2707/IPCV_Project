# IPCV - Project UE19CS333
Project Members:
1) Rishab K S - PES1UG19CS384
2) Rishab Kashyap B S - PES1UG19CS385
3) Pranav M R - PES1UG19CS340
4) Adhithya Sundar - PES1UG19CS026

Required Libaries:

                                                      tensorflow>=1.15.2
                                                      keras==2.3.1
                                                      imutils==0.5.3
                                                      numpy==1.18.2
                                                      opencv-python==4.2.0.*
                                                      matplotlib==3.2.1
                                                      scipy==1.4.1

## train_mask_detector.py: 
This file creates a deep learning model "mask_detector.model" which produces the probability of a face mask being worn.

Note: Please change the below line (which is the directory for image dataset) in the above file to your working directory

                                                      DIRECTORY = r"C:\Users\Rishab\IPCV_Project\dataset"

code to run : 
                                                      
                                                      python train_mask_detector.py


## detect_mask_video.py: 
This file provides a real time detection of face mask on a human face from a video camera (WebCam in this case).

code to run : 
                                                      
                                                      python detect_mask_video.py
