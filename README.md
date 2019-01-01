# ComputerVision-Fall2018
Predicting Vehicle Speed From Dashcam Video

# Challenges/Motivation:
	• Geometry unforgiving
	• Single camera lacks the sense of depth

• Input: Dashcam Video (datasets from highway and sub-urban driving)

• Processing: Convert video to images (frame by frame); find discernible features
between two successive image frames using various techniques to ascertain speed.
Obviously, it is a Regression problem.

• Output: Speed at every image frame

• Evaluation Metric(s): MSE(Mean Square Error) a.k.a L2 loss

• Tech Stack: Python 2.7, OpenCV 3.4.2, PyTorch 0.4.1, Scikit-Learn 0.19.2

# Techniques tried out

• With and without Semantic Segmentation

• Dense Optical Flow

• Sparse Optical Flow

• LSTM

• SVM


# References:

[1] Semantic Understanding of Scenes through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso and A. Torralba. International Journal on Computer Vision (IJCV), 2018. (https://arxiv.org/pdf/1608.05442.pdf)

[2] Scene Parsing through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. Computer Vision and Pattern Recognition (CVPR), 2017. (http://people.csail.mit.edu/bzhou/publication/scene-parse-camera-ready.pdf)
