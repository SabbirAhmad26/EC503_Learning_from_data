# EC503_Learning_from_data
1. Link to Google drive folder : https://drive.google.com/file/d/1HghMa1QZdrU1wXfB9-eFEUmD43SNsPLY/view?usp=sharing
The Google drive folder contains all the datasets and codes that used in this project

2. Link to original sources of data:
https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images

3. Libraries and dependencies: pandas, numpy, matplotlib, seaborn, cv2, sklearn,tqdm

4. In the Dimension_Reduction_Pipeline.py:

 &emsp;&emsp;(a) load_data() function is used for loading the images, converting from RGB iamges into
grayscale images, returning a numpy array.

  (b) create_label() function is used to generate the label for the data.
  
  (c) prepare_data() function calls load_data() function to load the positive and negative images
and then calls create_label() function to generate the numerical label for each class.
Finally, it flattens each image into one long vector and return a numpy array of features
(number of examples, image_size * image_size) and a numpy array of label (number of
examples, 1).
  
  (d) fit_pca(), fit_kernelPCA(), fit_NCA(), fit_lle(), fit_isomap(), fit_MDS() are the functions used
to call the corresponding dimension reduction methods.
The input of these functions is the features and the number of key components. All these
functions return the transformed features and the reconstruction error. 
