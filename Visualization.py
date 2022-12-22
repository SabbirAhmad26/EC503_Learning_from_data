
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
import seaborn as sns
import cv2
import sklearn

from random import shuffle
from tqdm import tqdm
import warnings
import time



########### Visualization of negative and positive images

negative_folder= './medical_large/0'
positive_folder= './medical_large/1'


negative_imgs = os.listdir(negative_folder)[:20]
positive_imgs = os.listdir(positive_folder)[:20]
plt.figure(figsize=(20, 10))

i = 0
for img in negative_imgs:
    path = os.path.join(negative_folder, img)
    im = cv2.imread(path)
    im = cv2.resize(im, (50, 50))
    plt.subplot(4, 5, i+1)
    plt.title('IDC (-)')
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB));
    plt.axis('off')
    i += 1
plt.savefig("Negative_imgs.pdf")
plt.show()



plt.figure(figsize=(20, 10))
i = 0
for img in positive_imgs:
    path = os.path.join(positive_folder, img)
    im = cv2.imread(path)
    im = cv2.resize(im, (50, 50))
    plt.subplot(4, 5, i+1)
    plt.title('IDC (+)')
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB));
    plt.axis('off')
    i += 1
plt.savefig("Positive_imgs.pdf")
plt.show()




########### Load the results of different dimension reduction methods
df_pca = pd.read_csv("PCA_result.csv")
df_kernel = pd.read_csv("KernelPCA_result.csv")
df_lle = pd.read_csv("LLE_result.csv")
df_mds = pd.read_csv("Isomap_result.csv")
df_isomap = pd.read_csv("MDS_result.csv")
df_nca= pd.read_csv("NCA_result.csv")
df_rfe= pd.read_csv("./supervised/RFE_result.csv")
df_lda= pd.read_csv("./supervised/LDA_result.csv")
df_lasso= pd.read_csv("./supervised/LASSO_result.csv")

###########  Plotting reconstruction error for PCA, KernelPCA, LLE, ISOMAP
plt.plot(df_pca['K'], df_pca['Recon_error'], 'b-D',label='PCA reconstruction error')
plt.plot(df_kernel['K'], df_kernel['Recon_error'], 'r--o',label='KernelPCA reconstruction error')
plt.plot(df_lle['K'], df_lle['Recon_error'], 'y-.*',label='LLE reconstruction error')
plt.plot(df_isomap['K'], df_isomap['Recon_error'], 'g:h',label='Isomap reconstruction error')
plt.yscale('log')
plt.xlabel("Number of key components")
plt.ylabel("Reconstructed error")
plt.title("Reconstructed error against number of components")
plt.legend(loc='upper left')
plt.savefig("./small_results/Reconstruction_error.pdf")
plt.show()




###########  Plotting testing CCR for non-linear dimension reduction methods when using logistic regression as the baseline classifier
plt.plot(df_pca['K'], df_pca['LR_CCR'], 'b-D',label='PCA CCR')
plt.plot(df_kernel['K'], df_kernel['LR_CCR'], 'r--o',label='KernelPCA CCR')
plt.plot(df_lle['K'], df_lle['LR_CCR'], 'y-.*',label='LLE CCR')
plt.plot(df_isomap['K'], df_isomap['LR_CCR'], 'g:h',label='Isomap CCR')
plt.plot(df_mds['K'], df_mds['LR_CCR'], 'c-x',label='MDS CCR')
plt.xlabel("Number of key components")
plt.ylabel("CCR")
plt.title("Testing CCR against number of components for logistic regression")
plt.legend(loc='upper left')
plt.savefig("./small_results/LR_CCR.pdf")
plt.show()

###########  Plotting testing CCR for non-linear dimension reduction methods when using decision tree as the baseline classifier
plt.plot(df_pca['K'], df_pca['DT_CCR'], 'b-D',label='PCA CCR')
plt.plot(df_kernel['K'], df_kernel['DT_CCR'], 'r--o',label='KernelPCA CCR')
plt.plot(df_lle['K'], df_lle['DT_CCR'], 'y-.*',label='LLE CCR')
plt.plot(df_isomap['K'], df_isomap['DT_CCR'], 'g:h',label='Isomap CCR')
plt.plot(df_mds['K'], df_mds['DT_CCR'], 'c-x',label='MDS CCR')
plt.xlabel("Number of key components")
plt.ylabel("CCR")
plt.title("Testing CCR against number of components for decision tree")
plt.legend(loc='upper left')
plt.savefig("./small_results/DT_CCR.pdf")
plt.show()


###########  Plotting the computational complexity of non-linear dimension reduction methods
plt.plot(df_pca['K'], df_pca['Time'], 'b-D',label='PCA Time')
plt.plot(df_kernel['K'], df_kernel['Time'], 'r--o',label='KernelPCA Time')
plt.plot(df_lle['K'], df_lle['Time'], 'y-.*',label='LLE Time')
plt.plot(df_isomap['K'], df_isomap['Time'], 'g:h',label='Isomap Time')
plt.plot(df_mds['K'], df_mds['Time'], 'c-x',label='MDS Time')
plt.xlabel("Number of key components")
plt.ylabel("Computation Time")
plt.title("Computation Time against number of components")
plt.legend(loc='upper left')
plt.savefig("./small_results/time_vs_components.pdf")
plt.show()





###########  Plotting testing CCR for supervised methods when using logistic regression as the baseline classifier
plt.plot(df_lasso['K'], df_lasso['LASSO_CCR'], 'y-.*',label='LASSO CCR')
plt.plot(df_nca['K'], df_nca['LR_CCR'], 'g:h',label='NCA CCR')
plt.plot(df_rfe['K'], df_rfe['RFE_CCR'], 'k--<',label='RFE CCR')
plt.plot(df_lda['K'], df_lda['LDA_CCR'], 'c-x',label='LDA CCR')
plt.xlabel("Number of key components")
plt.ylabel("CCR")
plt.title("Testing CCR against number of components for supervised method")
plt.legend(loc='upper left')
plt.savefig("./supervised/supervised_CCR.pdf")
plt.show()



##### Combining the results for all methods in one plot
plt.figure(figsize=(10, 10))
plt.plot(df_pca['K'], df_pca['LR_CCR'], 'b-D',label='PCA CCR')
plt.plot(df_kernel['K'], df_kernel['LR_CCR'], 'r--o',label='KernelPCA CCR')
plt.plot(df_lle['K'], df_lle['LR_CCR'], 'y-.*',label='LLE CCR')
plt.plot(df_isomap['K'], df_isomap['LR_CCR'], 'g:h',label='Isomap CCR')
plt.plot(df_mds['K'], df_mds['LR_CCR'], 'c-x',label='MDS CCR')
plt.plot(df_lasso['K'], df_lasso['LASSO_CCR'], color='navy', linestyle='solid', marker='*',label='LASSO CCR')
plt.plot(df_nca['K'], df_nca['LR_CCR'], color='violet', linestyle='dashed', marker='p',label='NCA CCR')
plt.plot(df_rfe['K'], df_rfe['RFE_CCR'], color='orange', linestyle='dashed', marker='s', label='RFE CCR')
plt.plot(df_lda['K'], df_lda['LDA_CCR'],color='pink', linestyle='dashed', marker='o',label='LDA CCR')
plt.xlabel("Number of key components",fontsize = 15)
plt.xticks(fontsize= 15 )
plt.ylabel("CCR",fontsize = 15)
plt.yticks(fontsize= 15 )
plt.title("Testing CCR against number of components for all methods",fontsize = 18)
plt.legend(loc='upper left', fontsize = 'large')

plt.savefig("./supervised/all_CCR.pdf")
plt.show()

print("Done!!!!!")
