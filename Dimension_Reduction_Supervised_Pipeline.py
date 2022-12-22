
#get_ipython().run_line_magic('matplotlib', 'inline')
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from random import shuffle
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.decomposition import PCA,KernelPCA
from sklearn.manifold import LocallyLinearEmbedding,Isomap,TSNE
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.linear_model import LogisticRegression
import time

import pdb

##RFE
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC


"""
Functions
"""

def make_parser():

    parser = argparse.ArgumentParser("Team 9: Exploration of Dimension Reduction")
    parser.add_argument("--data_name", type=str, default='./data/medical_3000')
    parser.add_argument("--save_dir", type=str, default='./final_results/results_supervised_feat_selection_3000')

    return parser


def load_data(image_folder,image_size):
    train_img = []
    for image in tqdm(os.listdir(image_folder)):
        path = os.path.join(image_folder, image)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (image_size, image_size))
        train_img.append(img)
        np_img = np.asarray(train_img)
    return np_img


def create_label(negative_data,positive_data):
    class_0 = np.zeros(negative_data.shape[0])
    class_1 = np.ones(positive_data.shape[0])
    return np.concatenate((class_0, class_1), axis=0)

def prepare_data(negative_folder,positive_folder, image_size):
    negative_data = load_data(negative_folder, image_size)
    positive_data = load_data(positive_folder, image_size)
    y_label = create_label(negative_data, positive_data)
    features = np.concatenate((negative_data, positive_data), axis=0)
    feature_flatten = features.reshape(features.shape[0], features.shape[1] * features.shape[2])
    idx = np.random.permutation(len(y_label))
    X, y = feature_flatten[idx], y_label[idx]
    return X, y

def fit_lr( X_train, X_test, y_train, y_test):
    """
    :param X: the features
    :param y: the labels
    :return: CCR_train = training accuracy
             CCR_test = testing accuracy
    """

    clf = LogisticRegression(random_state=123, penalty='none')
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    train_CCR = round(accuracy_score(y_train, train_pred) *100, 2)
    test_CCR = round(accuracy_score(y_test, test_pred) *100, 2)
    return train_CCR, test_CCR

def fit_lr_CV( X_CV, y_CV):
    """
    :param X: the features
    :param y: the labels
    :return: CCR_train = training accuracy
             CCR_test = testing accuracy
    """

    clf = LogisticRegression(random_state=123, penalty='none')
    clf.fit(X_CV, y_CV)
    CV_pred = clf.predict(X_CV)
    CV_CCR = round(accuracy_score(y_CV, CV_pred) *100, 2)
    return CV_CCR

def fit_dt( X_train, X_test, y_train, y_test):
    """
    :param X: the features
    :param y: the labels
    :return: CCR_train = training accuracy
             CCR_test = testing accuracy
    """

    clf = DecisionTreeClassifier(random_state=123, max_depth = 3)
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    train_CCR = round(accuracy_score(y_train, train_pred) *100, 2)
    test_CCR = round(accuracy_score(y_test, test_pred) *100, 2)
    return train_CCR,test_CCR


args = make_parser().parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
        
log_dir = f'{args.save_dir}/log.txt'
log_file = open(log_dir, 'w')
log_file.write('saving results \n')
    
"""
Load data
"""

negative_folder= f'{args.data_name}/0'
positive_folder= f'{args.data_name}/1'
image_size = 50

X, y = prepare_data(negative_folder,positive_folder, image_size)
X = X/255 # data normalization


"""
LDA
"""
print('start LDA')
log_file.write('start LDA\n')
st = time.time()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=123)

lda = LinearDiscriminantAnalysis(n_components=1)

X_new_train = lda.fit(X_train, y_train).transform(X_train)
X_new_test = lda.fit(X_train, y_train).transform(X_test)

train_CCR_lr, test_CCR_lr = fit_lr(X_new_train, X_new_test, y_train, y_test)
train_CCR_dt, test_CCR_dt = fit_dt(X_new_train, X_new_test, y_train, y_test)
log_file.write(f'Logistic Regression test ccr: {test_CCR_lr}\n')
log_file.write(f'Logistic Regression train ccr: {train_CCR_lr}\n')
log_file.write(f'Decision Tree test ccr: {test_CCR_dt}\n')
log_file.write(f'Decision Tree train ccr: {train_CCR_dt}\n')

et = time.time()
elapsed_time = et - st
log_file.write(f'Execution time: {elapsed_time} seconds\n')
log_file.write('')
log_file.write('')
"""
Cross Validation
"""
print('start LASSO with Logistic Regression')
log_file.write('start LASSO with Logistic Regression\n')
st = time.time()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=123)

c_list = np.arange(0.005,0.05,0.005)
test_CCR_lr = np.empty([c_list.size])
num_feat_lr = np.empty([c_list.size])
for idx, i in enumerate(c_list):
    lsvc = LinearSVC(C=i, penalty="l1", dual=False).fit(X_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    X_train_new = model.transform(X_train)
    num_feat_lr[idx] = X_train_new.shape[1]
    X_test_new = model.transform(X_test)    
    _, test_CCR_lr[idx]= fit_lr(X_train_new, X_test_new, y_train, y_test)
    log_file.write(f'lamda: {1/i} Test CCR of logstic regression: {test_CCR_lr[idx]}\n')
    
#plt.plot(1/c_list, test_CCR_lr, 'bo-', linewidth=2, markersize=8)
#plt.xlabel("lamda")
#plt.ylabel("CCR")
#plt.legend(["LASSO"], loc ="upper right")
#plt.title("Test CCR of LASSO")
#plt.savefig('./results_supervised_feat_selection_large/LASSO_CCR.png')
#plt.close()

plt.plot(1/c_list, num_feat_lr, 'bo-', linewidth=2, markersize=8)
plt.xlabel("lamda")
plt.ylabel("number of feature selected")
plt.legend(["LASSO"], loc ="upper right")
plt.title("Number of feature selected")
plt.savefig(f'{args.save_dir}/LASSO_num_feat.png')
plt.close()

et = time.time()
elapsed_time = et - st
log_file.write(f'Execution time: {elapsed_time} seconds\n')
log_file.write('')
log_file.write('')


print('start LASSO with Decision Tree')
log_file.write('start LASSO with Decision Tree\n')
st = time.time()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=123)

c_list = np.arange(0.005,0.05,0.005)
test_CCR_lr = np.empty([c_list.size])
num_feat_lr = np.empty([c_list.size])
for idx, i in enumerate(c_list):
    lsvc = LinearSVC(C=i, penalty="l1", dual=False).fit(X_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    X_train_new = model.transform(X_train)
    num_feat_lr[idx] = X_train_new.shape[1]
    X_test_new = model.transform(X_test)    
    _, test_CCR_lr[idx]= fit_dt(X_train_new, X_test_new, y_train, y_test)
    log_file.write(f'lamda: {1/i} Test CCR of logstic regression: {test_CCR_lr[idx]}\n')


"""
NCA
"""
print('start NCA with Logistic Regression')
log_file.write('start NCA with Logistic Regression\n')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=123)
X_test, X_CV, y_test, y_CV = train_test_split(X_test, y_test, test_size=0.50, random_state=123)

k_list = np.arange(10, 100, 10)
CCR_train = np.empty([k_list.size])
CCR_test = np.empty([k_list.size])
for idx, k in enumerate(k_list):

    st = time.time()
    
    embedding = NeighborhoodComponentsAnalysis(n_components=k)
    embedding = embedding.fit(X_train,y_train)
    X_train_new = embedding.transform(X_train)
    X_test_new = embedding.transform(X_test)
    CCR_train[idx], CCR_test[idx] = fit_lr(X_train_new, X_test_new, y_train, y_test)
    
    et = time.time()
    elapsed_time = et - st
    log_file.write(f'Execution time: {elapsed_time} seconds\n')

CCR_test_for_csv = CCR_test.tolist()
df_nca = pd.DataFrame(list(zip(k_list, CCR_test_for_csv)),
columns=['K', 'Variance'])

df_nca.to_csv(f'{args.save_dir}/Variance_NCA_LR.csv')

plt.plot(k_list,CCR_test, 'ro-', linewidth=2, markersize=8)    
plt.xlabel("Number of key components")
plt.ylabel("CCR")
plt.legend(["NCA"], loc ="upper right")
plt.title("Testing CCR against number of components")
plt.savefig(f'{args.save_dir}/NCA_LR.png')
plt.close()
log_file.write('')
log_file.write('')

print('start NCA with Decision Tree')
log_file.write('start NCA with Decision Tree\n')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=123)
X_test, X_CV, y_test, y_CV = train_test_split(X_test, y_test, test_size=0.50, random_state=123)

k_list = np.arange(10, 100, 10)
CCR_train = np.empty([k_list.size])
CCR_test = np.empty([k_list.size])
for idx, k in enumerate(k_list):

    st = time.time()
    
    embedding = NeighborhoodComponentsAnalysis(n_components=k)
    embedding = embedding.fit(X_train,y_train)
    X_train_new = embedding.transform(X_train)
    X_test_new = embedding.transform(X_test)
    CCR_train[idx], CCR_test[idx] = fit_dt(X_train_new, X_test_new, y_train, y_test)
    
    et = time.time()
    elapsed_time = et - st
    log_file.write(f'Execution time: {elapsed_time} seconds\n')

CCR_test_for_csv = CCR_test.tolist()
df_nca = pd.DataFrame(list(zip(k_list, CCR_test_for_csv)),
columns=['K', 'Variance'])
df_nca.to_csv(f'{args.save_dir}/Variance_NCA_DT.csv')

plt.plot(k_list,CCR_test, 'ro-', linewidth=2, markersize=8)    
plt.xlabel("Number of key components")
plt.ylabel("CCR")
plt.legend(["NCA"], loc ="upper right")
plt.title("Testing CCR against number of components")
plt.savefig(f'{args.save_dir}/NCA_DT.png')
plt.close()
log_file.write('')
log_file.write('')


"""
REF
"""
print('start RFE with Logistic Regression')
log_file.write('start RFE with Logistic Regression\n')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=123)
X_test, X_CV, y_test, y_CV = train_test_split(X_test, y_test, test_size=0.50, random_state=123)

k_list =  np.arange(10, 100, 10)
CCR_train = np.empty([k_list.size])
CCR_test = np.empty([k_list.size])

for idx, k in enumerate(k_list):
    st = time.time()
    
    estimator = LogisticRegression(penalty='l2',C = .15)
    selector = RFE(estimator, n_features_to_select=k, step=5)
    selector = selector.fit(X_train,y_train)
    X_train_new = selector.transform(X_train)
    X_test_new = selector.transform(X_test)
    CCR_train[idx], CCR_test[idx] = fit_lr(X_train_new, X_test_new, y_train, y_test)

    et = time.time()
    elapsed_time = et - st
    log_file.write(f'Execution time: {elapsed_time} seconds\n')
    
CCR_test_for_csv = CCR_test.tolist()
df_rfe = pd.DataFrame(list(zip(k_list, CCR_test_for_csv)),
columns=['K', 'Variance'])
df_rfe.to_csv(f'{args.save_dir}/Variance_RFE_LR.csv')

plt.plot(k_list, CCR_test, 'bo-', linewidth=2, markersize=8)    
plt.xlabel("Number of key components")
plt.ylabel("CCR")
plt.legend(["RFE"], loc ="upper right")
plt.title("Testing CCR against number of components")
plt.savefig(f'{args.save_dir}/REF_LR.png')


print('start RFE with Decision Tree')
log_file.write('start RFE with Decision Tree\n')
st = time.time()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=123)
X_test, X_CV, y_test, y_CV = train_test_split(X_test, y_test, test_size=0.50, random_state=123)

k_list =  np.arange(10, 100, 10)
CCR_train = np.empty([k_list.size])
CCR_test = np.empty([k_list.size])

for idx, k in enumerate(k_list):
    st = time.time()
    
    estimator = LogisticRegression(penalty='l2',C = .15)
    selector = RFE(estimator, n_features_to_select=k, step=5)
    selector = selector.fit(X_train,y_train)
    X_train_new = selector.transform(X_train)
    X_test_new = selector.transform(X_test)
    CCR_train[idx], CCR_test[idx] = fit_dt(X_train_new, X_test_new, y_train, y_test)

    et = time.time()
    elapsed_time = et - st
    log_file.write(f'Execution time: {elapsed_time} seconds\n')

CCR_test_for_csv = CCR_test.tolist()
df_rfe = pd.DataFrame(list(zip(k_list, CCR_test_for_csv)),
columns=['K', 'Variance'])
df_rfe.to_csv(f'{args.save_dir}/Variance_RFE_DT.csv')

plt.plot(k_list, CCR_test, 'bo-', linewidth=2, markersize=8)    
plt.xlabel("Number of key components")
plt.ylabel("CCR")
plt.legend(["RFE"], loc ="upper right")
plt.title("Testing CCR against number of components")
plt.savefig(f'{args.save_dir}/REF_DT.png')

log_file.close()
