import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from random import shuffle
from PIL import Image
from tqdm import tqdm
import warnings
import time

warnings.filterwarnings('ignore')
import os
from sklearn.decomposition import PCA,KernelPCA
from sklearn.manifold import LocallyLinearEmbedding,Isomap,MDS
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import NeighborhoodComponentsAnalysis

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



def load_data(image_folder,image_size):
    """

    :param image_folder: the dataset fold
    :param image_size: the input image size
    :return: a numpy array of images
    """
    train_img = []
    for image in tqdm(os.listdir(image_folder)[:3000]):
        path = os.path.join(image_folder, image)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (image_size, image_size))
        train_img.append(img/255.)
        np_img = np.asarray(train_img)
    return np_img


def create_label(negative_data,positive_data):
    """

    :param negative_data:  a numpy array of negative examples
    :param positive_data: a numpy array of postive examples
    :return: a numpy array of 0/1 label
    """
    class_0 = np.zeros(negative_data.shape[0])
    class_1 = np.ones(positive_data.shape[0])
    return np.concatenate((class_0, class_1), axis=0)

def prepare_data(negative_folder,positive_folder, image_size):
    """

    :param negative_folder: the folder contains the negative examples
    :param positive_folder:
    :param image_size:
    :return:
    """
    negative_data = load_data(negative_folder, image_size)
    positive_data = load_data(positive_folder, image_size)
    y_label = create_label(negative_data, positive_data)
    features = np.concatenate((negative_data, positive_data), axis=0)
    feature_flatten = features.reshape(features.shape[0], features.shape[1] * features.shape[2])
    idx = np.random.permutation(len(y_label))
    X, y = feature_flatten[idx], y_label[idx]
    return X, y


#intializate the PCA class
def fit_pca(k, X):
    """
    :param k: the number of components
    :param X: the features (without labels
    :return: X_new : the transformed features
             explained_variance : the total variance explained by the top K components
    """
    pca = PCA(n_components=k)
    X_new = pca.fit_transform(X)
    explained_variance = round(np.sum(pca.explained_variance_ratio_) * 100,2)
    X_reconstructed = pca.inverse_transform(X_new)
    reconstructed_error = round(mean_squared_error(X,X_reconstructed),2)

    return X_new, explained_variance,reconstructed_error

def fit_kernelPCA(k, X):
    """
    :param k: the number of components
    :param X: the features (without labels
    :return: X_new : the transformed features

    """
    kernel_pca = KernelPCA(n_components=k,kernel="rbf",fit_inverse_transform=True)
    X_new = kernel_pca.fit_transform(X)
    X_reconstructed = kernel_pca.inverse_transform(X_new)
    reconstructed_error = round(mean_squared_error(X,X_reconstructed),2)
    return X_new,reconstructed_error

def fit_NCA(k, X, y):
    """
    :param k: the number of components
    :param X: the features (without labels
    """
    embedding = NeighborhoodComponentsAnalysis(n_components=k)
    X_new = embedding.fit_transform(X,y)
    reconstructed_error = np.NAN  # no way to calculate the reconstructed error
    return X_new,reconstructed_error

def fit_lle(k, X, neighbors):
    """
    :param k: the number of components
    :param X: the features (without labels

    """
    # need cross-validation

    embedding = LocallyLinearEmbedding(n_components=k,n_neighbors= neighbors)
    X_new =  embedding.fit_transform(X)
    reconstructed_error = embedding.reconstruction_error_

    return X_new, reconstructed_error



def fit_isomap(k, X,neighbors):
    """
    :param k: the number of components
    :param X: the features (without labels

    """
    # need cross-validation

    embedding = Isomap(n_components=k,n_neighbors= neighbors)
    X_new = embedding.fit_transform(X)
    reconstructed_error = embedding.reconstruction_error()

    return X_new, reconstructed_error




def fit_MDS(k, X):
    """
    :param k: the number of components
    :param X: the features (without labels

    """
    embedding = MDS(n_components=k)
    X_new = embedding.fit_transform(X)
    reconstructed_error =  np.NAN # no way to calculate the reconstructed error

    return X_new, reconstructed_error





def fit_lda(k, X):
    """
    :param k: the number of components
    :param X: the features (without labels

    """
    clf = LinearDiscriminantAnalysis(n_components=k)
    X_new = clf.fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.30, random_state=123)
    test_CCR_lr = fit_lr(X_train, X_test, y_train, y_test)
    test_CCR_dt = fit_dt(X_train, X_test, y_train, y_test)



    return test_CCR_lr,test_CCR_dt




def fit_lr( X_train, X_test, y_train, y_test):
    """
    :param X: the features
    :param y: the labels
    :return: CCR_train = training accuracy
             CCR_test = testing accuracy
    """

    clf = LogisticRegression(random_state=123, penalty='none')
    clf.fit(X_train, y_train)
    # train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    # train_CCR = round(accuracy_score(y_train, train_pred) *100, 2)
    test_CCR = round(accuracy_score(y_test, test_pred) *100, 2)
    return test_CCR

def fit_dt( X_train, X_test, y_train, y_test):
    """
    :param X: the features
    :param y: the labels
    :return: CCR_train = training accuracy
             CCR_test = testing accuracy
    """

    clf = DecisionTreeClassifier(random_state=123, max_depth = 3)
    clf.fit(X_train, y_train)
    # train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    # train_CCR = round(accuracy_score(y_train, train_pred) *100, 2)
    test_CCR = round(accuracy_score(y_test, test_pred) *100, 2)
    return test_CCR

def plot_variance(n_component_ls, explained_variance_ls, file= "Variance_PCA.pdf"):
    """

    :param n_component_ls: a list contains the number of components
    :param explained_variance_ls: a list contains the explained variance of different number of components
    :param file: the file name to save the PCA result.
    :return: none
    """
    plt.plot(n_component_ls, explained_variance_ls, 'r--')
    plt.xlabel("Number of key components")
    plt.ylabel("% of explained variance")
    plt.title("% of explained variance against number of components")
    plt.show()
    plt.savefig(file)


def tune_neighbors(X, initial_com, end_com, step, neighbors_ls, transform_func):
    """
    :param X: features
    :param initial_com: the smallest number of components
    :param end_com: the largest number of components
    :param step:
    :param neighbors_ls: a list contains the candidate number of neighbors using in neighbor-based method
    :param transform_func: dimension reduction method
    :return: the optimal number of neighbor should be use for a specific method
    """
    best_neighbors_ls = []
    for k in range(initial_com, end_com, step):
        reconstructed_error_ls = []
        for neighbors in neighbors_ls:
            _, reconstructed_error = transform_func(k, X,neighbors)
            reconstructed_error_ls.append(reconstructed_error)
        best_neighbors_ls.append(neighbors_ls[np.argmin(reconstructed_error_ls)])
    best_neighbors = np.bincount(best_neighbors_ls).argmax()
    return best_neighbors






def fit_dimension_classifier(dimensional_reduction, X, initial_com, end_com, step,neighbors_dict):
    """

    :param dimensional_reduction: dimension reduction method
    :param X: features
     param initial_com: the smallest number of components
    :param end_com: the largest number of components
    :param step:
    :param neighbors_dict: a dictionary, key is the method, the value is the optimal neighbor.
    :return: a dataframe contains the results
    """
    n_component_ls, explained_variance_ls, reconstructed_error_ls, test_CCR_lr, test_CCR_dt,time_ls = [], [], [], [], [],[]
    for k in range(initial_com, end_com, step):
        if dimensional_reduction == "PCA":
            start = time.time()
            X_new, explained_variance, reconstructed_error= fit_pca(k, X)
            end = time.time()
            time_ls.append(round(end - start,2))
            explained_variance_ls.append(explained_variance)
            df_pca = pd.DataFrame(list(zip(n_component_ls, explained_variance_ls)),
                                  columns=['K', 'Variance'])
            df_pca.to_csv("Variance_PCA.csv")

        if dimensional_reduction == "LLE":
            start = time.time()
            X_new, reconstructed_error = fit_lle(k, X, neighbors_dict["LLE"])
            end = time.time()
            time_ls.append(round(end - start, 2))

        if dimensional_reduction == "KernelPCA":
            start = time.time()
            X_new, reconstructed_error = fit_kernelPCA(k, X)
            end = time.time()
            time_ls.append(round(end - start, 2))

        if dimensional_reduction == "NCA":
            start = time.time()
            X_new, reconstructed_error = fit_NCA(k, X, y)
            end = time.time()
            time_ls.append(round(end - start, 2))

        if dimensional_reduction == "Isomap":
            start = time.time()
            X_new, reconstructed_error = fit_isomap(k, X, neighbors_dict["Isomap"])
            end = time.time()
            time_ls.append(round(end - start, 2))


        if dimensional_reduction == "MDS":
            start = time.time()
            X_new, reconstructed_error = fit_MDS(k, X)
            end = time.time()
            time_ls.append(round(end - start, 2))

        # if dimensional_reduction == "LDA":
        #     X_new, reconstructed_error = fit_lda(1, X)


        n_component_ls.append(k)
        reconstructed_error_ls.append(reconstructed_error)
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.25, random_state=123)
        test_CCR_lr.append(fit_lr(X_train, X_test, y_train, y_test))
        test_CCR_dt.append(fit_dt(X_train, X_test, y_train, y_test))

        df = pd.DataFrame(list(zip(n_component_ls, reconstructed_error_ls, test_CCR_lr, test_CCR_dt,time_ls)), columns=['K', 'Recon_error', 'LR_CCR', 'DT_CCR','Time'])
    return df





negative_folder= './medical_large/0'
positive_folder= './medical_large/1'
image_size = 50

X_all, y_all = prepare_data(negative_folder,positive_folder, image_size)
print(f"X_all.shape:{X_all.shape}")
print(f"y_all.shape:{y_all.shape}")


X, X_val, y, y_val = train_test_split(X_all, y_all, test_size=0.20, random_state=123)




initial_com = 10
end_com = 100
step = 10
neighbors_ls = np.arange(2,10,1)
best_neighbors_lle = tune_neighbors(X_val, initial_com, end_com, step, neighbors_ls, fit_lle)
best_neighbors_isomap = tune_neighbors(X_val, initial_com, end_com, step, neighbors_ls, fit_isomap)
print(best_neighbors_lle,best_neighbors_isomap)
neighbors_dict ={"LLE": best_neighbors_lle, "Isomap": best_neighbors_isomap}




dimensional_reduction = "PCA"
df_pca = fit_dimension_classifier(dimensional_reduction, X, initial_com, end_com, step,neighbors_dict)
df_pca.to_csv("PCA_result.csv")

dimensional_reduction = "KernelPCA"
df_kernel = fit_dimension_classifier(dimensional_reduction, X, initial_com, end_com, step,neighbors_dict)
df_kernel.to_csv("KernelPCA_result.csv")

dimensional_reduction = "LLE"
df_lle = fit_dimension_classifier(dimensional_reduction, X, initial_com, end_com, step,neighbors_dict)
df_lle.to_csv("LLE_result.csv")

dimensional_reduction = "Isomap"
df_isomap = fit_dimension_classifier(dimensional_reduction, X, initial_com, end_com, step,neighbors_dict)
df_isomap.to_csv("Isomap_result.csv")


dimensional_reduction = "MDS"
df_mds = fit_dimension_classifier(dimensional_reduction, X, initial_com, end_com, step,neighbors_dict)
df_mds.to_csv("MDS_result.csv")

dimensional_reduction = "NCA"
df_nca = fit_dimension_classifier(dimensional_reduction, X, initial_com, end_com, step,neighbors_dict)
df_nca.to_csv("NCA_result.csv")


# dimensional_reduction = "LDA"
# df_lda = fit_dimension_classifier(dimensional_reduction, X, initial_com, end_com, step)
# df_lda.to_csv("LDA_result.csv")

lda_test_CCR_lr,lda_test_CCR_dt = fit_lda(1, X)
print(f"Testing CCR for the LDA combined with LR is {lda_test_CCR_lr:.2f}")
print(f"Testing CCR for the LDA combined with DT is {lda_test_CCR_dt:.2f}")




print("Done!!!!!")






