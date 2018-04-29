from PIL import Image
import numpy as np
import pandas as pd
import time
import scipy.io as sio
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# Dataset dimensions
N = 6283
D = 400


# Load feature dataset and convert to grayscale
X = []
for i in range(1, N+1):
    img = Image.open("./trainResized/{}.Bmp".format(str(i)))
    img = img.convert("LA")
    imgData = list(img.getdata())
    img = []
    for value in imgData:
        img.append(value[0]/value[1])
    X.append(img)
X = pd.DataFrame(X)


# Load label dataset
yReal = pd.read_csv("trainLabels.csv", usecols=[1])
yClass = yReal.iloc[0:N]
y = yClass.copy()
for i in range(0, N):
    y.iloc[i] = ord(str(yClass.iloc[i].values)[2])


# Divide training/test sets
Ntrain = 5000


def split_dataset(data, trainSize):
    np.random.seed(0)
    testSize = int(len(data) - trainSize)
    shuffled_ind = np.random.permutation(len(data))
    train_ind = shuffled_ind[testSize:]
    test_ind = shuffled_ind[:testSize]
    return data.iloc[train_ind].reset_index(drop=True), data.iloc[test_ind].reset_index(drop=True)


Xtrain, Xtest = split_dataset(X, Ntrain)
ytrain, ytest = split_dataset(y, Ntrain)
ytrain = ytrain.values.ravel()
ytest = ytest.values.ravel()


# Dimensionality reduction
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(Xtrain)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
print("PCA results: min dimensions = ", d, ", dimensions used = ", 100)
pca = PCA(n_components=100)
Xtrain_red = pca.fit_transform(Xtrain)
Xtest_red = pca.transform(Xtest)


# Support vector machine classifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

print("\n***** SUPPORT VECTOR MACHINE *****")
rbf_svm_best_mean = 0.
rbf_svm_best_param = [0., 0.]
cv_mean_acc = np.zeros((3, 10))
i = -1
for g in np.logspace(-3, -1, 3):
    i = i + 1
    j = -1
    for c in np.linspace(3.5, 5, 10):
        j = j + 1
        rbf_svm_clf = Pipeline((
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=g, C=c))
        ))
        time_start = time.clock()
        cv_accuracies = cross_val_score(rbf_svm_clf, Xtrain_red, ytrain, cv=5, scoring="accuracy")
        time_elapsed = time.clock() - time_start
        cv_mean_acc[i, j] = np.mean(cv_accuracies)
        print("CV mean = ", cv_mean_acc[i, j], ", CV training time = ", time_elapsed)
        if cv_mean_acc[i, j] > rbf_svm_best_mean:
            rbf_svm_best_mean = cv_mean_acc[i, j]
            rbf_svm_best_param = [g, c]
sio.savemat('rbf_svm_cv_mean_acc.mat', {'cv_mean_acc': cv_mean_acc})
print("CV results: gamma = ", rbf_svm_best_param[0], ", C = ", rbf_svm_best_param[1])
rbf_svm_best = Pipeline((
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=rbf_svm_best_param[0], C=rbf_svm_best_param[1], probability=True))
        ))
time_start = time.clock()
rbf_svm_best.fit(Xtrain_red, ytrain)
time_elapsed = time.clock() - time_start
ytrain_pred = rbf_svm_best.predict(Xtrain_red)
train_accuracy = accuracy_score(ytrain, ytrain_pred)
print("Training accuracy = ", train_accuracy, ", training time = ", time_elapsed)
ytest_pred = rbf_svm_best.predict(Xtest_red)
test_accuracy = accuracy_score(ytest, ytest_pred)
print("Test accuracy = ", test_accuracy)


# k-nearest neighbors classifier
from sklearn.neighbors import KNeighborsClassifier

print("\n***** K-NEAREST NEIGHBORS *****")
knn_best_mean = 0.
knn_best_param = ['uniform', 1]
cv_mean_acc = np.zeros((2, 10))
i = -1
for weigh in ['uniform', 'distance']:
    i = i + 1
    j = -1
    for neigh in range(1, 11):
        j = j + 1
        knn_clf = KNeighborsClassifier(weights=weigh, n_neighbors=neigh)
        time_start = time.clock()
        cv_accuracies = cross_val_score(knn_clf, Xtrain_red, ytrain, cv=5, scoring="accuracy")
        time_elapsed = time.clock() - time_start
        cv_mean_acc[i, j] = np.mean(cv_accuracies)
        print("CV mean accuracy = ", cv_mean_acc[i, j], ", CV training time = ", time_elapsed)
        if cv_mean_acc[i, j] > knn_best_mean:
            knn_best_mean = cv_mean_acc[i, j]
            knn_best_param = [weigh, neigh]
sio.savemat('knn_cv_mean_acc.mat', {'cv_mean_acc': cv_mean_acc})
print("CV results: weights = ", knn_best_param[0], ", n_neighbors = ", knn_best_param[1])
knn_best = KNeighborsClassifier(weights=knn_best_param[0], n_neighbors=knn_best_param[1])
time_start = time.clock()
knn_best.fit(Xtrain_red, ytrain)
time_elapsed = time.clock() - time_start
ytrain_pred = knn_best.predict(Xtrain_red)
train_accuracy = accuracy_score(ytrain, ytrain_pred)
print("Training accuracy = ", train_accuracy, ", training time = ", time_elapsed)
ytest_pred = knn_best.predict(Xtest_red)
test_accuracy = accuracy_score(ytest, ytest_pred)
print("Test accuracy = ", test_accuracy)


# Random forest classifier
from sklearn.ensemble import RandomForestClassifier

print("\n***** RANDOM FOREST *****")
rnd_best_mean = 0.
rnd_best_param = 0
cv_mean_acc = np.zeros(10)
i = -1
for nest in 5*np.logspace(2, 3, 10).astype(int):
    i = i + 1
    rnd_clf = RandomForestClassifier(n_estimators=nest)
    time_start = time.clock()
    cv_accuracies = cross_val_score(rnd_clf, Xtrain_red, ytrain, cv=5, scoring="accuracy")
    time_elapsed = time.clock() - time_start
    cv_mean_acc[i] = np.mean(cv_accuracies)
    print("CV mean = ", cv_mean_acc[i], ", CV training time = ", time_elapsed)
    if cv_mean_acc[i] > rnd_best_mean:
        rnd_best_mean = cv_mean_acc[i]
        rnd_best_param = nest
sio.savemat('rnd_cv_mean_acc.mat', {'cv_mean_acc': cv_mean_acc})
print("CV results: n_estimators = ", rnd_best_param)
rnd_best = RandomForestClassifier(n_estimators=rnd_best_param)
time_start = time.clock()
rnd_best.fit(Xtrain_red, ytrain)
time_elapsed = time.clock() - time_start
ytrain_pred = rnd_best.predict(Xtrain_red)
train_accuracy = accuracy_score(ytrain, ytrain_pred)
print("Training accuracy = ", train_accuracy, ", training time = ", time_elapsed)
ytest_pred = rnd_best.predict(Xtest_red)
test_accuracy = accuracy_score(ytest, ytest_pred)
print("Test accuracy = ", test_accuracy)


# Voting classifier
from sklearn.ensemble import VotingClassifier

print("\n***** SOFT VOTING *****")
voting_clf = VotingClassifier(
    estimators=[('svc', rbf_svm_best), ('knn', knn_best), ('rf', rnd_best)],
    voting='soft'
)
time_start = time.clock()
voting_clf.fit(Xtrain_red, ytrain)
time_elapsed = time.clock() - time_start
ytrain_pred = voting_clf.predict(Xtrain_red)
train_accuracy = accuracy_score(ytrain, ytrain_pred)
print("Training accuracy = ", train_accuracy, ", training time = ", time_elapsed)
ytest_pred = voting_clf.predict(Xtest_red)
test_accuracy = accuracy_score(ytest, ytest_pred)
print("Test accuracy = ", test_accuracy)
