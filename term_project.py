# TODO: How_to_run.txt
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


def loadData(tra_file, tst_file):
    Xtra = np.genfromtxt(tra_file, delimiter=',')
    Xtst = np.genfromtxt(tst_file, delimiter=',')
    # delete first rows
    Xtra = np.delete(Xtra, 0, 0)
    Xtst = np.delete(Xtst, 0, 0)
    Ytra = Xtra[:, -1]
    # delete class row
    Xtra = np.delete(Xtra, -1, 1)
    return Xtra, Xtst, Ytra


def preprocessing(Xtra, Xtst):
    scaler = StandardScaler()
    # fit only training data
    scaler.fit(Xtra)
    # scale both
    Xtra_scaled = scaler.transform(Xtra)
    Xtst_scaled = scaler.transform(Xtst)
    # create pca using training data
    pca = PCA(.95)
    pca.fit(Xtra_scaled)
    # pca.n_components_
    # apply pca to both training and test data
    Xtra_reduced = pca.transform(Xtra_scaled)
    Xtst_reduced = pca.transform(Xtst_scaled)

    return Xtra_reduced, Xtst_reduced


def trainModel(Xtra_r, Ytra):
    # train model using logistic regression
    model = LogisticRegression(solver='lbfgs')
    model.fit(Xtra_r, Ytra)
    return model


def predict(model, Xtst_r):
    prediction = model.predict(Xtst_r)
    return prediction


def writeOutput(prediction):
    with open('submission.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(["ID", "Predicted"])
        id = 1
        for row in prediction:
            filewriter.writerow([id, row.astype(int)])
            id += 1


Xtra, Xtst, Ytra = loadData('train.csv', 'test.csv')
Xtra_reduced, Xtst_reduced = preprocessing(Xtra, Xtst)
model = trainModel(Xtra_reduced, Ytra)
prediction = predict(model, Xtst_reduced)
writeOutput(prediction)
