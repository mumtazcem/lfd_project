# TODO: How_to_run.txt
import numpy as np
import csv
import warnings
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

seed = 1075
np.random.seed(seed)

# Classifiers
rf = RandomForestClassifier()
et = ExtraTreesClassifier()
knn = KNeighborsClassifier()
svc = SVC()
rg = RidgeClassifier()
lr = LogisticRegression(solver='lbfgs')
gnb = GaussianNB()
dt = DecisionTreeClassifier(max_depth=1)

# Bagging Classifiers
bagging_clf = BaggingClassifier(rf, max_samples=0.4, max_features=10, random_state=seed)

# Boosting Classifiers
ada_boost = AdaBoostClassifier()
ada_boost_svc = AdaBoostClassifier(base_estimator=svc, algorithm='SAMME')
grad_boost = GradientBoostingClassifier()
xgb_boost = xgb.XGBClassifier()

# Voting Classifiers
vclf = VotingClassifier(estimators=[('ada_boost', ada_boost), ('grad_boost', grad_boost),
                                    ('xgb_boost', xgb_boost), ('BaggingWithRF', bagging_clf)], voting='hard')

# Ensemble Classifier below scored 0.700 in kaggle
# however it scores 0.608 with cross validation
eclf = EnsembleVoteClassifier(clfs=[ada_boost_svc, grad_boost, xgb_boost], voting='hard')

# Grid Search
params = {'gradientboostingclassifier__n_estimators': [10, 200],
          'xgbclassifier__n_estimators': [10, 200]}
grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)


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


def writeOutput(prediction, filename):
    with open(filename, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(["ID", "Predicted"])
        id = 1
        for row in prediction:
            filewriter.writerow([id, row.astype(int)])
            id += 1


warnings.filterwarnings("ignore")
Xtra, Xtst, Ytra = loadData('train.csv', 'test.csv')
Xtra_reduced, Xtst_reduced = preprocessing(Xtra, Xtst)
model_lr = trainModel(Xtra_reduced, Ytra)
print("Classifiers cross-validation")

# Classifiers cross-validation
labels_clf = ['RandomForest', 'ExtraTrees', 'KNeighbors', 'SVC', 'Ridge', 'LinearRegression', 'GaussianNB',
              'DecisionTree']
for model, label in zip([rf, et, knn, svc, rg, model_lr, gnb, dt], labels_clf):
    scores = cross_val_score(model, Xtra_reduced, Ytra, cv=5, scoring='accuracy')
    model.fit(Xtra_reduced, Ytra)
    prediction = predict(model, Xtst_reduced)
    writeOutput(prediction, label + ".csv")
    print("Mean: {0:.3f}, std: {1:.3f} [{2} is used.]".format(scores.mean(), scores.std(), label))

print("-----------------------------------\n\n")
print("Bagging, Boosting and GridSearchCV cross-validation")

# Bagging, Boosting and GridSearchCV cross-validation
labels = ['Ada Boost', 'Ada BoostSVC', 'Grad Boost', 'XG Boost', 'Ensemble', 'Voting',
          'BaggingWithRF', 'Grid']
for model, label in zip([ada_boost, ada_boost_svc, grad_boost, xgb_boost, eclf, vclf, bagging_clf, grid],
                        labels):
    if label == 'Grid':
        print("Beware: Grid takes long time!")
    scores = cross_val_score(model, Xtra_reduced, Ytra, cv=5, scoring='accuracy')
    model.fit(Xtra_reduced, Ytra)
    prediction = predict(model, Xtst_reduced)
    writeOutput(prediction, label + ".csv")
    print("Mean: {0:.3f}, std: {1:.3f} [{2} is used.]".format(scores.mean(), scores.std(), label))