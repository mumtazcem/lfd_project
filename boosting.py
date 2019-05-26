from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from term_project import *

seed = 1075
np.random.seed(seed)

rf = RandomForestClassifier()
et = ExtraTreesClassifier()
knn = KNeighborsClassifier()
svc = SVC()
rg = RidgeClassifier()
lr = LogisticRegression(solver='lbfgs')
gnb = GaussianNB()
dt = DecisionTreeClassifier(max_depth=1)

ada_boost = AdaBoostClassifier()
ada_boost_svc = AdaBoostClassifier(base_estimator=svc, algorithm='SAMME')
grad_boost = GradientBoostingClassifier()
xgb_boost = xgb.XGBClassifier()
bagging_clf = BaggingClassifier(rf, max_samples=0.4, max_features=10, random_state=seed)

params = {'gradientboostingclassifier__n_estimators': [10, 200],
          'xgbclassifier__n_estimators': [10,200]}

boost_array = [ada_boost, grad_boost, xgb_boost]
vclf = VotingClassifier(estimators=[('RandomForests', ada_boost), ('SVC', grad_boost),
                                    ('LogisticRegression', xgb_boost),('BaggingWithRTC', bagging_clf)], voting='hard')
eclf = EnsembleVoteClassifier(clfs=[ada_boost, grad_boost, xgb_boost, bagging_clf], voting='hard')
grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
labels = ['Ada Boost', 'Ada BoostSVC', 'Grad Boost', 'XG Boost', 'Ensemble', 'Voting', 'BaggingWithRTC', 'Grid']
for clf, label in zip([ada_boost, ada_boost_svc, grad_boost, xgb_boost, eclf, vclf, bagging_clf, grid], labels):
    scores = cross_val_score(clf, Xtra_reduced, Ytra, cv=5, scoring='accuracy')
    clf.fit(Xtra_reduced, Ytra)
    prediction = predict(clf, Xtst_reduced)
    writeOutput(prediction, label + ".csv")
    print("Mean: {0:.3f}, std: (+/-) {1:.3f} [{2}]".format(scores.mean(), scores.std(), label))

