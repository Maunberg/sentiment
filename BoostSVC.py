from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from utils import get_full_pipline

param_grid = {
    'classifier__estimator__C': [0.1, 1, 10, 100],
    'classifier__estimator__gamma': ['scale', 'auto'],
    'classifier__estimator__kernel': ['linear', 'rbf'],
    'classifier__n_estimators': [50, 100, 150],
    'classifier__learning_rate': [0.01, 0.1, 1, 10]
}

est = SVC()
model = AdaBoostClassifier(estimator=est, random_state=0)

grid = get_full_pipline(model, param_grid, search='grid')
random = get_full_pipline(model, param_grid, search='random')
halv = get_full_pipline(model, param_grid, search='halving')
pipline = get_full_pipline(model, param_grid)
