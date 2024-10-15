from sklearn.svm import SVC
from utils import get_full_pipline
from sklearn.ensemble import BaggingClassifier

param_grid = {
    'classifier__estimator__C': [0.1, 1, 10, 100],
    'classifier__estimator__gamma': ['scale', 'auto'],
    'classifier__estimator__kernel': ['linear', 'rbf'], 
    'classifier__n_estimators': [10, 20, 30]
}

model = BaggingClassifier(estimator=SVC(), n_estimators=10, random_state=42)

grid = get_full_pipline(model, param_grid, search='grid')
random = get_full_pipline(model, param_grid, search='random')
halv = get_full_pipline(model, param_grid, search='halving')
pipline = get_full_pipline(model, param_grid)
