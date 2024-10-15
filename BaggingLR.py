from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from utils import get_full_pipline

param_grid = {
    'classifier__estimator__C': [0.01, 0.1, 1, 10, 100],
    'classifier__estimator__solver': ['liblinear', 'lbfgs', 'saga', 'newton-cg'],
    'classifier__estimator__penalty': ['l1', 'l2', 'elasticnet'],
    'classifier__n_estimators': [10, 20, 30]
}

model = BaggingClassifier(estimator=LogisticRegression(), n_estimators=10, random_state=42)

grid = get_full_pipline(model, param_grid, search='grid')
random = get_full_pipline(model, param_grid, search='random')
halv = get_full_pipline(model, param_grid, search='halving')
pipline = get_full_pipline(model, param_grid)
