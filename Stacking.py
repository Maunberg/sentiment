from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from utils import get_full_pipline

param_grid = {
    'classifier__svc__solver': ['liblinear', 'lbfgs', 'saga', 'newton-cg'],
    'classifier__rf__n_estimators': [50, 100, 150],
    'classifier__final_estimator__solver': ['liblinear', 'lbfgs', 'saga', 'newton-cg']
}

model = StackingClassifier(
    estimators=[
        ('svc', LogisticRegression()),
        ('rf', RandomForestClassifier())
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

grid = get_full_pipline(model, param_grid, search='grid')
random = get_full_pipline(model, param_grid, search='random')
halv = get_full_pipline(model, param_grid, search='halving')
pipline = get_full_pipline(model, param_grid)
