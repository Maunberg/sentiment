from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from utils import get_full_pipline

param_grid = {
    'classifier__lr1__solver': ['liblinear', 'lbfgs', 'saga'],
#    'classifier__lr2__n_estimators': ['liblinear', 'lbfgs', 'saga'],
#    'classifier__final_estimator__solver': ['liblinear', 'lbfgs', 'saga']
}

model = StackingClassifier(
    estimators=[
        ('lr1', LogisticRegression()),
        ('lr2', RandomForestClassifier())
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

grid = get_full_pipline(model, param_grid, search='grid')
random = get_full_pipline(model, param_grid, search='random')
halv = get_full_pipline(model, param_grid, search='halving')
pipline = get_full_pipline(model, param_grid)

