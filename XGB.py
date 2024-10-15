from xgboost import XGBClassifier
from utils import get_full_pipline

param_grid = {
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.1, 0.01, 0.001],
#    'classifier__n_estimators': [100, 300, 500],
#    'classifier__subsample': [0.8, 1.0],
#    'classifier__colsample_bytree': [0.8, 1.0],
#    'classifier__gamma': [0, 0.2, 0.4]
}

model = XGBClassifier()

grid = get_full_pipline(model, param_grid, search='grid')
random = get_full_pipline(model, param_grid, search='random')
halv = get_full_pipline(model, param_grid, search='halving')
pipline = get_full_pipline(model, param_grid)

