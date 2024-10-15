from catboost import CatBoostClassifier
from utils import get_full_pipline

param_grid = {
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__depth': [6, 8, 10],
}

model = CatBoostClassifier()

grid = get_full_pipline(model, param_grid, search='grid')
random = get_full_pipline(model, param_grid, search='random')
halv = get_full_pipline(model, param_grid, search='halving')
pipline = get_full_pipline(model, param_grid)
