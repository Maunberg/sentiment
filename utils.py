import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split

def tf_idf_pipeline(model):
    pipeline = Pipeline(steps=[
        ('preprocessor', TfidfVectorizer()),
        ('classifier', model)
    ])
    return pipeline

def tf_idf_transformed_pipeline(model, transformer):
    pipeline = Pipeline(steps=[
        ('preprocessor', TfidfVectorizer()),
        ('data_transformer', transformer),
        ('classifier', model)
    ])
    return pipeline

def get_full_pipline(model, param_grid, search='',transformer=False, n_iter=10, scoring='f1'):
    if isinstance(transformer, bool):
        pipeline = tf_idf_pipeline(model)
    else:
        pipeline = tf_idf_transformed_pipeline(model, transformer)
    if search=='grid':
        return GridSearchCV(pipeline, param_grid, cv=5, scoring=scoring)
    elif search=='random':
        return RandomizedSearchCV(pipeline, param_grid, cv=5, scoring=scoring)
    elif search=='halving':
        return HalvingGridSearchCV(pipeline, param_grid, cv=5, scoring=scoring)
    else:
        return pipeline

def grid_report(grid, X_test, y_test, way):
    text = f"Лучшие параметры: {grid.best_params_}"
    text += f"\nЛучший результат на кросс-валидации: {grid.best_score_}"
    text += f"\nРезультат на тестовых данных: {grid.score(X_test, y_test)}"
    res = (
        pd.DataFrame({
            "mean_test_score": grid.cv_results_["mean_test_score"],
            "mean_fit_time": grid.cv_results_["mean_fit_time"]})
          .join(pd.json_normalize(grid.cv_results_["params"]).add_prefix("param_"))
    ).dropna()
    res = res.sort_values('mean_test_score', ascending=False)
    with open(way+'.txt', 'w') as f:
        f.write(text)
    res.to_csv(way+'.csv')

def set_params(pipline, params):
    pipline.set_params(**params)
    return pipline
