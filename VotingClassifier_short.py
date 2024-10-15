from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from utils import get_full_pipline

param_grid = {
    # Гиперпараметры для LogisticRegression
    'classifier__lr__penalty': ['l1', 'l2', 'elasticnet'],  # Тип регуляризации
    'classifier__lr__solver': ['liblinear', 'lbfgs', 'saga'],  # Алгоритмы для оптимизации

    # Гиперпараметры для MultinomialNB
    'classifier__mnb__alpha': [0.5, 1.0, 1.5],  # Параметр сглаживания
#    'classifier__mnb__fit_prior': [True, False],  # Учитывать ли априорные вероятности классов
}

model = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('mnb', MultinomialNB()),
    ],
    voting='hard'
)

grid = get_full_pipline(model, param_grid, search='grid')
random = get_full_pipline(model, param_grid, search='random')
halv = get_full_pipline(model, param_grid, search='halving')
pipline = get_full_pipline(model, param_grid)

