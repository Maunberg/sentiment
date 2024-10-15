import pandas as pd
import numpy as np
import argparse
import os
import importlib

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
import joblib

from utils import grid_report, set_params

parser = argparse.ArgumentParser(description='Run Search')
parser.add_argument('model_name', type=str, help='Model name from dir')
args = parser.parse_args()

print('Stage 0: Uploading configs')

name_model = args.model_name
n = 50_000

model = importlib.import_module(name_model)

print('Loaded', name_model)

if not os.path.exists(name_model+'_ex') or len(os.listdir(name_model+'_ex'))==0:
    print('There is no search parameters information, model will be generated randomly')
    name_model = name_model+'_basic'
    model = model.pipline
else:
    best_params = {}
    best_score = 0
    method = ''
    for i in ['grid', 'random', 'halv']:
        with open(name_model+'_ex/'+i+'.txt') as f:
            data = f.readlines()
        sub_param = eval(data[0].replace('Лучшие параметры: ', ''))
        sub_score = eval(data[2].replace('Результат на тестовых данных: ', ''))
        if best_score < sub_score:
            best_score = sub_score
            best_params = sub_param
            method = i
    print('Best method:', method)
    print('Best score:', best_score)

    model = set_params(model.pipline, best_params)
    print('Best params loaded')

print('Stage 1: Preparing data')

df = pd.read_csv('FULL.csv').drop(['Unnamed: 0', 'id', 'name', 'rep'], axis=1).dropna()
df = pd.concat([df[df['typr']==1][:n], df[df['typr']==0][:n]])
df = df.sample(frac=1)

#X_train, X_test, y_train, y_test = train_test_split(df['text'], df['typr'], test_size=0.3, random_state=42)
X = list(df['text'])
y = list(df['typr'])
print('Data samples: ', len(X))

print('Stage 2: Training')

model.fit(X, y)

print('Model trained')

if not os.path.exists('models'):
    os.mkdir('models')

joblib.dump(model, 'models/'+name_model+'.pkl')

print('Model is saved: ', 'models/'+name_model+'.pkl')

print('Testing model pkl...')
test = joblib.load('models/'+name_model+'.pkl')
test_sen = 'Как же неприятно, когда люди не делают то, что нужно. Бесит!'
pred = test.predict([test_sen])
print('Test sen:', test_sen)
print('Predicted class:', pred)
