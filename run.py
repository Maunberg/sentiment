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

from utils import grid_report

parser = argparse.ArgumentParser(description='Run Search')
parser.add_argument('model_name', type=str, help='Model name from dir')
args = parser.parse_args()

print('Stage 0: Uploading configs')

name_model = args.model_name

model = importlib.import_module(name_model)

print('Loaded', name_model)

if not os.path.exists(name_model+'_ex'):
  os.mkdir(name_model+'_ex')

print('Ex dir is ready')

grids_report = name_model+'_ex/grid'
random_report = name_model+'_ex/random'
halv_report = name_model+'_ex/halv'
n = 10_000

print('Stage 1: Preparing data')

df = pd.read_csv('FULL.csv').drop(['Unnamed: 0', 'id', 'name', 'rep'], axis=1).dropna()
df = pd. concat([df[df['typr']==1][:n], df[df['typr']==0][:n]])
df = df.sample(frac=1)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['typr'], test_size=0.3, random_state=42)

print('Stage 2.1: Training with GridSearch')
model.grid.fit(X_train, y_train)

print('Training is finished. Creating report')
grid_report(model.grid, X_test, y_test, grids_report)
print('Report is ready')

print('Stage 2.2: Training with RandomSearch')
model.random.fit(X_train, y_train)

print('Training is finished. Creating report')
grid_report(model.random, X_test, y_test, random_report)
print('Report is ready')

print('Stage 2.3: Training with HalvingSearch')
model.halv.fit(X_train, y_train)

print('Training is finished. Creating report')
grid_report(model.halv, X_test, y_test, halv_report)
print('Report is ready')

print('Pipline is over successfully')
