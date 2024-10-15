import gradio as gr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import importlib
import json
import os
import re
from norm import norm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score

model_list = [i.replace('.pkl', '') for i in os.listdir('models')]

def present(text_was, name_model, join):
    model = joblib.load('models/'+name_model+'.pkl')
    text = [i for i in re.split('(?<=[.!?])', text_was) if len(i) > 0]
    text_was = []
    sub = []
    for i in text:
        if len(sub) >= join:
            sub.append(i)
            text_was.append(' '.join(sub))
            sub = []
        else:
            sub.append(i)
    text_was.append(' '.join(sub))
    text = [norm(i) for i in text_was]
    pred = model.predict(text)
    text = ''
    if os.path.exists(name_model+'_ex/') and len(os.listdir(name_model+'_ex/')) > 0:
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
        text += f'Search method: {method}\n'
        text += f'Test result: {round(best_score, 4)}\n'
        text += f'Model params: {best_params}\n'
    if text == '':
        text += 'No information.'
    res = []
    for emo, sent in zip(pred, text_was):
        if emo == 0:
            res.append((sent, 'NEG'))
        else:
            res.append((sent, 'POS'))
    return res, text

def predcit(name_model):
    model = joblib.load('models/'+name_model+'.pkl')
    df = pd.read_csv('FULL.csv').drop(['Unnamed: 0', 'id', 'name', 'rep'], axis=1).dropna()
    df = pd.concat([df[df['typr']==1][-5_000:], df[df['typr']==0][-5_000:]])
    X = df['text']
    y = df['typr']
    pred = model.predict(X)
    if not os.path.exists('predict'):
        os.mkdir('predict')
    pd.DataFrame({'X':X, 'y':y, 'pred':pred}).to_csv('predict/'+name_model+'.csv')

def get_metrics(name_model):
    df = pd.read_csv('predict/'+name_model+'.csv')
    y_true = df['y']
    y_pred = df['pred']
    metrics = {}
    metrics['ROC-AUC'] = roc_auc_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred)
    metrics['Recall'] = recall_score(y_true, y_pred)
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['F1'] = f1_score(y_true, y_pred)
    with open('predict/'+name_model+'.json', 'w') as f:
        json.dump(metrics, f, ensure_ascii=False)

def results(model_list, metrics_list):
    res = {}
    res['Model'] = []
    for name_model in model_list:
        if not os.path.exists('predict/'+name_model+'.csv'):
            predcit(name_model)
        if not os.path.exists('predict/'+name_model+'.json'):
            get_metrics(name_model)
        with open('predict/'+name_model+'.json') as f:
            sub_metr = json.load(f)
        res['Model'].append(name_model)
        for i in metrics_list:
            if i in res:
                res[i].append(sub_metr[i])
            else:
                res[i] = []
                res[i].append(sub_metr[i])
    df = pd.DataFrame(res)

    df_melted = df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Score', y='Model', hue='Metric', data=df_melted, orient='h')
    plt.title('Model Metrics Comparison')
    plt.xlabel('Score')
    plt.ylabel('Model')
    plt.legend(title='Metric')

    return df.set_index('Model').round(4).reset_index(), plt.gcf()

page_1 = gr.Interface(
    fn = present,
    inputs = [
        gr.Textbox(placeholder='Write text here', label='Write text'),
        gr.Radio(model_list, value='BaggingLR', label='Choose model'),
        gr.Slider(1, 10, value=1, label='Join sentences', step=1)
    ],
    outputs = [
        gr.HighlightedText(label='Sentiment analysis',
                           combine_adjacent=True,
                           show_legend=True,
                           color_map={'NEG':'red', 'POS':'green'}),
        gr.Textbox(label='Additional information')
    ],
    title='Sentiment detection',
    description='Write a text to get sentiment-analysis')

page_2 = gr.Interface(
    fn = results,
    inputs = [
    gr.CheckboxGroup(model_list, value='BaggingLR', label='Choose models'),
    gr.CheckboxGroup(['F1', 'Recall', 'Precision', 'ROC-AUC', 'Accuracy'],
                     value=['F1'], label='Choose metrics'),
    ],
    outputs = [
    gr.DataFrame(),
    gr.Plot(),
    ],
    title='Sentiment detection',
    description='Check model quality',
    show_progress = True
)

show = gr.TabbedInterface([page_1, page_2], ['Model demo', 'Model quality'])

show.launch(share=True)
