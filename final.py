import os
import torch
import pickle
import statistics
import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
import matplotlib.pyplot as plt

from GRAPE.training.gnn_mdi import train_gnn_mdi
from GRAPE.mc.mc_subparser import add_mc_subparser
from GRAPE.uci.uci_subparser import add_uci_subparser
from GRAPE.utils.utils import auto_select_gpu
from GRAPE.train_mdi import main


import warnings
warnings.filterwarnings("ignore")

# select device
if torch.cuda.is_available():
    cuda = auto_select_gpu()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
    print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    device = torch.device('cuda:{}'.format(cuda))
else:
    print('Using CPU')
    device = torch.device('cpu')


## ----------- task2 -----------
impute_rmse = []
impute_l1 = []
predict_rmse = []
predict_l1 = []

dataset = 'energy'
log_dir = 'task2'
for log_dir in ['task2_0.1', 'task2_0.2', 'task2_0.3', 'task2_0.4', 'task2', 'task2_0.6', 'task2_0.7', 'task2_0.8', 'task2_0.9']:
    data = pd.read_csv(r'.\GRAPE\uci\raw_data\{}\data\data.txt'.format(dataset), sep="\t", header=None)
    with open(r'.\GRAPE\uci\test\{}\{}\result.pkl'.format(dataset, log_dir), 'rb') as f:
        result = pickle.load(f)

    impute_rmse.append(result['curves']['impute_rmse'][-1])
    impute_l1.append(result['curves']['impute_l1'][-1])
    predict_rmse.append(result['curves']['predict_rmse'][-1])
    predict_l1.append(result['curves']['predict_l1'][-1])
impute_rmse, impute_l1, predict_rmse, predict_l1

plt.plot(np.arange(0.1, 1, 0.1), impute_rmse, label = "impute RMSE")
plt.plot(np.arange(0.1, 1, 0.1), impute_l1, label = "impute L1 loss")
plt.legend()
plt.xlabel('alpha (loss weight of imputation)')
plt.ylabel('loss')
plt.title('Imputation')
plt.show()

plt.plot(np.arange(0.1, 1, 0.1), predict_rmse, label = "Predict RMSE")
plt.plot(np.arange(0.1, 1, 0.1), predict_l1, label = "Predict L1 loss")
plt.legend()
plt.xlabel('alpha (loss weight of imputation)')
plt.ylabel('loss')
plt.title('Prediction')
plt.show()


## ----------- task3 -----------
# Few-shot learning
GRAPE_acc = []
RF_acc = []
XGB_acc = []

dataset = 'Breast'
log_dir = 'task3'
for log_dir in ['task3_0.1', 'task3_0.2', 'task3', 'task3_0.4', 'task3_0.5', 'task3_0.6', 'task3_0.7', 'task3_0.8', 'task3_0.9', 'task3_0.9']:
    data = pd.read_csv(r'.\GRAPE\uci\raw_data\{}\data\data.txt'.format(dataset), sep="\t", header=None)
    with open(r'.\GRAPE\uci\test\{}\{}\result.pkl'.format(dataset, log_dir), 'rb') as f:
        result = pickle.load(f)

    GRAPE_acc.append(max(result['curves']['predict_accuracy']))

    X = result['data']['data'].df_X
    y = result['data']['data'].df_y
    train = np.array(result['data']['train'].cpu())
    test = np.array(result['data']['test'].cpu())

    model = RandomForestClassifier()
    model.fit(X[train], y[train])
    pred = model.predict(X[test])
    RF_acc.append((pred == y[test]).sum() / y[test].shape[0])

    model = XGBClassifier()
    model.fit(X[train], y[train])
    pred = model.predict(X[test])
    XGB_acc.append((pred == y[test]).sum() / y[test].shape[0])
GRAPE_acc, RF_acc, XGB_acc 

plt.plot(range(10, 110, 10), GRAPE_acc, label = "GRAPE")
plt.plot(range(10, 110, 10), RF_acc, label = "Random Forest")
plt.plot(range(10, 110, 10), XGB_acc , label = "XGBoost")
plt.legend()
plt.xlabel('% labeled set')
plt.ylabel('Accuracy')
plt.show()


# Unsupervised
dataset = 'Breast'
log_dir = 'task3-2'
with open(r'.\GRAPE\uci\test\{}\{}\result.pkl'.format(dataset, log_dir), 'rb') as f:
    result = pickle.load(f)

result['kmeans']


## ----------- task4 -----------
# Original
dataset = 'Breast'
for dataset in ['Breast', 'glass', 'auto-mpg']:
    print('Dataset: {}'.format(dataset))
    data = pd.read_csv(r'.\GRAPE\uci\raw_data\{}\data\data.txt'.format(dataset), sep="\t", header=None)
    log_dir = 'original'
    with open(r'.\GRAPE\uci\test\{}\{}\result.pkl'.format(dataset, log_dir), 'rb') as f:
        result = pickle.load(f)
    label = np.concatenate([result['outputs']['label_train'], result['outputs']['label_test']])
    pred = np.concatenate([result['outputs']['final_pred_train'], result['outputs']['final_pred_test']])
    impute_mse = mean_squared_error(label, pred)
    impute_rmse = np.sqrt(impute_mse)

    null = data.iloc[:, -1].isnull()
    x, y = pred.reshape((data.shape[0], -1))[~null], data.iloc[:,-1][~null]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # model = LogisticRegression().fit(X_train, y_train)
    model = DecisionTreeClassifier(max_depth = 10).fit(X_train, y_train)
    model = DecisionTreeClassifier().fit(X_train, y_train)
    pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    acc = model.score(X_test, y_test)
    f1 = f1_score(y_test, pred, average = 'macro')
    if dataset == 'Breast':
        auc = roc_auc_score(y_test, pred)
    else:
        auc = roc_auc_score(y_test, y_pred_proba, average = 'macro', multi_class = 'ovo')
    print(classification_report(pred, y_test))
    print('[Original] MSE: {}, RMSE: {}, Accuracy: {}, F1-score: {}, AUC: {}'.format(impute_mse, impute_rmse, acc, f1, auc))


    # Contrastive learning
    log_dir = 'task4_2'
    with open(r'.\GRAPE\uci\test\{}\{}\result.pkl'.format(dataset, log_dir), 'rb') as f:
        result = pickle.load(f)

    label = np.concatenate([result['outputs']['label_train'], result['outputs']['label_test']])
    pred = np.concatenate([result['outputs']['final_pred_train'], result['outputs']['final_pred_test']])
    impute_mse = mean_squared_error(label, pred)
    impute_rmse = np.sqrt(impute_mse)

    null = data.iloc[:, -1].isnull()
    x, y = pred.reshape((data.shape[0], -1))[~null], data.iloc[:,-1][~null]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # model = LogisticRegression().fit(X_train, y_train)
    model = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)
    model = DecisionTreeClassifier().fit(X_train, y_train)
    pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    acc = model.score(X_test, y_test)
    f1 = f1_score(y_test, pred, average = 'macro')
    if dataset == 'Breast':
        auc = roc_auc_score(y_test, pred)
    else:
        auc = roc_auc_score(y_test, y_pred_proba, average = 'macro', multi_class = 'ovo')
    print(classification_report(pred, y_test))
    print('[Contrastive learning] MSE: {}, RMSE: {}, Accuracy: {}, F1-score: {}, AUC: {}'.format(impute_mse, impute_rmse, acc, f1, auc))


