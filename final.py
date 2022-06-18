import os
import torch
import pickle
import statistics
import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
import matplotlib.pyplot as plt

from GRAPE.utils.utils import auto_select_gpu

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


## ----------- task1 -----------
## For imputation model
predict_acc = []
predict_f1 = []

dataset = 'Breast'
log_dir = 'original'
for log_dir in ['task1_0.1', 'task1_0.2', 'task1_0.3', 'task1_0.4', 'task1_0.5', 'task1_0.6', 'task1_0.7', 'task1_0.8', 'task1_0.9']:
    data = pd.read_csv(r'.\GRAPE\uci\raw_data\{}\data\data.txt'.format(dataset), sep="\t", header=None)
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
    model = XGBClassifier().fit(X_train, y_train)
    pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    predict_acc.append(model.score(X_test, y_test))
    predict_f1.append(f1_score(y_test, pred, average = 'macro'))

predict_acc, predict_f1

plt.plot(np.arange(0.1, 1.0, 0.1), predict_acc, label = "Accuracy")
# plt.plot(range(1, 9), predict_f1, label = "F1-score")
# plt.legend()
plt.xlabel('Perturbed ratio')
plt.ylabel('Accuracy')
plt.title('Model 1')
plt.show()

## For prediction model
predict_acc = []

dataset = 'Breast'
log_dir = 'task1_y_0.1'
for log_dir in ['task2_y_1', 'task2_y_2', 'task2_y_3', 'task2_y_4', 'task2_y_5', 'task2_y_6', 'task2_y_7', 'task2_y_8']:
    data = pd.read_csv(r'.\GRAPE\uci\raw_data\{}\data\data.txt'.format(dataset), sep="\t", header=None)
    with open(r'.\GRAPE\uci\test\{}\{}\result.pkl'.format(dataset, log_dir), 'rb') as f:
        result = pickle.load(f)

    predict_acc.append(max(result['curves']['predict_accuracy']))

predict_acc

plt.plot(range(1, 9), predict_acc, label = "Accuracy")
# plt.legend()
plt.xlabel('# Removed features')
plt.ylabel('Accuracy')
plt.title('Model 2')
plt.show()


## ----------- task2 -----------
## For imputation model
# Method 1 - feature importance
predict_acc = []
predict_f1 = []

dataset = 'Breast'
log_dir = 'original'
for log_dir in ['task2_1', 'task2_2', 'task2_3', 'task2_4', 'task2_5', 'task2_6', 'task2_7', 'task2_8']:
    data = pd.read_csv(r'.\GRAPE\uci\raw_data\{}\data\data.txt'.format(dataset), sep="\t", header=None)
    with open(r'.\GRAPE\uci\test\{}\{}\result.pkl'.format(dataset, log_dir), 'rb') as f:
        result = pickle.load(f)
    label = np.concatenate([result['outputs']['label_train'], result['outputs']['label_test']])
    pred = np.concatenate([result['outputs']['final_pred_train'], result['outputs']['final_pred_test']])
    impute_mse = mean_squared_error(label, pred)
    impute_rmse = np.sqrt(impute_mse)

    null = data.iloc[:, -1].isnull()
    x, y = pred.reshape((data.shape[0], -1))[~null], data.iloc[:,-1][~null]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LogisticRegression().fit(X_train, y_train)
    # model = XGBClassifier().fit(X_train, y_train)
    pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    predict_acc.append(model.score(X_test, y_test))
    predict_f1.append(f1_score(y_test, pred, average = 'macro'))

predict_acc, predict_f1

plt.plot(range(1, 9), predict_acc, label = "Accuracy")
# plt.plot(range(1, 9), predict_f1, label = "F1-score")
# plt.legend()
plt.xlabel('# Removed features')
plt.ylabel('Accuracy')
plt.title('Model 1')
plt.show()

## For prediction model
predict_acc = []

dataset = 'Breast'
log_dir = 'original_y'
for log_dir in ['task2_y_1', 'task2_y_2', 'task2_y_3', 'task2_y_4', 'task2_y_5', 'task2_y_6', 'task2_y_7', 'task2_y_8']:
    data = pd.read_csv(r'.\GRAPE\uci\raw_data\{}\data\data.txt'.format(dataset), sep="\t", header=None)
    with open(r'.\GRAPE\uci\test\{}\{}\result.pkl'.format(dataset, log_dir), 'rb') as f:
        result = pickle.load(f)

    predict_acc.append(max(result['curves']['predict_accuracy']))

predict_acc

plt.plot(range(1, 9), predict_acc, label = "Accuracy")
# plt.legend()
plt.xlabel('# Removed features')
plt.ylabel('Accuracy')
plt.title('Model 2')
plt.show()

# Method 2 - paper...!!!


## ----------- task3 -----------
cluster_acc = []
dataset = 'Breast'
log_dir = 'original_y'
for log_dir in ['task3_0', 'task3_0.25', 'task3_0.5', 'task3_0.75', 'task3_0.99' ]:
    with open(r'.\GRAPE\uci\test\{}\{}\result.pkl'.format(dataset, log_dir), 'rb') as f:
        result = pickle.load(f)
    cluster_acc.append(statistics.mean(sorted(result['curves']['predict_accuracy'])[-10:]))

cluster_acc
plt.plot([0, 0.25, 0.5, 0.75, 0.99], cluster_acc, label = "Accuracy")
# plt.legend()
plt.xlabel('Weight of clustering loss')
plt.ylabel('Accuracy')
plt.title(dataset)
plt.xticks([0, 0.25, 0.5, 0.75, 0.99])
plt.show()

cluster_acc = []
dataset = 'QSAR Bio'
log_dir = 'task3'
for log_dir in ['task3_0', 'task3_0.25', 'task3_0.5', 'task3_0.75', 'task3_0.99' ]:
    with open(r'.\GRAPE\uci\test\{}\{}\result.pkl'.format(dataset, log_dir), 'rb') as f:
        result = pickle.load(f)
    cluster_acc.append(max(result['curves']['predict_accuracy']))

cluster_acc
plt.plot([0, 0.25, 0.5, 0.75, 0.99], cluster_acc, label = "Accuracy")
# plt.legend()
plt.xlabel('Weight of clustering loss')
plt.ylabel('Accuracy')
plt.title(dataset)
plt.xticks([0, 0.25, 0.5, 0.75, 0.99])
plt.show()

## ----------- task4 -----------