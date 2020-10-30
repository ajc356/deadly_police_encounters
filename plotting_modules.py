import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import statistics

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB

from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, accuracy_score, fbeta_score, log_loss, make_scorer
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix, auc
from sklearn.metrics import fbeta_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_recall_fscore_support

def plot_roc_curves(X, y):
    plt.figure(figsize=(10,6))
    lw = 2
    
    # train-val split and oversample
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=.2, random_state=0)
    adasyn = ADASYN(random_state=44)
    X_oversampled_train, y_oversampled_train = adasyn.fit_sample(
        X_train, y_train)
    
    # Logistic Regression
    # fit model and predict probabilities of validation data 
    log_reg = LogisticRegression(max_iter=5000, n_jobs=-1, random_state=44)
    log_reg.fit(X_oversampled_train, y_oversampled_train)
    y_pred = log_reg.predict_proba(X_val)

    fpr, tpr, thresholds = roc_curve(y_val, y_pred[:,1])
    model_auc = roc_auc_score(y_val, y_pred[:,1])
    plt.plot(fpr, tpr, color='b',
             lw=lw, label=f'Logistic Regression, AUC: {model_auc:.4f}')

    
    # Naive Bayes 
    # fit model and predict probabilities of validation data 
    nb = BernoulliNB()
    nb.fit(X_oversampled_train, y_oversampled_train)
    y_pred = nb.predict_proba(X_val)
     
    fpr, tpr, thresholds = roc_curve(y_val, y_pred[:,1])
    model_auc = roc_auc_score(y_val, y_pred[:,1])
    plt.plot(fpr, tpr, color='r',
             lw=lw, label=f'Bernoulli Naive Bayes, AUC: {model_auc:.4f}')
       
    # SVC 
    # fit model and predict probabilities of validation data
    svc = SVC(probability=True, random_state=1)
    svc.fit(X_oversampled_train, y_oversampled_train)
    y_pred = svc.predict_proba(X_val)
     
    fpr, tpr, thresholds = roc_curve(y_val, y_pred[:,1])
    model_auc = roc_auc_score(y_val, y_pred[:,1])
    plt.plot(fpr, tpr, color='g',
             lw=lw, label=f'SVC, AUC: {model_auc:.4f}')
    
    plt.plot([0,1],[0,1],c='violet',ls='--', label='Chance Line')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curves for top 3 Contending Models')
    plt.legend(loc='lower right', prop={'size': 10}, frameon=True);
    plt.savefig('ROC Curves for top 3 Contending Models');