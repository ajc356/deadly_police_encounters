import pandas as pd
import numpy as np

import statistics

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold

from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, accuracy_score, fbeta_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix, auc
from sklearn.metrics import fbeta_score

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import plot_roc_curve

from imblearn.over_sampling import ADASYN



def baseline_pipeline(X, y, model_name, model, over_under_sampler):
    """ For a given dataset (X,y), classification model, and over- or undersampler,
    fits the model to over/ underasampled data, and returns classification reports."""
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=4, stratify=y)
    
    X_over_under_sampled_train, y_over_under_sampled_train = over_under_sampler.fit_sample(
            X_train, y_train)

    model.fit(X_over_under_sampled_train, y_over_under_sampled_train)
    y_pred = model.predict(X_val)
    
    print(model_name, "using", over_under_sampler,":")
    print("Training set: ", model.score(X_train, y_train))
    print("Validation set: ", model.score(X_val, y_val))
    print(classification_report(y_val, y_pred))    


def average_metrics_pipeline(X, y, model_name, model, over_under_sampler, beta_value=10):
    """ For a given dataset (X,y), classification model, and over- or undersampler,
    fits the model to over/ underasampled data, and returns the weighted average 
    values for accuracy, precision, recall and fbeta classfication scoring metrics 
    across 5 cross-validated folds."""

    # setting up k-folds and dictionaries to hold results
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)
    cv_accuracy = []
    cv_precision = []
    cv_recall = []
    cv_fbeta = []

    # Render X and Y as np.arrays in order to generate indices for use with k-fold splitting
    # in order to manually keep track of which indices are being split into training/validation 
    # sets so I can over/ undersample only to the training set, which is then used to fit model
    X = np.array(X)
    y = np.array(y)
    
    for train_ind, val_ind in kf.split(X, y):

        X_train, y_train = X[train_ind], y[train_ind]
        X_over_under_sampled_train, y_over_under_sampled_train = over_under_sampler.fit_sample(
            X_train, y_train)
        X_val, y_val = X[val_ind], y[val_ind]
        
        std_scale = StandardScaler()
        X_over_under_sampled_train = std_scale.fit_transform(X_over_under_sampled_train)
        X_val = std_scale.transform(X_val)
        
        model.fit(X_over_under_sampled_train, np.array(
            y_over_under_sampled_train).ravel())
        y_pred = model.predict(X_val)
            
        # find average scoring metrics for the minority/positive class 
#         from sklearn.metrics import precision_recall_fscore_support
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        fbeta = fbeta_score(y_val, y_pred, beta=beta_value)

        cv_accuracy.append(accuracy)
        cv_precision.append(precision)
        cv_recall.append(recall)
        cv_fbeta.append(fbeta)

    # calculate means and standard deviations for each metric across all 5 kfolds 
    cv_recall = "Recall: {:.3f} +/- {:.3f}".format(statistics.mean(
        cv_recall), statistics.variance(cv_recall))
    cv_precision = "Precision: {:.3f} +/- {:.3f}".format(statistics.mean(
        cv_precision), statistics.variance(cv_precision))
    cv_fbeta = "F-beta Score: {:.3f} +/- {:.3f}".format(
        statistics.mean(cv_fbeta), statistics.variance(cv_fbeta))
    cv_accuracy = "Accuracy: {:.3f} +/- {:.3f}".format(statistics.mean(
        cv_accuracy), statistics.variance(cv_accuracy))    

    print(model_name, ":")
    print("Model score on training set: {:.3f}".format(model.score(X_over_under_sampled_train, y_over_under_sampled_train)))
    print("Model score on validation set: {:.3f}".format(model.score(X_val, y_val)))
    print(cv_recall)
    print(cv_precision)
    print(cv_fbeta)
    print(cv_accuracy)


def classification_reports_pipeline(X, y, model_name, model, oversampler):
    """ For a given dataset and classification model, fits the model, predicts classes of
    validation set, and returns the classification report showing precision, recall, f1 and 
    support for each class, as well as the average accuracy score. """

    X = np.array(X)
    y = np.array(y)

    # train-val split and oversample
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=.2, random_state=0)
    X_oversampled_train, y_oversampled_train = oversampler.fit_sample(
        X_train, y_train)

    # fit models

    model.fit(X_oversampled_train, np.array(y_oversampled_train).ravel())
    y_pred = model.predict(X_val)
    
    precision_recall_fscore = precision_recall_fscore_support(y_val, y_pred)
    
    majority_precision = "Majority Class Precision: {:.3f}".format(precision_recall_fscore[0][0])
    minority_precision = "Minority Class Precision: {:.3f}".format(precision_recall_fscore[0][1])

    majority_recall = "Majority Class Recall: {:.3f}".format(precision_recall_fscore[1][0])
    minority_recall = "Minority Class Recall: {:.3f}".format(precision_recall_fscore[1][1])
    
    majority_fbeta = "Majority Class F-beta: {:.3f}".format(precision_recall_fscore[2][0])
    minority_fbeta = "Minority Class F-beta: {:.3f}".format(precision_recall_fscore[2][1])

    # print classification reports
    print(model_name, ":")
    print(minority_precision)
    print(minority_recall)
    print(minority_fbeta)
    print(majority_precision)
    print(majority_recall)
    print(majority_fbeta)


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


