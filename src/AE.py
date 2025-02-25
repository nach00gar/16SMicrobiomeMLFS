import pandas as pd
import numpy as np
from composition_stats import closure, clr

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.metrics import roc_auc_score

from DM import *
import tensorflow as tf
import random

datasets = ["CRC1", "CRC2", "CD1", "CD2", "PAR1", "PAR2", "PAR3", "HIV", "OB", "CDI", "CIR", "IBD1", "IBD2", "MHE", "ART"]
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'Boosting': XGBClassifier(random_state=42),
    'KNN': Pipeline([('norm', MinMaxScaler()), ('classifier', KNeighborsClassifier())]),
    'LogisticRegression': LogisticRegression(random_state=42), 
    'SVM': svm.SVC(random_state=42, probability=True),
}

FS_GRID = {
    'RandomForest': {
        'classifier__n_estimators': [200, 300, 400],
        'classifier__max_features': ['sqrt', 'log2'],
        'classifier__max_depth': [None, 3, 5, 7, 8]
    },
    'KNN': {
        'classifier__classifier__n_neighbors': [7, 9, 11, 13, 15, 17, 19, 21],
        'classifier__classifier__weights': ['uniform', 'distance']
    },
    'SVM': {
        'classifier__C': [0.001, 0.1, 1, 10, 100, 1000],
        'classifier__kernel': ['rbf'],
        'classifier__gamma': ['scale', 'auto']
    },
    'LogisticRegression': {
        'classifier__C': np.logspace(-4, 4, 20),
    },
    'Boosting': {'classifier__max_depth': [3, 5, 7, 8], 'classifier__n_estimators': [300, 400, 500]}
    
}


def set_seeds(seed):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)  # For TensorFlow 1.x



def run_experiment_ae(datasets, dim=50, norm = "TSS"):
    results = pd.DataFrame(columns=['Dataset', 'Model', 'AUC', 'AUC Std. Dev.', 'Accuracy', 'Time'])
    
    for dataset in datasets:
        y = pd.read_csv("data/"+dataset+"y.csv", header=None)
        
        set_seeds(42)
        if norm =="TSS":
            dm = DeepMicrobiome(dataset +"tssx.csv", seed=42, data_dir="")
        if norm =="CLR":
            dm = DeepMicrobiome(dataset +"x.csv", seed=42, data_dir="")

        dm.loadCustomData()
        start_time = time.time()
        dm.ae(dims=[dim])
        X = pd.DataFrame(dm.X_train)
            
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.squeeze())
        outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
        
        for model_name, model in models.items():
            auc_scores = []
            accuracy_scores = []
            times = []
            for train_idx, test_idx in outer_cv.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
            
                pipe = Pipeline([
                    ('classifier', model)
                ])
                inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                grid = GridSearchCV(estimator=pipe, param_grid=FS_GRID[model_name], 
                                cv=inner_cv, scoring='roc_auc', refit=True, 
                                n_jobs=16, verbose=1, pre_dispatch=16)
                grid.fit(X_train, y_train)
                execution_time = time.time() - start_time
                best_model = grid.best_estimator_
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                auc_scores.append(roc_auc_score(y_test, y_pred_proba))
                accuracy_scores.append(best_model.score(X_test, y_test))
                times.append(execution_time)
            

            mean_auc = np.mean(auc_scores)
            std_auc = np.std(auc_scores)
            mean_accuracy = np.mean(accuracy_scores)
            std_accuracy = np.std(accuracy_scores)
            mean_time = np.mean(times)
            
    
            

            results.loc[len(results)] = [
                dataset, 
                model_name, 
                mean_auc, 
                std_auc, 
                mean_accuracy, 
                mean_time
            ]
    
    return results

def run_experiment_vae(datasets, dim=50, norm = "TSS"):
    results = pd.DataFrame(columns=['Dataset', 'Model', 'AUC', 'AUC Std. Dev.', 'Accuracy', 'Time'])
    
    for dataset in datasets:
        y = pd.read_csv("data/"+dataset+"y.csv", header=None)
        
        set_seeds(42)
        if norm =="TSS":
            dm = DeepMicrobiome(dataset +"tssx.csv", seed=42, data_dir="")
        if norm =="CLR":
            dm = DeepMicrobiome(dataset +"x.csv", seed=42, data_dir="")
        dm.loadCustomData()
        start_time = time.time()
        dm.vae(dims=[dim])
        X = pd.DataFrame(dm.X_train)
            
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.squeeze())

        outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
        
        for model_name, model in models.items():
            auc_scores = []
            accuracy_scores = []
            times = []
            for train_idx, test_idx in outer_cv.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
            
                pipe = Pipeline([
                    ('classifier', model)
                ])
                inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                grid = GridSearchCV(estimator=pipe, param_grid=FS_GRID[model_name], 
                                cv=inner_cv, scoring='roc_auc', refit=True, 
                                n_jobs=16, verbose=1, pre_dispatch=16)
                grid.fit(X_train, y_train)
                execution_time = time.time() - start_time
                best_model = grid.best_estimator_

                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                auc_scores.append(roc_auc_score(y_test, y_pred_proba))
                accuracy_scores.append(best_model.score(X_test, y_test))
                times.append(execution_time)

            mean_auc = np.mean(auc_scores)
            std_auc = np.std(auc_scores)
            mean_accuracy = np.mean(accuracy_scores)
            std_accuracy = np.std(accuracy_scores)
            mean_time = np.mean(times)
            
    
            

            results.loc[len(results)] = [
                dataset, 
                model_name, 
                mean_auc, 
                std_auc, 
                mean_accuracy, 
                mean_time
            ]
    
    return results

ae50tss = run_experiment_ae(datasets, dim=50, norm="TSS")
ae50tss.to_csv("NestedTSSAE50.csv")
ae50clr = run_experiment_ae(datasets, dim=50, norm="CLR")
ae50clr.to_csv("NestedCLRAE50.csv")
ae100tss = run_experiment_ae(datasets, dim=100, norm="TSS")
ae100tss.to_csv("NestedTSSAE100.csv")
ae100clr = run_experiment_ae(datasets, dim=100, norm="CLR")
ae100clr.to_csv("NestedCLRAE100.csv")
vaetss = run_experiment_vae(datasets, norm="TSS")
vaetss.to_csv("NestedTSSVAE.csv")
vaeclr = run_experiment_vae(datasets, norm="CLR")
vaeclr.to_csv("NestedCLRVAE.csv")

