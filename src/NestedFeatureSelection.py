import pandas as pd
import numpy as np
from composition_stats import closure, clr

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.metrics import roc_auc_score

from mrmr_feature_selector import MRMRFeatureSelector
from sklearn.linear_model import Lasso
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline

import time


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
    'Boosting': {'classifier__max_depth': [3, 5, 7, 8], 'classifier__n_estimators': [300, 500, 800]}
    
}

def FeatureSelector(n):
    if n == "Lasso":
        return SelectFromModel(Lasso(alpha=0.01, random_state=42))
    if n == "Lasso50":
        return SelectFromModel(Lasso(alpha=0.01, random_state=42), max_features=50)
    if n == "Lasso100":
        return SelectFromModel(Lasso(alpha=0.01, random_state=42), max_features=100)
    if n == "mRMR50":
        return MRMRFeatureSelector(num_features=50)
    if n == "mRMR100":
        return MRMRFeatureSelector(num_features=100)
    if n == "MIFS50":
        return SelectKBest(mutual_info_classif, k=50)
    if n == "MIFS100":
        return SelectKBest(mutual_info_classif, k=100)
    if n == "ReliefF50":
        return ReliefF(n_features_to_select=50, n_neighbors=1.0, n_jobs=-1)   
    if n == "ReliefF100":
        return ReliefF(n_features_to_select=50, n_neighbors=1.0, n_jobs=-1)

def run_nested_cv_experiment_avg(datasets, normalization="TSS", method="Lasso"):
    results = pd.DataFrame(columns=['Dataset', 'Model', 'AUC', 'AUC Std. Dev.', 'Accuracy', 'Std Accuracy', 'Time', 'Mean Selected'])
    
    for dataset in datasets:
        df = pd.read_csv("data/" + dataset + ".csv")
        y = df["label"]
        df.drop("label", axis=1, inplace=True)
        
        if normalization == "TSS":
            df = pd.DataFrame(closure(df))
        if normalization == "CLR":
            df = pd.DataFrame(clr(closure(df + 0.5)))
            
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.squeeze())
        
        outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
        
        for model_name, model in models.items():
            auc_scores = []
            accuracy_scores = []
            times = []
            selected_features_counts = []
            
            for train_idx, test_idx in outer_cv.split(df, y):
                X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if method == "ReliefF50" or method == "ReliefF100":
                    X_train = X_train.to_numpy()
                    X_test = X_test.to_numpy()

                selector = FeatureSelector(method)
                pipe = Pipeline([
                    ('feature_selection', selector),
                    ('classifier', model)
                ], memory="caching"+method, verbose=True)
                start_time = time.time()
                inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                grid = GridSearchCV(estimator=pipe, param_grid=FS_GRID[model_name], cv=inner_cv, 
                                    scoring='roc_auc', refit=True, n_jobs=-1, verbose=1)
                grid.fit(X_train, y_train)
                execution_time = time.time() - start_time

                best_model = grid.best_estimator_
                selected_features_counts.append(np.sum(best_model.named_steps['feature_selection'].get_support()))
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                auc_scores.append(roc_auc_score(y_test, y_pred_proba))
                accuracy_scores.append(best_model.score(X_test, y_test))
                times.append(execution_time)
            
            mean_auc = np.mean(auc_scores)
            std_auc = np.std(auc_scores)
            mean_accuracy = np.mean(accuracy_scores)
            std_accuracy = np.std(accuracy_scores)
            mean_time = np.mean(times)
            mean_selected = np.mean(selected_features_counts)
            
            results.loc[len(results)] = [
                dataset, 
                model_name, 
                mean_auc, 
                std_auc, 
                mean_accuracy, 
                std_accuracy, mean_time, mean_selected
            ]
    
    return results

meths = ["Lasso", "MIFS50", "MIFS100", "mRMR50", "mRMR100", "ReliefF50", "ReliefF100"]

for M in meths:
    tss = run_nested_cv_experiment_avg(datasets, normalization="TSS", method=M)
    tss.to_csv("NestedTSS"+M+"Results.csv")
    clr = run_nested_cv_experiment_avg(datasets, normalization="CLR", method=M)
    clr.to_csv("NestedCLR"+M+"Results.csv")