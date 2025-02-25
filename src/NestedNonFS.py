import pandas as pd
import numpy as np
from composition_stats import closure, clr, ilr

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


datasets = ["CRC1", "CRC2", "CD1", "CD2", "PAR1", "PAR2", "PAR3", "HIV", "OB", "CDI", "CIR", "IBD1", "IBD2", "MHE", "ART"]
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'Boosting': XGBClassifier(random_state=42),
    'KNN': Pipeline([('norm', MinMaxScaler()), ('classifier', KNeighborsClassifier())]),
    'LogisticRegression': LogisticRegression(random_state=42), 
    'SVM': svm.SVC(random_state=42, probability=True),
}

NONFS_GRID = {
    'RandomForest': {
        'n_estimators': [200, 300, 400],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 3, 5, 7, 8]
    },
    'KNN': {
        'classifier__n_neighbors': [7, 9, 11, 13, 15, 17, 19, 21],
        'classifier__weights': ['uniform', 'distance']
    },
    'SVM': {
        'C': [0.001, 0.1, 1, 10, 100, 1000],
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto']
    },
    'LogisticRegression': {
        'C': np.logspace(-4, 4, 20),
    },
    'Boosting': {'max_depth': [3, 5, 7, 8], 'n_estimators': [300, 500, 800]}

}

def run_nested_cv_experiment_avg(datasets, normalization="TSS"):
    results = pd.DataFrame(columns=['Dataset', 'Model', 'Mean AUC', 'Std AUC', 'Mean Accuracy', 'Std Accuracy'])
    
    for dataset in datasets:
        df = pd.read_csv("../data/" + dataset + ".csv")
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
            
            for train_idx, test_idx in outer_cv.split(df, y):
                X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                grid = GridSearchCV(estimator=model, param_grid=NONFS_GRID[model_name], cv=inner_cv, 
                                    scoring='roc_auc', refit=True, n_jobs=-1, verbose=1)
                grid.fit(X_train, y_train)
                
                best_model = grid.best_estimator_
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                auc_scores.append(roc_auc_score(y_test, y_pred_proba))
                accuracy_scores.append(best_model.score(X_test, y_test))
            
            mean_auc = np.mean(auc_scores)
            std_auc = np.std(auc_scores)
            mean_accuracy = np.mean(accuracy_scores)
            std_accuracy = np.std(accuracy_scores)
            
            results.loc[len(results)] = [
                dataset, 
                model_name, 
                mean_auc, 
                std_auc, 
                mean_accuracy, 
                std_accuracy
            ]
    
    return results

tss = run_nested_cv_experiment_avg(datasets, normalization="TSS")
clr = run_nested_cv_experiment_avg(datasets, normalization="CLR")
tss.to_csv("NestedTSSResults.csv")
clr.to_csv("NestedCLRResults.csv")