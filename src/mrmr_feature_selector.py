from mrmr import mrmr_classif
from sklearn.base import BaseEstimator, TransformerMixin

class MRMRFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, num_features):
        self.num_features = num_features
        self.selected_features = None

    def fit(self, X, y):
        # Selección de características con mrmr
        self.selected_features = mrmr_classif(X, y, K=self.num_features)
        return self

    def transform(self, X):
        # Retornar solo las características seleccionadas
        return X[self.selected_features]