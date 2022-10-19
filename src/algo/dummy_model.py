import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

import datetime
from datetime import datetime
import datetime
from yahoo_fin import stock_info as si
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler


def create_features(df_stock):
    df_resampled = df_stock.copy()
    df = df_resampled[['close']]
    df['Diff'] = df['close'].diff()
    df = df.dropna(axis=0)
    return df


def create_X_Y(df):
    scaler = MinMaxScaler()
    df['Diff_scaler'] = scaler.fit_transform()
    Y = df_lags[['lags_0']]
    return X, Y


class Stock_model(BaseEstimator, TransformerMixin):

    def __init__(self, data_fetcher):
        self.log = logging.getLogger()
        self.svr = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
        self._data_fetcher = data_fetcher
        self.log.warning('here')

    def fit(self, X, Y):
        data = self._data_fetcher(X)
        df_features = create_features(data)
        df_features, Y = create_X_Y(df_features)
        self.lr.fit(df_features, Y)
        return self

    def predict(self, X, Y=None):
        print(X)
        data = self._data_fetcher(X, last=True)
        print(data)
        df_features = create_features(data)
        print(df_features)
        df_features, Y = create_X_Y(df_features)
        predictions = self.lr.predict(df_features)

        return predictions.flatten()[-1]
