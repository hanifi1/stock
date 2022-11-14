import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression


def create_features(df_stock, nlags=10):
    df_resampled = df_stock.copy()
    lags_col_names = []
    for i in range(nlags + 1):
        df_resampled['lags_' + str(i)] = df_resampled['close'].shift(i)
        lags_col_names.append('lags_' + str(i))
    df = df_resampled[lags_col_names]
    print(df)
    df = df.dropna(axis=0)

    return df


def create_X_Y(df_lags):
    X = df_lags.drop('lags_0', axis=1)
    Y = df_lags[['lags_0']]
    return X, Y


class Stock_model(BaseEstimator, TransformerMixin):

    def __init__(self, data_fetcher):
        self.log = logging.getLogger()
        self.lr = LinearRegression()
        self._data_fetcher = data_fetcher
        self.log.warning('here')

    def fit(self, X, Y=None):
        data = self._data_fetcher(X)
        df_features = create_features(data)
        df_features, Y = create_X_Y(df_features)
        self.lr.fit(df_features, Y)
        return self

    def predict(self, X, Y=None):
        data = self._data_fetcher(X, last=True)
        print(data)
        df_features = create_features(data)
        print(df_features)
        df_features, Y = create_X_Y(df_features)
        predictions = self.lr.predict(df_features)
#         print(predictions.flatten()[-1])
#         print(data.loc[data.index.max(), 'close'])
        Diff = data.loc[data.index.max(), 'close'] - predictions.flatten()[-1]
        print(Diff)
        if Diff > 0:
            return 'Sell'
        else:
            return 'Buy'


        # return predictions.flatten()[-1]






#
# import logging
#
# from statsmodels.tsa.arima.model import ARIMA
#
# def create_features(df_stock):
#     return df_stock['close'][-50:].diff().dropna()
#
#
# class Stock_model(BaseEstimator, TransformerMaixin):
#
#     def __int__(self, data_fetcher):
#         self.log = logging.getLogger()
#         self.model = None
#         self.data_fetcher = data_fetcher
#         self.log.warning('here')
#
#     def fit(self, X):
#         data - self.data_fetcher(X)
#         df_features = create_features(data)
#         self.model = ARIMA(df_features, order=(1,0,1))
#         self.model - self.model.fit()
#         return self
#
#     def predict(self, X, Y=None):
#         prediction = self.model.forecast(steps=1)
#         return prediction.reset_index(drop=True)[0]
