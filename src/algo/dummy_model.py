import logging

from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.linear_model import LinearRegression
from src.IO.get_data_from_yahoo import get_last_stock_price
from sklearn.svm import SVR

def create_features(df_stock, nlags=15):
    df_resampled = df_stock.copy()
    lags_col_names = []
    for i in range(nlags + 1):
        df_resampled['lags_' + str(i)] = df_resampled['Diff'].shift(i)
        lags_col_names.append('lags_' + str(i))
    df = df_resampled[lags_col_names]
    df = df.dropna(axis=0)
    return df

def create_X_Y(df_lags, last=False):
    X = df_lags.drop('lags_0', axis=1).iloc[:-1, :]
    Y = df_lags[['lags_0']].iloc[:-1, :]
    if last:
        X = df_lags.iloc[-1:, :-1]
        Y = None
    return X, Y

def Stock_model(df_features):
    X, y = create_X_Y(df_features)
    model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
    model.fit(X.to_numpy(), y.to_numpy())
    return model

def Predict(Ticker):
    df = get_last_stock_price(Ticker)
    print(df)
    df_features = create_features(df)
    model = Stock_model(df_features)
    x  = create_X_Y(df_features, last=True)[0]
    pred_y = model.predict(x)
    if pred_y > 0 :
        return  'Sell'
    else:
        return 'Buy'


















################################################################################################
# def create_features(df_stock, nlags=10):
#     df_resampled = df_stock.copy()
#     lags_col_names = []
#     for i in range(nlags + 1):
#         df_resampled['lags_' + str(i)] = df_resampled['Diff'].shift(i)
#         lags_col_names.append('lags_' + str(i))
#     df = df_resampled[lags_col_names]
#     df = df.dropna(axis=0)
#     return df
#
#
# def MinMax(df):
#     scaler = MinMaxScaler()
#     df = df.reset_index()
#     MinMax_array = scaler.fit_transform(df['Diff'].values.reshape(-1, 1))
#     DF = pd.DataFrame(MinMax_array, columns=['MinMax'])
#     df = pd.concat([df, DF], axis=1).set_index('index')
#     return df
#
#
# def create_X_Y(df_lags):
#     X = df_lags.drop('lags_0', axis=1)
#     Y = df_lags[['lags_0']]
#     return X, Y
#
#
# class Stock_model(BaseEstimator, TransformerMixin):
#
#     def __init__(self, data_fetcher):
#         self.log = logging.getLogger()
#         self.lr = LinearRegression()
#         self._data_fetcher = data_fetcher
#         self.log.warning('here')
#
#     def fit(self, X, Y=None):
#         data = self._data_fetcher(X)
#         df_features = create_features(data)
#         df_features, Y = create_X_Y(df_features)
#         self.lr.fit(df_features, Y)
#         return self
#
#     def predict(self, X, Y=None):
#         data = self._data_fetcher(X, last=True)
#         df_features = create_features(data)
#         df_features, Y = create_X_Y(df_features)
#         predictions = self.lr.predict(df_features)
#         # predictions = scalar.inverse_transform(predictions)
#         Diff = data.loc[data.index.max(), 'close'] - predictions.flatten()[-1]
#         if Diff > 0:
#             return 'Sell'
#         else:
#             return 'Buy'
########################################################################################################
# class Stock_model(BaseEstimator, TransformerMixin):
#
#     def __init__(self, data_fetcher):
#         self.log = logging.getLogger()
#         self.lr = LinearRegression()
#         self._data_fetcher = data_fetcher
#         self.log.warning('here')
#
#     def create_features(self, df_stock, nlags=10):
#         df_resampled = df_stock.copy()
#         df_resampled = self.MinMax(df_resampled)
#         print('YYY', df_resampled)
#         lags_col_names = []
#         for i in range(nlags + 1):
#             df_resampled['lags_' + str(i)] = df_resampled['Diff'].shift(i)
#             lags_col_names.append('lags_' + str(i))
#         df = df_resampled[lags_col_names]
#         df = df.dropna(axis=0)
#         return df
#     def create_X_Y(self, df_lags):
#         X = df_lags.drop('lags_0', axis=1)
#         Y = df_lags[['lags_0']]
#         return X, Y
#     def fit(self, X, Y=None):
#         data = self._data_fetcher(X)
#         df_features = self.create_features(data)
#         df_features, Y = self.create_X_Y(df_features)
#         self.lr.fit(df_features, Y)
#         return self
#
#     def predict(self, X, Y=None):
#         data = self._data_fetcher(X, last=True)
#         df_features = self.create_features(data)
#         df_features, Y = self.create_X_Y(df_features)
#         predictions = self.lr.predict(df_features)
#         print('yyy',predictions)
#         predictions = self.MMScaler.inverse_transform(predictions)
#         Diff = data.loc[data.index.max(), 'close'] - predictions.flatten()[-1]
#         # print(Diff)
#         if Diff > 0:
#             return 'Sell'
#         else:
#             return 'Buy'



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
