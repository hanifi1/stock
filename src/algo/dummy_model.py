import logging

from statsmodels.tsa.arima.model import ARIMA

def create_features(df_stock):
    return df_stock['close'][-50:].diff().dropna()


class Stock_model(BaseEstimator, TransformerMaixin):

    def __int__(self, data_fetcher):
        self.log = logging.getLogger()
        self.model = None
        self.data_fetcher = data_fetcher
        self.log.warning('here')

    def fit(self, X):
        data - self.data_fetcher(X)
        df_features = create_features(data)
        self.model = ARIMA(df_features, order=(1,0,1))
        self.model - self.model.fit()
        return self

    def predict(self, X, Y=None):
        prediction = self.model.forecast(steps=1)
        return prediction.reset_index(drop=True)[0]
