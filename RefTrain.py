import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
# from tinkoff.invest import Client, CandleInterval

warnings.filterwarnings('ignore')
pio.renderers.default = "browser"


class DataPreprocessor:
    @staticmethod
    def preprocess_data(data, outlier_method='iqr'):
        if outlier_method == 'iqr':
            Q1 = data['close'].quantile(0.25)
            Q3 = data['close'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[(data['close'] >= lower_bound) & (data['close'] <= upper_bound)]
        else:
            q_low = data['close'].quantile(0.01)
            q_high = data['close'].quantile(0.99)
            data = data[(data['close'] >= q_low) & (data['close'] <= q_high)]
        return data


class VolumeIndicators:
    VOLUME_INDICATORS_ADDED = False

    @classmethod
    def add_volume_indicators(cls, data):
        try:
            price_change = np.where(data['close'] > data['close'].shift(1), 1,
                                    np.where(data['close'] < data['close'].shift(1), -1, 0))
            data['obv'] = (data['volume'] * price_change).cumsum()
            data['obv_ma'] = data['obv'].rolling(window=10).mean()
            data['obv_ratio'] = np.where(data['obv_ma'] != 0, data['obv'] / data['obv_ma'], 1.0)

            typical_price = (data['high'] + data['low'] + data['close']) / 3
            vwap_num = (typical_price * data['volume']).rolling(window=20).sum()
            vwap_den = data['volume'].rolling(window=20).sum()
            data['vwap'] = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price)
            data['price_vs_vwap'] = np.where(data['vwap'] != 0, (data['close'] - data['vwap']) / data['vwap'], 0)

            money_flow = typical_price * data['volume']
            positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
            negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)

            positive_mf = pd.Series(positive_flow).rolling(window=14).sum()
            negative_mf = pd.Series(negative_flow).rolling(window=14).sum()

            mfi_ratio = np.where(negative_mf != 0, positive_mf / negative_mf, 1.0)
            data['mfi'] = 100 - (100 / (1 + mfi_ratio))

            price_change_pct = data['close'].pct_change().fillna(0)
            data['vpt'] = (price_change_pct * data['volume']).cumsum()
            data['vpt_ma'] = data['vpt'].rolling(window=10).mean()

            high_low_diff = data['high'] - data['low']
            clv = np.where(high_low_diff != 0,
                           ((data['close'] - data['low']) - (data['high'] - data['close'])) / high_low_diff,
                           0)
            data['ad_line'] = (clv * data['volume']).cumsum()
            data['ad_oscillator'] = data['ad_line'].rolling(window=10).mean()

            if not cls.VOLUME_INDICATORS_ADDED:
                cls.VOLUME_INDICATORS_ADDED = True

        except Exception as e:
            data['obv_ratio'] = 1.0
            data['price_vs_vwap'] = 0.0
            data['mfi'] = 50.0
            data['vpt_ma'] = 0.0
            data['ad_oscillator'] = 0.0

        return data


class TechnicalIndicators:
    @staticmethod
    def add_technical_indicators(data):
        data['sma_10'] = data['close'].rolling(window=10).mean()
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()

        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))

        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']

        data['bb_middle'] = data['close'].rolling(window=20).mean()
        data['bb_std'] = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
        data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])

        data['volume_sma_10'] = data['volume'].rolling(window=10).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma_10']

        data['h_l'] = data['high'] - data['low']
        data['h_pc'] = abs(data['high'] - data['close'].shift(1))
        data['l_pc'] = abs(data['low'] - data['close'].shift(1))
        data['tr'] = data[['h_l', 'h_pc', 'l_pc']].max(axis=1)
        data['atr'] = data['tr'].rolling(window=14).mean()

        data['momentum_5'] = data['close'] / data['close'].shift(5) - 1
        data['momentum_10'] = data['close'] / data['close'].shift(10) - 1
        data['roc_10'] = ((data['close'] - data['close'].shift(10)) / data['close'].shift(10)) * 100

        high_14 = data['high'].rolling(window=14).max()
        low_14 = data['low'].rolling(window=14).min()
        data['williams_r'] = -100 * (high_14 - data['close']) / (high_14 - low_14)

        data['stoch_k'] = 100 * (data['close'] - low_14) / (high_14 - low_14)
        data['stoch_d'] = data['stoch_k'].rolling(window=3).mean()

        data['volatility'] = data['close'].rolling(window=20).std()
        data['high_20'] = data['high'].rolling(window=20).max()
        data['low_20'] = data['low'].rolling(window=20).min()
        data['price_position'] = (data['close'] - data['low_20']) / (data['high_20'] - data['low_20'])

        data = VolumeIndicators.add_volume_indicators(data)
        data.dropna(inplace=True)
        return data


class FeatureSelector:
    @staticmethod
    def improved_feature_selection(data, n_features=20, method='hybrid'):

        base_features = ['sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi', 'macd',
                         'macd_signal', 'macd_histogram', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
                         'volume', 'volume_sma_10', 'volume_ratio', 'atr', 'momentum_5', 'momentum_10',
                         'roc_10', 'williams_r', 'stoch_k', 'stoch_d', 'volatility', 'price_position']

        volume_features = ['obv_ratio', 'price_vs_vwap', 'mfi', 'vpt_ma', 'ad_oscillator']

        available_features = [f for f in base_features + volume_features if f in data.columns]

        data_clean = data[available_features + ['close']].dropna()

        if len(data_clean) == 0:
            fallback_features = ['sma_10', 'sma_20', 'rsi', 'macd', 'bb_position', 'volume_ratio', 'atr', 'momentum_5']
            available_fallback = [f for f in fallback_features if f in data.columns]
            return available_fallback[:min(n_features, len(available_fallback))]

        X = data_clean[available_features]
        y = data_clean['close']

        if method == 'hybrid':
            rf_importance = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_importance.fit(X, y)

            feature_importance = pd.DataFrame({
                'feature': available_features,
                'importance': rf_importance.feature_importances_
            }).sort_values('importance', ascending=False)

            top_features = feature_importance.head(min(25, len(available_features)))['feature'].tolist()
            X_filtered = X[top_features]

            estimator = LinearRegression()
            rfe = RFE(estimator, n_features_to_select=n_features)
            rfe.fit(X_filtered, y)

            selected_features = [top_features[i] for i in range(len(top_features)) if rfe.support_[i]]

        elif method == 'correlation':
            corr_matrix = X.corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            high_corr_pairs = [(corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                               for i in range(len(corr_matrix.index))
                               for j in range(len(corr_matrix.columns))
                               if i < j and corr_matrix.iloc[i, j] > 0.95]


            estimator = LinearRegression()
            rfe = RFE(estimator, n_features_to_select=n_features)
            rfe.fit(X, y)
            selected_features = [available_features[i] for i in range(len(available_features)) if rfe.support_[i]]


        price_features = [f for f in selected_features if
                          any(x in f for x in ['sma', 'ema', 'bb', 'close', 'momentum', 'roc'])]
        volume_features_selected = [f for f in selected_features if
                                    any(x in f for x in ['volume', 'obv', 'vwap', 'mfi', 'vpt', 'ad'])]
        oscillator_features = [f for f in selected_features if any(x in f for x in ['rsi', 'williams', 'stoch', 'macd'])]
        other_features = [f for f in selected_features if
                          f not in price_features + volume_features_selected + oscillator_features]


        return selected_features


class DataSetBuilder:
    @staticmethod
    def create_dataset(data, selected_features, scaler_features=None, scaler_target=None, window_size=30, fit_scalers=True):
        feature_data = data[selected_features]

        if fit_scalers or scaler_features is None:
            scaler_features = StandardScaler()
            scaled_features = scaler_features.fit_transform(feature_data)
        else:
            scaled_features = scaler_features.transform(feature_data)

        scaled_features = pd.DataFrame(scaled_features, columns=selected_features, index=feature_data.index)

        if fit_scalers or scaler_target is None:
            scaler_target = StandardScaler()
            scaled_target = scaler_target.fit_transform(data[['close']])
        else:
            scaled_target = scaler_target.transform(data[['close']])

        scaled_target = pd.Series(scaled_target.flatten(), index=data.index, name='scaled_close')

        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(scaled_features.iloc[i - window_size:i].values)
            y.append(scaled_target.iloc[i])
        X, y = np.array(X), np.array(y)

        X_rf = scaled_features.values[window_size:]
        y_rf = scaled_target.values[window_size:]

        return X, y, X_rf, y_rf, scaler_features, scaler_target


class ModelsFactory:
    @staticmethod
    def create_lstm_attention_model(input_shape):
        inputs = Input(shape=input_shape)
        lstm_out = LSTM(100, return_sequences=True, kernel_regularizer=l2(0.01))(inputs)
        lstm_out = Dropout(0.3)(lstm_out)
        attention = Attention()([lstm_out, lstm_out])
        attention_out = Concatenate()([lstm_out, attention])
        lstm_out_2 = LSTM(50, return_sequences=False, kernel_regularizer=l2(0.01))(attention_out)
        lstm_out_2 = Dropout(0.3)(lstm_out_2)
        outputs = Dense(1)(lstm_out_2)

        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    @staticmethod
    def create_rf_model(n_features):
        return RandomForestRegressor(n_estimators=100, random_state=42)


class EnsemblePredictor:
    def __init__(self, model_lstm, model_rf, selected_features, scaler_features, scaler_target, window_size):
        self.model_lstm = model_lstm
        self.model_rf = model_rf
        self.selected_features = selected_features
        self.scaler_features = scaler_features
        self.scaler_target = scaler_target
        self.window_size = window_size

    @staticmethod
    def adaptive_ensemble_weights(volatility):
        normalized_vol = min(1.0, volatility / 0.05)
        rf_weight = 0.2 + normalized_vol * 0.4
        lstm_weight = 1 - rf_weight
        return lstm_weight, rf_weight

    def predict_single_ensemble(self, data, n_days):
        last_window = data[self.selected_features].iloc[-self.window_size:].copy()
        scaled_last_window = self.scaler_features.transform(last_window)
        future_preds = []
        future_data = data.copy()

        recent_trend = np.mean(np.diff(data['close'].iloc[-10:]))

        current_volatility = data['volatility'].iloc[-20:].mean()
        lstm_weight, rf_weight = self.adaptive_ensemble_weights(current_volatility)

        for i in range(n_days):
            input_window = scaled_last_window.reshape((1, self.window_size, len(self.selected_features)))
            next_pred_scaled_lstm = self.model_lstm.predict(input_window, verbose=0)[0][0]
            next_pred_lstm = self.scaler_target.inverse_transform([[next_pred_scaled_lstm]])[0][0]

            last_features = scaled_last_window[-1].reshape(1, -1)
            next_pred_scaled_rf = self.model_rf.predict(last_features)[0]
            next_pred_rf = self.scaler_target.inverse_transform([[next_pred_scaled_rf]])[0][0]

            trend_correction = recent_trend * (1 - i * 0.1) * 0.3
            next_pred_lstm += trend_correction
            next_pred_rf += trend_correction

            next_pred = lstm_weight * next_pred_lstm + rf_weight * next_pred_rf
            future_preds.append(next_pred)

            last_date = future_data.index[-1]
            next_date = last_date + BDay(1)
            new_row = future_data.iloc[-1].copy()
            new_row['close'] = next_pred

            future_data = pd.concat([future_data, pd.DataFrame([new_row], index=[next_date])])
            future_data = TechnicalIndicators.add_technical_indicators(future_data)

            last_window = future_data[self.selected_features].iloc[-self.window_size:].copy()
            scaled_last_window = self.scaler_features.transform(last_window)

        return future_preds

    def predict_with_uncertainty(self, data, n_days, n_bootstrap=30):
        all_predictions = []
        for _ in range(n_bootstrap):
            bootstrap_data = data.copy()
            noise_std = bootstrap_data['atr'].iloc[-60:].mean() * 0.02
            noise = np.random.normal(0, noise_std, 60)
            bootstrap_data.loc[bootstrap_data.index[-60:], 'close'] += noise
            bootstrap_data = TechnicalIndicators.add_technical_indicators(bootstrap_data)

            pred = self.predict_single_ensemble(bootstrap_data, n_days)
            all_predictions.append(pred)

        all_predictions = np.array(all_predictions)
        mean_pred = np.mean(all_predictions, axis=0)
        lower_ci = np.percentile(all_predictions, 10, axis=0)
        upper_ci = np.percentile(all_predictions, 90, axis=0)

        return mean_pred, lower_ci, upper_ci

    def predict_ensemble(self, data, n_days):

        mean_pred, lower_ci, upper_ci = self.predict_with_uncertainty(data, n_days)

        current_volatility = data['volatility'].iloc[-20:].mean()
        lstm_weight, rf_weight = self.adaptive_ensemble_weights(current_volatility)


        return mean_pred, lower_ci, upper_ci


class StockPredictorApp:

    def run(self,data):
        forecast_days = 5
        data.rename(columns={'Open': 'open', 'High': 'high', 'Close': 'close', 'Volume': 'volume', 'Low': 'low'}, inplace=True)
        data['date'] = pd.to_datetime(data['date'], errors='coerce')

        data.dropna(subset=['date'], inplace=True)

        data.set_index('date', inplace=True)




        if data is None or len(data) < 200:
            print("Недостаточно данных для анализа.")
            return

        data = DataPreprocessor.preprocess_data(data, outlier_method='iqr')
        data = TechnicalIndicators.add_technical_indicators(data)

        selected_features = FeatureSelector.improved_feature_selection(data, n_features=15)

        scaler_features = StandardScaler()
        scaler_target = StandardScaler()
        scaler_features.fit(data[selected_features])
        scaler_target.fit(data[['close']])

        train_size = int(0.85 * len(data))
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]

        X_train, y_train, X_rf_train, y_rf_train, _, _ = DataSetBuilder.create_dataset(
            train_data, selected_features, scaler_features, scaler_target, window_size=30, fit_scalers=True)
        X_test, y_test, X_rf_test, y_rf_test, _, _ = DataSetBuilder.create_dataset(
            test_data, selected_features, scaler_features, scaler_target, window_size=30, fit_scalers=False)

        model_lstm = ModelsFactory.create_lstm_attention_model(input_shape=(30, X_train.shape[2]))
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        model_lstm.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=32,
                       callbacks=[early_stopping, reduce_lr], verbose=0)

        model_rf = ModelsFactory.create_rf_model(len(selected_features))
        model_rf.fit(X_rf_train, y_rf_train)

        y_pred_scaled_lstm = model_lstm.predict(X_test, verbose=0).flatten()
        y_pred_scaled_rf = model_rf.predict(X_rf_test)
        y_pred_lstm = scaler_target.inverse_transform(y_pred_scaled_lstm.reshape(-1, 1)).flatten()
        y_pred_rf = scaler_target.inverse_transform(y_pred_scaled_rf.reshape(-1, 1)).flatten()
        y_test_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()

        test_volatility = test_data['volatility'].mean()
        lstm_w, rf_w = EnsemblePredictor.adaptive_ensemble_weights(test_volatility)
        y_pred_ensemble = lstm_w * y_pred_lstm + rf_w * y_pred_rf

        mae = mean_absolute_error(y_test_actual, y_pred_ensemble)
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_ensemble))
        mape = np.mean(np.abs((y_test_actual - y_pred_ensemble) / y_test_actual)) * 100
        r2 = r2_score(y_test_actual, y_pred_ensemble)


        ensemble_predictor = EnsemblePredictor(model_lstm, model_rf, selected_features,
                                               scaler_features, scaler_target, window_size=30)

        future_preds, lower_ci, upper_ci = ensemble_predictor.predict_ensemble(data, forecast_days)

        last_date = data.index[-1]
        future_dates = [last_date + BDay(i + 1) for i in range(forecast_days)]


        base_columns = ['open', 'high', 'low', 'close', 'volume']

        full_dataframe = data[base_columns].copy()


        full_dataframe['forecast'] = np.nan
        full_dataframe['lower_ci'] = np.nan
        full_dataframe['upper_ci'] = np.nan
        full_dataframe['is_forecast'] = False

        plot_start_date = last_date - timedelta(days=5 * 365)
        test_dates = test_data.index[30:]
        test_mask = test_dates >= plot_start_date
        plot_test_dates = test_dates[test_mask]
        plot_y_pred_ensemble = y_pred_ensemble[test_mask]

        for date, pred in zip(plot_test_dates, y_pred_ensemble):
            if date in full_dataframe.index:
                full_dataframe.loc[date,'forecast']=pred
                full_dataframe.loc[date, 'is_forecast'] = True

        for i, date in enumerate(future_dates):
            last_row = data.iloc[-1].copy()

            new_row = pd.Series({
                'open': last_row['close'],
                'high': max(last_row['close'], future_preds[i]),
                'low': min(last_row['close'], future_preds[i]),
                'close': future_preds[i],
                'volume': last_row['volume'],
                'forecast': future_preds[i],
                'lower_ci': lower_ci[i],
                'upper_ci': upper_ci[i],
                'is_forecast': True
            }, name=date)

            full_dataframe = pd.concat([full_dataframe, new_row.to_frame().T])


        display_df = pd.concat([
            full_dataframe.iloc[-forecast_days - len(plot_test_dates) - 10:-forecast_days - len(plot_test_dates)],
            full_dataframe.iloc[-forecast_days - len(plot_test_dates):]
        ])

        display_df_formatted = display_df.copy()
        for col in ['open', 'high', 'low', 'close', 'forecast', 'lower_ci', 'upper_ci']:
            if col in display_df_formatted.columns:
                display_df_formatted[col] = display_df_formatted[col].round(2)

        return [display_df_formatted,mae,rmse,mape,r2]
