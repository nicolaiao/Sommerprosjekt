# import packages
from math import floor
import os
import pandas as pd
import numpy as np

from datetime import datetime
from msal import PublicClientApplication

import holidays

from cognite.client import CogniteClient

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from keras.models import load_model

import matplotlib.pyplot as plt

import xgboost
from sklearn.model_selection import train_test_split

import catboost as cb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import GRU
import itertools

# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

class HandlerClass:
    """
    This class handles model- training/evaluation
    """

    def __init__(self, 
                curves:     list,
                columns:    int, 
                train_xgb:  bool, 
                train_cbr:  bool, 
                train_lstm: bool, 
                start:      str, 
                end:        str,):
        """
        Initialize modelclass
        """
        self.df = None
        self.validation_data = None
        self.curves = curves
        self.columns = columns
        self.train_config = {"xgb": train_xgb, "cbr": train_cbr, "lstm": train_lstm}
        self.start_year, self.start_month, self.start_day = start.split("-")
        self.end_year, self.end_month, self.end_day = end.split("-")

        #default values:
        self.look_back = 8 
        self.max_evals_xgb = 100
        self.fast_cb = False #Only for testing purposes, will not give a good estimation
        self.fast_lstm = False #Only for testing purposes, will not give a good estimation
        self.n_features = len(columns) - 1
        self.exclude_inputs = [None]
        self.predict_future = False
        self.xgb_pred = None
        self.cbr_pred = None 
        self.lstm_pred = None

        self.xgb_save_model_name = f"xgb_model.json"
        self.cbr_save_model_name = f"cb_model.cbm"
        self.lstm_save_model_name = f"lstm_model.h5"
        self.xgb_load_model_name = f"xgb_model.json"
        self.cbr_load_model_name = f"cb_model.cbm"
        self.lstm_load_model_name = f"lstm_model.h5"      

        self.allowed_keys = {
            "look_back",
            "max_evals_xgb",
            "fast_cb",
            "fast_lstm",
            "exclude_inputs",
            "predict_future",
            "xgb_save_model_name",
            "cbr_save_model_name",
            "lstm_save_model_name",
            "xgb_load_model_name",
            "cbr_load_model_name",
            "lstm_load_model_name"
        }


    def set_attributes(self, **kwargs):
        """
        Inputs defined in Main.py are updated with this function
        """
        allowed_excluded = ["cons_actual","temp_forecast", "price_forecast", "cc_forecast", "volue_forecast", "Day sin", "Day cos", "Week sin", "Week cos", "Year sin", "Year cos", "holiday?", "weekend?"]
        for excluded in self.exclude_inputs:
            if excluded not in allowed_excluded and excluded is not None:
                raise Exception(f"Input: {excluded} not allowed! Allowed: {allowed_excluded}")
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.allowed_keys and v != 'default')
        

    def get_data_cdf(self, start_year,start_month,start_day,end_year,end_month,end_day):
        """
        Retrieve data stored in Cognite-Data-Fusion
        """

        # Log-in detaljer
        TENANT_ID = os.getenv("AZURE_TENANT_ID")
        CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
        CDF_CLUSTER = "az-power-no-northeurope"
        COGNITE_PROJECT = "heco-dev"

        # Code to log-in WIHTOUT client_secret
        SCOPES = [f"https://{CDF_CLUSTER}.cognitedata.com/.default"]

        AUTHORITY_HOST_URI = "https://login.microsoftonline.com"
        AUTHORITY_URI = AUTHORITY_HOST_URI + "/" + TENANT_ID
        PORT = 53000


        def __authenticate_azure():
            """
            Helper function to authenticate Cognite-Data-Fusion
            """

            app = PublicClientApplication(client_id=CLIENT_ID, authority=AUTHORITY_URI)

            # interactive login - make sure you have http://localhost:port in Redirect URI in App Registration as type "Mobile and desktop applications"
            creds = app.acquire_token_interactive(scopes=SCOPES, port=PORT)
            return creds


        creds = __authenticate_azure()

        client = CogniteClient(
            token_url=creds["id_token_claims"]["iss"],
            token=creds["access_token"],
            token_client_id=creds["id_token_claims"]["aud"],
            project=COGNITE_PROJECT,
            base_url=f"https://{CDF_CLUSTER}.cognitedata.com",
            client_name="cognite-python-dev",
        )
        curves = self.curves 

        #Defines start and end date
        start_date = datetime(int(start_year),int(start_month),int(start_day))
        end_date = datetime(int(end_year),int(end_month),int(end_day))
        
        #Creates empty dataframe
        df_watt = pd.DataFrame()
        
        #Implements curves from cdf to dataframe        
        for curve in curves:
            print(curve)
            hm = client.datapoints.retrieve_dataframe(
                start=start_date,
                end=end_date,
                aggregates=["average"],
                granularity="1h",
                id=client.time_series.retrieve(external_id=curve).id,)
            df_watt = pd.merge(df_watt, hm, left_index=True, right_index=True, how="outer")
        df_watt.columns = self.columns

        #Implements boolean holiday and weekend to dataframe
        df_watt = self.__add_holidays(df_watt, start_date, end_date)
        self.df = df_watt
        return df_watt
        
        

    def __add_holidays(self, df, start_date, end_date):
        """
        Helper function to add holiday/weekend in DataFrame. A boolean value either 0 or 1
        """
        no_holidays = holidays.NO()
        periods = pd.date_range(start_date, end_date, freq="H")
        d = np.zeros(len(periods))
        e = np.zeros(len(periods))

        for l in range(len(periods)):
            a = str(int(periods[l].strftime('%Y%m%d')))
            da = int(a[-2:])
            mo = int(a[-4:-2])
            yr = int(a[-8:-4])
            if date(yr,mo,da) in no_holidays:
                d[l] = 1
            if date(yr,mo,da).weekday() > 4:
                e[l] = 1

        df_temp = pd.DataFrame(d,index=[i for i in periods],columns=['holiday?'])
        ef = pd.DataFrame(e,index=[i for i in periods],columns=['weekend?'])

        df = pd.merge(df, df_temp, left_index=True, right_index=True, how='outer')
        df = pd.merge(df, ef, left_index=True, right_index=True, how='outer')
        return df

    def __cdf_to_utc(self):
        """
        Helper function to make timestamps tz-aware
        """
        df = self.df
        t=[]
        for time in df.index:
            t.append(time.tz_localize('UTC'))
        df.index = t
        
        self.df = df
        return df

    def __feature_eng(self):
        """
        Feature engineering for adding periodic curves that models seasonality in data
        """
        df = self.df
        df['Seconds'] = df.index.map(pd.Timestamp.timestamp)
        day = 60*60*24
        year = 365.2425*day
        week = day*7

        df['Day sin'] = np.sin(df['Seconds'] * (2* np.pi / day))
        df['Day cos'] = np.cos(df['Seconds'] * (2 * np.pi / day))
        df['Week sin'] = np.sin(df['Seconds'] * (2 * np.pi / week))
        df['Week cos'] = np.cos(df['Seconds'] * (2 * np.pi / week))
        df['Year sin'] = np.sin(df['Seconds'] * (2 * np.pi / year))
        df['Year cos'] = np.cos(df['Seconds'] * (2 * np.pi / year))
        df.drop(['Seconds'], axis=1, inplace=True) #We will not use the Seconds column any further
        df.dropna(inplace=True)
        
        self.df = df
        return df

    def __data_split(self, X, y):
        """
        splits the data into train, test and val sets
        is set to random to let the model also evaluate trends from the last year
        """
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=1)
        return X_train, X_val, y_train, y_val
        

    def __xgb(self, X_train, y_train, X_test, y_test):
        """
        Implementation of XGBoost with hyperparameter tuning
        """

        evaluation = [( X_train, y_train), ( X_test, y_test)]

        #Define search space for hp-tuning
        space={'max_depth': hp.choice('max_depth', np.arange(8, 32, dtype=int)),
            'gamma': hp.uniform ('gamma', 1,9),
            'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
            'reg_lambda' : hp.uniform('reg_lambda', 0,1),
            'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
            'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
            'n_estimators': 180
        }

        def __objective(space):
            """
            The objective for hyperparameter search 
            """
            print(f'Finding best params for XGBoost')
            reg = xgboost.XGBRegressor(n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                            reg_alpha = int(space['reg_alpha']),min_child_weight=space['min_child_weight'],
                            colsample_bytree=space['colsample_bytree'], eval_metric="rmse", early_stopping_rounds=10)
                    
            reg.fit(X_train, y_train,
                    eval_set=evaluation,
                    verbose=0)
            

            pred = reg.predict(X_test)
            accuracy = np.sqrt(mean_squared_error(y_test, pred))
            print ("SCORE:", accuracy)
            return {'loss': -accuracy, 'status': STATUS_OK }

        trials = Trials()

        best_hyperparams = fmin(fn = __objective,
                                space = space,
                                algo = tpe.suggest,
                                max_evals = self.max_evals_xgb,
                                trials = trials)

        dX_train = xgboost.DMatrix(X_train, label=y_train)
        dX_test = xgboost.DMatrix(X_test, label=y_test)
        booster = xgboost.train(params=best_hyperparams, dtrain=dX_train, evals=[(dX_train, 'train') , (dX_test, 'valid')])
        booster.save_model(self.xgb_save_model_name)

        return booster

    def __catboostregressor(self, X_train, y_train):
        """
        Implementation of CatBoostRegressor with hyperparameter tuning
        """
        if not self.fast_cb:
            #Slow CatBoost for training
            dataset_pool = cb.Pool(data=X_train, label=y_train)
            catboost = cb.CatBoostRegressor(task_type='GPU')

            #Define search space
            params = {
                'max_depth': [6, 12, 24],
                'learning_rate' : [0.1, 0.05, 0.001],
                'iterations'    : [50, 150, 300],
                'l2_leaf_reg': [2,10,24]
            }

            print('Fit CatBoostRegressor ... \n')
            catboost.grid_search(param_grid=params,X=dataset_pool, train_size=0.8, verbose=1) 
            #Grid search is very exhaustive, reduce parameters or use random_search to reduce runtime 

            catboost.save_model(self.cbr_save_model_name)
            return catboost
        
        else:
            #Fast CatBoost for testing purposes. Should not be used when training
            dataset_pool = cb.Pool(data=X_train, label=y_train)
            catboost = cb.CatBoostRegressor(task_type='GPU')

            #Define search space. Randomly selected for Fast CBR
            params = {
                'max_depth': [4],
                'learning_rate' : [0.05],
                'iterations'    : [70],
                'l2_leaf_reg': [10]
            }

            print('Fit CatBoostRegressor ... \n')
            catboost.grid_search(param_grid=params,X=dataset_pool, train_size=0.8, verbose=1) 
            #Grid search is very exhaustive, reduce parameters or use random_search to reduce runtime 

            catboost.save_model(self.cbr_save_model_name)
            return catboost

    def __evaluate_lstm(self, lstm_val_pred, y_val):
        """
        Prints evaluation-scores for LSTM model
        """
        y_val_df = pd.DataFrame({'y_val': y_val})
        y_val_df['lstm_val_pred'] = lstm_val_pred
        y_val_df = y_val_df.sort_index()
        plt.plot(y_val_df['lstm_val_pred'][-10:], label='lstm_prediction')
        plt.plot(y_val_df['y_val'][-10:], label='actual')
        plt.legend()
        plt.plot()
        plt.show()

        print(f"MSE lstm: {mean_squared_error(lstm_val_pred, y_val)}")
        print(f"MAE lstm: {mean_absolute_error(lstm_val_pred, y_val)}")
        print(f"MSLE lstm: {mean_squared_log_error(lstm_val_pred, y_val)}")
        print(f"R2 lstm: {r2_score(lstm_val_pred, y_val)}")

    def __evaluate_xgb_cbr(self, xgb_val_pred, cbr_val_pred, y_val):
        """
        Prints evaluation scores for XGBoost and CatBoost models
        """
        
        y_val_df = pd.DataFrame({'y_val': y_val}) #Make dataframe with validation data
        y_val_df['xgb_val_pred'] = xgb_val_pred.tolist() #Add predictions
        y_val_df['cbr_val_pred'] = cbr_val_pred.tolist()
        y_val_df = y_val_df.sort_index() #Sort whole dataset on index to readability
        plt.plot(y_val_df['xgb_val_pred'][-10:], label='xgb_prediction') #plot predictions
        plt.plot(y_val_df['y_val'][-10:], label='actual')
        plt.plot(y_val_df['cbr_val_pred'][-10:], label='cbr_prediction')
        plt.grid()
        plt.xticks(rotation=45) #Beautify plot
        plt.legend()
        plt.plot()
        plt.show()

        print(f"MSE XGB: {mean_squared_error(xgb_val_pred, y_val)}")
        print(f"MAE XGB: {mean_absolute_error(xgb_val_pred, y_val)}")
        print(f"MSLE XGB: {mean_squared_log_error(xgb_val_pred, y_val)}")
        print(f"R2 XGB: {r2_score(xgb_val_pred, y_val)}")
        print(f"******************")
        print(f"MSE cbr: {mean_squared_error(cbr_val_pred, y_val)}")
        print(f"MAE cbr: {mean_absolute_error(cbr_val_pred, y_val)}")
        print(f"MSLE cbr: {mean_squared_log_error(cbr_val_pred, y_val)}")
        print(f"R2 cbr: {r2_score(cbr_val_pred, y_val)}")

    def __load_xgb_model(self):
        """
        Load XGBoost model
        """
        model=self.xgb_load_model_name
        loaded = xgboost.XGBRegressor()
        loaded.load_model(model)
        return loaded

    def __load_cb_model(self):
        """
        Load CatBoost model
        """
        model = self.cbr_load_model_name
        cat = cb.CatBoostRegressor()
        cb_model = cat.load_model(model)
        return cb_model

    def create_lstm_dataset(self, dataset, look_back):
        """
        Creates dataset for evaluating LSTM model
        """
        dataset = dataset.to_numpy()
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), :-1]
            dataX.append(a)
            dataY.append(dataset[i + look_back, -1])
        return np.array(dataX), np.array(dataY)
        
    def __normalize_split(self, test_size=0.2):
        """
        Normalizes and splits data (specifically for LSTM) 
        returns the scaler to use for inverse transform
        """

        df = self.df
        if 'volue_forecast' in df.columns:
            """
            Volue forecast is deleted from LSTM because LSTM has a tendency to overfit when data correlate too much
            """
            df.drop('volue_forecast', axis=1, inplace=True)
        df1 = df
        df1 = df1.to_numpy()
        train_scaler = MinMaxScaler(feature_range=(0,1))
        test_scaler = MinMaxScaler(feature_range=(0,1))

        #To avoid data leak between train and test we cannot fit scaler to complete dataset, therefore splitted into train and test
        train = df1[:-floor(df.shape[0]*test_size),:]
        test = df1[-floor(df.shape[0]*test_size):, :]

        scaled_train = train_scaler.fit(train).transform(train)
        scaled_test = test_scaler.fit(test).transform(test)

        X_train = scaled_train[:, 1:]
        y_train = scaled_train[:,0]

        X_test = scaled_test[:, 1:]
        y_test = scaled_test[:, 0]

        #Convert data to DataFrame-format for further use
        train_data = pd.DataFrame({col : X_train[:,count] for count, col in enumerate(list(df.columns)[1:])}) #Probably too long and unnecessary comprehension, but it works :)
        train_data['y'] = y_train
        val_data = pd.DataFrame({col : X_test[:,count] for count, col in enumerate(list(df.columns)[1:])})
        val_data['y'] = y_test

        return train_data, val_data, train_scaler, test_scaler

    def __lstm(self, df, n_past):
        """
        Handles the process of making dataset for LSTM, reshaping the data and the configuration for training LSTM network on data
        """
        if 'volue_forecast' in df.columns:
            df.drop('volue_forecast', axis=1, inplace=True)
        n_features = df.shape[1] - 1
        print(f'n_features: {n_features}')
        train_data, val_data, train_scaler, test_scaler = self.__normalize_split()
        print("train_data", train_data)
        trainX,trainY=self.create_lstm_dataset(train_data, look_back=n_past)
        #testX,testY=create_dataset(test_data.to_numpy(), look_back=n_past)
        valX,valY=self.create_lstm_dataset(val_data, look_back=n_past)
        
        trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], n_features))
        #testX = np.reshape(testX, (1, testX.shape[0], testX.shape[1]))
        valX = valX.reshape((valX.shape[0], valX.shape[1], n_features))

        #Define training configuration
        if self.fast_lstm:
            config =  [[False], [False], [False], [64], [16], [0.2]]
        else:
            config = [[True, False], [True, False], [True, False], [64, 128], [16, 32, 128], [0.2]]  

        #Config: list of lists --> [[first_additional_layer], [second_additional_layer], [third_additional_layer], [n_neurons], [n_batch_size], [dropout]]
        hist, model = self.__LSTM_HyperParameter_Tuning(config, trainX, trainY, valX, valY) 
        hist = pd.DataFrame(hist)
        hist = hist.sort_values(by=[7], ascending=True)
        print(f'Best Combination: \n first_additional_layer = {hist.iloc[0, 0]}\n second_additional_layer = {hist.iloc[0, 1]}\n third_additional_layer = {hist.iloc[0, 2]}\n n_neurons = {hist.iloc[0, 3]}\n n_batch_size = {hist.iloc[0, 4]}\n dropout = {hist.iloc[0, 5]}')
        return model, valX, valY, train_scaler, test_scaler, train_data, val_data

    def __LSTM_HyperParameter_Tuning(self, config, x_train, y_train, x_val, y_val):
        """
        Builds LSTM-network with tuning
        Hyperparameter space is defined with a cartesian product of configurations
        """
        #Initial configuration
        first_additional_layer, second_additional_layer, third_additional_layer, n_neurons, n_batch_size, dropout = config

        #Make search space with all different combinations (Cartesian product of input iterables. Equivalent to nested for-loops.)
        possible_combinations = list(itertools.product(first_additional_layer, second_additional_layer, third_additional_layer,
                                                    n_neurons, n_batch_size, dropout))
        print(possible_combinations)
        print('\n')
        
        hist = []
        
        for i in range(0, len(possible_combinations)):
            """
            Build a model for each of the combinations
                This method might be enhanced significantly 
                by using e.g keras tuner with searchCVs 
                or other methods of tuning
            """
            
            print(f'Running Configuration {i+1}/{len(possible_combinations)}\n')
            print('--------------------------------------------------------------------')
            
            first_additional_layer, second_additional_layer, third_additional_layer, n_neurons, n_batch_size, dropout = possible_combinations[i]
            
            regressor = Sequential()
            regressor.add(LSTM(units=n_neurons, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]))) #input_shape = (time_steps, n_features)
            regressor.add(Dropout(dropout)) #Used to prevent overfitting

            if first_additional_layer:
                regressor.add(LSTM(units=n_neurons, return_sequences=True))
                regressor.add(Dropout(dropout))

            if second_additional_layer:
                regressor.add(LSTM(units=n_neurons, return_sequences=True))
                regressor.add(Dropout(dropout))

            if third_additional_layer:
                regressor.add(GRU(units=n_neurons, return_sequences=True))
                regressor.add(Dropout(dropout))

            regressor.add(LSTM(units=n_neurons, return_sequences=False))
            regressor.add(Dropout(dropout))
            regressor.add(Dense(units=1, activation='relu'))
            regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])

            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
            '''''
            If a validation dataset is specified to the fit() function via the validation_data or 
            validation_split arguments,then the loss on the validation dataset will be made available via the name “val_loss.”
            '''''

            file_path = self.lstm_save_model_name
            mc = ModelCheckpoint(file_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            '''''
            cb = Callback(...)  # First, callbacks must be instantiated.
            cb_list = [cb, ...]  # Then, one or more callbacks that you intend to use must be added to a Python list.
            model.fit(..., callbacks=cb_list)  # Finally, the list of callbacks is provided to the callback argument when fitting the model.
            '''''
            regressor.summary()
            regressor.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=n_batch_size, callbacks=[es, mc], verbose=0)

            train_accuracy = regressor.evaluate(x_train, y_train, verbose=3)
            test_accuracy = regressor.evaluate(x_val, y_val, verbose=3)

            hist.append(list((first_additional_layer, second_additional_layer, third_additional_layer, n_neurons, n_batch_size, dropout,
                            train_accuracy, test_accuracy)))

            print(f'{str(i)}-th combination = {possible_combinations[i]} \n train accuracy: {train_accuracy} and test accuracy: {test_accuracy}')
            
            print('--------------------------------------------------------------------')
            print('--------------------------------------------------------------------')
            print('--------------------------------------------------------------------')
            print('--------------------------------------------------------------------')
        # See https://github.com/MohammadFneish7/Keras_LSTM_Diagram for a good explanation on how LSTM works
        return hist, regressor

    def load_lstm(self):
        """
        Load LSTM model from class feature lstm_model_name defined in Main.py
        """
        model=self.lstm_load_model_name
        return load_model(model)

    def __inv_trans(self, lstm_pred, y_test_scaled, test_data, test_scaler):
        """
        Inverse transform predictions and validation data to better understand data.
        """
        lstm_pred_copy = np.repeat(lstm_pred, test_data.shape[1], axis=-1).reshape(-1, test_data.shape[1])
        print(lstm_pred_copy)
        lstm_pred_unscaled = test_scaler.inverse_transform(lstm_pred_copy)[:,0]
        y_test_copy = np.repeat(y_test_scaled, test_data.shape[1], axis=-1).reshape(-1, test_data.shape[1])
        y_test_unscaled = test_scaler.inverse_transform(y_test_copy)[:,0]

        return y_test_unscaled, lstm_pred_unscaled

    def __plot_lstm_pred(self, y_test, lstm_pred, n_hours):
        """
        Plot LSTM predictions
        """
        plt.title('Scaled')
        plt.plot(range(len(y_test[-n_hours:])), y_test[-n_hours:], linewidth=1.0, label='actual')
        plt.plot(range(len(lstm_pred[-n_hours:])), lstm_pred[-n_hours:], linewidth=1.0, label='pred')
        plt.legend()
        plt.show()


    def get_feature_importance(self, df_X, xgb_model: str, cbr_model: str):
        """
        Plots feature importance in XGB and CBR model
        """
        xgb_model = self.__load_xgb_model()
        cbr_model = self.__load_cb_model()
        xgb_feature_importance = xgb_model.feature_importances_
        cbr_feature_importance = cbr_model.get_feature_importance()

        sorted_idx_xgb = np.argsort(xgb_feature_importance)
        fig1 = plt.figure(figsize=(12, 6))
        plt.barh(range(len(sorted_idx_xgb)), xgb_feature_importance[sorted_idx_xgb], align='center')
        plt.yticks(range(len(sorted_idx_xgb)), np.array(df_X.columns)[sorted_idx_xgb])
        plt.title('Feature Importance XGBoost')
        plt.show()

        sorted_idx_cbr = np.argsort(cbr_feature_importance)
        fig2 = plt.figure(figsize=(12, 6))
        plt.barh(range(len(sorted_idx_cbr)), cbr_feature_importance[sorted_idx_cbr], align='center')
        plt.yticks(range(len(sorted_idx_cbr)), np.array(df_X.columns)[sorted_idx_cbr])
        plt.title('Feature Importance CatBoost')
        plt.show()

    def get_validation_data_lstm(self):
        #return a dataframe with the validation data for lstm
        pass

    def get_validation_data_xgbcbr(self):
        #return a dataframe with validation data for xgboost and catboost
        pass

    def get_predictions(self):
        pass
        #return self.xgb_pred, self.cbr_pred, self.lstm_pred


    def main_training(self, train_config):
        """
        For LSTM neural networks we cannot randomly split the data. This is because the time-ordering plays a role for the memory-cells. 
        In XGBoost and CatBoost however, the time-ordering is not relevant
        """

        df = self.get_data_cdf(start_year   =   self.start_year,
                                start_month =   self.start_month,
                                start_day   =   self.start_day,
                                end_year    =   self.end_year,
                                end_month   =   self.end_month,
                                end_day     =   self.end_day)    
        df = self.__cdf_to_utc()
        df = self.__feature_eng()

        #Inputs that should be excluded:
        for i in self.exclude_inputs:
            if i is not None:
                df.drop(i, axis=1, inplace=True)
        self.df = df

        train_cols=list(df.columns)[1:]
        X = df[train_cols]
        y = df['cons_actual']                

        #Split data into training and validation
        X_train, X_val, y_train, y_val = self.__data_split(X, y)

        for key, value in train_config.items():
            #Iterate train_config to only train models explicitly stated in Main.py
            
            if value and key=="xgb":
                X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=0.5, random_state=1)
                xgb_model = self.__xgb(X_train, y_train, X_test, y_test)
                xgb_val_pred = xgb_model.predict(xgboost.DMatrix(X_val, label=y_val))
            
            if value and key=='cbr':
                cbr_model = self.__catboostregressor(X_train, y_train)
                cbr_val_pred = cbr_model.predict(X_val)

            if value and key=='lstm':
                lstm_model, lstm_X_val, lstm_y_val, train_scaler, test_scaler, train_data, test_data  = self.__lstm(self.df, self.look_back)
                lstm_val_pred = lstm_model.predict(lstm_X_val)
                y_val_unscaled, lstm_pred_unscaled = self.__inv_trans(lstm_val_pred, lstm_y_val, test_data, test_scaler)
                self.__plot_lstm_pred(y_val_unscaled, lstm_pred_unscaled, n_hours=28)
                self.__evaluate_lstm(lstm_pred_unscaled, y_val_unscaled)

        try:
            self.__evaluate_xgb_cbr(xgb_val_pred, cbr_val_pred, y_val)
        except NameError:
            print("Missing XGB or CBR preds")
        

    def main_load_models(self, start_val, end_val):
        """
        Load pretrained models to be evaluated
        """

        xgb_model = self.__load_xgb_model()
        cbr_model = self.__load_cb_model()
        lstm_model = self.load_lstm()    

        start_year, start_month, start_day = start_val.split('-')
        end_year, end_month, end_day = end_val.split('-')

        df = self.get_data_cdf(start_year   = start_year,
                                start_month = start_month,
                                start_day   = start_day,
                                end_year    = end_year,
                                end_month   = end_month,
                                end_day     = end_day)
        df = self.__cdf_to_utc()
        df = self.__feature_eng() #Also deletes NaN-values
        
        #Inputs that should be excluded:
        for i in self.exclude_inputs:
            if i is not None:
                df.drop(i, axis=1, inplace=True)
        self.df = df
        print("self.df = ", self.df)

        if not self.predict_future:
            train_cols=list(df.columns)[1:]
            X = df[train_cols]
            y = df['cons_actual']
            X_train, X_val, y_train, y_val = self.__data_split(X, y)
            train_data, test_data, train_scaler, test_scaler = self.__normalize_split()
            print("test_data = ", test_data)
            xgb_val_pred = xgb_model.predict(X_val)
            cbr_val_pred = cbr_model.predict(X_val)
            self.xgb_pred = xgb_val_pred
            self.cbr_pred = cbr_val_pred
            
            X_test_scaled, y_test_scaled = self.create_lstm_dataset(test_data, look_back=self.look_back)
            print("lstm_before_pred = ", X_test_scaled)
            lstm_pred = lstm_model.predict(X_test_scaled)
            y_test_unscaled, lstm_pred_unscaled = self.__inv_trans(lstm_pred, y_test_scaled, test_data, test_scaler)
            self.__plot_lstm_pred(y_test_unscaled, lstm_pred_unscaled, n_hours=25)

            self.lstm_pred = lstm_pred_unscaled

        else:
            """
            WORK IN PROGRESS:
                This statement should handle the case where the models are used for predicting into the future.
                Because of some delays and a lot of bugs, we didnt have time to implement this correctly. 

                The thought was to replace the 'cons_actual' column with zeros before predicting. The plots will end up wrong, but there are easy quick-fixes for that
            """
            zeros = np.zeros(df.shape[0])
            X = df
            y = pd.DataFrame({'y': zeros}) #Insert y for non-existing

            X_train, X_val, y_train, y_val = self.__data_split(X, y)
            train_data, test_data, train_scaler, test_scaler = self.__normalize_split()

            print(type(X_val))
            print(type(train_data))

            print("test_data = ", test_data)
            xgb_val_pred = xgb_model.predict(X_val)
            cbr_val_pred = cbr_model.predict(X_val)
            self.xgb_pred = xgb_val_pred
            self.cbr_pred = cbr_val_pred
            
            X_test_scaled, y_test_scaled = self.create_lstm_dataset(test_data, look_back=self.look_back)
            print("lstm_before_pred = ", X_test_scaled)
            lstm_pred = lstm_model.predict(X_test_scaled)
            y_test_unscaled, lstm_pred_unscaled = self.__inv_trans(lstm_pred, y_test_scaled, test_data, test_scaler)
            self.__plot_lstm_pred(y_test_unscaled, lstm_pred_unscaled, n_hours=25)

            self.lstm_pred = lstm_pred_unscaled


        self.__evaluate_xgb_cbr(xgb_val_pred, cbr_val_pred, y_val)
        self.__evaluate_lstm(lstm_pred_unscaled, y_test_unscaled)
        self.get_feature_importance(X, xgb_model=xgb_model, cbr_model=cbr_model)