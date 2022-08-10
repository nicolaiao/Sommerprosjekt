"""
Before running:
run "pip install -r requirements.txt" 
    might take a while
"""

from ModelClass import HandlerClass

# start, end must be "YYYY-MM-DD"
start="2015-01-01"
end="2022-05-01" 
start_val="2022-05-02"
end_val="2022-06-06" 

curves = ["NO1_consumption_per_15min",
    "NO1_temperature_per_15min",
    "NO1_el_price_per_hour",
    "NO1_cloud_coverage_per_15min",
    "NO1_consumption_forecast_per_15min"
    ]
#Decide display name for columns (this has to be the same as used in training since xgboost and catboost saves column_name as feature information)
columns = ["cons_actual", "temp_forecast", "price_forecast", "cc_forecast", "volue_forecast"]


#Config: Decide if you wish to train the different models, and which ones
config = {
    "train_xgb":True, 
    "train_cbr":True, 
    "train_lstm":True}

"""
exclude_inputs allowed:
["cons_actual","temp_forecast", "price_forecast", "cc_forecast", "volue_forecast", "Day sin", "Day cos", "Week sin", "Week cos", "Year sin", "Year cos", "holiday?", "weekend?"]
Requires retraining if no model exists with the same input shape
"""
attributes = {
    "look_back": 8,
    "max_evals_xgb": 105,
    "fast_cb": False,
    "fast_lstm": False,
    "exclude_inputs": ["volue_forecast"],
    "xgb_save_model_name": "xgb_model_exclude_volue.json",
    "cbr_save_model_name": "cbr_model_exclude_volue.cbm",
    "lstm_save_model_name": "lstm_model_exclude_volue_lookback_8.h5",
    "xgb_load_model_name": "xgb_model_exclude_volue.json",
    "cbr_load_model_name": "cbr_model_exclude_volue.cbm",
    "lstm_load_model_name": "lstm_model_exclude_volue_lookback_8.h5"
}
#fast_cb and fast_lstm are for testing purposes only!

#Initiate HandlerClass
model_class = HandlerClass(curves=curves, columns=columns, start=start, end=end, **config)
#set attributes to specified
model_class.set_attributes(**attributes)
if all(x == False for x in model_class.train_config.values()):
    #Load model and evaluate
    model_class.main_load_models(start_val=start_val, end_val=end_val)
else:
    #Train model (gives an evaluation, but you should evaluate again by loading for more accurate results)
    model_class.main_training(model_class.train_config)
    #model_class.get_feature_importance()