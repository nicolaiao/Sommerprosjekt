## This project is intended for those who might have access to the data, others will not have an API-key. But you are welcome to check out or use the code if you like!


RUN "pip install -r requirements.txt" 

Main.py:
    config: a dictionary of boolean values telling which of the models should be retrained
    True -> run training on selected model
    False -> evaluate existing model


    attributes: a dictionary of optional values to pass to HandlerClass
        "look_back" -> (integer) a parameter for LSTM (how far back do you want to use data to predict the next timestep)
        "exclude_inputs" -> (list) a list of strings to exclude when training *see allowed values under
        the rest are model names used to differenciate models and evaluate them separately

    *exclude_inputs allowed:
    ["cons_actual","temp_forecast", "price_forecast", "cc_forecast", "volue_forecast", "Day sin", "Day cos", "Week sin", "Week cos", "Year sin", "Year cos", "holiday?", "weekend?"]
        !Requires retraining!

ModelClass.py:
    consists of mostly local class-methods

    TODO: 
    1.  Add getters and setters for predictions
    2.  DONE
    3.  See how models perform on different datasets
        -   Weekends vs. Weekdays
        -   Summer vs. Winter vs. Fall vs. Spring
        IMPORTANT:
            If you are going to evaluate different timeperiods you should retrain the model so that the periods you are evaluating/predicting on is not part of your training/testing data!
                This is because of information leak. You will see a much better performance than what is actually achieved
            -   It might be better to just use the already existing validation set and filter on the different periods
    
    4. Data to cdf -> need to upload prediction data BEFORE prediction!

BEFORE production:
    -   Install tensorflow with correct compiler-flags and software to support GPU-training -> significantly reduce training time
    -   Pipeline/Server to execute daily updates to CDF
    -   Server to retrain models
    -   Monitor evaluation metrics, in case of drop -> retrain
        1.  Change input parameters
        2.  Try new inputs
        3.  Tweak hyperparameter space
        4.  Retrain on new timeperiod (prefer new data)

To make training faster:
Right now only CatBoost is using GPU for training. XGBoost, and especially LSTM might get a boost on how fast the models performs training. (LSTM up to 6x faster)
    - use keras.layers.CuDNNLSTM instead of keras.layers.LSTM -> This is for better GPU utilization
        This requires admin-rights and GPU.software of some kind
        You might have to reinstall tensorflow with correct compiler-flags
    - Search space for hyperparameter tuning can be reduced by a lot. If you find combinations that works well, stick to them and tweak the rest of the space. It might be hard to believe,    
        but the search space we use for tuning LSTM in this project is actually pretty small. There are a LOT of parameters that could be tuned.
    - Cloud services and computing might speed up the process

    Notes: LSTM is infamous for being REEAAAALLY slow, as it is with most DL-techniques. 
    
