train_test
==========

**Description**

training and testing process on the preprocessed data based on best configurations (best model, feature set and history)

**Usage**

.. py:function:: prediction.train_test(data, instance_testing_size, forecast_horizon, feature_or_covariate_set, history_length=1,layer_number=None, model='knn', base_models=None, model_type='regression', model_parameters=None, feature_scaler='logarithmic', target_scaler='logarithmic', labels=None, performance_measures=['MAPE'], performance_mode='normal', performance_report=True, save_predictions=True, verbose=0)


**Parameters**

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: train_test_in.txt

.. Note:: In the current version, 'AIC' and 'BIC' can only be calculated for the 'glm' model and 'classification' model_type.

**Returns** 

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: train_test_out.txt

**Example** 

.. code-block:: python

   import pandas as pd
   from EQPredict.prediction import train_test
   
   df = pd.read_csv('./historical_data h=1 l=2.csv')
   trained_model = train_test(data = df, instance_testing_size = 0.2, forecast_horizon = 4,layer_number=2
                              feature_or_covariate_set = ['magnitude', 'depth', 
                                                            'time elapsed 5.5'])