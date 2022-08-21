predict_future
===============

**Description**

Predict the target variable values of the future by training the model on the training set with the best configuration.

**Usage**

.. py:function:: predict.predict_future(data, future_data, forecast_horizon, feature_or_covariate_set, model = 'knn', base_models = [], model_type = 'regression', model_parameters = None, feature_scaler = None, target_scaler = None, labels = None, save_predictions = True, verbose = 0)


**Parameters**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: predict_future_in.txt

**Returns**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: predict_future_out.txt
