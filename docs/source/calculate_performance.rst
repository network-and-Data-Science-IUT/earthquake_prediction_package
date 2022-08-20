calculate_performance
===========

**Description**

This function receives true_values and predicted_values and a list of performance measures as input from the user and returns the performance measures between true_values and predicted_values.

**Usage**

.. py:function:: prediction.calculate_performance(true_values, predicted_values, performance_measures = ['MAPE'], trivial_values = [], model_type = 'regression', num_params = 1, labels = None)

**Parameters**

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: calculate_performance_in.txt


**Returns** 

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: calculate_performance_out.txt

**Example** 

.. code-block:: python

   import pandas as pd
   from EQPredict.prediction import calculate_performance
   
   true_values_list = [1, 3, 2, 2, 3, 1]
   predicted_values_list = [[0.7,0.3,0.0], [0.2,0.0,0.8], [0.1,0.3,0.6], 
                           [0.9,0.1,0.0], [1.0,0.0,0.0], [0.6,0.2,0.2]]
   
   errors = calculate_performance(true_values = true_values_list,
            predicted_values = predicted_values_list, performance_measures=['AUC'],
            model_type='classification',labels=[1,2,3])
