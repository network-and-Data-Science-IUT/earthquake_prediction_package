make_historical_data
=======================

**Description**

Transforming data to the historical format and extract features.
This subprocess prepare the final data frame including
features and target variable for modeling.

The set of features is consisted of:

#. Spatial covariates, temporal covariates at current temporal unit (t)
#. Historical values of these covariates at h-1(where h is history length) previous temporal units (t-1 , t-2 , … , t-h+1).
#. The covariates of the L-layer pixels(where L is layer_number L = (0,1,...,maximum_layer_number)) at ‘history_length_of_neighbors’-1 previous temporal units(t-1 , t-2 , … , t-‘history_length_of_neighbors’+1)

The target of the final data frame is the values of the target variable at the temporal unit t+r.

**Usage**

.. py:function:: data_preprocessing.make_historical_data(data, layer_number, aggregate_layers, neighboring_covariates, history_for_layers, aggregation_modes, neighbours_dictionary, forecast_horizon, history_length, verbose=0)

**Parameters**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: make_historical_data_in.txt

**Returns**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: make_historical_data_out.txt
