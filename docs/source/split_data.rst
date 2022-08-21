split_data
==========

**Description**

    Splitting the preprocessed data into training, validation, and testing sets with regard to user preference.
    The function allows temporal splitting of the data, wherein the test set is selected from the last temporal
    units in the data, and a number of previous temporal units of the test set (here denoted by the Gap)
    is excluded from the data to simulate the situation of real prediction in which we do not have access to the 
    information of the forecast_horizon - 1 units before the time point of the target variable.
    fig 1.a shows the temporal splitting of a sample preprocessed data.
    The cross validation method is also supported in the function, which is depicted in fig 1.b.

**Usage**

..   py:function:: predict.split_data(data, splitting_type = 'instance', instance_testing_size = None, instance_validation_size = None, instance_random_partitioning = False, fold_total_number = None, fold_number = None, forecast_horizon = 1, granularity = 1, verbose = 0)

**Parameters**

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: split_data_in.txt

**Returns** 

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: split_data_out.txt
