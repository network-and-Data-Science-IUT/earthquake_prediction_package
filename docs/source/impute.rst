impute
======

**Description**

Impute the missing values of each covariate of the input data, using different approaches such as KNN, mean, median, most_frequent, min, max. For each of these methods, it is provided to choose K for the size of the window in which this method is applied.

**Usage**

.. py:function:: preprocess.impute(data , column_identifier = None , missing_value = np.nan, fill_missing_target = 0, K = None, impute_strategy= "KNN")

**Parameters**

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: impute_in.txt

**Returns** 

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: impute_out.txt
