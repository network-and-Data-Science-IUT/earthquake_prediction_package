impute
======

**Description**

The impute function is a versatile tool within the EarthquakeForecast package that addresses missing values in covariates of input data. By employing various imputation approaches such as KNN, mean, median, most frequent, minimum, and maximum, this function systematically fills in missing values for each covariate. Users can customize the imputation process by selecting the desired method and, in the case of KNN imputation, specifying the number of nearest neighbors (K) for imputing missing values.

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
