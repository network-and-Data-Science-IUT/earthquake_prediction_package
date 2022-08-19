select_features
================

**Description**

creating a data frame from the input data frame containing only features in the corresponding feature set to the ordered_covariates_or_features input. If the items in the list are covariate names, the corresponding feature set is the set of covariates and all of their historical values in the input data, otherwise the feature set is the set of features mentioned in the list.


**Usage**

.. py:function:: data_preprocessing.select_features(data, ordered_covariates_or_features)

**Parameters**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: select_features_in.txt

**Returns**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: select_features_out.txt
**Example**

.. code-block:: python

   import pandas as pd
   from EQPredict.data_preprocessing import select_features
   df = pd.read_csv('data.csv')
   new_df = select_features(data=df,
                            ordered_covariates_or_features = ['magnitude t', 'depth',
                                                             'temperature t', 'temperature t-1'])
 

