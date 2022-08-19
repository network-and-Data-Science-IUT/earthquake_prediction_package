rank_covariates
================

**Description**

after ranking covariates in this function based on supported methods “correlation”, “mRMR” and “variance”,  the covariates in each resulting subset along with their corresponding historical values form a candidate feature set in the feature selection process.


**Usage**

.. py:function:: data_preprocessing.rank_covariates(data, rank_covariates='mRMR', forced_covariates=[],verbose=0)

**Parameters**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: rank_covariates_in.txt

**Returns**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: rank_covariates_out.txt
**Example**

.. code-block:: python

   import pandas as pd
   from EQPredict.data_preprocessing import rank_covariates
   df = pd.read_csv('data.csv')
   ranked_covariates = rank_covariates(data=df,ranking_method='variance',
                                       forced_covariates=['magnitude','temperature'])
 

