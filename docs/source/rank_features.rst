rank_features
================

**Description**

after ranking features in this function based on supported methods “correlation”, “mRMR”, and “variance”,  a list of sorted features will be the output and will form a candidate feature set in the feature selection process.


**Usage**

.. py:function:: prediction.rank_features(data, ranking_method='mRMR', forced_features=[],verbose=0)

**Parameters**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: rank_features_in.txt

**Returns**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: rank_features_out.txt
**Example**

.. code-block:: python

   import pandas as pd
   from EQPredict.prediction import rank_features
   df = pd.read_csv('data.csv')
   ranked_features = rank_features(data=df,ranking_method='variance',
                                       forced_features=['magnitude t-1','depth t'])
 

