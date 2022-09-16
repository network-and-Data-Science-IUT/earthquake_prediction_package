extract_feature
================

**Description**

extract extra spatial or temporal features to enrich feature list.

spatial features: 'delta time x', 'b value'. temporal features: 'event frequency', 'total energy'

**Usage**

.. py:function:: data_preprocessing.extract_feature(data, column_identifier=None, feature_list=None)

**Parameters**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: extract_feature_in.txt

**Returns**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: extract_feature_out.txt
**Example**

.. code-block:: python

   import pandas as pd
   from EarthquakeForecast.data_preprocessing import extract_feature
   df = pd.read_csv('data.csv')
   extracted_features = extract_feature(data=df,column_identifier={'temporal ID':'time',
   'spatial ID':'city',
   'magnitude':'mag'},
   feature_list = ['delta time 4.5','total energy'])

