set_temporal_id
===============

**Description**

The set_temporal_id function within the EarthquakeForecast package allows users to assign temporal identifiers to groups of samples based on desired time units and steps. These temporal IDs help label and categorize samples according to their specific time intervals.

**Usage**

.. py:function:: data_preprocessing.set_temporal_id(data, column_identifier=None, unit='temporal ID', step=1, verbose=0)


**Parameters**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: set_temporal_id_in.txt 

**Returns**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: set_temporal_id_out.txt

**Example**

.. code-block:: python

   import pandas as pd
   from EearthquakeForecast.data_preprocessing import set_temporal_id
   df = pd.read_csv('data.csv')
   set_temporal_id(data=df,column_identifier={'temporal ID':'time'},
   unit = 'month', step = 3)

