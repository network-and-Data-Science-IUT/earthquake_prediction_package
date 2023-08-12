aggregate
=========

**Description**

The aggregate function in the EarthquakeForecast package provides a method for consolidating and summarizing data when multiple samples share the same spatial and temporal identifiers. This function streamlines the aggregation process, allowing users to choose from various aggregation modes to combine values within these shared identifiers.

**Usage**

.. py:function:: data_preprocessing.aggregate(data,column_identifier,aggregation_mode='mean', base=None, verbose=0)

**Parameters**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: aggregate_in.txt

**Returns**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: aggregate_out.txt
