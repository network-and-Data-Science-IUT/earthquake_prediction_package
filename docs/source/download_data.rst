download_data
=============

**Description**

download the earthquake data from a data center based on the time interval and the geographical locations defined by user input.

**Usage**

.. py:function:: data_downloading.download_data(time_interval,save_dataframe = False,magnitude_range=None,    depth_range=None,data_center='USGS',rectangular_region=None,circular_region=None)

**Parameters**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: download_data_in.txt
.. note:: one of *rectangular_region* or *circular_region* input must be initiated. If both are initiated the *rectangular_region* is considered.
**Returns**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: download_data_out.txt
