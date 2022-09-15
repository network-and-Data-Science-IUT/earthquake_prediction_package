set_spatial_id
================

**Description**

This function will divide the region of interest
into some pixels or clusters
and set a spatial ID for each of them 
based on the longitude, latitude and depth.

There are two ways to set spatial ID:
1. pixelating.
2. kmeans clustering.

**Usage**

.. py:function:: data_preprocessing.set_spatial_id(data, column_identifier, pixel_scale, kmeans_clusters, target_area, plot)

**Parameters**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: set_spatial_id_in.txt
.. note:: either 'pixel_scale' or 'kmeans_clusters' should be set. in case both of them are set, 'pixel_scale' will be considered. 
**Returns**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: set_spatial_id_out.txt