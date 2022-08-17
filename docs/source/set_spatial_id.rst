set_spatial_id
================

**Description**

This function will divide the region of interest
into some pixels (sub-regions)
and set a spatial ID for each pixel
based on the longitude, latitude, and depth.

**Usage**

.. py:function:: data_preprocessing.set_spatial_id(data, column_identifier, pixel_scale, kmeans_clusters, target_area, verbose)

**Parameters**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: set_spatial_id_in.txt

**Returns**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: set_spatial_id_out.txt
