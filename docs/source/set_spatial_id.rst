set_spatial_id
================

**Description**

The set_spatial_id function within the EarthquakeForecast package facilitates the creation of spatial identifiers for distinct regions within a given region of interest. This function operates by dividing the area into pixels or clusters and assigning spatial IDs based on the provided longitude, latitude, and depth information.

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