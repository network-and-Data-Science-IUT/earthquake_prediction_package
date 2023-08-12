import os
import enum
import folium
from matplotlib.patches import Polygon
from scipy.spatial import Voronoi, voronoi_plot_2d
from collections import defaultdict
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np


# Using enum class to define pixelating type
class Pixelating_type(enum.Enum):
   Two_diminsion = 1
   Three_diminsion = 2

def check_magnitude(data):
    import numpy as np
    if not data['magnitude'].dtype in [np.float64,np.int64]:
        raise Exception('Error: invalid type for \'magnitude\' column in dataframe. expected types are int64 and float64.')
    if data['magnitude'].isnull().values.any():
        print('Warning: there are some NaN values in the magnitude column.')
        data_without_nan = data[data['magnitude'].notna()]
    if len(data_without_nan[data_without_nan['magnitude'] < 0]) > 0:
        print('Warning: negative magnitude value is not acceptable. rows that their depths are negative will be removed.')
        data = data[data['magnitude'] >= 0]
        if len(data) == 0:
            raise Exception('dataframe got zero len after removing none-valid magnitudes.')
    return data



def check_latitude(data):
    import numpy as np
    if not data['latitude'].dtype in [np.float64,np.int64]:
        raise Exception('Error: invalid type for \'latitude\' column in dataframe. expected types are int64 and float64')
    if data['latitude'].isnull().values.any():
        print('Warning: there are some NaN values in the latitude column. those rows will be removed from dataframe.')
        data = data[data['latitude'].notna()]
        if len(data) == 0:
            raise Exception('dataframe got zero len after removing NaN columns.')
    if len(data[(data['latitude'] < -90) | (data['latitude'] > 90)]) > 0:
        print('Warning: latitude are not in range [-90,90]. rows that are not in the range will be removed.')
        data = data[(data['latitude'] >= -90) & (data['latitude'] <= 90)]
        if len(data) == 0:
            raise Exception('dataframe got zero len after removing non valid latitudes.')
    return data



def check_longitude(data):
    import numpy as np
    if not data['longitude'].dtype in [np.float64,np.int64]:
        raise Exception('Error: invalid type for \'longitude\' column in dataframe. expected types are int64 and float64')
    if data['longitude'].isnull().values.any():
        print('Warning: there are some NaN values in the longitude column. those rows will be removed from dataframe.')
        data = data[data['longitude'].notna()]
        if len(data) == 0:
            raise Exception('dataframe got zero len after removing NaN columns.')
    if len(data[(data['longitude'] < -180) | (data['longitude'] > 180)]) > 0:
        print('Warning: longitude are not in range [-180,180]. rows that are not in the range will be removed.')
        data = data[(data['longitude'] >= -180) & (data['longitude'] <= 180)]
        if len(data) == 0:
            raise Exception('dataframe got zero len after removing non valid longitudes.')
    return data



def check_depth(data):
    import numpy as np
    if not data['depth'].dtype in [np.float64,np.int64]:
        raise Exception('Error: invalid type for \'depth\' column in dataframe. expected types are int64 and float64.')
    if data['depth'].isnull().values.any():
        print('Warning: there are some NaN values in the depth column. those rows will be removed from dataframe.')
        data = data[data['depth'].notna()]
        if len(data) == 0:
            raise Exception('dataframe got zero len after removing NaN columns.')
    if len(data[data['depth'] < 0]) > 0:
        print('Warning: negative depth value is not acceptable. rows that their depths are negative will be removed.')
        data = data[data['depth'] >= 0]
        if len(data) == 0:
            raise Exception('dataframe got zero len after removing none-valid depths.')
    return data



def check_temporal_id(data):
    import numpy as np
    from pandas.api.types import is_numeric_dtype
    if not is_numeric_dtype:
        raise Exception('Error: invalid type for \'temporal id\' column in dataframe. expected numeric type.')
    if data['temporal id'].isnull().values.any():
        print('Warning: there are some NaN values in the temporal id column. those rows will be removed from dataframe.')
        data = data[data['temporal id'].notna()]
        if len(data) == 0:
            raise Exception('dataframe got zero len after removing NaN rows from temporal id.')
    return data



def check_spatial_id(data):
    import numpy as np
    if data['spatial id'].isnull().values.any():
        print('Warning: there are some NaN values in the spatial id column. those rows will be removed from dataframe.')
        data = data[data['spatial id'].notna()]
        if len(data) == 0:
            raise Exception('dataframe got zero len after removing NaN rows from spatial id.')
    return data



def check_target(data):
    import numpy as np
    if not (data['target'].dtype == np.float64 or data['target'].dtype == np.int64):
        raise TypeError('Error: the \'target\' column type should be either float64 or int64.')
    if data['target'].isnull().values.any():
         print('Warning: The target variable column includes Null values and therefore the resulting values of applying' + 
         ' target_mode may not be valid.')
    return data



def check_data_columns(data,list_of_columns):
    for column in list_of_columns:
        if column == 'magnitude':
            data = check_magnitude(data)
        if column == 'longitude':
            data = check_longitude(data)
        if column == 'latitude':
            data = check_latitude(data)
        if column == 'depth':
            data = check_depth(data)
        if column == 'temporal id':
            data = check_temporal_id(data)
        if column == 'spatial id':
            data = check_spatial_id(data)
        if column == 'target':
            data = check_target(data)
    return data

def voronoi_polygons(voronoi, diameter):
    """
    Generate shapely.geometry.Polygon objects corresponding to the
    regions of a scipy.spatial.Voronoi object, in the order of the
    input points. The polygons for the infinite regions are large
    enough that all points within a distance 'diameter' of a Voronoi
    vertex are contained in one of the infinite polygons.
    """
    centroid = voronoi.points.mean(axis=0)
    # Mapping from (input point index, Voronoi point index) to list of
    # unit vectors in the directions of the infinite ridges starting
    # at the Voronoi point and neighboring the input point.
    ridge_direction = defaultdict(list)
    for (p, q), rv in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        u, v = sorted(rv)
        if u == -1:
            # Infinite ridge starting at ridge point with index v,
            # equidistant from input points with indexes p and q.
            t = voronoi.points[q] - voronoi.points[p] # tangent
            n = np.array([-t[1], t[0]]) / np.linalg.norm(t) # normal
            midpoint = voronoi.points[[p, q]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - centroid, n)) * n
            ridge_direction[p, v].append(direction)
            ridge_direction[q, v].append(direction)

    for i, r in enumerate(voronoi.point_region):
        region = voronoi.regions[r]
        if -1 not in region:
            # Finite region.
            yield Polygon(voronoi.vertices[region])
            continue
        # Infinite region.
        inf = region.index(-1)              # Index of vertex at infinity.
        j = region[(inf - 1) % len(region)] # Index of previous vertex.
        k = region[(inf + 1) % len(region)] # Index of next vertex.
        if j == k:
            # Region has one Voronoi vertex with two ridges.
            dir_j, dir_k = ridge_direction[i, j]
        else:
            # Region has two Voronoi vertices, each with one ridge.
            dir_j, = ridge_direction[i, j]
            dir_k, = ridge_direction[i, k]

        # Length of ridges needed for the extra edge to lie at least
        # 'diameter' away from all Voronoi vertices.
        length = 2 * diameter / np.linalg.norm(dir_j + dir_k)

        # Polygon consists of finite part plus an extra edge.
        finite_part = voronoi.vertices[region[inf + 1:] + region[:inf]]
        extra_edge = [voronoi.vertices[j] + dir_j * length,
                      voronoi.vertices[k] + dir_k * length]
        yield Polygon(np.concatenate((finite_part, extra_edge)))

def plot_earthquake_data(data,target_area,pixel_scale):

    depth_level = 1
    if len(pixel_scale) == 2:
        max_lon,min_lon,max_lat,min_lat = target_area
    else:
        max_lon,min_lon,max_lat,min_lat,max_depth,min_depth = target_area
        depth_level = pixel_scale[2]
    
    for l in range(depth_level):

        # Create a map centered at the midpoint of the selected area
        midpoint = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
        map = folium.Map(location=midpoint, zoom_start=4)
        # Divide the map into x*y pixels and label each pixel

        for i in range(pixel_scale[1]):
            for j in range(pixel_scale[0]):
                pixel_lat = min_lat + (((max_lat - min_lat) * (i+0.5)) / pixel_scale[1])
                pixel_lon = min_lon + (((max_lon - min_lon) * (j+0.5)) / pixel_scale[0])
                folium.Marker(
                    location=[pixel_lat, pixel_lon],
                    icon=None,
                    popup='<b>spatial id: %d\nLocation:\nlon=%.2f,lat=%.2f</b>'%(i * pixel_scale[0] + j + 1 + pixel_scale[0]*pixel_scale[1]*l,pixel_lon,pixel_lat),
                    ).add_to(map)
        # Draw the boundary lines between the pixels
        # Draw the lines across latitudes:
        for i in range(pixel_scale[1]+1):
            line_lat = min_lat + ((max_lat - min_lat) * i) /pixel_scale[1]
            folium.PolyLine(
                    locations=[[line_lat, min_lon], [line_lat, max_lon]],
                    color='black',
                    weight=2,
                    opacity=0.8,
                ).add_to(map)
        # Draw the lines across longitudes:
        for i in range(pixel_scale[0]+1):
            line_lon = min_lon + ((max_lon - min_lon) * i) /pixel_scale[0]
            folium.PolyLine(
                    locations=[[min_lat, line_lon], [max_lat, line_lon]],
                    color='black',
                    weight=2,
                    opacity=0.8,
                ).add_to(map)

        # Plot data points for each event on the map:
        data_in_depth_l = data
        if depth_level != 1:
            depth_low = min_depth + ((max_depth - min_depth)/depth_level)*l
            depth_high = depth_low + (max_depth - min_depth)/depth_level
            output_name = '3D_pixelating_depth ' + str(depth_low) + '-' + str(depth_high)
            data_in_depth_l = data[(data['depth'] >= depth_low) & (data['depth'] <= depth_high)]
        else:
            output_name = '2D_pixelating'
            
        # ploting the earthquake events on the map:
        for i,event in data_in_depth_l.iterrows():
            folium.CircleMarker(
                location=[event['latitude'], event['longitude']],
                radius=2,
                color="red",
                fill=True,
                fill_color="red",
                popup='Location:\nlon=%.2f,lat=%.2f</b>'%(event['longitude'], event['latitude'])
            ).add_to(map)        
        map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
        yield map,output_name+'&pixel_scale='+str(pixel_scale)


def plot_map(data,target_area=None,pixel_scale =None,pixelatin_type = None,kmeans=None):

    # create a directory to save the plots
    if not os.path.exists('./plots'):
            os.makedirs('./plots')  
    if pixel_scale is not None:
        for map,output_name in plot_earthquake_data(data,target_area,pixel_scale):
            map.save('./plots/'+output_name + '.html')
        
    elif kmeans is not None:
        max_lon,min_lon,max_lat,min_lat = target_area
        centroids = kmeans.cluster_centers_
        copy_centroids_lon = centroids[:,0].copy()
        copy_centroids_lat = centroids[:,1].copy()
        # swap lon & lat's in centroids:
        centroids[:,0], centroids[:,1] = copy_centroids_lat,copy_centroids_lon
        # creating the Voronoi object using centroids coming out of k means algorithm:
        vor = Voronoi(centroids)
        # the boundry represent exactly the target area:
        boundary = np.array([[min_lat,min_lon],[min_lat,max_lon],[max_lat,max_lon],[max_lat,min_lon]])
        x, y = boundary.T
        diameter = np.linalg.norm(boundary.ptp(axis=0))
        boundary_polygon = Polygon(boundary)
        midpoint = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
        # creating the map and zooming on the midpoint:
        map = folium.Map(location=midpoint, zoom_start=4)
        # voronoi_polygons is a generator that gives us the polygons that are needed to 
        # plot on the map so that we have each cluster seperated from another using boundry lines.
        for p in voronoi_polygons(vor, diameter):
            x, y = zip(*p.intersection(boundary_polygon).exterior.coords)
            # draw all lines of the polygon one by one:
            for i in range(len(x)-1):
                folium.PolyLine(
                        locations=[[x[i], y[i]], [x[i+1], y[i+1]]],
                        color='black',
                        weight=2,
                        opacity=0.8,
                    ).add_to(map)
                
        # ploting the spatial id markers on the map:
        for i,centroid in enumerate(centroids):
            folium.Marker(
                location=centroid,
                icon=None,
                popup='<b>spatial id: %d\n Location:\n lon=%.2f,lat=%.2f</b>'%(i+1,centroid[1],centroid[0]),
                ).add_to(map)
        
      

        # ploting the earthquake events on the map:
        for i,event in data.iterrows():
            folium.CircleMarker(
                location=[event['latitude'], event['longitude']],
                radius=2,
                color="red",
                fill=True,
                fill_color="red",
                popup='Location:\nlon=%.2f,lat=%.2f</b>'%(event['longitude'], event['latitude'])
            ).add_to(map)
        # fiting the map on the target area and saving the result:
        map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

        map.save("./plots/kmeans_clustered"+"&K="+str(len(centroids)) +".html")


def scale_dataframe_using_pixel_scale(data, pixel_scale,target_area,pixelating_type,plot):
    '''
    divide the region of interest into some set of pixels and then set a spatial id
    for each of the pixels in the spatial id column.
    '''
    import numpy as np
    if pixelating_type == Pixelating_type.Two_diminsion:
        max_lon,min_lon,max_lat,min_lat = target_area
        lon_diff = max_lon - min_lon
        lat_diff = max_lat - min_lat
        x_pixels,y_pixels = pixel_scale
        lon_per_pixel = lon_diff/x_pixels
        lat_per_pixel = lat_diff/y_pixels
        lons = data['longitude']
        lats = data['latitude']
        i = np.floor(np.abs((lons - min_lon))/lon_per_pixel)
        j = np.floor(np.abs((lats - min_lat))/lat_per_pixel)
        i_s = i
        j_s = j*x_pixels
        i_s[i_s > x_pixels -1 ] = x_pixels - 1
        j_s[j_s > (y_pixels-1)*x_pixels] = (y_pixels-1)*x_pixels
        index = i_s + j_s + 1
        spatial_scaled_data = data
        spatial_scaled_data['spatial id'] = index.astype(np.int64)
        # ploting
        if plot:
            plot_map(data=data,pixel_scale=pixel_scale,target_area=target_area,pixelatin_type=Pixelating_type.Two_diminsion)
        return spatial_scaled_data
    if pixelating_type == Pixelating_type.Three_diminsion:
        max_lon,min_lon,max_lat,min_lat,max_depth,min_depth = target_area
        lon_diff = max_lon - min_lon
        lat_diff = max_lat - min_lat
        depth_diff = max_depth - min_depth
        x_pixels,y_pixels,z_pixels = pixel_scale
        lon_per_pixel = lon_diff/x_pixels
        lat_per_pixel = lat_diff/y_pixels
        depth_per_pixel = depth_diff/z_pixels
        lons = data['longitude']
        lats = data['latitude']
        depths = data['depth']
        i = np.floor(np.abs((lons - min_lon))/lon_per_pixel)
        j = np.floor(np.abs((lats - min_lat))/lat_per_pixel)
        z = np.floor(np.abs((depths - min_depth))/depth_per_pixel)
        i_s = i
        j_s = j*x_pixels
        z_s = x_pixels*y_pixels*z
        i_s[i_s > x_pixels -1 ] = x_pixels - 1
        j_s[j_s > (y_pixels-1)*x_pixels] = (y_pixels-1)*x_pixels
        z_s[z_s > x_pixels*y_pixels*(z_pixels-1)] = x_pixels*y_pixels*(z_pixels-1)
        index =  i_s + j_s + z_s + 1
        spatial_scaled_data = data
        spatial_scaled_data['spatial id'] = index.astype(np.int64)
        # ploting
        if plot:
            plot_map(data=data,pixel_scale=pixel_scale,target_area=target_area,pixelatin_type=Pixelating_type.Three_diminsion)
        return spatial_scaled_data
    


def scale_dataframe_using_kmeans_clusters(data, kmeans_clusters, plot,target_area):
    '''
    divide the region of interest into some set of clusters and then set a spatial id
    for each of the clusters in the spatial id column.
    '''
    from sklearn.cluster import KMeans
    kmeans = KMeans(kmeans_clusters)
    lon_and_lat = data[['longitude','latitude']]
    kmeans.fit(lon_and_lat)
    spatial_ids = kmeans.fit_predict(lon_and_lat)
    data['spatial id'] = spatial_ids+1
    spatial_scaled_data = data
    # ploting
    if plot:
        plot_map(data=data,kmeans=kmeans,target_area=target_area)
    return spatial_scaled_data



def find_target_area_from_dataframe(data,pixelating_type):
    '''
    Two_diminsion case:
    finding the maximum and minimum longitudes and latitudes in the data to estimate
    target area.
    Three_diminsion case:
    finding the maximum and minimum longitudes, latitudes and depths in the data to estimate
    target area.
    '''
    if pixelating_type == Pixelating_type.Two_diminsion:
        max_lon = data['longitude'].max()
        min_lon = data['longitude'].min()
        max_lat = data['latitude'].max()
        min_lat = data['latitude'].min()
        return (max_lon,min_lon,max_lat,min_lat)
    if pixelating_type == Pixelating_type.Three_diminsion:
        max_lon = data['longitude'].max()
        min_lon = data['longitude'].min()
        max_lat = data['latitude'].max()
        min_lat = data['latitude'].min()
        max_depth = data['depth'].max()
        min_depth = data['depth'].min()
        return (max_lon,min_lon,max_lat,min_lat,max_depth,min_depth)



def input_checking_set_spatial_id(data,column_identifier, pixel_scale,
                   kmeans_clusters ,target_area):
    '''checking all inputs of set_spatial_id function and
       renaming \'data\' input columns to standard format.'''
    import pandas as pd
    # checking 'data' input:
    # type checking:
    if not isinstance(data,(pd.DataFrame,str)):
        raise TypeError('Error: invalid type for \'data\' input. expected a pandas dataframe or str')
    if type(data) == str:
        data = pd.read_csv(data)
    else:
        data = data.copy()
    # checking 'pixel_scale' input:
    if pixel_scale is not None:
        # type checking 1:
        if not isinstance(pixel_scale,(list,tuple)):
            raise TypeError('Error: invalid type for \'pixel_scale\' input. expected a list.')
        # type checking 2:
        if not all(isinstance(x, (int)) for x in pixel_scale) :
            raise TypeError('Error: invalid type for \'pixel_scale\' input. not all elements in the list are int.')        
        # length checking:
        if not ((len(pixel_scale) == 2) or (len(pixel_scale) == 3)):
            raise Exception('Error: invalid length for \'pixel_scale\' input. expected lengths are 2 and 3.')
        # value checking:
        if not all([x > 0 for x in pixel_scale]):
            raise ValueError('Error: invalid value for \'pixel_scale\' input. expected values are greater than zero.')
    # checking 'target_area' input:
    if target_area is not None:
        # type checking 1:
        if not isinstance(target_area,list):
            raise TypeError('Error: invalid type for \'target_area\' input. expected a list.')
        # type checking 2:
        if not all(isinstance(x, (float,int)) for x in target_area):
            raise TypeError('Error: invalid type for \'target_area\' input. not all elements are int or float')
        # length checking:
        if not ((len(target_area) == 4) or (len(target_area) == 6)):
            raise Exception('Error: invalid length for \'target_area\' input. expected lengths are 4 and 6.')
        # value checking:
        if len(target_area) == 4:
            if not (( -180 <= target_area[0] <= 180) and ( -180 <= target_area[1] <= 180)):
               raise ValueError('Error: invalid value for \'target_area\' input. longitude should be in range [-180,180]')
            if not (( -90 <= target_area[2] <= 90) and ( -90 <= target_area[3] <= 90)):
               raise ValueError('Error: invalid value for \'target_area\' input. latitude should be in range [-90,90]')
            if target_area[0] <= target_area[1]:
               raise ValueError('Error: max longitude should be greater than min longitude')
            if target_area[2] <= target_area[3]:
               raise ValueError('Error: max latitude should be greater than min latitude')
            if pixel_scale is not None:
                if len(pixel_scale) != 2:
                    raise Exception('Error: target_area should be in size six because pixel_scale is in size three')
        if len(target_area) == 6:
            if not (( -180 <= target_area[0] <= 180) and ( -180 <= target_area[1] <= 180)):
               raise ValueError('Error: invalid value for \'target_area\' input. longitude should be in range [-180,180]')
            if not (( -90 <= target_area[2] <= 90) and ( -90 <= target_area[3] <= 90)):
               raise ValueError('Error: invalid value for \'target_area\' input. latitude should be in range [-90,90]')
            if not (( 0 <= target_area[4]) and ( 0 <= target_area[5])):
               raise ValueError('Error: invalid value for \'target_area\' input. depth should be greater than or equal to zero.')
            if target_area[0] <= target_area[1]:
               raise ValueError('Error: max longitude should be greater than min longitude') 
            if target_area[2] <= target_area[3]:
               raise ValueError('Error: max latitude should be greater than min latitude') 
            if target_area[4] <= target_area[5]:
               raise ValueError('Error: max depth should be greater than min depth')
            if pixel_scale is not None:
                if len(pixel_scale) != 3:
                    raise Exception('Error: target_area should be in size four because pixel_scale is in size 2')        
    # checking 'kmeans_cluster' input:
    if kmeans_clusters is not None:
        # type checking:
        if not isinstance(kmeans_clusters,int):
            raise TypeError('Error: invalid type for \'kmeans_clusters\' input. expected int.')
        # value checking:
        if not (kmeans_clusters > 0):
            raise ValueError('Error: invalid value for \'kmeans_clusters\' input. expected value is greater than zero.')
    # checking 'column_identifier' input:
    if column_identifier is not None:
        # type checking:
        if not isinstance(column_identifier,dict):
            raise TypeError('Error: invalid type for \'column_identifier\' input. expected dictionary.')
        # length checking:
        if not (len(column_identifier) > 0):
            raise Exception('Error: invalid length for \'column_identifier\' input. expected length is greater than zero.')
        if (pixel_scale is not None) or (kmeans_clusters is not None):
            # key checking:
            if not all([((x in column_identifier.keys()) or (x in data.columns)) for x in ['longitude','latitude'] ]):
              raise Exception('Error: not all keys(longitude and latitude)  found in \'column_identifier\' input or in the dataframe.')
            # column checking:
            if not all([True if (x in data.columns) else (x in column_identifier.keys()) for x in ['longitude','latitude'] ]):
                raise Exception('Error: not all columns specified in \'column_identifier\' input found in \'data\' input.')
            # renaming in dataframe.
            if not ('longitude' in data.columns):
                data = data.rename(columns={column_identifier['longitude']:'longitude'})
            # renaming in dataframe.        
            if not ('latitude' in data.columns):
                data = data.rename(columns={column_identifier['latitude']:'latitude'})
            if pixel_scale is not None:
                if len(pixel_scale) == 3:
                    # key checking:
                    if not all([((x in column_identifier.keys()) or (x in data.columns)) for x in ['depth'] ]):
                        raise Exception('Error: not all keys(depth) found in \'column_identifier\' input or in the dataframe.')
                    # column checking:
                    if not all([True if (x in data.columns) else (x in column_identifier.keys()) for x in ['depth'] ]):
                        raise Exception('Error: not all columns specified in \'column_identifier\' input found in \'data\' input.')
                    # renaming in dataframe:
                    if not ('depth' in data.columns):              
                        data = data.rename(columns={column_identifier['depth']:'depth'})
        else:
            # key checking:
            if not all([((x in column_identifier.keys()) or (x in data.columns)) for x in ['spatial id'] ]):
              raise Exception('Error: not all keys(spatial id) found in \'column_identifier\' input or in the dataframe.')
            # column checking:
            if not all([True if (x in data.columns) else (x in column_identifier.keys()) for x in ['spatial id'] ]):
                raise Exception('Error: not all columns specified in \'column_identifier\' input found in \'data\' input.')
            # renaming in dataframe:
            if not ('spatial id' in data.columns):
                data = data.rename(columns={column_identifier['spatial id']:'spatial id'})            
    else:
        if (pixel_scale is not None) or (kmeans_clusters is not None):
            if not (('longitude' in data.columns) and ('latitude' in data.columns)):
                raise Exception('Error: dataframe must contain longitude and latitude columns.')
            if pixel_scale is not None:
                if len(pixel_scale) == 3:
                    if not ('depth' in data.columns):
                        raise Exception('Error: dataframe must contain depth column.')
        else:
            if not ('spatial id' in data.columns):
                raise Exception('Error: dataframe must contain spatial id column.')
    # checking the content of 'data' input columns:
    if (pixel_scale is not None) or (kmeans_clusters is not None):
        data = check_data_columns(data,['longitude','latitude'])
        if pixel_scale is not None:
            if len(pixel_scale) == 3:
                data = check_data_columns(data,['depth'])
    else:
        data = check_data_columns(data,['spatial id'])
    # checking if all events happen in the target area:
    if target_area is not None:
        if len(target_area) == 4:
            max_lon,min_lon,max_lat,min_lat = target_area
            if any((max_lon < data['longitude']) | (data['longitude'] < min_lon) | (data['latitude'] < min_lat) | (data['latitude'] > max_lat)):
                print('Warning: there are some events out of target_area. those will be removed.')
                data = data[(max_lon < data['longitude']) | (data['longitude'] < min_lon) | (data['latitude'] < min_lat) | (data['latitude'] > max_lat)]
                if len(data) == 0:
                    raise Exception('dataframe got zero len after removing out of target_area events.')
        if len(target_area) == 6:
            max_lon,min_lon,max_lat,min_lat,max_depth,min_depth = target_area
            if any((max_lon < data['longitude']) | (data['longitude'] < min_lon) | (data['latitude'] < min_lat) | (data['latitude'] > max_lat) | (data['depth'] > max_depth) | (data['depth'] < min_depth)):
                print('Warning: there are some events out of target_area. those will be removed.')
                data = data[(max_lon < data['longitude']) | (data['longitude'] < min_lon) | (data['latitude'] < min_lat) | (data['latitude'] > max_lat) | (data['depth'] > max_depth) | (data['depth'] < min_depth)]
                if len(data) == 0:
                    raise Exception('dataframe got zero len after removing out of target_area events.')
    return data



def set_spatial_id(data,column_identifier = None, pixel_scale = None,
                   kmeans_clusters = None,target_area = None,plot = True):
    '''
    This function will divide the region of interest into some sub-regions (e.g pixels or clusters)
    and set a spatial id for each sub-region based on the longitude, latitude, and depth.
    '''
    data = input_checking_set_spatial_id(data,column_identifier, pixel_scale,
                   kmeans_clusters ,target_area)
    # check if the user initially set a spatial id column:
    if 'spatial id' in data.columns:
        spatial_scaled_dataframe = data
        return spatial_scaled_dataframe
    # find the target area if it is not initiated by the user:
    if target_area is None:
        if kmeans_clusters is not None or len(pixel_scale ) == 2:
                target_area = find_target_area_from_dataframe(data,Pixelating_type.Two_diminsion)
        else:
                target_area = find_target_area_from_dataframe(data,Pixelating_type.Three_diminsion)
    if pixel_scale is not None:
        if len(pixel_scale) == 2:
            # 2D pixelating:
            spatial_scaled_dataframe = scale_dataframe_using_pixel_scale(data,pixel_scale,target_area,Pixelating_type.Two_diminsion,plot)
        else:
            # 3D pixelating:
            spatial_scaled_dataframe = scale_dataframe_using_pixel_scale(data,pixel_scale,target_area,Pixelating_type.Three_diminsion,plot)
        return spatial_scaled_dataframe
    if kmeans_clusters is not None:
       # seting spatial id with kmeans clustering:
       spatial_scaled_dataframe = scale_dataframe_using_kmeans_clusters(data,kmeans_clusters,plot,target_area)
       return spatial_scaled_dataframe