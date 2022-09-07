import enum
# Using enum class to define pixelating type
class Pixelating_type(enum.Enum):
   Two_diminsion = 1
   Three_diminsion = 2
def plot_map(data,target_area,save_address,pixel_scale =None,kmeans_model=None):
    # TO DO
    pass
def magnitude(data):
    # TO DO
    pass
def latitude(data):
    import numpy as np
    if not data['latitude'].dtype in [np.float64,np.int64]:
        raise Exception('Error: invalid type for \'latitude\' column in dataframe. expected types are int64 and float64')
    if len(data[(data['latitude'] < -90) | (data['latitude'] > 90)]) > 0:
        print('Warning: latitude are not in range [-90,90]. rows that are not in the range will be removed.')
        data = data[(data['latitude'] >= -90) & (data['latitude'] <= 90)]
    return data
def longitude(data):
    import numpy as np
    if not data['longitude'].dtype in [np.float64,np.int64]:
        raise Exception('Error: invalid type for \'longitude\' column in dataframe. expected types are int64 and float64')
    if len(data[(data['longitude'] < -180) | (data['longitude'] > 180)]) > 0:
        print('Warning: longitude are not in range [-180,180]. rows that are not in the range will be removed.')
        data = data[(data['longitude'] >= -180) & (data['longitude'] <= 180)]
def depth(data):
    import numpy as np
    if not data['depth'].dtype in [np.float64,np.int64]:
        raise Exception('Error: invalid type for \'depth\' column in dataframe. expected types are int64 and float64.')
    if len(data[data['depth'] < 0]) > 0:
        print('Warning: negative depth value is not acceptable. rows that their depths are negative will be removed.')
        data = data[data['depth'] >= 0]
def temporal_id(data):
    # TO DO
    pass
def check_data_columns(data,list_of_columns):
    for column in list_of_columns:
        if column == 'magnitude':
            magnitude(data)
        if column == 'longitude':
            longitude(data)
        if column == 'latitude':
            latitude(data)
        if column == 'depth':
            depth(data)
        if column == 'temporal id':
            temporal_id(data)
def scale_dataframe_using_pixel_scale(data, pixel_scale,target_area,pixelating_type):
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
        return spatial_scaled_data
def scale_dataframe_using_kmeans_clusters(data, kmeans_clusters):
    '''
    divide the region of interest into some set of clusters and then set a spatial id
    for each of the clusters in the spatial id column.
    '''
    from sklearn.cluster import KMeans
    kmeans = KMeans(kmeans_clusters)
    lon_and_lat = data['longitude','latitude']
    kmeans.fit(lon_and_lat)
    spatial_ids = kmeans.fit_predict(lon_and_lat)
    data['spatial id'] = spatial_ids+1
    spatial_scaled_data = data
    return spatial_scaled_data

def find_target_area_from_dataframe(data,pixelating_type):
    '''
    finding the maximum and minimum longitude and latitude in the data to estimate
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
                   kmeans_clusters ,target_area,verbose):
    '''checking all inputs of set_spatial_id function'''
    import pandas as pd
    # checking 'data' input:
    # type checking:
    if not isinstance(data,(pd.DataFrame,str)):
        raise Exception('Error: invalid type for \'data\' input. expected a pandas dataframe or str')
    if type(data) == str:
        data = pd.read_csv(data)
    else:
        data = data.copy()
    # checking 'pixel_scale' input:
    if pixel_scale is not None:
        # type checking 1:
        if not isinstance(pixel_scale,(list,tuple)):
            raise Exception('Error: invalid type for \'pixel_scale\' input. expected a list.')
        # type checking 2:
        if not all(isinstance(x, (int)) for x in pixel_scale) :
            raise Exception('Error: invalid type for \'pixel_scale\' input. not all elements in the list are int.')        
        # length checking:
        if not ((len(pixel_scale) == 2) or (len(pixel_scale) == 3)):
            raise Exception('Error: invalid length for \'pixel_scale\' input. expected lengths are 2 and 3.')
        # value checking:
        if not all([x > 0 for x in pixel_scale]):
            raise Exception('Error: invalid value for \'pixel_scale\' input. expected values are greater than zero.')
    # checking 'target_area' input:
    if target_area is not None:
        # type checking 1:
        if not isinstance(target_area,list):
            raise Exception('Error: invalid type for \'target_area\' input. expected a list.')
        # type checking 2:
        if not all(isinstance(x, (float,int)) for x in target_area):
            raise Exception('Error: invalid type for \'target_area\' input. not all elements are int or float')
        # length checking:
        if not ((len(target_area) == 4) or (len(target_area) == 6)):
            raise Exception('Error: invalid length for \'target_area\' input. expected lengths are 4 and 6.')
        # value checking:
        if len(target_area) == 4:
            if not (( -180 <= target_area[0] <= 180) and ( -180 <= target_area[1] <= 180)):
               raise Exception('Error: invalid value for \'target_area\' input. longitude should be in range [-180,180]')
            if not (( -90 <= target_area[2] <= 90) and ( -90 <= target_area[3] <= 90)):
               raise Exception('Error: invalid value for \'target_area\' input. latitude should be in range [-90,90]')
            if target_area[0] <= target_area[1]:
               raise Exception('Error: max longitude should be greater than min longitude') 
            if target_area[2] <= target_area[3]:
               raise Exception('Error: max latitude should be greater than min latitude') 
            if len(pixel_scale) != 2:
               raise Exception('Error: target_area should be in size six because pixel_scale is not in size 3')  
        if len(target_area) == 6:
            if not (( -180 <= target_area[0] <= 180) and ( -180 <= target_area[1] <= 180)):
               raise Exception('Error: invalid value for \'target_area\' input. longitude should be in range [-180,180]')
            if not (( -90 <= target_area[2] <= 90) and ( -90 <= target_area[3] <= 90)):
               raise Exception('Error: invalid value for \'target_area\' input. latitude should be in range [-90,90]')
            if not (( 0 <= target_area[4]) and ( 0 <= target_area[5])):
               raise Exception('Error: invalid value for \'target_area\' input. depth should be greater than or equal to zero.')
            if target_area[0] <= target_area[1]:
               raise Exception('Error: max longitude should be greater than min longitude') 
            if target_area[2] <= target_area[3]:
               raise Exception('Error: max latitude should be greater than min latitude') 
            if target_area[4] <= target_area[5]:
               raise Exception('Error: max depth should be greater than min depth')
            if pixel_scale is not None:
                if len(pixel_scale) != 3:
                    raise Exception('Error: target_area should be in size four because pixel_scale is in size 2')  
            
    # checking 'kmeans_cluster' input:
    if kmeans_clusters is not None:
        # type checking:
        if not isinstance(kmeans_clusters,int):
            raise Exception('Error: invalid type for \'kmeans_clusters\' input. expected int.')
        # value checking:
        if not (kmeans_clusters > 0):
            raise Exception('Error: invalid value for \'kmeans_clusters\' input. expected value is greater than zero.')
    # checking 'verbose' input:
    # type checking:
    if not isinstance(verbose,int):
        raise Exception('Error: invalid type for \'verbose\' input. expected int.')
    # value checking:
    if not (verbose in [0,1,2]):
        raise Exception('Error: invalid value for \'verbose\' input. expected values are 0,1,2')
    # checking 'column_identifier' input:
    if column_identifier is not None:
        # type checking:
        if not isinstance(column_identifier,dict):
            raise Exception('Error: invalid type for \'column_identifier\' input. expected dictionary.')

        # length checking:
        if not (len(column_identifier) > 0):
            raise Exception('Error: invalid length for \'column_identifier\' input. expected length is greater than zero.')
        # key checking:
        if not all([((x in column_identifier.keys()) or (x in data.columns)) for x in ['longitude','latitude'] ]):
            raise Exception('Error: not all keys(longitude and latitude) found in \'column_identifier\' input or in the dataframe.')
        # column checking:
        if not all([(column_identifier[x] in data.columns)  if (x in column_identifier.keys()) else (x in data.columns) for x in ['longitude','latitude'] ]):
            raise Exception('Error: not all columns specified in \'column_identifier\' input found in \'data\' input.')
        # renaming in dataframe.
        if 'longitude' in column_identifier.keys():
            data = data.rename(columns={column_identifier['longitude']:'longitude'})
        if 'latitude' in column_identifier.keys():
            data = data.rename(columns={column_identifier['latitude']:'latitude'})
        if pixel_scale is not None:
            if len(pixel_scale) == 3:
                # key checking:
                if not all([((x in column_identifier.keys()) or (x in data.columns)) for x in ['depth'] ]):
                    raise Exception('Error: not all keys (depth) found in \'column_identifier\' input.')
                if 'depth' in column_identifier.keys():
                    # column checking
                    if not (column_identifier['depth'] in data.columns):
                        raise Exception('Error: not all columns specified in \'column_identifier\' input found in \'data\' input.')
                    # renaming in dataframe.
                    data = data.rename(columns={column_identifier['depth']:'depth'})
        if (pixel_scale is None) and (kmeans_clusters is None):
                if not all([((x in column_identifier.keys()) or (x in data.columns)) for x in ['spatial id'] ]):
                    raise Exception('Error: not all keys (spatial id) found in \'column_identifier\' input.')
                if 'spatial id' in column_identifier.keys():
                    # column checking
                    if not (column_identifier['spatial id'] in data.columns):
                        raise Exception('Error: not all columns specified in \'column_identifier\' input found in \'data\' input.')
                    # renaming in dataframe.
                    data = data.rename(columns={column_identifier['spatial id']:'spatial id'})            
    else:
        if pixel_scale is not None:
            if len(pixel_scale) == 2:
                if not (('longitude' in data.columns) and ('latitude' in data.columns)):
                    raise Exception('Error: dataframe must contain longitude and latitude columns.')
            else:
                if not (('longitude' in data.columns) and ('latitude' in data.columns) and ('depth' in data.columns)):
                    raise Exception('Error: dataframe must contain longitude, latitude and depth columns.')
        if pixel_scale is None and kmeans_clusters is not None:
            if not (('longitude' in data.columns) and ('latitude' in data.columns)):
                raise Exception('Error: dataframe must contain longitude and latitude columns.')
        if (pixel_scale is None) and (kmeans_clusters is None):
            if not ('spatial id' in data.columns):
                raise Exception('Error: dataframe must contain spatial id column.')
    return data
def set_spatial_id(data,column_identifier = None, pixel_scale = None,
                   kmeans_clusters = None,target_area = None,plot = False,verbose = 0):
    '''
    This function will divide the region of interest into some pixels (sub-regions)
    and set a spatial ID for each pixel based on the longitude, latitude, and depth.
    '''
    data = input_checking_set_spatial_id(data,column_identifier, pixel_scale,
                   kmeans_clusters ,target_area,verbose)
    if column_identifier is  not None:
        if 'spatial id' in column_identifier.keys():
            if data['spatial id'].isnull().values.any():
                print('Warning: there are some NaN values in \'spatial id\' column. those rows will be removed from dataframe.')
                data = data[data['spatial id'].notna()]
                # check if dataframe is not len zero.
                if len(data) == 0:
                    raise Exception('dataframe got zero len after removing nan columns.')
            if column_identifier['spatial id'] in data.columns:
                spatial_scaled_dataframe = data.rename(columns = {column_identifier['spatial id']:'spatial id'})
                return spatial_scaled_dataframe
            else:
                column_name = column_identifier['spatial id']
                raise Exception(f'Error: there is no such column named \'{str(column_name)}\' in the dataframe')
        
    if pixel_scale is not None:
        if target_area is None:
            if len(pixel_scale ) == 2:
                    target_area = find_target_area_from_dataframe(data,Pixelating_type.Two_diminsion)
            else:
                    target_area = find_target_area_from_dataframe(data,Pixelating_type.Three_diminsion)
        if data['longitude'].isnull().values.any():
            print('Warning: there are some NaN values in the longitude column. those rows will be removed from dataframe.')
            data = data[data['longitude'].notna()]
            if len(data) == 0:
                raise Exception('dataframe got zero len after removing NaN columns.')
        if data['latitude'].isnull().values.any():
            print('Warning: there are some NaN values in the latitude column. those rows will be removed from dataframe.')
            data = data[data['latitude'].notna()]
            if len(data) == 0:
                raise Exception('dataframe got zero len after removing NaN columns.')
        check_data_columns(data,['longitude','latitude'])
        if len(pixel_scale) == 2:
            spatial_scaled_dataframe = scale_dataframe_using_pixel_scale(data,pixel_scale,target_area,Pixelating_type.Two_diminsion)
        else:
            if data['depth'].isnull().values.any():
                print('Warning: there are some NaN values in the depth column')
                data = data[data['depth'].notna()]
                if len(data) == 0:
                    raise Exception('dataframe got zero len after removing nan columns.')
            check_data_columns(data,['depth'])
            spatial_scaled_dataframe = scale_dataframe_using_pixel_scale(data,pixel_scale,target_area,Pixelating_type.Three_diminsion)
        return spatial_scaled_dataframe
    if kmeans_clusters is not None:
       spatial_scaled_dataframe = scale_dataframe_using_kmeans_clusters(data,kmeans_clusters)
       return spatial_scaled_dataframe