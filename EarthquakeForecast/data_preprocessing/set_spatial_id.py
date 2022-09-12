import enum



# Using enum class to define pixelating type
class Pixelating_type(enum.Enum):
   Two_diminsion = 1
   Three_diminsion = 2



def plot_map(data,target_area=None,pixel_scale =None,pixelatin_type = None,kmeans=None):
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial import Voronoi, voronoi_plot_2d
    if pixel_scale is not None:
        if pixelatin_type == Pixelating_type.Two_diminsion:
            max_lon,min_lon,max_lat,min_lat = target_area
            x_pixels,y_pixels = pixel_scale
            vlines = list()
            step = (max_lon - min_lon)/x_pixels
            for i in range(1,x_pixels):
                vlines.append(min_lon+i*step)
            hlines = list()
            step = (max_lat - min_lat)/y_pixels
            for i in range(1,y_pixels):
                hlines.append(min_lat+i*step)
            x_s = [min_lon] + vlines + [max_lon]
            x_s = [(x_s[i] + x_s[i+1])/2 for i in range(0,len(x_s)-1)]
            y_s = [min_lat] + hlines + [max_lat]
            y_s = [(y_s[i] + y_s[i+1])/2 for i in range(0,len(y_s)-1)]
            position_of_spatial_ids = list()
            for j in y_s:
                for i in x_s:
                    position_of_spatial_ids.append((i,j))
            fig, ax = plt.subplots()
            ax.set_xlim(min_lon, max_lon)
            ax.set_ylim(min_lat,max_lat)
            ax.set_box_aspect(1)
            for i in hlines:
                plt.axhline(y=i)
            for i in vlines:
                plt.axvline(x=i)
            for i,pos in enumerate(position_of_spatial_ids):
                plt.text(pos[0], pos[1], str(i+1), color="black", fontsize=12)
            plt.scatter(x = data['longitude'],y=data['latitude'],s=2)
            ax.set_xticks([min_lon]+vlines+[max_lon])
            ax.set_yticks([min_lat]+ hlines + [max_lat])
            plt.xlabel('longitude')
            plt.ylabel('latitude')
            path = './ploting'
            is_exist = os.path.exists(path)
            if not is_exist: 
                os.makedirs(path)
            fig.savefig('./ploting/2d_pixelating_map.png')
            return
        if pixelatin_type == Pixelating_type.Three_diminsion:
            max_lon,min_lon,max_lat,min_lat,max_depth,min_depth = target_area
            x_pixels,y_pixels,z_pixels = pixel_scale
            zlines = list()
            step = (max_depth - min_depth)/z_pixels
            for i in range(1,z_pixels):
                zlines.append(min_depth+i*step)
            vlines = list()
            step = (max_lon - min_lon)/x_pixels
            for i in range(1,x_pixels):
                vlines.append(min_lon+i*step)
            hlines = list()
            step = (max_lat - min_lat)/y_pixels
            for i in range(1,y_pixels):
                hlines.append(min_lat+i*step)
            x_s = [min_lon] + vlines + [max_lon]
            x_s = [(x_s[i] + x_s[i+1])/2 for i in range(0,len(x_s)-1)]
            y_s = [min_lat] + hlines + [max_lat]
            y_s = [(y_s[i] + y_s[i+1])/2 for i in range(0,len(y_s)-1)]
            z_s = [min_depth] + zlines + [max_depth]
            position_of_spatial_ids = list()
            for j in y_s:
                for i in x_s:
                    position_of_spatial_ids.append((i,j))
            depth_level = 0
            for k in range(len(z_s)-1):
                fig, ax = plt.subplots()
                ax.set_xlim(min_lon, max_lon)
                ax.set_ylim(min_lat,max_lat)
                ax.set_box_aspect(1)
                for i in hlines:
                    plt.axhline(y=i)
                for i in vlines:
                    plt.axvline(x=i)
                for i,pos in enumerate(position_of_spatial_ids):
                    plt.text(pos[0], pos[1], str(i+1+depth_level*x_pixels*y_pixels), color="black", fontsize=12)
                plt.scatter(x = data[(data['depth'] >= z_s[k]) & (data['depth'] <= z_s[k+1])]['longitude'],y=data[(data['depth'] <= z_s[k+1])& (data['depth'] >= z_s[k])]['latitude'],s=2)
                ax.set_xticks([min_lon]+vlines+[max_lon])
                ax.set_yticks([min_lat]+ hlines + [max_lat])
                plt.xlabel('longitude')
                plt.ylabel('latitude')
                plt.title(f'depth:{z_s[k]}-{z_s[k+1]}')
                path = './ploting'
                is_exist = os.path.exists(path)
                if not is_exist: 
                    os.makedirs(path)
                fig.savefig(f'./ploting/2d_pixelating_map_depth_{z_s[k]}-{z_s[k+1]}.png')
                depth_level += 1
            return
    if kmeans is not None:
        spatial_ids = np.unique(data['spatial ID'])
        centroids = kmeans.cluster_centers_
        vor = Voronoi(centroids)
        fig = voronoi_plot_2d(vor)
        for i in spatial_ids:
            plt.scatter((data[data['spatial ID'] == i])['longitude'] , (data[data['spatial ID'] == i])['latitude'] , label = i,s=2)
        for i,pos in enumerate(centroids):
            plt.text(pos[0], pos[1], str(i+1), color="black", fontsize=12)
        path = './ploting'
        is_exist = os.path.exists(path)
        if not is_exist: 
            os.makedirs(path)
        fig.savefig('./ploting/kmeans_map.png')
        return
            
def magnitude(data):
    # changing the order first check nan
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



def latitude(data):
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



def longitude(data):
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



def depth(data):
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



def temporal_id(data):
    import numpy as np
    if not data['temporal ID'].dtype in [np.int64]:
        raise Exception('Error: invalid type for \'temporal ID\' column in dataframe. expected type is int64.')
    if data['temporal ID'].isnull().values.any():
        print('Warning: there are some NaN values in the temporal ID column. those rows will be removed from dataframe.')
        data = data[data['temporal ID'].notna()]
        if len(data) == 0:
            raise Exception('dataframe got zero len after removing NaN rows from temporal ID.')
    return data




def spatial_id(data):
    import numpy as np
    if data['spatial ID'].isnull().values.any():
        print('Warning: there are some NaN values in the spatial ID column. those rows will be removed from dataframe.')
        data = data[data['spatial ID'].notna()]
        if len(data) == 0:
            raise Exception('dataframe got zero len after removing NaN rows from spatial ID.')
    return data



def check_data_columns(data,list_of_columns):
    for column in list_of_columns:
        if column == 'magnitude':
            data = magnitude(data)
        if column == 'longitude':
            data = longitude(data)
        if column == 'latitude':
            data = latitude(data)
        if column == 'depth':
            data = depth(data)
        if column == 'temporal id':
            data = temporal_id(data)
        if column == 'spatial ID':
            data = spatial_id(data)
    return data



def scale_dataframe_using_pixel_scale(data, pixel_scale,target_area,pixelating_type,plot):
    '''
    divide the region of interest into some set of pixels and then set a spatial ID
    for each of the pixels in the spatial ID column.
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
        spatial_scaled_data['spatial ID'] = index.astype(np.int64)
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
        spatial_scaled_data['spatial ID'] = index.astype(np.int64)
        # ploting
        if plot:
            plot_map(data=data,pixel_scale=pixel_scale,target_area=target_area,pixelatin_type=Pixelating_type.Three_diminsion)
        return spatial_scaled_data
    


def scale_dataframe_using_kmeans_clusters(data, kmeans_clusters, plot):
    '''
    divide the region of interest into some set of clusters and then set a spatial ID
    for each of the clusters in the spatial ID column.
    '''
    from sklearn.cluster import KMeans
    kmeans = KMeans(kmeans_clusters)
    lon_and_lat = data[['longitude','latitude']]
    kmeans.fit(lon_and_lat)
    spatial_ids = kmeans.fit_predict(lon_and_lat)
    data['spatial ID'] = spatial_ids+1
    spatial_scaled_data = data
    # ploting
    if plot:
        plot_map(data=data,kmeans=kmeans)
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
                   kmeans_clusters ,target_area,verbose):
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
    # checking 'verbose' input:
    # type checking:
    if not isinstance(verbose,int):
        raise TypeError('Error: invalid type for \'verbose\' input. expected int.')
    # value checking:
    if not (verbose in [0,1,2]):
        raise ValueError('Error: invalid value for \'verbose\' input. expected values are 0,1,2')
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
            if not all([((x in column_identifier.keys()) or (x in data.columns)) for x in ['spatial ID'] ]):
              raise Exception('Error: not all keys(spatial ID) found in \'column_identifier\' input or in the dataframe.')
            # column checking:
            if not all([True if (x in data.columns) else (x in column_identifier.keys()) for x in ['spatial ID'] ]):
                raise Exception('Error: not all columns specified in \'column_identifier\' input found in \'data\' input.')
            # renaming in dataframe:
            if not ('spatial ID' in data.columns):
                data = data.rename(columns={column_identifier['spatial ID']:'spatial ID'})            
    else:
        if (pixel_scale is not None) or (kmeans_clusters is not None):
            if not (('longitude' in data.columns) and ('latitude' in data.columns)):
                raise Exception('Error: dataframe must contain longitude and latitude columns.')
            if pixel_scale is not None:
                if len(pixel_scale) == 3:
                    if not ('depth' in data.columns):
                        raise Exception('Error: dataframe must contain depth column.')
        else:
            if not ('spatial ID' in data.columns):
                raise Exception('Error: dataframe must contain spatial ID column.')
    # checking the content of 'data' input columns:
    if (pixel_scale is not None) or (kmeans_clusters is not None):
        data = check_data_columns(data,['longitude','latitude'])
        if pixel_scale is not None:
            if len(pixel_scale) == 3:
                data = check_data_columns(data,['depth'])
    else:
        data = check_data_columns(data,['spatial ID'])
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
                   kmeans_clusters = None,target_area = None,plot = False,verbose = 0):
    '''
    This function will divide the region of interest into some pixels (sub-regions)
    and set a spatial ID for each pixel based on the longitude, latitude, and depth.
    '''
    data = input_checking_set_spatial_id(data,column_identifier, pixel_scale,
                   kmeans_clusters ,target_area,verbose)

    if 'spatial ID' in data.columns:
        spatial_scaled_dataframe = data
        return spatial_scaled_dataframe
    if pixel_scale is not None:
        if target_area is None:
            if len(pixel_scale ) == 2:
                    target_area = find_target_area_from_dataframe(data,Pixelating_type.Two_diminsion)
            else:
                    target_area = find_target_area_from_dataframe(data,Pixelating_type.Three_diminsion)

        if len(pixel_scale) == 2:
            spatial_scaled_dataframe = scale_dataframe_using_pixel_scale(data,pixel_scale,target_area,Pixelating_type.Two_diminsion,plot)
        else:
            spatial_scaled_dataframe = scale_dataframe_using_pixel_scale(data,pixel_scale,target_area,Pixelating_type.Three_diminsion,plot)
        return spatial_scaled_dataframe
    if kmeans_clusters is not None:
       spatial_scaled_dataframe = scale_dataframe_using_kmeans_clusters(data,kmeans_clusters,plot)
       return spatial_scaled_dataframe