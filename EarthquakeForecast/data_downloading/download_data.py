def input_checking_download_data(time_interval,save_dataframe,magnitude_range,depth_range,data_center,
                                 rectangular_region,circular_region):
    '''checking all inputs of download_data function'''
    from datetime import datetime
    # checking 'data_center' input:
    # type checking:
    if not isinstance(data_center,str):
        raise TypeError(f'Error: invalid type for \'data_center\'. expected type is str, got {type(data_center)}')
    # value checking:
    if not (data_center in ['USGS']):
        raise ValueError('Error: invalid value for data_center. in this version only \'USGS\' is available')
    
    # checking 'time_interval' input:
     # type checking 1:
    if not isinstance(time_interval,(list,tuple)):
        raise TypeError(f'Error: invalid type for \'time_interval\'. expected types are list and tuple, got {type(time_interval)}')
    # length checking:
    if len(time_interval) != 2:
        raise Exception('Error: invalid length for \'time_interval\'. expected length is two.')
    # type checking 2:
    if not (isinstance(time_interval[0],str) and isinstance(time_interval[1],str)):
        raise TypeError('Error: invalid type for \'time_interval\'. expected all elements to be str type.')
    # value checking:
    r1,r2 = [datetime.strptime(r,'%Y/%m/%d') for r in time_interval]
    if not (r1 < r2):
        raise ValueError('Error: invalid value for \'time_interval\'. expected first element to be less than' +
        ' second element')



    # checking 'rectangular_region' input:
    if rectangular_region is not None:
        # type checking 1:
        if not isinstance(rectangular_region,(list,tuple)):
            raise TypeError('Error: invalid type for \'rectangular_region\'. expected types are list and tuple, got' + 
            f' {type(rectangular_region)}')
        # type checking 2:
        if not all([isinstance(coor,(int,float)) for coor in rectangular_region]):
            raise TypeError('Error: invalid type for \'rectangular_region\'. expected all elements to be int or float.')
        # length checking:
        if len(rectangular_region) != 4:
            raise Exception('Error: invalid length for \'rectangular_region\'. expected length is four.')
        # value checking:
        if not ((-180 <= rectangular_region[0] < rectangular_region[1] <= 180) and (-90 <= rectangular_region[2] < rectangular_region[3] <= 90)):
            raise ValueError('Error: invalid value for \'rectangular_region\'.expected values are:\n'+
                             '-180 <= longitude <= +180, -90 <= latitude <= 90.')

    # checking 'circular_region' input:
    if circular_region is not None:
        # type checking:
        if not isinstance(circular_region,dict):
            raise TypeError(f'Error: invalid type for \'circular_region\'.expected type is dict, got {type(circular_region)}.')
        # key checking:
        if not (('radius' in circular_region.keys()) and ('longitude' in circular_region.keys()) and ('latitude' in circular_region.keys())):
            raise KeyError('Error: invalid keys for \'circular_region\'. expected *radius*, *longitude* and *latitude* keys'+
            ' to be in the dictionary keys.')
        # value checking:
        if not ((-180 <= circular_region['longitude'] <= 180) and (-90 <= circular_region['latitude'] <= 90)):
            raise ValueError('Error: invalid value for \'circular_region\'.expected values are:\n'+
                             '-180 <= longitude <= +180, -90 <= latitude <= 90.')
        if not (0 < circular_region['radius']):
            raise ValueError('Error: invalid value for \'circular_region\'.expected value is:\n'+
                             '0 < radius.')
    
    # either 'circular_region' or 'rectangular_region' must be initiated.
    if (circular_region is None) and (rectangular_region is None):
        raise Exception('Error: either \'circular_region\' or \'rectangular_region\' must be initiated.')
    
    # checking 'magnitude_range' input:
    if magnitude_range is not None:
        # type checking 1:
        if not isinstance(magnitude_range,(list,tuple)):
            raise TypeError(f'Error: invalid type for \'magnitude_range\'. expected types are list and tuple, got {type(magnitude_range)}')
        # type checking 2:
        if not all([isinstance(r,(int,float)) for r in magnitude_range]):
            raise TypeError('Error: invalid type for \'magnitude_range\'. expected all elements to be int or float.')
        # length checking:
        if len(magnitude_range) != 2:
            raise Exception('Error: invalid length for \'magnitude_range\'. expected length is 2')
        # value checing:
        if not (0 <= magnitude_range[0] < magnitude_range[1]):
            raise ValueError('Error: invalid value for \'magnitude_range\'. expected value is:\n' + 
            '0 <= magnitude_range[0] < magnitude_range[1]')

    # checking 'depth_range' input:
    if depth_range is not None:
        # type checking 1:
        if not isinstance(depth_range,(list,tuple)):
            raise TypeError(f'Error: invalid type for \'depth_range\'. expected types are list and tuple, got {type(depth_range)}')
        # type checking 2:
        if not all([isinstance(r,(int,float)) for r in depth_range]):
            raise TypeError('Error: invalid type for \'depth_range\'. expected all elements to be int or float.')
        # length checking:
        if len(depth_range) != 2:
            raise Exception('Error: invalid length for \'depth_range\'. expected length is two.')
        # value checing:
        if not (0 <= depth_range[0] < depth_range[1]):
            raise ValueError('Error: invalid length for \'depth_range\'. expected value is:\n' +
            '0 <= depth_range[0] < depth_range[1]')

    # checking 'save_address' input:
    # type checking:
    if not isinstance(save_dataframe,bool):
        raise TypeError(f'Error: invalid type for \'save_address\'. expected type is bool, got {type(save_dataframe)}')



def download_data(time_interval,save_dataframe = False,magnitude_range=None,depth_range=None,data_center='USGS',
                  rectangular_region=None,circular_region=None):
    '''download the earthquake data from a data center based on the
       time interval and the geographical locations defined by user input'''
    from urllib.request import urlopen
    import pandas as pd

    input_checking_download_data(time_interval,save_dataframe,magnitude_range,depth_range,data_center,
                                rectangular_region,circular_region)

    if data_center == 'USGS':
        url = 'https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv'
        url = url + '&starttime=' + time_interval[0] + '&endtime=' + time_interval[1]
        if magnitude_range is not None:
            url = url + '&minmagnitude=' + str(magnitude_range[0]) + '&maxmagnitude=' + str(magnitude_range[1])
        if depth_range is not None:
            url = url + '&mindepth=' + str(depth_range[0]) + '&maxdepth=' + str(depth_range[1])
        if rectangular_region is not None:
            url = url + '&minlatitude=' + str(rectangular_region[0]) + '&maxlatitude=' + str(rectangular_region[1]) + '&minlongitude=' + str(rectangular_region[2]) + \
                    '&maxlongitude=' + str(rectangular_region[3])
        elif circular_region is not None:
            url = url + '&maxradiuskm=' + str(circular_region['radius'])+ '&longitude=' +str( circular_region['longitude']) + \
                 '&latitude=' + str(circular_region['latitude'])

        csv_url_rect = urlopen(url)
        downloaded_data = pd.read_csv(csv_url_rect)
        if save_dataframe:
           downloaded_data.to_csv('earthquake_data.csv')
        return downloaded_data