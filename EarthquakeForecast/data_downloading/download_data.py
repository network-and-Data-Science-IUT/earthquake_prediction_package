def input_checking_download_data(time_interval,save_dataframe,magnitude_range,depth_range,data_center,
                                 rectangular_region,circular_region):
    '''checking all inputs of download_data function'''
    from datetime import datetime
    # checking 'data_center' input:
    # type checking:
    if not isinstance(data_center,str):
        raise TypeError(f'Error: invalid type for \'data_center\'. expected type is str, got {type(data_center)}')
    # value checking:
    if not (data_center in ['USGS','IRIS','ISC']):
        raise ValueError('Error: invalid value for data_center. in this version we support [\'USGS\',\'IRIS\',\'ISC\'].')
    
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
    r1,r2 = [datetime.strptime(r,'%Y-%m-%d') for r in time_interval]
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
        if not ((-90 <= rectangular_region[0] < rectangular_region[1] <= 90) and (-180 <= rectangular_region[2] < rectangular_region[3] <= 180)):
            raise ValueError('Error: invalid value for \'rectangular_region\'.expected values are:\n'+
                             '-180 <= longitude <= +180, -90 <= latitude <= 90.')

    # checking 'circular_region' input:
    if circular_region is not None:
        # type checking:
        if not isinstance(circular_region,dict):
            raise TypeError(f'Error: invalid type for \'circular_region\'.expected type is dict, got {type(circular_region)}.')
        # key checking:
        if data_center == 'USGS':
          if not (('radius' in circular_region.keys()) and ('longitude' in circular_region.keys()) and ('latitude' in circular_region.keys())):
              raise KeyError('Error: invalid keys for \'circular_region\'. expected *radius*, *longitude* and *latitude* keys'+
              ' to be in the dictionary keys.')
        else:
          if not (('maxradius' in circular_region.keys()) and ('minradius' in circular_region.keys())  and ('longitude' in circular_region.keys()) and ('latitude' in circular_region.keys())):
              raise KeyError('Error: invalid keys for \'circular_region\'. expected *radius*, *longitude* and *latitude* keys'+
              ' to be in the dictionary keys.')          
        # value checking:
        if not ((-180 <= circular_region['longitude'] <= 180) and (-90 <= circular_region['latitude'] <= 90)):
            raise ValueError('Error: invalid value for \'circular_region\'.expected values are:\n'+
                             '-180 <= longitude <= +180, -90 <= latitude <= 90.')
        if data_center == 'USGS':
          if not (0 < circular_region['radius']):
              raise ValueError('Error: invalid value for \'circular_region\'.expected value is:\n'+
                              '0 < minradius < 180, 0 < maxradius <= 180.')
        else:
          if not ((0 < circular_region['minradius'] < 180) or (0 < circular_region['maxradius'] <= 180)):
              raise ValueError('Error: invalid value for \'circular_region\'.expected value is:\n'+
                              '0 < minradius < 180, 0 < maxradius <= 180.')
      
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
    from bs4 import BeautifulSoup
    from urllib.error import URLError
    import os
    input_checking_download_data(time_interval,save_dataframe,magnitude_range,depth_range,data_center,
                                rectangular_region,circular_region)
    
    # creating the url to download the data.
    if data_center == 'USGS':
        url = 'https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv'
    elif data_center == 'ISC':
        url = 'http://isc-mirror.iris.washington.edu/fdsnws/event/1/query?format=xml'
    elif data_center == 'IRIS':
        url = 'http://service.iris.edu/fdsnws/event/1/query?format=xml'
    url = url + '&starttime=' + time_interval[0] + '&endtime=' + time_interval[1]
    if magnitude_range is not None:
        url = url + '&minmagnitude=' + str(magnitude_range[0]) + '&maxmagnitude=' + str(magnitude_range[1])
    if depth_range is not None:
        url = url + '&mindepth=' + str(depth_range[0]) + '&maxdepth=' + str(depth_range[1])
    
    geo_info = ''
    if rectangular_region is not None:
        url = url + '&minlatitude=' + str(rectangular_region[0]) + '&maxlatitude=' + str(rectangular_region[1]) + '&minlongitude=' + str(rectangular_region[2]) + \
                '&maxlongitude=' + str(rectangular_region[3])
        geo_info = '&minlatitude=' + str(rectangular_region[0]) + '&maxlatitude=' + str(rectangular_region[1]) + '&minlongitude=' + str(rectangular_region[2]) + \
                '&maxlongitude=' + str(rectangular_region[3])
    elif circular_region is not None:
        url = url + '&longitude=' +str( circular_region['longitude']) + \
                    '&latitude=' + str(circular_region['latitude'])
        if data_center == 'USGS':
            url = url + '&maxradiuskm=' + str(circular_region['radius'])
            geo_info = '&maxradiuskm=' + str(circular_region['radius'])
        if data_center == 'ISC' or data_center == 'IRIS':
            url = url + '&maxradius=' + str(circular_region['maxradius']) + '&minradius=' + str(circular_region['minradius'])
            geo_info = '&maxradius=' + str(circular_region['maxradius']) + '&minradius=' + str(circular_region['minradius'])
    try:         
      downloaded_url = urlopen(url)
    except URLError as e:
      print(f"An error occurred: {e.reason}")
      print('You can try other datacenters or change the input parameters.')
      return

    if data_center == 'USGS':
        downloaded_data = pd.read_csv(downloaded_url)
        downloaded_data.rename({'mag':'magnitude'},axis=1,inplace=True)
    else:
        bs_data = BeautifulSoup(downloaded_url, "xml")
        events = bs_data.find_all('event')
        event_records = []
        for event in events:
            try:
              event_records.append({'time':event.find('time').find('value').text,
                           'latitude':float(event.find('latitude').find('value').text) ,
                           'longitude':float(event.find('longitude').find('value').text),
                           'depth':float(event.find('depth').find('value').text),
                           'magnitude':float(event.find('mag').find('value').text)})
            except:
              # some records do not have of the of the columns so we do not add them to the output dataset.
              pass
        downloaded_data = pd.DataFrame.from_records(event_records)
    if save_dataframe:
        # create a directory to save the data
        if not os.path.exists('./data'):
                os.makedirs('./data')  
        downloaded_data.to_csv('./data/'+'eq_data'+geo_info+'&s='+time_interval[0]+'&e='+time_interval[1]+'&dc='+data_center+'.csv')
        print('dataframe saved successfully!')
    return downloaded_data