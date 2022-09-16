import configurations



def check_magnitude(data):
    import numpy as np
    if not data['magnitude'].dtype in [np.float64,np.int64]:
        raise Exception('Error: invalid type for \'magnitude\' column in dataframe. expected types are int64 and float64.')
    if data['magnitude'].isnull().values.any():
        print('Warning: there are some NaN values in the magnitude column. those will be removed from dataframe')
        data = data[data['magnitude'].notna()]
    if len(data[data['magnitude'] < 0]) > 0:
        print('Warning: negative magnitude value is not acceptable. rows that their magnitude are negative will be removed.')
        data = data[data['magnitude'] >= 0]
        if len(data) == 0:
            raise Exception('dataframe got zero len after removing none-valid magnitudes.')
    return data



def check_temporal_id(data):
    import numpy as np
    if not data['temporal ID'].dtype in [np.int64]:
        raise Exception('Error: invalid type for \'temporal ID\' column in dataframe. expected type is int64.')
    if data['temporal ID'].isnull().values.any():
        print('Warning: there are some NaN values in the temporal ID column. those rows will be removed from dataframe.')
        data = data[data['temporal ID'].notna()]
        if len(data) == 0:
            raise Exception('dataframe got zero len after removing NaN rows from temporal ID.')
    return data



def check_spatial_id(data):
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
            data = check_magnitude(data)
        if column == 'temporal id':
            data = check_temporal_id(data)
        if column == 'spatial ID':
            data = check_spatial_id(data)
    return data



def event_frequency(data):
    '''this function calculate a feature that specifes the number of events in a specific
       spatial & temporal ID'''
    import pandas as pd
    extracted_feature_temporal = pd.DataFrame(columns = ['spatial ID', 'temporal ID', 'event frequency'])
    for spatial_id in data['spatial ID'].unique():
        for temporal_id in data[data['spatial ID'] == spatial_id]['temporal ID'].unique():
            partial_data = data[(data['spatial ID'] == spatial_id) & (data['temporal ID'] == temporal_id)]
            extracted_feature_temporal = extracted_feature_temporal.append({'spatial ID' : spatial_id, 
            'temporal ID' : temporal_id,'event frequency' : len(partial_data)},ignore_index = True)
    return extracted_feature_temporal



def b_value(data):
    '''this function calculate b value as a spatial feature using maximum likelihood technique
       for 50 nearst events to the center of all earthquake events in a specific spatial ID'''
    import pandas as pd
    import numpy as np
    extracted_feature_spatial = pd.DataFrame(columns=['spatial ID','b value'])
    # centroid = center of all earthquake events.
    centroids = data.groupby('spatial ID')[['longitude','latitude']].mean().rename(columns= \
    {'longitude':'cen_longitude','latitude':'cen_latitude'})
    for i in range(len(centroids)):
        # calculate the distance between all events and the centroid.
        data['distance'] = np.linalg.norm(data[['longitude','latitude']].values - \
        centroids.iloc[i].values,axis = 1)
        # sort based on distance to find 50 nearst events.
        data = data.sort_values(by='distance',axis=0)
        magnitudes = data[:min(50,len(data))]['magnitude']
        extracted_feature_spatial = extracted_feature_spatial.append({'spatial ID':centroids.iloc[i].name,'b value': \
        # maximum likelihood formula: b_value = log(e)/(m_mean-m_min).
        np.log10(np.e)/(np.mean(magnitudes) -np.min(magnitudes))},ignore_index=True)
    return extracted_feature_spatial



def delta_time(data,magnitude_threshold):
    '''this function calculates a feature that specifies the average temporal unit diffrence
       between earthquake events with magnitude larger than a threshold('magnitude_threshold' input)'''
    import pandas as pd
    extracted_feature_spatial = pd.DataFrame(columns = ['spatial ID', 'delta time'])
    for spatial_id in data['spatial ID'].unique():
        data_with_specific_spatial_id = data[data['spatial ID'] == spatial_id]
        temporal_ids_of_main_events = data_with_specific_spatial_id[data_with_specific_spatial_id['magnitude'] >= \
                        magnitude_threshold]['temporal ID']
        delta_time = (temporal_ids_of_main_events.diff()+1)[1:]
        mean_delta_time = delta_time.mean()
        extracted_feature_spatial = extracted_feature_spatial.append({'spatial ID':spatial_id,
        'delta time':mean_delta_time},ignore_index=True)
    # if there is no event with magnitude greater than 'magnitude_threshold'
    # the value of the feature for that spatial ID will be the maximum value of temporal ID.
    # so we fill the NaN's (it happens when there is no event with magnitude greater than 
    # threshold) with the maximum temporal ID value.
    extracted_feature_spatial = extracted_feature_spatial.fillna(data['temporal ID'].max())
    return extracted_feature_spatial




def total_energy(data):
    '''this function calculates a feature that specifies total energy released in a specific 
       spatial ID and temporal ID.'''
    import pandas as pd
    extracted_feature_temporal = pd.DataFrame(columns = ['spatial ID', 'temporal ID', 'total energy'])
    for spatial_id in data['spatial ID'].unique():
        for temporal_id in data[data['spatial ID'] == spatial_id]['temporal ID'].unique():

            data_with_specific_temporal_spatial_id = data[(data['spatial ID'] == spatial_id) & (data['temporal ID'] == temporal_id)]
            extracted_feature_temporal = extracted_feature_temporal.append({'spatial ID' : spatial_id, 
            'temporal ID' : temporal_id,'total energy' : \
             # formula: energy = 10**(1.44*magnitude(richter) + 5.24)
             (10**(1.44*data_with_specific_temporal_spatial_id['magnitude'] + 5.24)).sum()},
            ignore_index = True)
    return extracted_feature_temporal



def input_checking_extract_feature(data,column_identifier,feature_list):
    '''checking all inputs of set_spatial_id function and
       renaming \'data\' input columns to standard format.'''
    import pandas as pd
    # checking 'data' input:
    # type checking:
    if not isinstance(data,(pd.DataFrame,str)):
        raise TypeError('Error: invalid type for \'data\' input. expected a pandas dataframe or str')
    if isinstance(data,str):
        data = pd.read_csv(data)
    else:
        data = data.copy()
    # checking 'feature_list' input:
    # type checking 1:
    if not isinstance(feature_list,(list,tuple)):
       raise TypeError('Error: invalid type for \'feature_list\' input. expected list or tuple') 
    # type checking 2:
    if not all(isinstance(feature,str) for feature in feature_list):
       raise TypeError('Error: invalid type for \'feature_list\' input. elements are expected to be string type')    
    # value checking 1:
    if not all([(feature in configurations.FEATURES or feature.startswith('delta time')) for feature in feature_list]):
        raise ValueError('Error: invalid value for \'feature_list\' input. name of the feature is not in feature list\n'+ \
        f'feature list = {configurations.FEATURES}')
    # value checking 2:
    for feature in feature_list:
        if feature.startswith('delta time'):
            try:
                float(feature[len('delta time')+1:])
            except:
                raise ValueError('Error: an error occured while trying to convert the hyperparameter of delta time to a float.expected format: \'delta time (a float number)\'')
    # checking column_identifier:
    if column_identifier is not None:
        # type checking:
        if not isinstance(column_identifier,dict):
            raise Exception('Error: invalid type for \'column_identifier\' input. expected dictionary.')
        # length checking:
        if not (len(column_identifier) > 0):
            raise Exception('Error: invalid length for \'column_identifier\' input. expected length is greater than zero.')
        # key checking:
        if not all([((x in column_identifier.keys()) or (x in data.columns)) for x in ['spatial ID','temporal ID']]):
            raise Exception('Error: not all keys(\'spatial ID\' and \'temporal ID\') found in \'column_identifier\' input or in the dataframe.')
        # column checking:
        if not all([True if (x in data.columns) else (column_identifier[x] in data.columns) for x in ['spatial ID','temporal ID'] ]):
            raise Exception('Error: not all columns specified in \'column_identifier\' input found in \'data\' input.')
        # renaming in dataframe.
        if not ('spatial ID' in data.columns):
            data = data.rename(columns={column_identifier['spatial ID']:'spatial ID'})
        if not ('temporal ID' in data.columns):
            data = data.rename(columns={column_identifier['temporal ID']:'temporal ID'})
        if any([feature in feature_list for feature in ['total energy',
        'b value']]) or any([feature.startswith('delta time') for feature in feature_list]) :
            # key checking:
            if not (('magnitude' in column_identifier.keys()) or ('magnitude' in data.columns)):
                raise Exception('Error: not all keys(\'magnitude\') found in \'column_identifier\' input or in the dataframe.')
            # column checking:
            if not (True if ('magnitude' in data.columns) else (column_identifier['magnitude'] in data.columns)):
                raise Exception('Error: not all columns specified in \'column_identifier\' input found in \'data\' input.')
            # renaming in dataframe.
            if not ('magnitude' in data.columns):
                data = data.rename(columns={column_identifier['magnitude']:'magnitude'})
    else:
        # checking columns exist in dataframe:
        if not all([(x in data.columns) for x in ['spatial ID','temporal ID']]):
            raise Exception('Error: not all columns(\'spatial ID\' and \'temporal ID\') found in \'data\' input.')
        if any([feature in feature_list for feature in ['total energy',
        'b value']]) or any([feature.startswith('delta time') for feature in feature_list]):
            # checking column exists in dataframe:
            if not (('magnitude' in data.columns) or ('magnitude' in data.columns)):
                raise Exception('Error: not all colum(\'magnitude\') found in \'data\' input.')
    # checking the content of 'data' input columns:
    if 'event frequency' in feature_list:
        data = check_data_columns(data,['temporal ID','spatial ID'])
    if any([feature in feature_list for feature in ['total energy',
        'b value']]) or any([feature.startswith('delta time') for feature in feature_list]):
        data = check_data_columns(data,['temporal ID','spatial ID','magnitude'])
    return data



def extract_feature(data,feature_list,column_identifier=None):
    '''extract extra spatial or temporal features to enrich feature list.'''
    import pandas as pd
    import numpy as np

    data = input_checking_extract_feature(data,column_identifier,feature_list)
    unique_spatial_ids = data['spatial ID'].unique()
    spatial_id_temporal_id = np.array([(spatial_id,temporal_id) for spatial_id in data['spatial ID'].unique() for temporal_id in data[data['spatial ID'] == spatial_id]['temporal ID'].unique()])
    extracted_features_temporal = pd.DataFrame({'spatial ID':spatial_id_temporal_id[:,0],'temporal ID' : spatial_id_temporal_id[:,1]})
    extracted_features_spatial = pd.DataFrame({'spatial ID':unique_spatial_ids})

    for feature in feature_list:

        if feature == 'event frequency':
            extracted_features_temporal['event frequency'] = event_frequency(data)['event frequency']

        if feature.startswith('delta time'):
            extracted_features_spatial['delta time'] = delta_time(data,float(feature[len('delta time')+1:]))['delta time']

        if feature == 'total energy':
            extracted_features_temporal['total energy'] = total_energy(data)['total energy']

        if feature == 'b value':
            extracted_features_spatial['b value'] = b_value(data)['b value']

    return extracted_features_temporal,extracted_features_spatial