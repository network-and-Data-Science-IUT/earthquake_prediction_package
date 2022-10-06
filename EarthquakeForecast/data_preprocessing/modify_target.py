import configurations



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
        if column == 'temporal ID':
            data = check_temporal_id(data)
        if column == 'spatial ID':
            data = check_spatial_id(data)
        if column == 'target':
            data = check_target(data)
    return data



def input_checking_modify_target(data,target_mode,class_boundaries,column_identifier):
    '''checking all inputs of modify_target function and
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
    # checking 'target_mode' input:
    if target_mode is not None:
        # type checking:
        if not (isinstance(target_mode,str)):
            raise TypeError('Error: invalid type for \'target_mode\' input.')
        # value checking:
        if not ((target_mode in  configurations.TARGET_MODES)):
            if target_mode.startswith('moving average '):
                try:
                    int(target_mode[len('moving average '):])
                except:
                    raise TypeError(f'Error: expected an integer after \'moving average\'')
            else:
                raise TypeError(f'Error: invalid value for \'target_mode\' input. expected values are {configurations.TARGET_MODES}.')
        # checking 'class_boundaries' input:
        if target_mode == 'classify':
            # type checking 1:
            if not isinstance(class_boundaries,(list,tuple)):
                raise TypeError('Error: invalid type for \'class_boundaries\' input. expected a list or tuple.')
            # type checking 2:
            if not all([isinstance(bound,(int,float)) for bound in class_boundaries]):
                raise TypeError('Error: invalid type for \'class_boundaries\' input. expected all elements in \'class_boundaries\'' + \
                    " to be int or float.")
            # length checking:
            if not (len(class_boundaries) > 0):
                raise TypeError('Error: invalid length for \'class_boundaries\' input. expected length is greater than zero.')        
            # value checking:
            if not all([(class_boundaries[i] < class_boundaries[i+1]) for i in range(len(class_boundaries)-1)]):
                raise TypeError('Error: invalid value for \'class_boundaries\' input. expected elements in \'class_boundaries\'' + \
                    " to be in increasing order.")
    # checking column_identifier input:
    if column_identifier is not None:
        # type checking:
        if not isinstance(column_identifier,dict):
            raise TypeError('Error: invalid type for \'column_identifier\' input. expected dictionary.')
        # length checking:
        if not (len(column_identifier) > 0):
            raise Exception('Error: invalid length for \'column_identifier\' input. expected length is greater than zero.')
        if not ('target' in data.columns):
            # key checking:
            if not ('target' in column_identifier.keys()):
                raise Exception('Error: \'target\' key not found in \'column_identifier\' keys.')
            # column checking:
            if not (column_identifier['target'] in data.columns):
                column_name = column_identifier['target']
                raise Exception(f'Error: there is no column named {column_name} in \'data\' columns.')
            # renaming in data:
            data = data.rename(columns={column_identifier['target']:'target'})
        if not ('temporal ID' in data.columns):
            # key checking:
            if not ('temporal ID' in column_identifier.keys()):
                raise Exception('Error: \'temporal ID\' key not found in \'column_identifier\' keys.')
            # column checking:
            if not (column_identifier['temporal ID'] in data.columns):
                column_name = column_identifier['temporal ID']
                raise Exception(f'Error: there is no column named {column_name} in \'data\' columns.')
            # renaming in data:
            data = data.rename(columns={column_identifier['temporal ID']:'temporal ID'})
    else:
        if not (('target' in data.columns) and ('temporal ID' in data.columns)):
            raise TypeError('Error: there are no such columns \'target\' or \'temporal ID\' in \'data\' columns.(data must contain both' + \
            ' these columns.)')
    # checking 'target' & 'temporal ID' columns.
    check_data_columns(data,['target','temporal ID'])
    # sorting based on temporal ID.
    data = data.sort_values(by='temporal ID')
    return data



def cumulative(target_signal):
    cumulative_target = target_signal.cumsum()
    return cumulative_target



def differential(target_signal):
    target_without_nan = target_signal.dropna()
    first_target = target_without_nan.iloc[0]
    differentiated_target = target_without_nan.diff()
    differentiated_target.iloc[0] = first_target
    return differentiated_target



def moving_average(target_signal,window_size):

    moving_averaged_target = target_signal.rolling(window=window_size, min_periods=0).mean()
    return moving_averaged_target



def classify(target_signal,class_boundaries):
    import pandas as pd
    target_without_nan = target_signal.dropna()
    classifed_target = pd.cut(target_without_nan,bins=class_boundaries,
    labels=list(range(len(class_boundaries)-1)))
    return classifed_target



def modify_target(data,class_boundaries=None,target_mode=None,column_identifier=None): 

        import numpy as np
        import pandas as pd
        data = input_checking_modify_target(data=data,class_boundaries=class_boundaries,target_mode=target_mode,column_identifier=column_identifier)
        ######################################################################### normal
        if target_mode is None:
            data['Target (normal)'] = data['target']
            data = data.rename(columns={'target':'Normal target'})   
            return data
        ######################################################################### cumulative
        if target_mode == 'cumulative':
            mask = data['target'].isna()
            data['Target (cumulative)'] = data['target'].where(mask,other=cumulative)
            data = data.rename(columns={'target':'Normal target'})

        ######################################################################### differential
        elif target_mode == 'differential': # make target differential
            mask = data['target'].isna()
            data['Target (differential)'] = data['target'].where(mask,other=differential)
            data = data.rename(columns={'target':'Normal target'})
        ######################################################################### classify
        elif target_mode == 'classify': # make target classified
            mask = data['target'].isna()
            data['Target (classify)'] = data['target'].where(mask,other=lambda target_signal : classify(target_signal,class_boundaries))
            data = data.rename(columns={'target':'Normal target'})
        ######################################################################### moving average
        elif target_mode.startswith('moving average'):
            window_size = int(target_mode[len('moving average '):])
            mask = data['target'].isna()
            data['Target (moving average)'] = data['target'].where(mask,other=lambda target_signal : moving_average(target_signal,window_size))
            data = data.rename(columns={'target':'Normal target'})  
        return data