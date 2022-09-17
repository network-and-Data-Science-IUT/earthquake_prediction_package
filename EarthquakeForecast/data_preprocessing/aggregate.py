import warnings 
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
import pandas as pd

import configurations


'''
If two or more samples have the same spatial id and temporal id, this function aggregates them.
'''

# renaming the columns to formal format
def rename_columns(data, column_identifier):
  if type(column_identifier)==dict:
    if "temporal ID" not in data:
      if "temporal ID" not in list(column_identifier.keys()):
         raise ValueError("temporal ID is not specified in column_identifier.")
    if "spatial ID" not in data:
      if "spatial ID" not in list(column_identifier.keys()):
         raise ValueError("spatial ID is not specified in column_identifier.")
    if ("Tempoal ID" not in data) and ("spatial ID" not in data):
      for key, value in column_identifier.items():
        data.rename(columns = {value:key}, inplace = True)
  elif column_identifier is not None:
        raise TypeError("The column_identifier must be of type dict")
  return data



def aggregate(data,column_identifier=None,aggregation_mode="mean",base=None, verbose=0):
    #check validity
  #data:
   if type(data) == str:
      data = pd.read_csv(data)
   elif type(data) != pd.DataFrame :
      raise TypeError("The input data must be of type DataFrame or string.\n")
   if data.empty:
     raise ValueError("The input data is empty.\n")

  #rename column of data using column identifier
   data = rename_columns(data.copy(), column_identifier)
   if "temporal ID" not in data:
     raise ValueError("temporal ID is not specified in data.\n")
   if "spatial ID" not in data:
     raise ValueError("spatial ID is not specified in data.\n")
  #column_identifier validity
   if "temporal covariates" not in column_identifier:
     raise ValueError("temporal covariates are not specified in column identifier\n")
   if "spatial covariates" not in column_identifier:
     raise ValueError("spatial covariates are not specified in column identifier\n")
  #check validity of temporal covariates
   if type(column_identifier["temporal covariates"]) != list:
     raise TypeError("temporal covariates should be specified in column identifier in a list.\n")
  #check vaidity of spatail covariates
   if type(column_identifier["spatial covariates"]) != list:
     raise TypeError("spatail covariates should be specified in column identifier in a list.\n")
   for spatial_covar in column_identifier["spatial covariates"]:
      grouped_data = data.groupby("spatial ID")
      for name_of_group, contents_of_group in grouped_data:
           spatial_covar_list_in_group= contents_of_group[spatial_covar].tolist()
           if not(spatial_covar_list_in_group.count(spatial_covar_list_in_group[0]) == len(spatial_covar_list_in_group)):
              raise ValueError("contents of spatial covariates, should not change in diffrent temporal IDs")
  #check if data contains any types other than int or float
   int_or_float_data = data.select_dtypes(include=['float64', 'int64'])
   if not int_or_float_data.equals(data):
     print("Warning: It's not possible to aggregate columns containing types other than int or float. these columns were removed during the aggregation process.\n")
     data = int_or_float_data
  #check validity of base
   if aggregation_mode != "base_max":
     if base !=  None:
       raise ValueError("base is only available in the case of base_max as aggregation_mode.\n")
   else:
    if base == None:
      raise ValueError("base should be specified in the case of base_max as aggregation_mode.\n")
    if type(base) != str:
      raise TypeError("Type of base should be str and specifies a column in data.\n")
    if base not in data:
      if base not in column_identifier:
        raise ValueError("base should specifiy name of a column in data or identify a column in column_identifie.r\n")

  #check validity of aggregation_mode:
   if type(aggregation_mode) != str:
     raise TypeError("The aggregation_mode must be of type str.\n")
   if aggregation_mode not in AGGREGATION_MODE_OPTIONS:
       raise ValueError("The aggregation_mode must be among these options [mean , max , min , std , sum , base_max, mode].\n")



  #sort data
   data.sort_values(by=['temporal ID','spatial ID'],ascending=True, ignore_index=True,inplace=True)


  #aggregation_mode is mean
   if aggregation_mode == "mean":
     data=data.groupby(["temporal ID", "spatial ID"]).mean().reset_index()

  #aggregation_mode is max
   if aggregation_mode == "max":
     data=data.groupby(["temporal ID", "spatial ID"]).max().reset_index()

  #aggregation_mode is min
   if aggregation_mode == "min":
     data=data.groupby(["temporal ID", "spatial ID"]).min().reset_index()

  #aggregation_mode is std
   if aggregation_mode == "std":
     data=(data.groupby(['temporal ID', 'spatial ID']).std().fillna(data.groupby(['temporal ID', 'spatial ID']).last())).reset_index()

  #aggregation_mode is sum
   if aggregation_mode == "sum":
     data=data.groupby(["temporal ID", "spatial ID"]).sum().reset_index()

  #aggregaion_mode is base_max
   if aggregation_mode == "base_max":
     data=data.sort_values(base, ascending=False).drop_duplicates(['temporal ID','spatial ID'])
     data.sort_values(by=['temporal ID','spatial ID'],ascending=True, ignore_index=True,inplace=True)

  #aggregation_mode is mode
   if aggregation_mode == "mode":
    data=data.groupby(['temporal ID','spatial ID']).agg(lambda x: pd.Series.mode(x)[0])
    data=data.reset_index(level=['temporal ID', 'spatial ID'])

   return data