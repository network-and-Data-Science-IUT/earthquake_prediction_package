import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandas as pd
    import numpy as np
    import datetime



import configuration



# renaming the columns to formal format
def rename_columns(data, column_identifier):
  if type(column_identifier)==dict:
    if "temporal ID level 1" not in data:
      if "temporal ID level 1" not in list(column_identifier.keys()):
         raise ValueError("temporal ID level 1 is not specified in column_identifier.")
      for key, value in column_identifier.items():
        data.rename(columns = {value:key}, inplace = True)
  elif column_identifier is not None:
        raise TypeError("The column_identifier must be of type dict")
  return data




def validate_date(date_text, date_format):
    try:
        datetime.datetime.strptime(date_text,date_format)
        return True
    except ValueError:
        return False




def set_temporal_id(data, verbose=0, temporal_scale="temporal ID level 1", temporal_unit=1, column_identifier=None):
    # check validity
    # data:
    if type(data) == str:
        data = pd.read_csv(data)
    elif type(data) != pd.DataFrame:
        raise TypeError("The input data must be of type DataFrame or string.\n")
    if data.empty:
        raise ValueError("The input data is empty.\n")

    # rename column of data using column identifier
    data = rename_columns(data.copy(), column_identifier)
    if "temporal ID level 1" not in data:
        raise ValueError("temporal ID level 1 is not specified in data.\n")

    # remove NaN temporal ID
    data2 = data[data["temporal ID level 1"].notna()]
    if len(data2) != len(data):
        print("Warning: instances with NaN temporal ID level 1 were removed\n")
    data = data2

    # check validity of data after removing NaN temporal IDs
    if data.empty:
        raise ValueError("The input data is empty.\n")

        # check validity of data in presence of temporal ID level 2
        if "temporal ID level 2" in data:
            if data["temporal ID level 2"].dtype != int:
                raise TypeError("Type of temporal ID level 2 must be int")
        if "temporal ID level 2" in data:
            if data["temporal ID level 1"].dtype != int:
                raise TypeError("In presence of temporal ID level 2, type of temporal ID level 1 must be int")


    # check validity of temporal id level 1 of type str and convert to datetime
    if data["temporal ID level 1"].dtype == object and isinstance(data.iloc[0]["temporal ID level 1"], str):
        if all(data["temporal ID level 1"].apply(validate_date, args=["%Y-%m-%dT%H:%M:%S.%fZ"])):
            data["temporal ID level 1"] = pd.to_datetime(data["temporal ID level 1"], format="%Y-%m-%dT%H:%M:%S.%fZ")

        elif all(data["temporal ID level 1"].apply(validate_date, args=["%Y-%m-%d"])):
            data["temporal ID level 1"] = pd.to_datetime(data["temporal ID level 1"], format="%Y-%m-%d")

        else:
            raise ValueError("format of temporal ID level 1 is not among acceptable foramts of date.\n")


    # temporal_scale
    if type(temporal_scale) != str:
        raise TypeError("The temporal scale must be of type str.\n")
    if temporal_scale not in configuration.TEMPORAL_SCALE_OPTIONS:
        raise ValueError(
            "The temporal scale should be among these options [second, minute, hour, day, week, month, yaer, temporal ID level 1, temporal ID level 2].\n")
    if temporal_scale == "temporal ID level 1" and (data["temporal ID level 1"].dtype != int):
        raise TypeError("To choose tempoal ID level 1 as temporl_scale, type of this column in data must be int.\n")
    if (temporal_scale == "temporal ID level 2"):
        if ("tempoal ID level 2" not in data):
            raise ValueError("temporal ID level 2 is not specified in data")
        if (data["temporal ID level 2"].dtype != int):
            raise TypeError(
                "To choose tempoal ID level 2 as temporl_scale, type of temporal ID level 1, 2 in data must be int.\n")
    if (data["temporal ID level 1"].dtype == int):
        if temporal_scale not in ["temporal ID level 1", "temporal ID level 2"]:
            raise ValueError(
                temporal_scale + " as temporal scale is only available in the case of temporal ID level 1 of acceptable datetime formats.\n")
        if temporal_unit != 1:
            raise ValueError("Temporal unit in case of temporal ID of type int, is meaningless.\n")


    # tempoal_unit
    if type(temporal_unit) != int:
        raise TypeError("The temporal unit must be of type int")
        if temporal_unit < 0:
            raise ValueError("temporal unit should be a positive number.\n")


    # column_identifier validity is checked in rename_column function


    # temporal_scale == temopral ID level 1:
    if temporal_scale == "temporal ID level 1":
        data.sort_values(by="temporal ID level 1")
        data["Temporal ID"] = data["temporal ID level 1"]
    # temporal_scale == temporal ID level 2:
    if temporal_scale == "temporal ID level 2":
        data.sort_values(by="temporal ID level 2")
        data["Temporal ID"] = data["temporal ID level 2"]


    # sort data
    if pd.api.types.is_datetime64_dtype(data["temporal ID level 1"].dtype):
        data.sort_values(by="temporal ID level 1")
        data["temporal ID level 1"].dt


        # to milliseconds
        data["seconds"] = data["temporal ID level 1"].astype(int)
        data["seconds"] = data["seconds"].div(1000000).astype(int)


        # temporal_scale == second
        if temporal_scale == "second":
            scale = temporal_unit * 1000

        # temporal_scale == minute
        if temporal_scale == "minute":
            scale = temporal_unit * 60000

        # temporal_scale == hour
        if temporal_scale == "hour":
            scale = temporal_unit * 3600000

        # temporal_scale == day  (24 hours)
        if temporal_scale == "day":
            scale = temporal_unit * 86400000

        # temporal_scale == week (7 days)
        if temporal_scale == "week":
            scale = temporal_unit * 604800000

        # temporal_scale == month (30 days)
        if temporal_scale == "month":
            scale = temporal_unit * 2592000000

        # temporal_scale == year (365 days)
        if temporal_scale == "year":
            scale = temporal_unit * 31536000000

        first_id = data["seconds"][0]
        data["Temporal ID"] = (((data["seconds"] - first_id) / scale) + 1).astype(int)
    data = data.set_index("Temporal ID", drop=False)
    return data