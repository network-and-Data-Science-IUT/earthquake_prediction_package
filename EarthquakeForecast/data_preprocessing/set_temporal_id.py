import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandas as pd
    import numpy as np
    import datetime
    from dateutil.relativedelta import relativedelta



import configuration



# renaming the columns to formal format
def rename_columns(data, column_identifier):
  if type(column_identifier)==dict:
    if "temporal ID" not in data:
      if "temporal ID" not in list(column_identifier.keys()):
         raise ValueError("temporal ID is not specified in column_identifier.")
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


def set_temporal_id(data, verbose=0, unit="temporal ID", step=1, column_identifier=None):
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
    if "temporal ID" not in data:
        raise ValueError("temporal ID is not specified in data.\n")

    # remove NaN temporal ID
    data2 = data[data["temporal ID"].notna()]
    if len(data2) != len(data):
        print("Warning: instances with NaN temporal ID were removed\n")
    data = data2

    # check validity of data after removing NaN temporal IDs
    if data.empty:
        raise ValueError("The input data is empty.\n")

    # check validity of temporal id level 1 of type str and convert to datetime
    if data["temporal ID"].dtype == object and isinstance(data.iloc[0]["temporal ID"], str):
        # %Y-%m-%dT%H:%M:%S.%fZ:
        if all(data["temporal ID"].apply(validate_date, args=["%Y-%m-%dT%H:%M:%S.%fZ"])):
            data["temporal ID"] = pd.to_datetime(data["temporal ID"], format="%Y-%m-%dT%H:%M:%S.%fZ")

        # %Y-%m-%d
        elif all(data["temporal ID"].apply(validate_date, args=["%Y-%m-%d"])):
            data["temporal ID"] = pd.to_datetime(data["temporal ID"], format="%Y-%m-%d")
            if unit in ["second", "minute", "hour"]:
                raise ValueError("This format of temporal ID in data, could not be scaled with units smaller than day")
        else:
            raise ValueError("format of temporal ID is not among acceptable foramts of date.\n")

    # unit
    if type(unit) != str:
        raise TypeError("The temporal scale must be of type str.\n")
    if unit not in TEMPORAL_UNIT_OPTIONS:
        raise ValueError(
            "The temporal scale should be among these options [second, minute, hour, day, week, month, yaer, temporal ID].\n")
    if unit == "temporal ID" and (data["temporal ID"].dtype != int):
        raise TypeError("To choose tempoal ID as temporl_scale, type of this column in data must be int.\n")
    if (data["temporal ID"].dtype == int):
        if unit != "temporal ID":
            raise ValueError(
                unit + " as temporal scale is only available in the case of temporal ID in acceptable datetime formats.\n")
        if step != 1:
            raise ValueError("Temporal unit in case of temporal ID of type int, is meaningless.\n")

    # tempoal_unit
    if type(step) != int:
        raise TypeError("The temporal unit must be of type int")
        if step < 0:
            raise ValueError("temporal unit should be a positive number.\n")

    # column_identifier validity is checked in rename_column function
    data.insert(0, 'Temporal ID', '')
    # unit == temopral ID:
    if unit == "temporal ID":
        data.sort_values(by="temporal ID")
        data["Temporal ID"] = data["temporal ID"]

    # sort data
    if pd.api.types.is_datetime64_dtype(data["temporal ID"].dtype):
        data.sort_values(by="temporal ID")
        data["temporal ID"].dt

        # to milliseconds
        data["miliseconds"] = data["temporal ID"].astype(int)
        data["miliseconds"] = data["miliseconds"].div(1000000).astype(int)

    first_id = data["temporal ID"][0]
    # unit == second
    if unit == "second":
        data["Temporal ID"] = (((data["miliseconds"] - data["miliseconds"][0]) // 1000) // step) + 1

    # unit == minute (60 seconds)
    if unit == "minute":
        data["Temporal ID"] = (((data["miliseconds"] - data["miliseconds"][0]) // 60000) // step) + 1

    # unit == hour (3600 seconds)
    if unit == "hour":
        data["Temporal ID"] = (((data["miliseconds"] - data["miliseconds"][0]) // 3600000) // step) + 1

    # unit == day
    if unit == "day":
        data["Temporal ID"] = (((data["temporal ID"] - first_id).dt.days) // step) + 1

    # unit == week (7 days)
    if unit == "week":
        data["Temporal ID"] = (((data["temporal ID"] - first_id).dt.days) // (7 * step)) + 1

    # unit == month
    if unit == "month":
        data["y"] = data.apply(lambda x: relativedelta(x['temporal ID'], first_id).years, axis=1)
        data["m"] = data.apply(lambda x: relativedelta(x['temporal ID'], first_id).months, axis=1)
        data["Temporal ID"] = (((data["y"] * 12) + data["m"]) // step) + 1

    # unit == year
    if unit == "year":
        data["Temporal ID"] = (data.apply(lambda x: relativedelta(x['temporal ID'], first_id).years,
                                          axis=1) // step) + 1
    data = data.drop(['miliseconds', 'y', 'm'], axis=1, errors='ignore')
    data.rename(columns={"temporal ID": "time", "Temporal ID": "temporal ID"}, errors="ignore", inplace=True)
    return data