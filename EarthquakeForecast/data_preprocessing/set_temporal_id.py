import warnings
# in the testing case where the package is not indexed i comment the below line and explicitly init the TEMPORAL_UNIT_OPTIONS
# uncomment before commit
# from EarthquakeForecast.data_preprocessing.configurations import TEMPORAL_UNIT_OPTIONS
TEMPORAL_UNIT_OPTIONS = ["second", "minute", "hour", "day", "week", "month", "year", "temporal id"]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandas as pd
    import numpy as np
    import datetime
    from dateutil.relativedelta import relativedelta

# import configurations


# renaming the columns to formal format
def rename_columns(data, column_identifier):
    if type(column_identifier) == dict:
        if "temporal id" not in data:
            if "temporal id" not in list(column_identifier.keys()):
                raise ValueError("temporal id is not specified in column_identifier.")

        for key, value in column_identifier.items():

            if type(value) == str:
                if (value not in data) and (key not in data):
                    raise ValueError("{} does not exist in data columns.\n".format(value))

            if type(value) == list:
                for i in value:
                    if i not in data:
                        raise ValueError("{} does not exist in data columns.\n".format(i))

            if key == "temporal id":
                continue
            elif key == "spatial id":
                continue
            elif key == "target":
                continue
            elif key == "temporal covariates":
                continue
            elif key == "spatial covariates":
                continue
            data.rename(columns={value: key}, inplace=True)

        if "temporal id" not in data:
            if "temporal id" not in list(column_identifier.keys()):
                raise ValueError("temporal id is not specified in column_identifier.")
            else:
                data.rename(columns={column_identifier["temporal id"]: "temporal id"}, inplace=True)
        elif "temporal id" in list(column_identifier.keys()):
            column_identifier.pop("temporal id")

        if "spatial id" not in data:
            if "spatial id" in list(column_identifier.keys()):
                if column_identifier["spatial id"] not in data:
                    raise ValueError("temporal id and spatial id should be unique columns.")
                data.rename(columns={column_identifier["spatial id"]: "spatial id"}, inplace=True)
        else:
            if "spatial id" in list(column_identifier.keys()):
                column_identifier.pop("spatial id")

        if "target" in data:
            if "target" in list(column_identifier.keys()):
                column_identifier.pop("target")
        else:
            if "target" in list(column_identifier.keys()):
                data.rename(columns={column_identifier["target"]: "target"}, inplace=True)

    return data


def validate_date(date_text, date_format):
    try:
        datetime.datetime.strptime(date_text, date_format)
        return True
    except ValueError:
        return False


def set_temporal_id(data, unit="temporal id", step=1, column_identifier=None):
    """
    Label a group of samples with the same temporal id according to the desired temporal unit and step.
    Temporal ID identifies which time period a sample belongs to.
    """

    # check validity
    # data:
    if type(data) == str:
        data = pd.read_csv(data)
    elif type(data) != pd.DataFrame:
        raise TypeError("The input data must be of type DataFrame or string.\n")
    if data.empty:
        raise ValueError("The input data is empty.\n")

    # rename column of data using column identifier
    data = rename_columns(data.copy(), column_identifier.copy())
    if "temporal id" not in data:
        raise ValueError("temporal id is not specified in data.\n")

    # remove NaN temporal id
    data2 = data[data["temporal id"].notna()]
    if len(data2) != len(data):
        print("Warning: instances with NaN temporal id were removed\n")
    data = data2

    # check validity of data after removing NaN temporal ids
    if data.empty:
        raise ValueError("The input data is empty.\n")

    # check validity of temporal id level 1 of type str and convert to datetime
    if data["temporal id"].dtype == object and isinstance(data.iloc[0]["temporal id"], str):

        # %Y-%m-%dT%H:%M:%S.%fZ:
        if all(data["temporal id"].apply(validate_date, args=["%Y-%m-%dT%H:%M:%S.%fZ"])):
            data["temporal id"] = pd.to_datetime(data["temporal id"], format="%Y-%m-%dT%H:%M:%S.%fZ")

        # %Y-%m-%d
        elif all(data["temporal id"].apply(validate_date, args=["%Y-%m-%d"])):
            data["temporal id"] = pd.to_datetime(data["temporal id"], format="%Y-%m-%d")
            if unit in ["second", "minute", "hour"]:
                raise ValueError(
                    "This format of temporal id in data, could not be scaled with units smaller than day.\n")

        # "%m/%d/%Y  %H:%M:%S %p"
        elif all(data["temporal id"].apply(validate_date, args=["%m/%d/%Y  %H:%M:%S %p"])):
            data["temporal id"] = pd.to_datetime(data["temporal id"], format="%m/%d/%Y  %H:%M:%S %p")

        # "%m/%d/%Y %H:%M:%S"
        elif all(data["temporal id"].apply(validate_date, args=["%m/%d/%Y %H:%M:%S"])):
            data["temporal id"] = pd.to_datetime(data["temporal id"], format="%m/%d/%Y  %H:%M:%S")

        # "%m-%d-%Y %H:%M:%S"
        elif all(data["temporal id"].apply(validate_date, args=["%m-%d-%Y %H:%M:%S"])):
            data["temporal id"] = pd.to_datetime(data["temporal id"], format="%m-%d-%Y  %H:%M:%S")

        # "%Y-%m-%d %H:%M:%S"
        elif all(data["temporal id"].apply(validate_date, args=["%Y-%m-%d %H:%M:%S"])):
            data["temporal id"] = pd.to_datetime(data["temporal id"], format="%Y-%m-%d %H:%M:%S")

        # %Y-%m-%dT%H:%M:%S.%f
        elif all(data["temporal id"].apply(validate_date, args=["%Y-%m-%dT%H:%M:%S.%f"])):
            data["temporal id"] = pd.to_datetime(data["temporal id"], format="%Y-%m-%dT%H:%M:%S.%f")
        
        else:
            raise ValueError("format of temporal id is not among acceptable formats of date.\n")
        # sort data (negative temporal id problem)
        data.sort_values(by="temporal id", inplace=True, ignore_index=True)

    # unit
    if type(unit) != str:
        raise TypeError("The temporal scale must be of type str.\n")
    if unit not in TEMPORAL_UNIT_OPTIONS:
        raise ValueError(
            "The temporal scale should be among these options [second, minute, hour, day, week, month, yer, "
            "temporal id].\n")
    if unit == "temporal id" and (data["temporal id"].dtype != int):
        raise TypeError("To choose temporal id as temporal_scale, type of this column in data must be int.\n")
    if data["temporal id"].dtype == int:
        if unit != "temporal id":
            raise ValueError(
                unit + "as temporal scale is only available in the case of temporal id in acceptable datetime "
                       "formats.\n")
        if step != 1:
            raise ValueError("Temporal unit in case of temporal id of type int, is meaningless.\n")

    # temporal_unit
    if type(step) != int:
        raise TypeError("The temporal unit must be of type int")
    if step <= 0:
        raise ValueError("temporal unit should be a positive number.\n")

    # column_identifier validity is checked in rename_column function
    data.insert(0, 'Temporal ID', '')
    # unit == temporal id:
    if unit == "temporal id":
        data.sort_values(by="temporal id", ignore_index=True)
        data["Temporal ID"] = data["temporal id"]

    # sort data
    if pd.api.types.is_datetime64_dtype(data["temporal id"].dtype):
        data.sort_values(by="temporal id", inplace=True, ignore_index=True)
        data["temporal id"].dt

        # to milliseconds
        data["milliseconds"] = data["temporal id"].astype(int)
        data["milliseconds"] = data["milliseconds"].div(1000000).astype(int)

    first_id = data["temporal id"].min()
    # unit == second
    if unit == "second":
        data["Temporal ID"] = (((data["milliseconds"] - data["milliseconds"][0]) // 1000) // step) + 1

    # unit == minute (60 seconds)
    if unit == "minute":
        data["Temporal ID"] = (((data["milliseconds"] - data["milliseconds"][0]) // 60000) // step) + 1

    # unit == hour (3600 seconds)
    if unit == "hour":
        data["Temporal ID"] = (((data["milliseconds"] - data["milliseconds"][0]) // 3600000) // step) + 1

    # unit == day
    if unit == "day":
        data["Temporal ID"] = ((data["temporal id"] - first_id).dt.days // step) + 1

    # unit == week (7 days)
    if unit == "week":
        data["Temporal ID"] = ((data["temporal id"] - first_id).dt.days // (7 * step)) + 1

    # unit == month
    if unit == "month":
        data["y"] = data.apply(lambda x: relativedelta(x['temporal id'], first_id).years, axis=1)
        data["m"] = data.apply(lambda x: relativedelta(x['temporal id'], first_id).months, axis=1)
        data["Temporal ID"] = (((data["y"] * 12) + data["m"]) // step) + 1

    # unit == year
    if unit == "year":
        data["Temporal ID"] = (data.apply(lambda x: relativedelta(x['temporal id'], first_id).years,
                                          axis=1) // step) + 1
    data = data.drop(['milliseconds', 'y', 'm'], axis=1, errors='ignore')
    data.rename(columns={"temporal id": "time", "Temporal ID": "temporal id"}, errors="ignore", inplace=True)
    data = data.reset_index(drop=True)
    return data
