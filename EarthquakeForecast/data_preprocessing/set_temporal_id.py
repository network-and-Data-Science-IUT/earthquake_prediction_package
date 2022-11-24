import warnings

from EarthquakeForecast.data_preprocessing.configurations import TEMPORAL_UNIT_OPTIONS

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandas as pd
    import numpy as np
    import datetime
    from dateutil.relativedelta import relativedelta

import configurations


# renaming the columns to formal format
def rename_columns(data, column_identifier):
    if type(column_identifier) == dict:
        if "temporal ID" not in data:
            if "temporal ID" not in list(column_identifier.keys()):
                raise ValueError("temporal ID is not specified in column_identifier.")

        for key, value in column_identifier.items():

            if type(value) == str:
                if value not in data:
                    raise ValueError("{} does not exist in data columns.\n".format(value))

            if type(value) == list:
                for i in value:
                    if i not in data:
                        raise ValueError("{} does not exist in data columns.\n".format(i))

            if key == "temporal ID":
                continue
            elif key == "spatial ID":
                continue
            elif key == "target":
                continue
            elif key == "temporal covariates":
                continue
            elif key == "spatial covariates":
                continue
            data.rename(columns={value: key}, inplace=True)

        if "temporal ID" not in data:
            if "temporal ID" not in list(column_identifier.keys()):
                raise ValueError("temporal ID is not specified in column_identifier.")
            else:
                data.rename(columns={column_identifier["temporal ID"]: "temporal ID"}, inplace=True)
        elif "temporal ID" in list(column_identifier.keys()):
            print(
                "Warning: temporal ID is defined in both data columns and colum_identifier. data columns have higher "
                "priority than column_identifier, so temporal ID has been removed from column_identifier.\n")
            column_identifier.pop("temporal ID")

        if "spatial ID" not in data:
            if "spatial ID" in list(column_identifier.keys()):
                if column_identifier["spatial ID"] not in data:
                    raise ValueError("temporal ID and spatial ID should be unique columns.")
                data.rename(columns={column_identifier["spatial ID"]: "spatial ID"}, inplace=True)
        else:
            if "spatial ID" in list(column_identifier.keys()):
                print(
                    "Warning: spatial ID is defined in both data columns and colum_identifier. data columns have "
                    "higher priority than column_identifier, so spatial ID has been removed from column_identifier.\n")
                column_identifier.pop("spatial ID")

        if "target" in data:
            if "target" in list(column_identifier.keys()):
                print(
                    "Warning: target is defined in both data columns and colum_identifier. data columns have higher "
                    "priority than column_identifier, so target has been removed from column_identifier.\n")
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


def set_temporal_id(data, verbose=0, unit="temporal ID", step=1, column_identifier=None):
    """
    Label a group of samples with the same temporal ID according to the desired temporal unit and step.
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
                raise ValueError(
                    "This format of temporal ID in data, could not be scaled with units smaller than day.\n")

        # "%m/%d/%Y  %H:%M:%S %p"
        elif all(data["temporal ID"].apply(validate_date, args=["%m/%d/%Y  %H:%M:%S %p"])):
            data["temporal ID"] = pd.to_datetime(data["temporal ID"], format="%m/%d/%Y  %H:%M:%S %p")

        # "%m/%d/%Y %H:%M:%S"
        elif all(data["temporal ID"].apply(validate_date, args=["%m/%d/%Y %H:%M:%S"])):
            data["temporal ID"] = pd.to_datetime(data["temporal ID"], format="%m/%d/%Y  %H:%M:%S")

        # "%m-%d-%Y %H:%M:%S"
        elif all(data["temporal ID"].apply(validate_date, args=["%m-%d-%Y %H:%M:%S"])):
            data["temporal ID"] = pd.to_datetime(data["temporal ID"], format="%m-%d-%Y  %H:%M:%S")

        # "%Y-%m-%d %H:%M:%S"
        elif all(data["temporal ID"].apply(validate_date, args=["%Y-%m-%d %H:%M:%S"])):
            data["temporal ID"] = pd.to_datetime(data["temporal ID"], format="%Y-%m-%d %H:%M:%S")

        else:
            raise ValueError("format of temporal ID is not among acceptable formats of date.\n")
        # sort data (negative temporal ID problem)
        data.sort_values(by="temporal ID", inplace=True, ignore_index=True)

    # unit
    if type(unit) != str:
        raise TypeError("The temporal scale must be of type str.\n")
    if unit not in TEMPORAL_UNIT_OPTIONS:
        raise ValueError(
            "The temporal scale should be among these options [second, minute, hour, day, week, month, yer, "
            "temporal ID].\n")
    if unit == "temporal ID" and (data["temporal ID"].dtype != int):
        raise TypeError("To choose temporal ID as temporal_scale, type of this column in data must be int.\n")
    if data["temporal ID"].dtype == int:
        if unit != "temporal ID":
            raise ValueError(
                unit + "as temporal scale is only available in the case of temporal ID in acceptable datetime "
                       "formats.\n")
        if step != 1:
            raise ValueError("Temporal unit in case of temporal ID of type int, is meaningless.\n")

    # temporal_unit
    if type(step) != int:
        raise TypeError("The temporal unit must be of type int")
    if step <= 0:
        raise ValueError("temporal unit should be a positive number.\n")

    # column_identifier validity is checked in rename_column function
    data.insert(0, 'Temporal ID', '')
    # unit == temporal ID:
    if unit == "temporal ID":
        data.sort_values(by="temporal ID", ignore_index=True)
        data["Temporal ID"] = data["temporal ID"]

    # sort data
    if pd.api.types.is_datetime64_dtype(data["temporal ID"].dtype):
        data.sort_values(by="temporal ID", inplace=True, ignore_index=True)
        data["temporal ID"].dt

        # to milliseconds
        data["milliseconds"] = data["temporal ID"].astype(int)
        data["milliseconds"] = data["milliseconds"].div(1000000).astype(int)

    first_id = data["temporal ID"].min()
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
        data["Temporal ID"] = ((data["temporal ID"] - first_id).dt.days // step) + 1

    # unit == week (7 days)
    if unit == "week":
        data["Temporal ID"] = ((data["temporal ID"] - first_id).dt.days // (7 * step)) + 1

    # unit == month
    if unit == "month":
        data["y"] = data.apply(lambda x: relativedelta(x['temporal ID'], first_id).years, axis=1)
        data["m"] = data.apply(lambda x: relativedelta(x['temporal ID'], first_id).months, axis=1)
        data["Temporal ID"] = (((data["y"] * 12) + data["m"]) // step) + 1

    # unit == year
    if unit == "year":
        data["Temporal ID"] = (data.apply(lambda x: relativedelta(x['temporal ID'], first_id).years,
                                          axis=1) // step) + 1
    data = data.drop(['milliseconds', 'y', 'm'], axis=1, errors='ignore')
    data.rename(columns={"temporal ID": "time", "Temporal ID": "temporal ID"}, errors="ignore", inplace=True)
    data = data.reset_index(drop=True)
    return data
