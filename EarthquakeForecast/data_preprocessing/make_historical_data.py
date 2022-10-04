import pandas as pd
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")


# renaming the columns to formal format
def rename_columns(data, column_identifier):
    if type(column_identifier) == dict:
        for key, value in column_identifier.items():
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

        if "spatial ID" not in data:
            if "spatial ID" not in list(column_identifier.keys()):
                raise ValueError("spatial ID is not specified in column_identifier.")
            else:
                data.rename(columns={column_identifier["spatial ID"]: "spatial ID"}, inplace=True)

        if "target" not in data:
            if "target" not in list(column_identifier.keys()):
                raise ValueError("target is not specified in column_identifier.")
            else:
                data.rename(columns={column_identifier["target"]: "target"}, inplace=True)
        if "spatial covariates" in column_identifier:
            if ("temporal covariates" not in column_identifier) or (len(column_identifier["temporal covariates"]) == 0):
                raise ValueError("At least one temporal covariate must be specified in column identifier")
        if "spatial covariates" in data:
            if len(list(
                    set(column_identifier["temporal covariates"]) & set(column_identifier["spatial covariates"]))) > 0:
                raise ValueError("A covariate can not be in both temporal covariates and spatial covariates")
        if "spatail covariates" in data:
            if (column_identifier["target"] in column_identifier["temporal covariates"]) or (
                    column_identifier["target"] in column_identifier["spatial covariates"]):
                raise ValueError("target should not be in temporal covariates or spataial covariates")
        else:
            if (column_identifier["target"] in column_identifier["temporal covariates"]):
                raise ValueError("target should not be in temporal or spataial covariates")
    elif column_identifier is not None:
        raise TypeError("The column_identifier must be of type dict")
    return data


def make_historical_data(data, forecast_horizon, history_length=1, column_identifier=None):
    # check validity:
    # data:
    if type(data) == str:
        data = pd.read_csv(data)
    elif type(data) != pd.DataFrame:
        raise TypeError("The input data must be of type DataFrame or string.\n")
    if data.empty:
        raise ValueError("The input data is empty.\n")
    # history_length:
    if type(history_length) == dict:
        for value in history_length.values():
            if type(value) != int:
                raise TypeError("The history_length of each covariate must be of type int.\n")
            else:
                if value <= 0:
                    raise ValueError("history_length should be a positive integer.\n")
    elif type(history_length) == int:
        if history_length <= 0:
            raise ValueError("history_length sholud be a positive integer.\n")
    else:
        raise TypeError("The history_length must be of type int or dict.\n")
    # column_identifier
    if column_identifier != None:
        data = rename_columns(data.copy(), column_identifier)
    if "temporal ID" not in data:
        raise ValueError("temporal ID is not specified in data.\n")
    if "spatial ID" not in data:
        raise ValueError("spatial ID is not specified in data.\n")
    # column_identifier validity
    if "temporal covariates" not in column_identifier:
        raise ValueError("temporal covariates are not specified in column identifier\n")

    # check validity of temporal covariates
    if type(column_identifier["temporal covariates"]) != list:
        raise TypeError("temporal covariates should be specified in column identifier in a list.\n")
        # check vaidity of spatail covariates
    if "spatial covariates" in column_identifier:
        if type(column_identifier["spatial covariates"]) != list:
            raise TypeError("spatail covariates should be specified in column identifier in a list.\n")
        for spatial_covar in column_identifier["spatial covariates"]:
            print(spatial_covar)
            grouped_data = data.groupby("spatial ID")
            for name_of_group, contents_of_group in grouped_data:
                spatial_covar_list_in_group = contents_of_group[spatial_covar].tolist()
                if not (spatial_covar_list_in_group.count(spatial_covar_list_in_group[0]) == len(
                        spatial_covar_list_in_group)):
                    raise ValueError("contents of spatial covariates, should not change in diffrent temporal IDs")
    # check if data contains any types other than int or float
    int_or_float_data = data.select_dtypes(include=['float64', 'int64'])
    if not int_or_float_data.equals(data):
        print(
            "Warning: It's not possible to aggregate columns containing types other than int or float. these columns were removed during the aggregation process.\n")
        data = int_or_float_data
    # temporal covariates and spatial covariates list:
    if 'spatial covariates' in column_identifier.keys():
        spatial_covariates = [item for item in list(column_identifier['spatial covariates']) if item in data.columns]
    else:
        spatial_covariates = []
    temporal_covariates = [item for item in list(column_identifier['temporal covariates']) if item in data.columns]
    if type(history_length) == int:
        history_length_dict = {covar: history_length for covar in temporal_covariates}

    elif type(history_length) == dict:
        history_length_dict = history_length.copy()
        # history length should  contain all temporal covariates
        for item in temporal_covariates:
            if item not in history_length_dict.keys():
                history_length_dict[item] = 1
                print(
                    "Warning: history_length of {} which is a temporal covariate, is not specified in dictionary of history_length. 1 as default history_length applied for that.\n".format(
                        item))

        # history length shoul contain nothing other than temporal covariates
        for item in list(history_length_dict):
            if item not in temporal_covariates:
                print(
                    "Warning: {} is not a temporal  covariate, and history_length can not be applied for that.\n".format(
                        item))
                del history_length_dict[item]
    else:
        raise TypeError("history_length must be of type int or dict.\n")

    # type of history_length items in dict case should be int
    for covar in history_length_dict.keys():
        if type(history_length_dict[covar]) != int:
            raise TypeError(
                "The specified history length for each covariate in the history_length dict must be of type int.\n")
        if history_length_dict[covar] <= 0:
            raise ValueError("history_length must be a positive number")

    temp_data = pd.DataFrame()
    target_data = pd.DataFrame()

    for temporal_covar, temporal_covar_his_len in history_length_dict.items():
        for i in range(temporal_covar_his_len):
            if i == 0:
                col_name = temporal_covar + " t"
                temp_data[col_name] = data[temporal_covar]
                data.drop(temporal_covar, inplace=True, axis=1)
            else:
                col_name = temporal_covar + " t-" + str(i)
                temp_data[col_name] = temp_data[temporal_covar + " t"].shift(i).copy()
    data = data.join(temp_data)

    # forecast_horizon
    if type(forecast_horizon) != int:
        raise TypeError("forecast_horizon must be of type int.\n")
    if forecast_horizon < 0:
        raise ValueError("forecast_horizon must be a positive number.\n")
    if forecast_horizon == 0:
        target_data["target t"] = data["target"].copy()
    else:
        col_name = "target t+" + str(forecast_horizon)
        target_data[col_name] = data["target"].shift(-forecast_horizon).copy()
    data.drop("target", inplace=True, axis=1)
    data = data.join(target_data)



    return data
