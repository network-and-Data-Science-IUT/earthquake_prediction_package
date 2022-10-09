import pandas as pd
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")


# renaming the columns to formal format
def rename_columns(data, column_identifier):
    if type(column_identifier) != dict:
      raise TypeError("column_identifier must be of type dict.")
    else:
        for key, value in column_identifier.items():
            if type(value) == str:
               if value not in data:
                 raise ValueError("{} does not exist in data columns.\n".format(value))
            if type (value) == list:
              for i in  value:
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
          print("Warning: temporal ID is defined in both data columns and colum_identifier. data columns have higher priority than column_identifier, so temporal ID has been removed from column_identifier.\n")
          column_identifier.pop("temporal ID")

        if "spatial ID" not in data:
            if "spatial ID" not in list(column_identifier.keys()):
                raise ValueError("spatial ID is not specified in column_identifier.")
            else:
              if column_identifier["spatial ID"] not in data:
                raise ValueError("temporal ID and spatial ID should be unique columns.")
              data.rename(columns={column_identifier["spatial ID"]: "spatial ID"}, inplace=True)
        elif "spatial ID" in list(column_identifier.keys()):
          print("Warning: spatial ID is defined in both data columns and colum_identifier. data columns have higher priority than column_identifier, so spatail ID has been removed from column_identifier.\n")
          column_identifier.pop("spatial ID")
        if "target" not in data:
            if "target" not in list(column_identifier.keys()):
                raise ValueError("target is not specified in column_identifier.")
            else:
                if column_identifier["target"] not in data:
                  raise ValueError("target should be an unique column.")
                data.rename(columns={column_identifier["target"]: "target"}, inplace=True)
        elif "target" in list(column_identifier.keys()):
          print("Warning: target is defined in both data columns and colum_identifier. data columns have higher priority than column_identifier, so target has been removed from column_identifier.\n")
          column_identifier.pop("target")

        if ("temporal covariates" not in column_identifier):
                raise ValueError("At least one temporal covariate must be specified in column identifier")
        elif (len(column_identifier["temporal covariates"]) == 0):
                raise ValueError("At least one temporal covariate must be specified in column identifier")
        for item in column_identifier["temporal covariates"]:
          if item in column_identifier.values():
            raise ValueError("temporal covariates should be unique columns.\n")
        if "spatial covariates" in column_identifier:
            if len(list(set(column_identifier["temporal covariates"]) & set(column_identifier["spatial covariates"]))) > 0:
                raise ValueError("A covariate can not be in both temporal covariates and spatial covariates")
        if "spatail covariates" in column_identifier:
          if "target" in column_identifier:
            if (column_identifier["target"] in column_identifier["temporal covariates"]) or (column_identifier["target"] in column_identifier["spatial covariates"]):
                raise ValueError("target should not be in temporal covariates or spataial covariates")
        else:
          if "target" in column_identifier:
            if (column_identifier["target"] in column_identifier["temporal covariates"]):
                raise ValueError("target should not be in temporal or spataial covariates")
    return data


from numpy import NaN


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
    if type(column_identifier) != dict:
        raise TypeError("column_identifier must be of type dict.\n")
    if "target" in data:
        target_as_temporal_covar = data["target"]
        target_as_temporal_covar_name = "target*"
    elif "target" in column_identifier:
        target_as_temporal_covar = data[column_identifier["target"]]
        target_as_temporal_covar_name = column_identifier["target"]
    else:
        raise ValueError("The target column must be specified in data or column_identifier.\n")
    data = rename_columns(data.copy(), column_identifier)
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
    # add content of target column befor renaming to data and temporal covariates
    data[target_as_temporal_covar_name] = target_as_temporal_covar
    temporal_covariates.append(target_as_temporal_covar_name)
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

        # history length should contain nothing other than temporal covariates
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
    # prepare data
    # remove columns other than temporal and spatial ids, target , temporal and spatial covariates
    for column in data:
        if column not in ["target", "temporal ID", "spatial ID"]:
            if column not in (spatial_covariates + temporal_covariates):
                data.drop(column, inplace=True, axis=1)
                print(
                    "Warning: {} is neither target nor temporal ID nor spatial ID nor among covariates. this column was removed from data.\n".format(
                        column))
    # sort base on spatial > temporal ids
    data.sort_values(by=['spatial ID', 'temporal ID'], ascending=True, ignore_index=True, inplace=True)
    # make data continous (contain all temporal id in each temporal covariates) ** add a new column and make that 1 if this row is new
    data["new_row"] = 0
    grouped_data_spatial = data.groupby("spatial ID")
    missed_temporal_ids_data = pd.DataFrame()
    for name_of_group, contents_of_group in grouped_data_spatial:
        for temporal_id in range(min(contents_of_group["temporal ID"]) - 1,
                                 max(contents_of_group["temporal ID"] + forecast_horizon + 1)):
            if temporal_id not in list(contents_of_group["temporal ID"]):
                new_row = pd.DataFrame()
                for column in contents_of_group:
                    if column == "temporal ID":
                        new_row[column] = [temporal_id]
                    elif column == "spatial ID":
                        new_row[column] = [name_of_group]
                    elif column == "new_row":
                        new_row[column] = [1]
                    else:
                        new_row[column] = NaN
                missed_temporal_ids_data = pd.concat([missed_temporal_ids_data, new_row]).reset_index(drop=True)
    data = pd.concat([missed_temporal_ids_data, data]).reset_index()
    data.sort_values(by=['spatial ID', 'temporal ID'], ascending=True, ignore_index=True, inplace=True)
    data.drop("index", inplace=True, axis=1)

    # target is a temporal covariate too
    temp_data = pd.DataFrame()
    data2 = pd.DataFrame()
    target_data = pd.DataFrame()

    grouped_continous_data = data.groupby("spatial ID")
    for name_of_group, contents_of_group in grouped_continous_data:
        temp_group = pd.DataFrame()
        for temporal_covar, temporal_covar_his_len in history_length_dict.items():
            for i in range(temporal_covar_his_len):
                temp_data = contents_of_group
                if i == 0:
                    col_name = temporal_covar + " t"
                    temp_group[col_name] = contents_of_group[temporal_covar]
                    contents_of_group.drop(temporal_covar, inplace=True, axis=1)
                else:
                    col_name = temporal_covar + " t-" + str(i)
                    temp_group[col_name] = temp_group[temporal_covar + " t"].shift(i).copy()
                temp_data = temp_data.join(temp_group)

        data2 = pd.concat([temp_data, data2]).reset_index()
        data2.drop("index", inplace=True, axis=1)
        data2.sort_values(by=['spatial ID', 'temporal ID'], ascending=True, ignore_index=True, inplace=True)
        # forecast_horizon:
        target_group = pd.DataFrame()
        if type(forecast_horizon) != int:
            raise TypeError("forecast_horizon must be of type int.\n")
        if forecast_horizon < 0:
            raise ValueError("forecast_horizon must be a positive number.\n")
        if forecast_horizon == 0:
            target_group["target t"] = contents_of_group["target"].copy()
        else:
            col_name = "target t+" + str(forecast_horizon)
            target_group[col_name] = contents_of_group["target"].shift(-forecast_horizon).copy()
        contents_of_group.drop("target", inplace=True, axis=1)
        target_data = pd.concat([target_data, target_group]).reset_index()
        target_data.drop("index", inplace=True, axis=1)
    data2[col_name] = target_data[col_name]

    # sort, remove new_row = 1 rows
    data2 = data2[data2.new_row != 1]
    data2.sort_values(by=['spatial ID', 'temporal ID'], ascending=True, ignore_index=True, inplace=True)
    data2.drop("new_row", inplace=True, axis=1)
    data2.drop("target", inplace=True, axis=1)
    data = data2
    return data
