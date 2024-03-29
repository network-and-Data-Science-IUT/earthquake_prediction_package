import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
import functools as ft

# Temporary commnet the below line because we are not testing in real indexed package.
# import configurations
from data_preprocessing import configurations

'''
If two or more samples have the same spatial id and temporal id, this function aggregates them.
'''


# renaming the columns to formal format
def rename_columns(data, column_identifier):
    if type(column_identifier) == dict:
        for key, value in column_identifier.items():
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

        if "spatial id" not in data:
            if "spatial id" not in list(column_identifier.keys()):
                raise ValueError("spatial id is not specified in column_identifier.")
            else:
                data.rename(columns={column_identifier["spatial id"]: "spatial id"}, inplace=True)

        if "target" not in data:
            if "target" not in list(column_identifier.keys()):
                raise ValueError("target is not specified in column_identifier.")
            else:
                data.rename(columns={column_identifier["target"]: "target"}, inplace=True)

        if ("temporal covariates" not in column_identifier) or (len(column_identifier["temporal covariates"]) == 0):
            raise ValueError("At least one temporal covariate must be specified in column identifier.\n")
        if "spatail covariates" in column_identifier:
            if len(list(
                    set(column_identifier["temporal covariates"]) & set(column_identifier["spatial covariates"]))) > 0:
                raise ValueError("A covariate can not be in both temporal covariates and spatial covariates")
            if (column_identifier["target"] in column_identifier["temporal covariates"]) or (
                    column_identifier["target"] in column_identifier["spatial covariates"]):
                raise ValueError("target should not be in temporal covariates or spatial covariates.\n")
        else:
            if column_identifier["target"] in column_identifier["temporal covariates"]:
                raise ValueError("target should not be in temporal covariates.")

    elif column_identifier is not None:
        raise TypeError("The column_identifier must be of type dict")
    return data


def aggregate(data, column_identifier, aggregation_mode="mean", base=None, verbose=0):
    # check validity

    # data:
    if type(data) == str:
        data = pd.read_csv(data)
    elif type(data) != pd.DataFrame:
        raise TypeError("The input data must be of type DataFrame or string.\n")
    if data.empty:
        raise ValueError("The input data is empty.\n")

    # rename columns of data using column identifier
    column_identifier = column_identifier.copy()
    data = rename_columns(data.copy(), column_identifier)
    if "temporal id" not in data:
        raise ValueError("temporal id is not specified in data.\n")
    if "spatial id" not in data:
        raise ValueError("spatial id is not specified in data.\n")

    # check validity of column identifier
    if column_identifier is None:
        raise ValueError("At least one temporal covariate must be specified in column identifier.\n")
    elif "temporal covariates" not in column_identifier:
        raise ValueError("temporal covariates are not specified in column identifier.\n")

    # check validity of temporal covariates
    if type(column_identifier["temporal covariates"]) != list:
        raise TypeError("temporal covariates should be specified in column identifier in a list.\n")
        # check validity of spatial covariates
    if "spatial covariates" in column_identifier:
        if type(column_identifier["spatial covariates"]) != list:
            raise TypeError("spatial covariates should be specified in column identifier in a list.\n")
        for spatial_covar in column_identifier["spatial covariates"]:
            grouped_data = data.groupby("spatial id")
            for name_of_group, contents_of_group in grouped_data:
                spatial_covar_list_in_group = contents_of_group[spatial_covar].tolist()
                if not (spatial_covar_list_in_group.count(spatial_covar_list_in_group[0]) == len(
                        spatial_covar_list_in_group)):
                    raise ValueError("contents of spatial covariates, should not change in different temporal ids")
    # check if data contains any types other than int or float
    int_or_float_data = data.select_dtypes(include=['float64', 'int64'])
    if not int_or_float_data.equals(data):
        print(
            "Warning: It's not possible to aggregate columns containing types other than int or float. these columns "
            "were removed during the aggregation process.\n")
        data = int_or_float_data
    # check validity of base
    if aggregation_mode != "base_max":
        if base is not None:
            raise ValueError("base is only available in the case of base_max as aggregation_mode.\n")
    else:
        if base is None:
            raise ValueError("base should be specified in the case of base_max as aggregation_mode.\n")
        if type(base) != str:
            raise TypeError("Type of base should be str and specifies a column in data.\n")
        if base not in data:
            if base not in column_identifier:
                raise ValueError(
                    "base should specify name of a column in data or identify a column in column_identifier.\n")

    # check validity of aggregation_mode in case of str:
    if type(aggregation_mode) == str:
        if aggregation_mode not in configurations.AGGREGATION_MODE_OPTIONS:
            raise ValueError(
                "The aggregation_mode must be among these options [mean , max , min , std , sum , base_max, mode].\n")

    # check validity of aggregation_mode in case of dict
    elif type(aggregation_mode) == dict:
        for aggr_mode in aggregation_mode.values():
            if type(aggr_mode) != str:
                raise TypeError("aggregation_mode options should be of type str.\n")
            if aggr_mode not in configurations.AGGREGATION_MODE_OPTIONS:
                raise ValueError(
                    "The aggregation_mode must be among these options [mean , max , min , std , sum , base_max, "
                    "mode].\n")
            # change aggregation_mode base on column_identifier
            column_identifier_non_list_items = [a for a in column_identifier.items() if isinstance(a[1], str)]
            for k, v in column_identifier_non_list_items:
                if v in aggregation_mode:
                    aggregation_mode[k] = aggregation_mode.pop(v)
    else:
        raise TypeError("The aggregation_mode must be of type str or dict.\n")

    # sort data
    data.sort_values(by=['temporal id', 'spatial id'], ascending=True, ignore_index=True, inplace=True)

    # list of each aggregation_mode in case of dict:
    covariate_names = list(filter(lambda x: not x.startswith(('temporal id', 'spatial id')), data.columns))
    if "spatial covariates" in column_identifier:
        spatial_covariates = [spatial_covar for spatial_covar in column_identifier["spatial covariates"]]
    else:
        spatial_covariates = []
    mean_data = pd.DataFrame()
    max_data = pd.DataFrame()
    min_data = pd.DataFrame()
    std_data = pd.DataFrame()
    sum_data = pd.DataFrame()
    mode_data = pd.DataFrame()
    if type(aggregation_mode) == dict:
        # spatial covariates should remain constant
        for covar, operator in aggregation_mode.items():
            if operator == 'std' and (covar in spatial_covariates):
                print(
                    "Warning: std is not available for spatial covariates. Spatial covariates remain constant during "
                    "aggregation and mean as default method applies for them.\n")
            if operator == 'sum' and (covar in spatial_covariates):
                print(
                    "Warning: sum is not available for spatial covariates. Spatial covariates remain constant during "
                    "aggregation and mean as default method applies for them.\n")

        mean_covariates = [covar for covar, operator in aggregation_mode.items() if operator == 'mean']
        max_covariates = [covar for covar, operator in aggregation_mode.items() if operator == 'max']
        min_covariates = [covar for covar, operator in aggregation_mode.items() if operator == 'min']
        std_covariates = [covar for covar, operator in aggregation_mode.items() if
                          (operator == 'std' and (covar not in spatial_covariates))]
        sum_covariates = [covar for covar, operator in aggregation_mode.items() if
                          (operator == 'sum' and (covar not in spatial_covariates))]
        mode_covariates = [covar for covar, operator in aggregation_mode.items() if operator == 'mode']

        for covar, operator in aggregation_mode.items():
            if operator == 'base_max':
                raise ValueError("base_max aggregation_mode is not valid in the case of aggregation_mode type dict.\n")

        unspecified_covariates = list(set(covariate_names) - set(
            mean_covariates + max_covariates + min_covariates + std_covariates + mode_covariates + sum_covariates))

        if len(unspecified_covariates) > 0:
            print(
                "Warning : The aggregation_mode is not specified for some of the covariates.\nThe mean operator will "
                "be used to aggregate these covariates' values.\n")
            mean_covariates += unspecified_covariates

        # aggregation mode = mean in dict
        if len(mean_covariates) > 0:
            mean_covariates += ["temporal id", "spatial id"]
            mean_covariates = list(dict.fromkeys(mean_covariates))
            mean_data = data.copy()[mean_covariates]
            mean_data = mean_data.groupby(["temporal id", "spatial id"]).mean().reset_index()

        # aggregation mode = max in dict
        if len(max_covariates) > 0:
            max_covariates += ["temporal id", "spatial id"]
            max_data = data.copy()[max_covariates]
            max_data = max_data.groupby(["temporal id", "spatial id"]).max().reset_index()

        # aggregation_mode = min in dict
        if len(min_covariates) > 0:
            min_covariates += ["temporal id", "spatial id"]
            min_data = data.copy()[min_covariates]
            min_data = min_data.groupby(["temporal id", "spatial id"]).min().reset_index()

        # aggregation_mode = std in dict
        if len(std_covariates) > 0:
            std_covariates += ["temporal id", "spatial id"]
            std_data = data.copy()[std_covariates]
            std_data = (std_data.groupby(["temporal id", "spatial id"]).std().fillna(
                data.groupby(['temporal id', 'spatial id']).last())).reset_index()

        # aggregation mode = sum in dict
        if len(sum_covariates) > 0:
            sum_covariates += ["temporal id", "spatial id"]
            sum_data = data.copy()[sum_covariates]
            sum_data = sum_data.groupby(["temporal id", "spatial id"]).sum().reset_index()

        # aggregation mode = mode in dict
        if len(mode_covariates) > 0:
            mode_covariates += ["temporal id", "spatial id"]
            mode_data = data.copy()[sum_covariates]
            mode_data = mode_data.groupby(['temporal id', 'spatial id']).agg(lambda x: pd.Series.mode(x)[0])

        datas = [mean_data, max_data, min_data, sum_data, mode_data, std_data]
        dataframes = []
        for x in datas:
            if len(x) > 0:
                dataframes.append(x)
        data = ft.reduce(lambda left, right: pd.merge(left, right, on=["temporal id", "spatial id"]), dataframes)

    # aggregation_mode is mean
    if aggregation_mode == "mean":
        data = data.groupby(["temporal id", "spatial id"]).mean().reset_index()

    # aggregation_mode is max
    if aggregation_mode == "max":
        data = data.groupby(["temporal id", "spatial id"]).max().reset_index()

    # aggregation_mode is min
    if aggregation_mode == "min":
        data = data.groupby(["temporal id", "spatial id"]).min().reset_index()

    # aggregation_mode is std
    if aggregation_mode == "std":
        if len(spatial_covariates) > 0:
            print(
                "Warning: std aggregation mode is not available for spatial covariates. Spatial covariates remain "
                "constant during aggregation and mean as default method applies for them.\n")
            spa_cov_list = [x for x in spatial_covariates if (x in data.columns)]
            spa_cov_list += ["temporal id", "spatial id"]
            mean_data = data[spa_cov_list].copy()
            mean_data = mean_data.groupby(["temporal id", "spatial id"]).mean().reset_index()

            std_data_list = [x for x in data.columns if not (x in spatial_covariates)]
            std_data = data[std_data_list].copy()
            std_data = (std_data.groupby(["temporal id", "spatial id"]).std().fillna(
                data.groupby(['temporal id', 'spatial id']).last())).reset_index()

            dataframes = [mean_data, std_data]
            data = ft.reduce(lambda left, right: pd.merge(left, right, on=["temporal id", "spatial id"]), dataframes)

        else:
            data = (data.groupby(['temporal id', 'spatial id']).std().fillna(
                data.groupby(['temporal id', 'spatial id']).last())).reset_index()

    # aggregation_mode is sum
    if aggregation_mode == "sum":
        if len(spatial_covariates) > 0:
            print(
                "Warning: Sum aggregation mode is not available for spatial covariates. Spatial covariates remain "
                "constant during aggregation and mean as default method applies for them.\n")
            spa_cov_list = [x for x in spatial_covariates if (x in data.columns)]
            spa_cov_list += ["temporal id", "spatial id"]
            mean_data = data[spa_cov_list].copy()
            mean_data = mean_data.groupby(["temporal id", "spatial id"]).mean().reset_index()

            sum_data_list = [x for x in data.columns if not (x in spatial_covariates)]
            sum_data = data[sum_data_list].copy()
            sum_data = sum_data.groupby(["temporal id", "spatial id"]).sum().reset_index()

            dataframes = [mean_data, sum_data]
            data = ft.reduce(lambda left, right: pd.merge(left, right, on=["temporal id", "spatial id"]), dataframes)

        else:
            data = data.groupby(["temporal id", "spatial id"]).sum().reset_index()

    # aggregation_mode is base_max
    if aggregation_mode == "base_max":
        data = data.sort_values(base, ascending=False).drop_duplicates(['temporal id', 'spatial id'])
        data.sort_values(by=['temporal id', 'spatial id'], ascending=True, ignore_index=True, inplace=True)

    # aggregation_mode is mode
    if aggregation_mode == "mode":
        data = data.groupby(['temporal id', 'spatial id']).agg(lambda x: pd.Series.mode(x)[0])
        data = data.reset_index(level=['temporal id', 'spatial id'])
    return data
