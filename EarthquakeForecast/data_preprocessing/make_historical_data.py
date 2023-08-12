import pandas as pd
import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
from numpy import NaN
import re

# Temporary commnet the below line because we are not testing in real indexed package.
# import configurations
from data_preprocessing import configurations

'''
Transforming data to the historical format and extract features. This subprocess prepare the final data frame 
including features and target variable for modeling. 

The set of features is consisted of:

1. Spatial covariates, temporal covariates at current temporal unit (t)

2. Historical values of these covariates at h-1(where h is history length) previous temporal units (t-1 , t-2 , … , t-h+1).

3. The covariates of the L-layer pixels(where L is layer_number L = (0,1,…,maximum_layer_number)) at 
‘history_length_of_neighbors’-1 previous temporal units(t-1 , t-2 , … , t-‘history_length_of_neighbors’+1) 

The target of the final data frame is the values of the target variable at the temporal unit t+r.
'''


# renaming the columns to formal format
def rename_columns(data, column_identifier):
    # type checking
    if type(column_identifier) != dict:
        raise TypeError("column_identifier must be of type dict.")
    else:
        # column specified in column identifier to rename, doesn't exist in data columns
        for key, value in column_identifier.items():
            if key == 'target':
                key = 'Normal target'
            if type(value) == str:
                if (value not in data) and (key not in data):
                    raise ValueError("{} does not exist in data columns.\n".format(value))
            if type(value) == list:
                for i in value:
                    if i not in data:
                        raise ValueError("{} does not exist in data columns.\n".format(i))
            if key in ["temporal id","spatial id","Normal target","temporal covariates","spatial covariates"]:
                continue
            data.rename(columns={value: key}, inplace=True)

        # temporal id
        if "temporal id" not in data:
            if "temporal id" not in list(column_identifier.keys()):
                raise ValueError("temporal id is not specified in column_identifier.")
            else:
                data.rename(columns={column_identifier["temporal id"]: "temporal id"}, inplace=True)

        elif "temporal id" in list(column_identifier.keys()):
            column_identifier.pop("temporal id")

        # spatial id
        if "spatial id" not in data:
            if "spatial id" not in list(column_identifier.keys()):
                raise ValueError("spatial id is not specified in column_identifier.")
            else:
                if column_identifier["spatial id"] not in data:
                    raise ValueError("temporal id and spatial id should be unique columns.")
                data.rename(columns={column_identifier["spatial id"]: "spatial id"}, inplace=True)

        elif "spatial id" in list(column_identifier.keys()):
            column_identifier.pop("spatial id")

        # target
        # absence of "Normal target" and "Target (...)" in data columns
        if "Normal target" not in data or not any(bool(re.search("Target \(.*?\)", col)) for col in data.columns):
            if "target" not in list(column_identifier.keys()):
                raise ValueError("target is not specified in column_identifier.")
            else:
                if column_identifier["target"] not in data:
                    raise ValueError("Normal target should be an unique column.")
                # create "Normal target" and "Target (normal)" from target using column identifier
                data.rename(columns={column_identifier["target"]: "Normal target"}, inplace=True)
                data["Target (normal)"] = data["Normal target"]

        elif "target" in list(column_identifier.keys()):
            column_identifier.pop("target")

        # temporal covariates
        if "temporal covariates" not in column_identifier:
            raise ValueError("At least one temporal covariate must be specified in column identifier.\n")

        elif len(column_identifier["temporal covariates"]) == 0:
            raise ValueError("At least one temporal covariate must be specified in column identifier.\n")

        for item in column_identifier["temporal covariates"]:
            if item in column_identifier.values():
                raise ValueError("temporal covariates should be unique columns.\n")

        # spatial covariates
        if "spatial covariates" in column_identifier:
            if len(list(
                    set(column_identifier["temporal covariates"]) & set(column_identifier["spatial covariates"]))) > 0:
                raise ValueError("A covariate can not be in both temporal covariates and spatial covariates.\n")

        # We have internally treated the target as a temporal covariate and the user is not allowed to do so
        if "spatial covariates" in column_identifier:
            if "target" in column_identifier:
                if (column_identifier["target"] in column_identifier["temporal covariates"]) or (
                        column_identifier["target"] in column_identifier["spatial covariates"]):
                    raise ValueError("target should not be in temporal covariates or spataial covariates.\n")

        else:
            if "target" in column_identifier:
                if column_identifier["target"] in column_identifier["temporal covariates"]:
                    raise ValueError("target should not be in temporal or spataial covariates.\n")
    return data


import networkx as nx


def find_neighbors(layer_number, neighbors_dictionary, spatial_id):
    graph_pairs = []
    for node, neighbors_list in neighbors_dictionary.items():
        for neighbor in neighbors_list:
            if ((node, neighbor) not in graph_pairs) and ((neighbor, node) not in graph_pairs):
                graph_pairs.append((node, neighbor))

    G = nx.Graph()
    G.add_edges_from(graph_pairs)
    path_length = nx.single_source_shortest_path_length(G, spatial_id)
    list_of_neighbors = []

    for node, length in path_length.items():
        if length == (layer_number-1):
            list_of_neighbors.append(node)

    return list_of_neighbors


def make_historical_data(data, column_identifier, forecast_horizon, history_length=1,
                         layer_number=1, aggregate_layer=True, neighboring_covariates=None, neighbors_dictionary=None,
                         aggregation_mode="mean"):
    """
    Make data historical in both temporal and spatial aspects
    so that output data is proper for the model.
    Remained columns after this function are the only necessary ones.
    """
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
            elif value <= 0:
                raise ValueError("history_length should be a positive integer.\n")

    elif type(history_length) == int:
        if history_length <= 0:
            raise ValueError("history_length sholud be a positive integer.\n")
    else:
        raise TypeError("The history_length must be of type int or dict.\n")

    # column_identifier
    if type(column_identifier) != dict:
        raise TypeError("column_identifier must be of type dict.\n")
        # column identifier is required to specify at least one temporal covariate

    # target meant to be among temporal covariates and become historical.
    # since the columns' name in input data has more priority than column identifier,
    # if both "Normal target" and "Target (...)" were among columns of data,
    # we add a column name "*Normal target" with the same content as "Normal target"
    # to temporal covariates' list, otherwise we produce "Normal target" and "target (normal)"
    # based on column identifier which specifies the "target" column in data.
    # In this case, the original column remains in data and will be added to the temporal covariates' list.

    # presence of "Noraml target" and "Target (...)" in data
    if "Normal target" in data and any(bool(re.search("Target \(.*?\)", col)) for col in data.columns):
        target_as_temporal_covar = data["Normal target"]
        target_as_temporal_covar_name = "*Normal target"

    # target in column identifier
    elif "target" in column_identifier:
        target_as_temporal_covar = data[column_identifier["target"]]
        target_as_temporal_covar_name = column_identifier["target"]

    # absence of target in data and column identifier
    else:
        raise ValueError("The target column must be specified in column identifier.\n")

    # rename columns
    data = rename_columns(data.copy(), column_identifier)

    # column_identifier validity
    if "temporal covariates" not in column_identifier:
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
                    raise ValueError("contents of spatial covariates, should not change in different temporal ids.\n")

    # check if data contains any types other than int or float
    int_or_float_data = data.select_dtypes(include=['float64', 'int64'])
    if not int_or_float_data.equals(data):
        print(
            "Warning: It's not possible to work with columns containing types other than int or float. these columns "
            "were removed during the making historical process.\n")
        data = int_or_float_data

    # temporal covariates and spatial covariates list:
    if 'spatial covariates' in column_identifier.keys():
        spatial_covariates = [item for item in list(column_identifier['spatial covariates']) if item in data.columns]
    else:
        spatial_covariates = []
    temporal_covariates = [item for item in list(column_identifier['temporal covariates']) if item in data.columns]

    # add content of target column before renaming to data and temporal covariates
    data[target_as_temporal_covar_name] = target_as_temporal_covar
    temporal_covariates.append(target_as_temporal_covar_name)

    ###################################################################### MAKE TEMPORAL HISTORICAL
    if type(history_length) == int:
        history_length_dict = {covar: history_length for covar in temporal_covariates}

    elif type(history_length) == dict:
        history_length_dict = history_length.copy()
        # history length should  contain all temporal covariates
        for item in temporal_covariates:
            if item not in history_length_dict.keys():
                history_length_dict[item] = 1
                print(
                    "Warning: history_length of {} which is a temporal covariate, is not specified in dictionary of "
                    "history_length. 1 as default history_length applied for that.\n".format(
                        item))

        # history length should contain nothing other than temporal covariates
        for item in list(history_length_dict):
            if item not in temporal_covariates:
                print(
                    "Warning: {} is not a temporal covariate, and history_length can not be applied for that.\n".format(
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
            raise ValueError("history_length must be a positive number.\n")

    # prepare data
    # remove columns other than temporal and spatial ids, target , temporal and spatial covariates
    for column in data:
        if bool(re.search("Target \(.*?\)", column)):
            if column not in ["Target (normal)", "Target (cumulative)", "Target (differential)", "Target (moving "
                                                                                                 "average)"]:
                raise ValueError(
                    "modified target must be among these options: Target (normal), Target (cumulative), "
                    "Target (differential), Target (moving average)]\n")

        if column not in ["Normal target", "temporal id", "spatial id", "Target (normal)", "Target (cumulative)",
                          "Target (differential)", "Target (moving average)"]:
            if column not in (spatial_covariates + temporal_covariates):
                data.drop(column, inplace=True, axis=1)

    # sort base on spatial id and temporal ids
    data.sort_values(by=['spatial id', 'temporal id'], ascending=True, ignore_index=True, inplace=True)

    # make data continuous (contain all temporal id in each temporal covariates) ** add a new column and make that 1
    # if this row is new
    data["new_row"] = 0
    grouped_data_spatial = data.groupby("spatial id")
    missed_temporal_ids_data = pd.DataFrame()
    for name_of_group, contents_of_group in grouped_data_spatial:
        for temporal_id in range(min(contents_of_group["temporal id"]) - 1,
                                 max(contents_of_group["temporal id"] + forecast_horizon + 1)):
            if temporal_id not in list(contents_of_group["temporal id"]):
                if temporal_id > 0:
                    new_row = pd.DataFrame()
                    for column in contents_of_group:
                        if column == "temporal id":
                            new_row[column] = [temporal_id]
                        elif column == "spatial id":
                            new_row[column] = [name_of_group]
                        elif column == "new_row":
                            new_row[column] = [1]
                        else:
                            new_row[column] = NaN

                    missed_temporal_ids_data = pd.concat([missed_temporal_ids_data, new_row]).reset_index(drop=True)

    data = pd.concat([missed_temporal_ids_data, data]).reset_index()
    data.sort_values(by=['spatial id', 'temporal id'], ascending=True, ignore_index=True, inplace=True)
    data.drop("index", inplace=True, axis=1)

    # history_length
    temp_data = pd.DataFrame()
    data2 = pd.DataFrame()
    target_data = pd.DataFrame()

    grouped_continuous_data = data.groupby('spatial id')
    for name_of_group, contents_of_group in grouped_continuous_data:
        temp_group = pd.DataFrame()
        for temporal_covar, temporal_covar_his_len in history_length_dict.items():
            
            for i in range(temporal_covar_his_len):
                temp_data = contents_of_group
                if i == 0:
                    col_name = temporal_covar.replace(' ','_') + " t"
                    temp_group[col_name] = contents_of_group[temporal_covar]
                    contents_of_group.drop(temporal_covar, inplace=True, axis=1)
                else:
                    col_name = temporal_covar.replace(' ','_') + " t-" + str(i)
                    temp_group[col_name] = temp_group[temporal_covar.replace(' ','_') + " t"].shift(i).copy()
                temp_data = temp_data.join(temp_group)

        data2 = pd.concat([temp_data, data2]).reset_index()
        data2.drop("index", inplace=True, axis=1)
        data2.sort_values(by=['spatial id', 'temporal id'], ascending=True, ignore_index=True, inplace=True)

        # forecast_horizon:
        target_group = pd.DataFrame()
        if type(forecast_horizon) != int:
            raise TypeError("forecast_horizon must be of type int.\n")
        if forecast_horizon < 0:
            raise ValueError("forecast_horizon must be a positive number.\n")
        for col in contents_of_group:
            if bool(re.search("Target \(.*?\)", col)):
                if col in ["Target (normal)", "Target (cumulative)", "Target (differential)",
                           "Target (moving average)"]:
                    name_of_modified_target = col
                else:
                    raise ValueError(
                        "modified target must be among these options: Target (normal), Target (cumulative), "
                        "Target (differential), Target (moving average)]\n")
        if forecast_horizon == 0:
            target_group["Normal target"] = contents_of_group["Normal target"].copy()
            target_group[name_of_modified_target] = contents_of_group[name_of_modified_target].copy()
        else:
            col_name = "Normal target"
            target_group[col_name] = contents_of_group["Normal target"].shift(-forecast_horizon).copy()
            target_group[name_of_modified_target] = contents_of_group[name_of_modified_target].shift(
                -forecast_horizon).copy()
        contents_of_group.drop("Normal target", inplace=True, axis=1)
        contents_of_group.drop(name_of_modified_target, inplace=True, axis=1)

        target_data = pd.concat([target_data, target_group]).reset_index()
        target_data.drop("index", inplace=True, axis=1)
    data2[col_name] = target_data[col_name]
    data2[name_of_modified_target] = target_data[name_of_modified_target]

    ###################################################################### MAKE SPATIAL HISTORICAL

    # sort
    data2.sort_values(by=['spatial id', 'temporal id'], ascending=True, ignore_index=True, inplace=True)
    # check validity
    # layer_number
    if type(layer_number) != int:
        raise TypeError("The layer_number must be of type int.\n")
    if layer_number <= 0:
        raise ValueError("The layer_number must be a positive number.\n")
    # aggregate_layer
    if type(aggregate_layer) != bool:
        raise TypeError("The aggregate_layer must be of type bool.\n")
    # neighbors_dictionary
    if neighbors_dictionary is not None:
        if type(neighbors_dictionary) != dict:
            raise TypeError("The neighbors_dictionary must be of type dict.\n")
        for key, value in neighbors_dictionary.items():
            if type(key) != int:
                raise TypeError("neighbors_dictionary keys must be spatial ids and of type int.\n")
            if type(value) != list:
                raise TypeError("neighbors_dictionary values must be list of integers.\n")
            for item in value:
                if type(item) != int:
                    raise TypeError("neighbors_dictionary must contain list of int as spatial ids.\n")
                if item not in set(neighbors_dictionary.keys()):
                    raise ValueError("{} is not specified in neighbors_dictionary.\n".format(item))
                if key not in set(neighbors_dictionary[item]):
                    raise ValueError(
                        "{} is among {}'s neighbors but {} is not in {}'s neighbors list.\n".format(item, key, key, item))
                if item not in data["spatial id"].tolist():
                    raise ValueError("{} is not among data's spatial ids.\n".format(item))
        if key not in data["spatial id"].tolist():
            raise ValueError("{} is not among data's spatial ids.\n".format(key))
    # neighboring_covariates
    if type(neighboring_covariates) == list:
        for covar in neighboring_covariates:
            if type(covar) != str:
                raise TypeError("neighboring_covariates must be a list of strings.\n")
            if (covar not in data) and (covar != 'target'):
                print(
                    "Warning: There is no column with name *{}* in data after renaming columns using column "
                    "identifier. This item dropped form neighboring_covariates list.\n".format(
                        covar))
                neighboring_covariates.remove(covar)
            # replace target with *target in case of covariate
            if covar == 'target':
                neighboring_covariates = list(
                    map(lambda x: x.replace('target', '*Normal_target'), neighboring_covariates))
            if covar == 'spatial id' or covar == 'temporal id':
                print(
                    "Warning: neighboring_covariate's items must be covariates specified in column identifier. {} "
                    "dropped from this list.".format(
                        covar))
                neighboring_covariates.remove(covar)

    elif neighboring_covariates is not None:
        raise TypeError("neighboring_covariates must be of type list.\n")
    # aggregation_mode
    if type(aggregation_mode) != str:
        raise TypeError("The aggregation_mode must be of type str.\n")
    if aggregation_mode not in configurations.SPATIAL_HISTORICAL_AGGREGATION_MODES:
        raise ValueError("The aggregation_mode must be among these options: [mean, max, min, sum, std]\n")
    # raise error in case of empty or none neighboring covariates and layer number>0
    # make data continuous for all temporal ids
    grouped_data_spatial = data2.groupby("spatial id")
    missed_temporal_ids_data = pd.DataFrame()
    # change: the below line is un needed.
    # min_temporal_id = data2["temporal id"].min()
    max_temporal_id = data2["temporal id"].max()

    for name_of_group, contents_of_group in grouped_data_spatial:
        # change no need for forecast hoizon: range(1, max_temporal_id + forecast_horizon + 1) --> range(1, max_temporal_id + 1)
        for temporal_id in range(1, max_temporal_id + 1):
            if temporal_id not in list(contents_of_group["temporal id"]):
                new_row = pd.DataFrame()
                for column in contents_of_group:
                    if column == "temporal id":
                        new_row[column] = [temporal_id]
                    elif column == "spatial id":
                        new_row[column] = [name_of_group]
                    elif column == "new_row":
                        new_row[column] = [1]
                    else:
                        new_row[column] = NaN
                missed_temporal_ids_data = pd.concat([missed_temporal_ids_data, new_row]).reset_index(drop=True)
    data2 = pd.concat([missed_temporal_ids_data, data2]).reset_index()
    data2.sort_values(by=['spatial id', 'temporal id'], ascending=True, ignore_index=True, inplace=True)
    data2.drop("index", inplace=True, axis=1)
    # below line is for test remove it latter.
    # return data2

    if (neighbors_dictionary is not None) and (neighboring_covariates is not None):
        # filter columns: columns to be processed.
        filter_columns = []

        for covar in neighboring_covariates:
            filter_column = [col for col in data2 if col.startswith(covar)]
            filter_columns.extend(filter_column)
        
        for covar_col in filter_columns:
            for layer in range(2, layer_number+1):
                # first set the newly created column for the current covariate and layer combination to np.nan 
                # for all records in the DataFrame.
                data2["layer_{}_{}".format(layer-1,covar_col)] = np.nan
                for index,record in enumerate(data2.to_records()):
                    spatial_id = record['spatial id']
                    temporal_id = record['temporal id']
                    # find the neighbors of the current reocrd.
                    neighbors_list = find_neighbors(layer, neighbors_dictionary, spatial_id)
                    neighbors_values_before_agg = []
                    for neighbor_spatial_id in neighbors_list:
                        # get the corresponding features to be added.
                        neighbor_covr_value = data2.loc[(data2["spatial id"] == neighbor_spatial_id) & (data2["temporal id"] == temporal_id)][
                            covar_col].iloc[0]
                        neighbors_values_before_agg.append(neighbor_covr_value)

                    # aggregate
                    if len(neighbors_values_before_agg) == 0 or all(np.isnan(x) for x in neighbors_values_before_agg):
                        continue
                    if aggregation_mode == "mean":
                        final_value = np.nanmean(neighbors_values_before_agg)
                    elif aggregation_mode == "min":
                        final_value = np.nanmin(neighbors_values_before_agg)
                    elif aggregation_mode == "max":
                        final_value = np.nanmax(neighbors_values_before_agg)
                    elif aggregation_mode == "sum":
                        final_value = np.nansum(neighbors_values_before_agg)
                    elif aggregation_mode == "std":
                        final_value = np.nanstd(neighbors_values_before_agg)
                    else:
                        final_value = np.nanmean(neighbors_values_before_agg)

                    data2.at[index, "layer_{}_{}".format(layer-1,covar_col)] = final_value
    
    # aggregate layers
    # aggregate(mean) the covariates in different layers.
    if aggregate_layer:
        columns = {}
        pattern = r'^layer_(\d+)_(.*)$'
        column_names = data2.columns
        for column in column_names:
            match = re.match(pattern, column)
            if match:
                prefix = match.group(2)
                if prefix not in columns.keys():
                    columns[prefix] = [column]
                else:
                    columns[prefix].append(column)
        columns = list(columns.values())
        for columns_to_agg in columns:
            agg_colum_name = re.sub(r"^layer_\d+", "layer_#",columns_to_agg[0] )
            data2[agg_colum_name] = data2[columns_to_agg].mean(axis=1)
            data2 = data2.drop(columns=columns_to_agg)
    # sort, remove new_row = 1 rows
    data2 = data2[data2.new_row != 1]
    data2.sort_values(by=['spatial id', 'temporal id'], ascending=True, ignore_index=True, inplace=True)
    data2.drop("new_row", inplace=True, axis=1)
    data = data2
    return data