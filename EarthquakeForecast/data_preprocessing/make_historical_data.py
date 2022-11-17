import pandas as pd
import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
from numpy import NaN
import re

import configurations

'''
Transforming data to the historical format and extract features. This subprocess prepare the final data frame including features and target variable for modeling.

The set of features is consisted of:

1. Spatial covariates, temporal covariates at current temporal unit (t)

2. Historical values of these covariates at h-1(where h is history length) previous temporal units (t-1 , t-2 , … , t-h+1).

3. The covariates of the L-layer pixels(where L is layer_number L = (0,1,…,maximum_layer_number)) at ‘history_length_of_neighbors’-1 previous temporal units(t-1 , t-2 , … , t-‘history_length_of_neighbors’+1)

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

        # temporal ID
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

        # spatial ID
        if "spatial ID" not in data:
            if "spatial ID" not in list(column_identifier.keys()):
                raise ValueError("spatial ID is not specified in column_identifier.")
            else:
                if column_identifier["spatial ID"] not in data:
                    raise ValueError("temporal ID and spatial ID should be unique columns.")
                data.rename(columns={column_identifier["spatial ID"]: "spatial ID"}, inplace=True)

        elif "spatial ID" in list(column_identifier.keys()):
            print(
                "Warning: spatial ID is defined in both data columns and colum_identifier. data columns have higher "
                "priority than column_identifier, so spatial ID has been removed from column_identifier.\n")
            column_identifier.pop("spatial ID")

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
            print(
                "Warning: \"Normal target\" and \"Target (...)\" exist in data columns and target is defined in "
                "colum_identifier.\ndata columns have higher priority than column_identifier, so target has been "
                "removed from column_identifier.\n")
            column_identifier.pop("Normal target")

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


def find_neighbors(layer_number, neighbours_dictionary, spatial_id):
    graph_pairs = []
    for node, neighbors_list in neighbours_dictionary.items():
        for neighbor in neighbors_list:
            if ((node, neighbor) not in graph_pairs) and ((neighbor, node) not in graph_pairs):
                graph_pairs.append((node, neighbor))

    G = nx.Graph()
    G.add_edges_from(graph_pairs)
    path_length = nx.single_source_shortest_path_length(G, spatial_id)
    list_of_neighbors = []

    for node, length in path_length.items():
        if length == layer_number:
            list_of_neighbors.append(node)

    return list_of_neighbors


def make_historical_data(data, column_identifier, forecast_horizon, history_length=1,
                         layer_number=0, aggregate_layer=True, neighboring_covariates=None, neighbours_dictionary=None,
                         aggregation_mode="mean", history_of_layers=False):
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
            grouped_data = data.groupby("spatial ID")
            for name_of_group, contents_of_group in grouped_data:
                spatial_covar_list_in_group = contents_of_group[spatial_covar].tolist()
                if not (spatial_covar_list_in_group.count(spatial_covar_list_in_group[0]) == len(
                        spatial_covar_list_in_group)):
                    raise ValueError("contents of spatial covariates, should not change in different temporal IDs.\n")

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

    # MAKE TEMPORAL HISTORICAL
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
        if column not in ["Normal target", "temporal ID", "spatial ID", "Target (normal)", "Target (cumulative)",
                          "Target (differential)", "Target (moving average)"]:
            if column not in (spatial_covariates + temporal_covariates):
                data.drop(column, inplace=True, axis=1)
                print(
                    "Warning: {} is neither target nor temporal ID nor spatial ID nor among covariates. this column "
                    "was removed from data.\n".format(
                        column))

    # sort base on spatial id and temporal ids
    data.sort_values(by=['spatial ID', 'temporal ID'], ascending=True, ignore_index=True, inplace=True)

    # make data continous (contain all temporal id in each temporal covariates) ** add a new column and make that 1
    # if this row is new
    data["new_row"] = 0
    grouped_data_spatial = data.groupby("spatial ID")
    missed_temporal_ids_data = pd.DataFrame()
    for name_of_group, contents_of_group in grouped_data_spatial:
        for temporal_id in range(min(contents_of_group["temporal ID"]) - 1,
                                 max(contents_of_group["temporal ID"] + forecast_horizon + 1)):
            if temporal_id not in list(contents_of_group["temporal ID"]):
                if temporal_id > 0:
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

    # history_length
    temp_data = pd.DataFrame()
    data2 = pd.DataFrame()
    target_data = pd.DataFrame()

    grouped_continuous_data = data.groupby('spatial ID')
    for name_of_group, contents_of_group in grouped_continuous_data:
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
        for col in contents_of_group:
            if bool(re.search("Target \(.*?\)", col)):
                name_of_modified_target = col
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

    # MAKE SPATIAL HISTORICAL

    # sort
    data2.sort_values(by=['spatial ID', 'temporal ID'], ascending=True, ignore_index=True, inplace=True)
    # check validity
    # layer_number
    if type(layer_number) != int:
        raise TypeError("The layer_number must be of type int.\n")
    if layer_number < 0:
        raise ValueError("The layer_number must be a none negative number.\n")
    # aggregate_layer
    if type(aggregate_layer) != bool:
        raise TypeError("The aggregate_layer must be of type bool.\n")
    # neighbours_dictionary
    if type(neighbours_dictionary) != dict:
        raise TypeError("The neighbours_dictionary must be of type dict.\n")
    for key, value in neighbours_dictionary.items():
        if type(key) != int:
            raise TypeError("neighbours_dictionary keys must be spatial IDs and of type int.\n")
        if type(value) != list:
            raise TypeError("neighbours_dictionary values must be list of integers.\n")
        for item in value:
            if type(item) != int:
                raise TypeError("neighbours_dictionary must contain list of int as spatial IDs.\n")
            if item not in set(neighbours_dictionary.keys()):
                raise ValueError("{} is not specified in neighbours_dictionary.\n".format(item))
            if key not in set(neighbours_dictionary[item]):
                raise ValueError(
                    "{} is among {}'s neighbours but {} is not in {}'s neighbours list.\n".format(item, key, key, item))
            if item not in data["spatial ID"].tolist():
                raise ValueError("{} is not among data's spatial IDs.\n".format(item))
        if key not in data["spatial ID"].tolist():
            raise ValueError("{} is not among data's spatial IDs.\n".format(key))
    # neighboring_covariates
    if type(neighboring_covariates) == list:
        for covar in neighboring_covariates:
            if type(covar) != str:
                raise TypeError("neighboring_covariates must be a list of strings.\n")
            if covar not in data:
                print(
                    "Warning: There is no column with name *{}* in data after renaming columns using column "
                    "identifier. This item dropped form neighboring_covariates list.\n".format(
                        covar))
                neighboring_covariates.remove(covar)
            # replace target with *target in case of covariate
            if covar == 'Normal target':
                neighboring_covariates = list(
                    map(lambda x: x.replace('Normal target', '*Normal target'), neighboring_covariates))
            if covar == 'spatial ID' or covar == 'temporal ID':
                print(
                    "Warning: neighboring_covariates' items must be covariates specified in column identifier. {} "
                    "dropped from this list.".format(
                        covar))
                neighboring_covariates.remove(covar)

    elif neighboring_covariates is not None:
        raise TypeError("neighboring_covariates must be of type list.\n")
    # aggregation_mode
    if type(aggregation_mode) != str:
        raise TypeError("The aggregation_mode must be of type str.\n")
    if aggregation_mode not in configurations.neighbours_values_before_agg:
        raise ValueError("The aggregation_mode must be among these options: [mean, max, min, sum, std]\n")
    # raise error in case of empty or none neighboring covariates and layer number>0
    # make data continous for all temporal IDs
    grouped_data_spatial = data2.groupby("spatial ID")
    missed_temporal_ids_data = pd.DataFrame()
    min_temporal_id = data2["temporal ID"].min()
    max_temporal_id = data2["temporal ID"].max()

    for name_of_group, contents_of_group in grouped_data_spatial:
        for temporal_id in range(1, max_temporal_id + forecast_horizon + 1):
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
    data2 = pd.concat([missed_temporal_ids_data, data2]).reset_index()
    data2.sort_values(by=['spatial ID', 'temporal ID'], ascending=True, ignore_index=True, inplace=True)
    data2.drop("index", inplace=True, axis=1)

    if neighbours_dictionary is not None:
        if history_of_layers:
            filter_columns = []
            for covar in neighboring_covariates:
                filter_column = [col for col in data2 if col.startswith(covar)]
                filter_columns.extend(filter_column)

            for covar_col in data2[filter_columns]:
                for layer in range(1, layer_number + 1):
                    index = -1
                    for spa_id in data2["spatial ID"]:
                        index += 1
                        tempo_id = data2["temporal ID"][index]
                        neighbours_list = find_neighbors(layer, neighbours_dictionary, spa_id)
                        neighbours_values_before_agg = []
                        for neighbour_id in neighbours_list:
                            df = data2.loc[(data2["spatial ID"] == neighbour_id) & (data2["temporal ID"] == tempo_id)][
                                covar_col]
                            neighbours_values_before_agg.append(df.tolist()[0])
                        # aggregate
                        if aggregation_mode == "mean":
                            final_value = np.nanmean(neighbours_values_before_agg)
                        elif aggregation_mode == "min":
                            final_value = np.nanmin(neighbours_values_before_agg)
                        elif aggregation_mode == "max":
                            final_value = np.nanmax(neighbours_values_before_agg)
                        elif aggregation_mode == "sum":
                            final_value = np.nansum(neighbours_values_before_agg)
                        elif aggregation_mode == "std":
                            final_value = np.nanstd(neighbours_values_before_agg)
                        else:
                            final_value = np.nanmean(neighbours_values_before_agg)
                        # make new column for each combination of layer and covariate
                        data2["layer {} {}".format(layer, covar_col)] = NaN
                        data2.at[index, "layer {} {}".format(layer, covar_col)] = final_value
        # aggregate layers
        # aggregate columns start with format layer (#layer numnber#) (covar x) to aggregated layers (covar x)
        if aggregate_layer == True:
            columns_of_data = data2.columns
            for col in data2:
                i = data2.columns.get_loc(col)
                column_to_aggregate = []
                columns_of_data_list = columns_of_data.tolist()
                for column in columns_of_data_list:
                    if columns_of_data_list.index(column) > i:
                        if re.sub("layer \d ", "", col) == re.sub("layer \d ", "", column):
                            if column.startswith("layer"):
                                if not column.startswith("#"):
                                    column_to_aggregate.append(column)
                                    columns_of_data_list[i] = "#{}".format(column)
                if len(column_to_aggregate) > 1:
                    name_of_mean = re.sub("layer \d ", "", column_to_aggregate[0])
                    name_of_mean = "layer 0 " + name_of_mean
                    data2[name_of_mean] = data2[column_to_aggregate].mean(axis=1)
                    data2.drop(column_to_aggregate, axis=1)

            # if history_of_layers == False:
            # merge columns with format layer n ... t t-1  to layer n ...

    # sort, remove new_row = 1 rows
    data2 = data2[data2.new_row != 1]
    data2.sort_values(by=['spatial ID', 'temporal ID'], ascending=True, ignore_index=True, inplace=True)
    data2.drop("new_row", inplace=True, axis=1)
    data = data2

    for col in data:
        number_of_layer = re.sub("layer ", "", col)[0]
        covar_name = re.sub(" t-\d", "", col)
        covar_name = re.sub(" t\b", "", covar_name)
        covar_name = re.sub("layer \d ", "", covar_name)
        history_len = re.sub("layer \d ", "", col)
        history_len = history_len.replace(covar_name, "")
        covar_name = re.sub(" t-\d", "", col)
        if number_of_layer != 0:
            data.rename({col: "{} l{}{}".format(covar_name, number_of_layer, history_len)})
        else:
            data.rename({col: "{} neighbours{}".format(covar_name, history_len)})

    return data
