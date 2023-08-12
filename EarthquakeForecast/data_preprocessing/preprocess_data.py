# for testing below lines are commented.
# from impute import impute
# from set_spatial_id import set_spatial_id
# from set_temporal_id import set_temporal_id
# from aggregate import aggregate
# from extract_feature import extract_feature
# from modify_target import modify_target
# from make_historical_data import make_historical_data
# import pandas as pd

import pandas as pd
from data_preprocessing.impute import impute
from data_preprocessing.set_spatial_id import set_spatial_id
from data_preprocessing.set_temporal_id import set_temporal_id
from data_preprocessing.aggregate import aggregate
from data_preprocessing.make_historical_data import make_historical_data
from data_preprocessing.modify_target import modify_target


def preprocess_data(data, forecast_horizon,column_identifier, fill_missing_target=0, K=5, impute_strategy='KNN', pixel_scale = None,
                   kmeans_clusters = None,target_area = None,plot = True,unit="temporal id", step=1,aggregation_mode="mean", base=None,
                   class_boundaries=None,target_mode=None, history_length=1,
                         layer_number=1, aggregate_layer=True, neighboring_covariates=None, neighbors_dictionary=None,
                         aggregation_mode_neighbors="mean"):

            ############################################################# Impute
            imputed_data = impute(data,column_identifier=column_identifier,
            fill_missing_target=fill_missing_target,impute_strategy=impute_strategy)
            ############################################################# Set spatial id
            spatial_scaled_data = set_spatial_id(imputed_data,column_identifier=column_identifier,
                    pixel_scale=pixel_scale,kmeans_clusters=kmeans_clusters,target_area=target_area,plot=plot)
            ############################################################# Set temporal id
            temporal_scaled_data = set_temporal_id(spatial_scaled_data,column_identifier=column_identifier,
                                   unit=unit,step=step)
            ############################################################# Aggregate
            aggregated_data = aggregate(temporal_scaled_data,column_identifier=column_identifier,
                                        aggregation_mode=aggregation_mode,base=base)
            ############################################################# Modify target
            modified_target_data = modify_target(aggregated_data,column_identifier=column_identifier,class_boundaries=class_boundaries,target_mode=target_mode)
            ############################################################# Make data historical
            # loop over different temporal history lengths to create the list historical_datas
            historical_datas = []
            for hl in range(1,history_length+1):
                historical_data = make_historical_data(modified_target_data,column_identifier=column_identifier,
                                        forecast_horizon=forecast_horizon,
                                        history_length=hl,layer_number=layer_number,aggregate_layer=aggregate_layer
                                        ,neighboring_covariates=neighboring_covariates,aggregation_mode=aggregation_mode_neighbors,
                                        neighbors_dictionary=neighbors_dictionary)
                historical_datas.append(historical_data)
            preprocess_datas = historical_datas
            return preprocess_datas