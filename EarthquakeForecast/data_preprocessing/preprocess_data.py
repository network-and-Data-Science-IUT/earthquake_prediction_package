from impute import impute
from set_spatial_id import set_spatial_id
from set_temporal_id import set_temporal_id
from aggregate import aggregate
from extract_feature import extract_feature
from modify_target import modify_target
from make_historical_data import make_historical_data
import pandas as pd



def preprocess_data(data, forecast_horizon,column_identifier, fill_missing_target=0, K=5, impute_strategy='KNN', pixel_scale = None,
                   kmeans_clusters = None,target_area = None,plot = True,unit="temporal ID", step=1,feature_list=None,aggregation_mode="mean", base=None,
                   class_boundaries=None,target_mode=None, history_length=1,
                         layer_number=0, aggregate_layer=True, neighboring_covariates=None, neighbours_dictionary=None,
                         aggregation_mode_neighbours="mean", history_of_layers=False):

            data = data.copy()
            data = impute(data,column_identifier=column_identifier,
            fill_missing_target=fill_missing_target,impute_strategy=impute_strategy)

            data = set_spatial_id(data,column_identifier=column_identifier,
                    pixel_scale=pixel_scale,kmeans_clusters=kmeans_clusters,target_area=target_area,plot=plot)

            data = set_temporal_id(data,column_identifier={'temporal covariates':column_identifier['temporal covariates']},
                                   unit=unit,step=step)

            data = aggregate(data,column_identifier={'temporal covariates':column_identifier['temporal covariates']},
                                        aggregation_mode=aggregation_mode,base=base)

            extracted_features_temporal,extracted_features_spatial = extract_feature(data,feature_list=feature_list,
                                                                column_identifier=column_identifier)
            
            # merge with the original data.
            data = data.merge(extracted_features_spatial,on='spatial ID')
            data = data.merge(extracted_features_temporal,on = ['spatial ID','temporal ID'])
            
            data = modify_target(data,column_identifier=column_identifier,class_boundaries=class_boundaries,target_mode=target_mode)

            data = make_historical_data(data,column_identifier={'temporal covariates':column_identifier['temporal covariates']},
                                forecast_horizon=forecast_horizon,
                                history_length=history_length,layer_number=layer_number,aggregate_layer=aggregate_layer
                                ,neighboring_covariates=neighboring_covariates,aggregation_mode=aggregation_mode_neighbours,
                                history_of_layers=history_of_layers,neighbours_dictionary=neighbours_dictionary
                                )

            preprocessed_data = data
            return preprocessed_data