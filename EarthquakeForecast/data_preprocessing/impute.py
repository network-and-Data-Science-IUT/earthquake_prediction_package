import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandas as pd
    import numpy as np
    from sklearn.impute import KNNImputer
    from sklearn.impute import SimpleImputer



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


def impute(data, column_identifier=None, fill_missing_target=0, K=None, impute_strategy="KNN"):
    """
     Impute the dataset if it contains missing values.
     Most of the times these missing values are encoded as blanks, NaNs or other placeholders.
     We can just discard the row containing any missing values,
     but a better solution is to impute them with an strategy using the known part of the data.
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
    # data = rename_columns(data.copy(), column_identifier)
    if "temporal ID" not in data:
        raise ValueError("temporal ID is not specified in data.\n")

    # remove instances without temporal ID
    data2 = data[data["temporal ID"].notna()]
    if len(data2) != len(data):
        print("Warning: instances with NaN temporal ID were removed\n")
    data = data2

    # check validity of data after removing NaN temporal IDs
    if data.empty:
        raise ValueError("The input data is empty.\n")

    if "spatial ID" not in data:
        raise ValueError("spatial ID is not specified in data.\n")

    # remove instances without spatial ID
    data2 = data[data["spatial ID"].notna()]
    if len(data2) != len(data):
        print("Warning: instances with NaN spatial ID were removed\n")
    data = data2

    # check validity of data after removing NaN temporal IDs
    if data.empty:
        raise ValueError("The input data is empty.\n")

    # check validity of impute strategy
    if type(impute_strategy) not in [str, float, int]:
        raise TypeError("impute_strategy must be of type string or a number for fill_value.\n")
    if type(impute_strategy) == str:
        if impute_strategy not in IMPUTE_STRATEGIES:
            raise ValueError("impute_strategy must be among these options: KNN, mean, median, most_frequent, min, max")

    # Impute:
    # set default value of K to size of dataframe
    if K == None:
        K = len(data)
    imp_data = data.copy()

    # KNN imputer
    if impute_strategy == "KNN":
        if K == len(data):
            K = 5
        imputer = KNNImputer(n_neighbors=K)
        imp_data[:] = imputer.fit_transform(imp_data)

    # mean imputer
    if impute_strategy == "mean":
        if K == len(data):
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp_data[:] = imputer.fit_transform(imp_data)
        # else

    # median imputer
    if impute_strategy == "median":
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        imp_data[:] = imputer.fit_transform(imp_data)

    # most_frequent imputer
    if impute_strategy == "most_frequent":
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imp_data[:] = imputer.fit_transform(imp_data)

    # min imputer
    if impute_strategy == "min":
        imputer = ColumnTransformer(
            [(k, SimpleImputer(strategy="constant", fill_value=imp_data[k].min()), [k]) for k in list(imp_data)])
        imp_data[:] = imputer.fit_transform(imp_data)

    # max imputer
    if impute_strategy == "max":
        imputer = ColumnTransformer(
            [(k, SimpleImputer(strategy="constant", fill_value=imp_data[k].max()), [k]) for k in list(imp_data)])
        imp_data[:] = imputer.fit_transform(imp_data)

    # constant value
    if type(impute_strategy) in [int, float]:
        imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=impute_strategy)
        imp_data[:] = imputer.fit_transform(imp_data)

    # impute target with fill_missing_target

    # imp_data['target'].fillna(value = fill_missing_target, inplace=True)

    return imp_data


