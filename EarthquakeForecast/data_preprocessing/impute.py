import warnings

from EarthquakeForecast.data_preprocessing.configurations import IMPUTE_STRATEGIES

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandas as pd
    import numpy as np
    from sklearn.impute import KNNImputer
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    import itertools
    import statistics
    from scipy import stats


# renaming the columns to formal format
def rename_columns(data, column_identifier):
    if column_identifier is None:
        return data
    if type(column_identifier) == dict:
        for key, value in column_identifier.items():
            # check existence of value in dataset
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
            if "temporal ID" in list(column_identifier.keys()):
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


def K_impute(row, df, k, impute_strategy):
    list_of_neighbors_value = []
    # if row had null value impute that with other values in that row
    # only on features
    if list(itertools.chain.from_iterable(df[row[row.isnull()].index].values.tolist())):
        for item in row[row.isnull()].index:
            for i in (((df.iloc[:, 0:-1] - row[0:-1]) ** 2).sum(axis=1) ** 0.5).sort_values(ascending=True)[
                     1:k + 1].index.values.tolist():
                list_of_neighbors_value.append(df[item].values.tolist()[i])

            value = 0
            # mean imputer
            if impute_strategy == "mean":
                value = np.nanmean(list_of_neighbors_value)

            # median imputer
            elif impute_strategy == "median":
                value = np.nanmedian(list_of_neighbors_value)

            # most frequent imputer
            elif impute_strategy == "most_frequent":
                value = stats.mode(list_of_neighbors_value, nan_policy="omit")[0][0]

            # min imputer
            elif impute_strategy == "min":
                value = np.nanmin(list_of_neighbors_value)

            # max imputer
            elif impute_strategy == "max":
                value = np.nanmax(list_of_neighbors_value)

            # impute nan with calculated value
            df.loc[row.name, item] = value

            # clear list of neighbors value
            list_of_neighbors_value.clear()


def impute(data, column_identifier=None, missing_value=np.nan, fill_missing_target=0, K=None, impute_strategy="KNN"):
    """
     Impute the dataset if it contains missing values.
     Most of the time these missing values are encoded as blanks, NaNs or other placeholders.
     We can just discard the row containing any missing values,
     but a better solution is to impute them with a strategy using the known part of the data.
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

    # impute strategy:
    if type(impute_strategy) not in [str, float, int]:
        raise TypeError("impute_strategy must be of type string or a number for fill_value.\n")
    if type(impute_strategy) == str:
        if impute_strategy not in IMPUTE_STRATEGIES:
            raise ValueError("impute_strategy must be among these options: KNN, mean, median, most_frequent, min, max")

    # replace missing_value with np.nan
    data.replace(missing_value, np.nan, inplace=True)

    # drop columns which has all the values as NaN
    size = data.size
    data.dropna(axis=1, how='all', inplace=True)
    if data.size != size:
        print("Warning: Columns which has all the values as NaN were droped.")

    # drop rows which has all the values as NaN
    size = data.size
    data.dropna(axis=0, how='all', inplace=True)
    if data.size != size:
        print("Warning: Rows which has all the values as NaN were droped.")

    # split data into features, ids, target (imputing ids is meaningless and target imputation follows a different
    # approach)
    id_df = pd.DataFrame()
    target_df = pd.DataFrame()
    if "temporal ID" in data:
        id_df["temporal ID"] = data["temporal ID"]
        data.drop("temporal ID", axis=1, inplace=True)

    if "spatial ID" in data:
        id_df["spatial ID"] = data["spatial ID"]
        data.drop("spatial ID", axis=1, inplace=True)

    if "target" in data:
        target_df = data["target"]
        data.drop("target", axis=1, inplace=True)

    # check if all data are numeric
    numeric_data = data.select_dtypes(include=np.number)
    if numeric_data.size != data.size:
        print(
            "Warning: Columns containing non-numeric values were removed. Be careful in specifying the missing value.\n")
        data = numeric_data

    # Impute:
    # set default value of K to size of dataframe
    if K is None:
        K = len(data)

    # KNN imputer
    if impute_strategy == "KNN":
        if K == len(data):
            K = 5
        imputer = KNNImputer(n_neighbors=K)
        data[:] = imputer.fit_transform(data)

    # mean imputer
    if impute_strategy == "mean":
        if K == len(data):
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            data[:] = imputer.fit_transform(data)
        else:
            data.apply(K_impute, df=data, k=K, impute_strategy="mean", axis=1)

    # median imputer
    if impute_strategy == "median":
        if K == len(data):
            imputer = SimpleImputer(missing_values=np.nan, strategy='median')
            data[:] = imputer.fit_transform(data)
        else:
            data.apply(K_impute, df=data, k=K, impute_strategy="median", axis=1)

    # most_frequent imputer
    if impute_strategy == "most_frequent":
        if K == len(data):
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            data[:] = imputer.fit_transform(data)
        else:
            data.apply(K_impute, df=data, k=K, impute_strategy="most_frequent", axis=1)

    # min imputer
    if impute_strategy == "min":
        if K == len(data):
            imputer = ColumnTransformer(
                [(k, SimpleImputer(strategy="constant", fill_value=data[k].min()), [k]) for k in list(data)])
            data[:] = imputer.fit_transform(data)
        else:
            data.apply(K_impute, df=data, k=K, impute_strategy="min", axis=1)

    # max imputer
    if impute_strategy == "max":
        if K == len(data):
            imputer = ColumnTransformer(
                [(k, SimpleImputer(strategy="constant", fill_value=data[k].max()), [k]) for k in list(data)])
            data[:] = imputer.fit_transform(data)
        else:
            data.apply(K_impute, df=data, k=K, impute_strategy="max", axis=1)

    # constant value
    if type(impute_strategy) in [int, float]:
        imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=impute_strategy)
        data[:] = imputer.fit_transform(data)

    # impute target with fill_missing_target
    target_df.fillna(value=fill_missing_target, inplace=True)

    # concat ids, data, target
    data = pd.concat([id_df, data, target_df], axis=1)

    return data
