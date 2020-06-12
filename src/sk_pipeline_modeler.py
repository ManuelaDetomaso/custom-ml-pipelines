import itertools
import numpy as np
from pickle import load
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
            
# Define sklearn workflow structure and basic processing methods

class ColumnTypeSelector(BaseEstimator, TransformerMixin):
    """
    Select specific column types
    """
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X: pd.DataFrame, y: pd.Series = None
            ) -> 'ColumnTypeSelector':
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame):
        """
        Apply selection.
        """
        # return a dataframe with only specified attributes
        return X[self.attribute_names]
    
    def get_params(self, deep=False):
        return {
            "attribute_names": self.attribute_names
            }


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Numerical missing value imputer, filler: median
    """

    def __init__(self, cat_attribs):
        self.cat_attribs = cat_attribs

    def fit(self, X: pd.DataFrame, y: pd.Series = None
            ) -> 'CategoricalImputer':
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()
        # fill missing categories with "Unknown" label
        for feature in self.cat_attribs:
            X[feature] = X[feature].fillna('Unknown')

        print("Categorical missing values imputed.")
        return X

    def get_params(self, deep=False):
        return {
            "cat_attribs": self.cat_attribs
            }


class NumericalImputer(BaseEstimator, TransformerMixin):
    """
    Numerical missing value imputer, filler: median
    """

    def __init__(self, imputed_features_dict):
        self.imputed_features_dict = imputed_features_dict
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None
            ) -> 'NumericalImputer':
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations
        """
        X = X.copy()
        # fill missing values with columns' median
        for feature in X.columns: 
            X[feature].fillna(self.imputed_features_dict[feature], inplace=True)

        print("Numerical missing values imputed.")
        return X

    def get_params(self, deep=True):
        return {
           "imputed_features_dict": self.imputed_features_dict
            }


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Categories as columns of 1 or 0 values."""

    def __init__(self, encoder_dict):
        self.encoder_dict = encoder_dict

    def fit(self, X: pd.DataFrame, y: pd.Series = None
            ) -> 'CategoricalEncoder':
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations
        """
        
        # take categorical columns to one hot encode
        cat_attribs = list(self.encoder_dict.keys())
        dataset = X[cat_attribs]
        
        # check if there are any of the original indicated variables to encode
        useable_encoder_dict = dict(
            [item for item in self.encoder_dict.items() if item[0] in cat_attribs]
        )
        if useable_encoder_dict: 
            # retrieve categories for each features
            usable_categories = [
                np.array(c, dtype=object) for c in useable_encoder_dict.values()
            ]
            # retrieve categorical features' names
            usable_feature_names = list(
                itertools.chain.from_iterable(
                    [k + "_" + c for c in v] for k, v in useable_encoder_dict.items()
                )
            )
            # set names for one hot encoded cat features
            expected_feature_names = list(
                itertools.chain.from_iterable(
                    [k + "_" + c for c in v] for k, v in self.encoder_dict.items()
                )
            )

            # Fit-transform with scikit-learn encoder, with parameters determined above
            encoder = OneHotEncoder(handle_unknown="ignore", categories=usable_categories).fit(
                dataset
            )
            encoded_df = pd.DataFrame(
                encoder.transform(dataset).toarray(), columns=usable_feature_names
            )

            # Handle eventual missing features in incoming data
            for feature in expected_feature_names:
                if feature not in encoded_df.columns:
                    encoded_df[feature] = 0

            # Join encoded columns to original data frame, drop original columns to encode
            all_data = pd.concat([encoded_df.reset_index(), X.reset_index()], axis=1)
            all_data = all_data[
                [c for c in all_data.columns if c not in cat_attribs and "index" not in c]
            ]

            print("Categorical data encoded.")
            return all_data

    def get_params(self, deep=False):
        return {
            "encoder_dict": self.encoder_dict
            }


class NumFeaturesGenerator(BaseEstimator, TransformerMixin):
    """
    Create new variables by combining some input attributes.
    """
    def __init__(self, combined_features, add_bedrooms_per_room):
        self.add_bedrooms_per_room = eval(str(add_bedrooms_per_room))
        self.combined_features = combined_features
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None
            ) -> 'NumFeaturesGenerator':
        """Fit statement to accomodate the sklearn pipeline."""

        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations
        """
        # isolate features to use
        total_rooms = self.combined_features[0]
        total_bedrooms = self.combined_features[1]
        population = self.combined_features[2]
        households = self.combined_features[3]

        # combine features to create new ones
        X["rooms_per_household"] = X[total_rooms] / X[households]
        X["population_per_household"] = X[population] / X[households]

        # set option to create also bedrooms per room
        if self.add_bedrooms_per_room:
            X["bedrooms_per_room"] = X[total_bedrooms] / X[total_rooms]
        
        print("New numerical attributes created.")
        return X

    def get_params(self, deep=False):
        return {
            "add_bedrooms_per_room" : self.add_bedrooms_per_room,
            "combined_features": self.combined_features
            }

class UnrelatedFeaturesDropper(BaseEstimator, TransformerMixin):

    def __init__(self, variables_to_drop=None):
        self.variables = variables_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # encode labels
        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X


class NumFeaturesPreprocessor(BaseEstimator, TransformerMixin):
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None
            ) -> 'NumFeaturesPreprocessor':
        """Fit statement to accomodate the sklearn pipeline."""

        return self
    
    def transform(self, X: pd.DataFrame, config):
        """
        Apply transforms to the numerical attributes
        """
        dataset = X.copy()
        # select numerical features
        num_features = ColumnTypeSelector(config.num_attribs).transform(dataset)

        # impute numerical missing values
        imputed_num_features = (
            NumericalImputer(config.imputed_features_dict).transform(
                num_features
                )
        )

        # add new numerical attributes
        enriched_num_features = (
            NumFeaturesGenerator(config.num_attribs, config.add_bedrooms_per_room).transform(
                imputed_num_features
                )
        )
        return enriched_num_features


class CatFeaturesPreprocessor(BaseEstimator, TransformerMixin):
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None
            ) -> 'CatFeaturesPreprocessor':
        """Fit statement to accomodate the sklearn pipeline."""

        return self
    
    def transform(self, X: pd.DataFrame, config):
        """
        Apply transforms to the categorical attributes
        """
        
        dataset = X.copy()
        # select categorical features
        cat_features = ColumnTypeSelector(config.cat_attribs).transform(dataset)

        # impute categorical missing values
        imputed_cat_features = (
            CategoricalImputer(config.cat_attribs).transform(
                cat_features
                )
        )
        
        # encode categorical attributes
        encoded_cat_features = (
            CategoricalEncoder(config.cat_labels).transform(
                imputed_cat_features
                )
        )
        return encoded_cat_features


class CustomTrainTestSplit:
    
    def __init__(self, **kwargs):
        """
        Apply train / test split either stratified on a given varibale's bins or randomic
        """
        self.stratified = eval(kwargs.get("stratified", True))
        self.var = kwargs.get("strata_var", "median_income")
        self.bins = kwargs.get("bins", [0., 1.5, 3.0, 4.5, 6., np.inf])
        self.labels = kwargs.get("labels", [1, 2, 3, 4, 5])
        self.test_size = kwargs.get("test_size", 0.2)
        self.random_state = kwargs.get("random_state", 42)
        self.target = kwargs.get("TARGET", "median_house_value")
        
    def transform(self, X):
        """
        Apply transforms to get train / test split
        """
        dataset = X.copy()
        bin_values = [eval(str(b)) for b in self.bins]
        
        # cut median income according to bin values
        if self.stratified:
            print("Stratified Train / Test split.")
            dataset["var_cut"] = pd.cut(
                dataset[self.var], bins=bin_values, labels=self.labels
            )
            
            # define split
            split = StratifiedShuffleSplit(
                n_splits=1, test_size=self.test_size, random_state=self.random_state
            )
            # select train / test indicies based on var bins
            for train_index, test_index in split.split(dataset, dataset["var_cut"]):
                train_set = dataset.loc[train_index]
                test_set = dataset.loc[test_index]
        else:
            # standard sklearn random train test split
            print("Random Train / Test split.")
            train_set, test_set = train_test_split(
                dataset, test_size=self.test_size, random_state=self.random_state
                )

        # define the train and test target columns
        train_target = train_set[self.target]
        test_target = test_set[self.target]

        print("Train set dimesions: {}".format(train_set.shape))
        print("Test set dimesions: {}".format(test_set.shape))
        print("Train target dimesions: {}".format(train_target.shape))
        print("Test target dimesions: {}".format(test_target.shape))

        # delete not useful binned var and the target
        for set_ in (train_set, test_set):
            set_.drop(["var_cut", self.target], axis=1, inplace=True)

        return train_set, test_set, train_target, test_target


class CustomModelTrainer:
    """
    Train the model pipeline with ranomized search of the best hyperparameters
    """
    def __init__(self, estimator, **kwargs):
        self.estimator = estimator
        self.params_grid = kwargs.get("PARAMS_GRID")
        self.cd_folds = kwargs.get("CV_FOLDS", 5)
        self.scoring_metrics = kwargs.get("scoring_metrics", "neg_mean_squared_error")
        
    def train(self, X, y):
        # set randomized trainer

        rnd_search = RandomizedSearchCV(
            self.estimator, 
            param_distributions=dict(self.params_grid), 
            cv=self.cd_folds,
            scoring=self.scoring_metrics,
            return_train_score=True
        )

        # Fit the model
        trainer = rnd_search.fit(X, y)
    
        # Return the best fitted estimator
        return trainer


### SKLEARN PIPELINE ###

def sklearn_dataflow(config, scaler=None):
    if scaler is not None:
        scaler=StandardScaler()
    else:
        scaler=scaler

    # Define the Pipeline for Numerical Features
    num_pipeline = Pipeline(
        [
            # select numerical features
            ("selector", ColumnTypeSelector(config.num_attribs)),
            # impute missing values with each feature median value
            ("num_imputer", NumericalImputer(config.imputed_features_dict)),
            # create new numerical attributes
            ("feature_enricher", NumFeaturesGenerator(config.combined_features, config.add_bedrooms_per_room)),
            # apply standardization of values across different features scales
            ('std_scaler', scaler)
        ]
    )

    # Define the Pipeline for Categorical Features
    cat_pipeline = Pipeline(
        [
            # select categorical variables
            ('selector', ColumnTypeSelector(config.cat_attribs)),
            # impute missing values with "Unknown" category
            ('cat_imputer', CategoricalImputer(config.cat_attribs)),
            # apply one-hot-encoding of categories
            ('encoder', CategoricalEncoder(config.cat_labels)),
        ]
    )

    # Combine Numerical and Categorical transformations
    features_pipe = FeatureUnion(
        transformer_list=[
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline),
        ]
    )

    # Set the complete Pipeline in identically used in serving
    housing_pipeline = Pipeline(
        [
            # define all features' transformations
            ("features", features_pipe),
            # add the chosen estimator to the pipeline
            ("estimator", RandomForestRegressor())
        ]
    )

    print("Sklearn Housing Prediction Pipeline created.")
    return num_pipeline, cat_pipeline, features_pipe, housing_pipeline