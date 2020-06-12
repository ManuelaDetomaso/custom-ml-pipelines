import json

class Config:
    """
    Define config settings for the Housing Model Prediction Pipeline
    """

    def __init__(self, conf):
        # Set input data path
        self.input_path = conf.get("INPUT_CONFIG").get("INPUT_PATH", "input")

        # Set other input configurations
        self.input_config = conf.get("INPUT_CONFIG", {"Fake":"Fake"})
        self.housing_data_path = self.input_path +"/"+ self.input_config["INPUT_FILENAME"]
        
        # Set output files configurations
        self.output_config = conf.get("OUTPUT_CONFIG", {})

        # Set output data path
        self.output_path = conf.get("OUTPUT_PATH", "data/outputs")

        self.numerical_features_path = self.output_path +"/"+ self.output_config["NUMFEATURES_FILENAME"]
        self.enriched_numerical_features_path = self.output_path +"/"+ self.output_config["ENRICHEDFEATURES_FILENAME"]
        self.imputed_numerical_features_path = self.output_path +"/"+ self.output_config["IMPUTED_NUMFEATURES_FILENAME"]
        self.scaled_numerical_features_path = self.output_path +"/"+ self.output_config["SCALED_NUMFEATURES_FILENAME"]
        self.categorical_features_path = self.output_path +"/"+ self.output_config["CATFEATURES_FILENAME"]
        self.imputed_categorical_features_path = self.output_path +"/"+ self.output_config["IMPUTED_CATFEATURES_FILENAME"]
        self.encoded_categorical_features_path = self.output_path +"/"+ self.output_config["ENCODED_CATFEATURES_FILENAME"]
        self.all_processed_data_path = self.output_path +"/"+ self.output_config["ALL_PROCESSED_FILENAME"]

        # Set training configurations
        self.training_config = conf.get("TRAINING_CONFIG", {})

        # Set testing configurations
        self.testing_config = conf.get("TESTING_CONFIG", {})

        # Set partitioned data paths
        self.partitions_path = conf.get("OUTPUT_CONFIG").get("PARTITIONS_PATH", "data/outputs/partitions")
        self.trainset_path = self.partitions_path +"/"+ self.output_config.get("TRAINSET_FILENAME")
        self.train_target_path = self.partitions_path +"/"+ self.output_config.get("TRAINSET_TARGET_FILENAME")
        self.testset_path = self.partitions_path +"/"+ self.output_config.get("TESTSET_FILENAME")
        self.test_target_path = self.partitions_path +"/"+ self.output_config.get("TESTSET_TARGET_FILENAME")

        # Set pipeline arguments for numerical features
        self.num_attribs = [k for k,v in conf.get("VARIABLES").items() if "object" not in v]
        self.imputed_features_dict = conf.get("NUMERICAL_VARS_MEDIANS", {})
        self.combined_features = conf.get("COMBINED_VARIABLES", [])
        self.add_bedrooms_per_room = conf.get("ADD_BEDROOMS_PER_ROOM", None)

        # Set pipeline arguments for categorical features
        self.cat_attribs = [k for k,v in conf.get("VARIABLES").items() if "object" in v]
        self.cat_labels = conf.get("CATEGORICAL_VARS_LABELS", [])

        # Set pipeline argument for feature enrichment
        self.combined_features = conf.get("COMBINED_VARIABLES", [])
        self.add_bedrooms_per_room = conf.get("ADD_BEDROOMS_PER_ROOM", None)

        # Set model hyperparameters configurations 
        self.params_grid = self.training_config.get("PARAMS_GRID", {})

        # Set trained model artifacts
        self.artifacts_path = conf.get("ARTIFACTS_PATH", "data/outputs/model_artifacts")
        self.scaler_path = self.artifacts_path + "/" + conf.get("SCALER_FILENAME", "numerical_scaler.pkl")
        self.model_path = self.artifacts_path + "/" + conf.get("MODEL_FILENAME", "trained_housing_model.pkl")

        # Predictions settings
        self.predictions_path = self.output_path + "/" + conf.get("PREDICTIONS_FILENAME", "predictions.csv")

        # Results settings
        self.results_path = self.output_path + "/" + conf.get("RESULTS_FILENAME", "results.csv")


def read_config(config_path):
    """
    Read config files and return Config object with settings
    """
    with open(config_path) as f:
        conf = json.load(f)

    return Config(conf)
