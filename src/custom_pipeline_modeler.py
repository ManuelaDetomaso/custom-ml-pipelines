import attr
import collections
import numpy as np
import os
from pickle import dump, load
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

# Define custom workflow structure 

@attr.s
class Stage:
    name: str = attr.ib(default="")
    env = attr.ib(default=None)

    def run(self):
        pass


@attr.s
class MultiStage(Stage):
    stages = attr.ib(factory=collections.OrderedDict)

    def run(self):
        for name, stage in self.stages.items():
            with self.env.monitoring.monitor(name):
                stage.run()
                print("")

    def append(self, stage: Stage):
        name = stage.name

        if not name:
            raise ValueError(f"Invalid name: {name}")
        elif name in self.stages:
            raise ValueError(f"Stage {name} already defined.")

        # Store parameters
        stage.env = self.env

        # Add to pipeline flow
        self.stages[name] = stage


@attr.s
class CollectDataStage(Stage):
    """
    Donwload and store housing data
    """
    processor_kwargs: dict = attr.ib(factory=dict)

    def process(self) -> pd.DataFrame:
        from .input_collector import DataAccessor

        # Load all housing data
        housing = DataAccessor(**self.processor_kwargs).load_housing_data()
        return housing

    def run(self):
        return self.process()

        
@attr.s
class PartitionDataStage(Stage):
    input_path: str = attr.ib(default=None)
    output_path: str = attr.ib(default=None)
    processor_kwargs: dict = attr.ib(factory=dict)
        
    def load_inputs(self):
        # Load prepared data
        if self.input_path is None:
            raise ValueError("Missing input_path")
        df = pd.read_csv(self.input_path)
        print(f"Read {self.input_path}")
        return df

    def process(self, X: pd.DataFrame):
        from .sk_pipeline_modeler import CustomTrainTestSplit

        # apply train test split
        train_set, test_set, train_target, test_target = (
            CustomTrainTestSplit(**self.processor_kwargs).transform(X)
        )
        train_set['flag_split']=1
        test_set['flag_split']=0
        train_test_sets = pd.concat([train_set, test_set], axis=0)
        return train_set, test_set, train_target, test_target, train_test_sets

    def dump_outputs(self, train_set, test_set, train_target, test_target, train_test_sets):
        
        # Define partitioned data names
        train_set.name="train_set.csv"
        train_target.name="train_target.csv"
        test_set.name="test_set.csv"
        test_target.name="test_target.csv"
        train_test_sets.name = "train_test_sets.csv"
        
        # Write partitioned data
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)
        for data in [train_set, train_target, test_set, test_target, train_test_sets]:
            print(self.output_path +"/"+ data.name)
            data.to_csv(self.output_path +"/"+ data.name, index=data.index.name)
            print("Wrote {}".format(self.output_path +"/"+ data.name))
        return None
        
    def run(self):
        df_in = self.load_inputs()
        train_set, test_set, train_target, test_target, train_test_sets = self.process(df_in)
        return self.dump_outputs(train_set, test_set, train_target, test_target, train_test_sets)

        
@attr.s
class FilterAttrStage(Stage):
    """
    Select only numerical attributes
    """
    input_path: str = attr.ib(default=None)
    output_path: str = attr.ib(default=None)
    processor_kwargs: list = attr.ib(factory=list)

    def load_inputs(self):
        # Load housing data
        if self.input_path is None:
            raise ValueError("Missing input_path")
        df = pd.read_csv(self.input_path)
        print(f"Read {self.input_path}")
        return df

    def process(self, X: pd.DataFrame) -> pd.DataFrame:
        from .sk_pipeline_modeler import ColumnTypeSelector

        # Filter all housing specific features
        num_df = ColumnTypeSelector(self.processor_kwargs).transform(X)
        return num_df

    def dump_outputs(self, df: pd.DataFrame):
        # Write selected numerical features 
        if not os.path.isdir("/".join(self.output_path.split("/")[:-1])):
            os.makedirs("/".join(self.output_path.split("/")[:-1]))
        df.to_csv(self.output_path, index=df.index.name)
        print(f"Wrote {self.output_path}")
        return None
        
    def run(self):
        df_in = self.load_inputs()
        df_out = self.process(df_in)
        return self.dump_outputs(df_out)
    
@attr.s
class ImputeNumericalAttrStage(Stage):
    """
    Impute missing values for numerical attributes
    """
    input_path: str = attr.ib(default=None)
    output_path: str = attr.ib(default=None)
    processor_kwargs: dict = attr.ib(factory=dict)
        
    def load_inputs(self):
        # Load selected numerical data
        if self.input_path is None:
            raise ValueError("Missing input_path")
        df = pd.read_csv(self.input_path)
        print(f"Read {self.input_path}")
        return df

    def process(self, X: pd.DataFrame) -> pd.DataFrame:
        from .sk_pipeline_modeler import NumericalImputer

        # Impute all housing numerical features with respective medians
        imp_df = NumericalImputer(self.processor_kwargs).transform(X)
        return imp_df

    def dump_outputs(self, df: pd.DataFrame):
        # Write imputed numerical features 
        if not os.path.isdir("/".join(self.output_path.split("/")[:-1])):
            os.makedirs("/".join(self.output_path.split("/")[:-1]))
        df.to_csv(self.output_path, index=df.index.name)
        print(f"Wrote {self.output_path}")
        return None
        
    def run(self):
        df_in = self.load_inputs()
        df_out = self.process(df_in)
        return self.dump_outputs(df_out)
       
@attr.s
class CreateNumericalAttrStage(Stage):
    """
    Generate new numerical features 
    """
    input_path: str = attr.ib(default=None)
    output_path: str = attr.ib(default=None)
    processor_args: tuple = attr.ib(factory=tuple)
        
    def load_inputs(self):
        # Load imputed data
        if self.input_path is None:
            raise ValueError("Missing input_path")
        df = pd.read_csv(self.input_path)
        print(f"Read {self.input_path}")
        return df

    def process(self, X: pd.DataFrame) -> pd.DataFrame:
        from .sk_pipeline_modeler import NumFeaturesGenerator
        combined_features, add_bedrooms_per_room = self.processor_args
        
        # Cretae new numerical features with given numerical attributes
        feat_df = NumFeaturesGenerator(
            combined_features, add_bedrooms_per_room
        ).transform(X)
        return feat_df

    def dump_outputs(self, df: pd.DataFrame):
        # Write old and new numerical features 
        if not os.path.isdir("/".join(self.output_path.split("/")[:-1])):
            os.makedirs("/".join(self.output_path.split("/")[:-1]))
        df.to_csv(self.output_path, index=df.index.name)
        print(f"Wrote {self.output_path}")
        return None
        
    def run(self):
        df_in = self.load_inputs()
        df_out = self.process(df_in)
        return self.dump_outputs(df_out)
    
@attr.s   
class ScaleNumericalAttrStage(Stage):
    input_path: str = attr.ib(default=None)
    output_path: str = attr.ib(default=None)
    scaler_path: str = attr.ib(default=None)
        
    def load_inputs(self):
        # Load enriched numerical data
        if self.input_path is None:
            raise ValueError("Missing input_path")
        df = pd.read_csv(self.input_path)
        print(f"Read {self.input_path}")
        return df

    def process(self, X: pd.DataFrame) -> pd.DataFrame:
        from sklearn.preprocessing import StandardScaler

        # Apply standard scaling of numerical features
        df = X.copy()
        scaler = StandardScaler().fit(df)
        scaled_features = scaler.transform(df)
        scaled_df = pd.DataFrame(
            scaled_features, index=df.index, columns=df.columns
        )
        return scaler, scaled_df

    def dump_outputs(self, scaler, df):
        # Write scaled numerical features and save fitted scaler
        if not os.path.isdir("/".join(self.output_path.split("/")[:-1])):
            os.makedirs("/".join(self.output_path.split("/")[:-1]))   
        if not os.path.isdir("/".join(self.scaler_path.split("/")[:-1])):
            os.makedirs("/".join(self.scaler_path.split("/")[:-1]))            
        dump(scaler, open(self.scaler_path, "wb"))
        print(f"Wrote {self.scaler_path}")
        df.to_csv(self.output_path, index=df.index.name)
        print(f"Wrote {self.output_path}")
        return None
        
    def run(self):
        df_in = self.load_inputs()
        scaler, df_out = self.process(df_in)
        return self.dump_outputs(scaler, df_out)

    
@attr.s    
class ImputeCategoricalAttrStage(Stage):
    input_path: str = attr.ib(default=None)
    output_path: str = attr.ib(default=None)
    processor_kwargs: list = attr.ib(factory=list)
        
    def load_inputs(self):
        # Load categorical data
        if self.input_path is None:
            raise ValueError("Missing input_path")
        df = pd.read_csv(self.input_path)
        print(f"Read {self.input_path}")
        return df

    def process(self, X: pd.DataFrame) -> pd.DataFrame:
        from .sk_pipeline_modeler import CategoricalImputer

        # Impute categorical features
        imp_df = CategoricalImputer(self.processor_kwargs).transform(X)
        return imp_df

    def dump_outputs(self, df: pd.DataFrame):
        # Write imputed categorical features
        if not os.path.isdir("/".join(self.output_path.split("/")[:-1])):
            os.makedirs("/".join(self.output_path.split("/")[:-1]))
        df.to_csv(self.output_path, index=df.index.name)
        print(f"Wrote {self.output_path}")
        return None
    
    def run(self):
        df_in = self.load_inputs()
        df_out = self.process(df_in)
        return self.dump_outputs(df_out)
    
    
@attr.s   
class EncodeCategoricalAttrStage(Stage):
    input_path: str = attr.ib(default=None)
    output_path: str = attr.ib(default=None)
    processor_kwargs: dict = attr.ib(factory=dict)
        
    def load_inputs(self):
        # Load imputed categorical data
        if self.input_path is None:
            raise ValueError("Missing input_path")
        df = pd.read_csv(self.input_path)
        print(f"Read {self.input_path}")
        return df

    def process(self, X: pd.DataFrame) -> pd.DataFrame:
        from .sk_pipeline_modeler import CategoricalEncoder

        # Encode categorical features
        enc_df = CategoricalEncoder(self.processor_kwargs).transform(X)
        return enc_df

    def dump_outputs(self, df: pd.DataFrame):
        # Write encoded categorical features
        if not os.path.isdir("/".join(self.output_path.split("/")[:-1])):
            os.makedirs("/".join(self.output_path.split("/")[:-1]))
        df.to_csv(self.output_path, index=df.index.name)
        print(f"Wrote {self.output_path}")
        return None
        
    def run(self):
        df_in = self.load_inputs()
        df_out = self.process(df_in)
        return self.dump_outputs(df_out)

@attr.s
class MergeFeaturesStage(Stage):
    num_input_path: str = attr.ib(default=None)
    cat_input_path: str = attr.ib(default=None)
    output_path: str = attr.ib(default=None)

    def load_inputs(self):
        # Load scaled and encoded data
        if (self.num_input_path is None or self.cat_input_path is None):
            raise ValueError("Missing input_path")
        df_num = pd.read_csv(self.num_input_path)
        print(f"Read {self.num_input_path}")
        df_cat = pd.read_csv(self.cat_input_path)
        print(f"Read {self.cat_input_path}")
        return df_num, df_cat

    def process(self, df_num: pd.DataFrame, df_cat: pd.DataFrame) -> pd.DataFrame:

        # Merge numerical and categorical transformations
        return pd.concat([df_num.reset_index(), df_cat.reset_index()], axis=1)

    def dump_outputs(self, df: pd.DataFrame):
        # Write processed features
        if not os.path.isdir("/".join(self.output_path.split("/")[:-1])):
            os.makedirs("/".join(self.output_path.split("/")[:-1]))
        df.to_csv(self.output_path, index=df.index.name)
        print(f"Wrote {self.output_path}")
        return None
        
    def run(self):
        df_num, df_cat = self.load_inputs()
        df_out = self.process(df_num, df_cat)
        return self.dump_outputs(df_out)
    
@attr.s
class TrainDataStage(Stage):
    input_path: str = attr.ib(default=None)
    target_path: str = attr.ib(default=None)
    output_path: str = attr.ib(default=None)
    estimator = attr.ib(default=RandomForestRegressor())
    processor_kwargs: dict = attr.ib(factory=dict)
        
    def load_inputs(self):
        # Load partitioned data categorical data
        if self.input_path is None:
            raise ValueError("Missing input_path")
        train_set = pd.read_csv(self.input_path)
        train_target = pd.read_csv(self.target_path)
        train_target = train_target.T.squeeze()
        print(f"Read {self.input_path}")
        return train_set, train_target

    def process(self, X: pd.DataFrame, y: pd.Series):
        from .sk_pipeline_modeler import CustomModelTrainer

        # Train the model
        estimator = self.estimator
        trainer = CustomModelTrainer(estimator, **self.processor_kwargs).train(X, y)

        return trainer.best_estimator_

    def dump_outputs(self, trained_model):
        from pickle import dump
        # Write trained model
        if not os.path.isdir("/".join(self.output_path.split("/")[:-1])):
            os.makedirs("/".join(self.output_path.split("/")[:-1]))
        dump(trained_model, open(self.output_path, 'wb'))
        print(f"Wrote {self.output_path}")
        return None
        
    def run(self):
        X, y = self.load_inputs()
        trained_model = self.process(X,y)
        return self.dump_outputs(trained_model)
    
    
@attr.s
class PredictStage(Stage):
    input_path: str = attr.ib(default=None)
    model_path = attr.ib(default=None)
    output_path: str = attr.ib(default=None)
        
    def load_inputs(self):
        from pickle import load
        # Load processed data and the trained model
        if self.input_path is None:
            raise ValueError("Missing input_path")
        if self.model_path is None:
            raise ValueError("Missing model_path")
        final_model = load(open(self.model_path, 'rb'))
        print(f"Load model from {self.model_path}")
        df = pd.read_csv(self.input_path)
        print(f"Read {self.input_path}")
        return final_model, df

    def process(self, model, X: pd.DataFrame) -> pd.DataFrame:
        
        # Generate predictions
        final_predictions = pd.DataFrame(model.predict(X))
        final_predictions.columns=['predictions']
        return final_predictions

    def dump_outputs(self, df: pd.DataFrame):
        # Write predictions
        if not os.path.isdir("/".join(self.output_path.split("/")[:-1])):
            os.makedirs("/".join(self.output_path.split("/")[:-1]))
        df.to_csv(self.output_path, index=df.index.name)
        print(f"Wrote {self.output_path}")
        return None
        
    def run(self):
        model, df_in = self.load_inputs()
        df_out = self.process(model=model, X=df_in)
        return self.dump_outputs(df_out)
    
    
@attr.s
class EvaluateStage(Stage):
    target_path: str = attr.ib(default=None)
    predictions_path: str = attr.ib(default=None)
    output_path: str = attr.ib(default=None)
        
    def load_inputs(self):
        # Load predictions and the target if any
        if self.predictions_path is None:
            raise ValueError("Missing predictions_path")
        predictions = pd.read_csv(self.predictions_path).T.squeeze()
        print(f"Read {self.predictions_path}")
        if self.target_path is not None:
            target = pd.read_csv(self.target_path).T.squeeze()
            print(f"Read {self.target_path}")
            return target, predictions
        else:
            return predictions

    def process(self, target: pd.Series, predictions: pd.Series):
        
        from sklearn.metrics import mean_squared_error
        #Calculate root mean squared error
        results = pd.Series(np.sqrt(mean_squared_error(target, predictions)))
        return results.rename("RMSE")
        
    def run(self):
        target, predictions = self.load_inputs()
        results = self.process(target, predictions)
        return results
    

def custom_dataflow(args, config, data_path, target_path=None, env=None) -> Stage:
    
    # Build dataflow
    print("Building Housing model training dataflow ........")
    custom_pipeline = MultiStage(name="Housing-Model-Full-Custom-Training", env=env)

    stage_1 = CollectDataStage(
        name="collect-data-stage",
        processor_kwargs=config.input_config,
    )
    custom_pipeline.append(stage_1)
    
    stage_2 = PartitionDataStage(
        name="split-train-test-stage",
        input_path=config.housing_data_path,
        output_path=config.partitions_path,
        processor_kwargs=config.testing_config
    )
    custom_pipeline.append(stage_2)
    
    stage_3 = FilterAttrStage(
        name="select-numerical-attributes-stage",
        input_path=data_path,
        output_path=config.numerical_features_path,
        processor_kwargs=config.num_attribs,
    )
    custom_pipeline.append(stage_3)
    
    stage_4 = ImputeNumericalAttrStage(
        name="impute-numerical-missing-values-stage",
        input_path=config.numerical_features_path,
        output_path=config.imputed_numerical_features_path,
        processor_kwargs=config.imputed_features_dict,
    )
    custom_pipeline.append(stage_4)
    
    stage_5 = CreateNumericalAttrStage(
        name="create-new-attributes-stage",
        input_path=config.imputed_numerical_features_path,
        output_path=config.enriched_numerical_features_path,
        processor_args=(config.num_attribs, config.add_bedrooms_per_room)
    )
    custom_pipeline.append(stage_5)
    
    stage_6 = ScaleNumericalAttrStage(
        name="scale-numerical-attributes-stage",
        input_path=config.enriched_numerical_features_path,
        output_path=config.scaled_numerical_features_path,
        scaler_path=config.scaler_path,
    )
    custom_pipeline.append(stage_6)
    
    stage_7 = FilterAttrStage(
        name="select-categorical-attributes-stage",
        input_path=data_path,
        output_path=config.categorical_features_path,
        processor_kwargs=config.cat_attribs,
    )
    custom_pipeline.append(stage_7)
    
    stage_8 = ImputeCategoricalAttrStage(
        name="impute-categorical-missing-values-stage",
        input_path=config.categorical_features_path,
        output_path=config.imputed_categorical_features_path,
        processor_kwargs=config.cat_attribs,
    )
    custom_pipeline.append(stage_8)
    
    stage_9 = EncodeCategoricalAttrStage(
        name="encode-categorical-features-stage",
        input_path=config.imputed_categorical_features_path,
        output_path=config.encoded_categorical_features_path,
        processor_kwargs=config.cat_labels,
    )
    custom_pipeline.append(stage_9)
    
    stage_10 = MergeFeaturesStage(
        name="merge-num-cat-features-stage",
        num_input_path=config.scaled_numerical_features_path,
        cat_input_path=config.encoded_categorical_features_path,
        output_path=config.all_processed_data_path,
    )
    custom_pipeline.append(stage_10)

    stage_11 = TrainDataStage(
        name="train-model-stage",
        input_path=config.all_processed_data_path,
        target_path=config.train_target_path,
        estimator=RandomForestRegressor(),
        output_path=config.model_path,
        processor_kwargs=config.training_config, 
    )
    custom_pipeline.append(stage_11)
    
    stage_12=PredictStage(
        name="make-predictions-stage",
        input_path=config.all_processed_data_path,
        model_path=config.model_path,
        output_path=config.predictions_path
    )
    custom_pipeline.append(stage_12)
    
    stage_13=EvaluateStage(
        name="evaluate-results-stage",
        target_path=target_path,
        predictions_path=config.predictions_path,
        output_path=config.results_path
    )
    custom_pipeline.append(stage_13)

    return custom_pipeline