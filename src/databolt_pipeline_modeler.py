import numpy as np
import os
from pickle import dump, load
import pandas as pd

# Databolt imports
import d6tflow
import luigi
# from luigi.util import inherits

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# Define Databolt workflow structure
    
class TaskCollectData(d6tflow.tasks.TaskCSVPandas):
    """
    Task to collect input data.
    """ 
    do_collection = luigi.BoolParameter(default=True)
    input_kwargs = luigi.DictParameter()
    
    def run(self):
        if self.do_collection:
            # dowload and store housing data loaclly
            from .input_collector import DataAccessor 

            housing = DataAccessor(**self.input_kwargs).load_housing_data()
            # save housing data as Task output
            self.save(housing)
        return housing
            
        
class TaskPartitionData(d6tflow.tasks.TaskCSVPandas): 
    """
    Task to do train/test split.
    """ 
    # list in presist the output names
    persist=['train_set', 'test_set', 'train_target', 'test_target']
    do_split=luigi.BoolParameter(default=True)
    input_kwargs=luigi.DictParameter()
    testing_kwargs=luigi.DictParameter()
    
    def requires(self):
        # specify Task to inherit from
        return TaskCollectData(input_kwargs=self.input_kwargs)
        
    def run(self):
        # Load housing data as TaskCollectData output
        housing_data = self.inputLoad()
        
        if self.do_split:
            from .sk_pipeline_modeler import CustomTrainTestSplit
            # Do custom stratified train/test split
            train_set, test_set, train_target, test_target = (
                CustomTrainTestSplit(**self.testing_kwargs).transform(housing_data)
            )
            # Svae partitions as Task output
            self.save(
                {
                 "train_set":train_set, 
                 "test_set":test_set, 
                 "train_target":train_target, 
                 "test_target":test_target
                }
            )

class TaskPreprocessNumericalFeatures(d6tflow.tasks.TaskCSVPandas):
    """
    Task to preprocess numerical features.
    """ 
    do_split=luigi.BoolParameter(default=True)
    train=luigi.BoolParameter(default=False)
    test=luigi.BoolParameter(default=False)
    config=luigi.Parameter()
    input_kwargs = luigi.DictParameter()
    testing_kwargs = luigi.DictParameter()

    def requires(self):
        # specify Tasks to inherit from
        return {
            "housing_data":TaskCollectData(input_kwargs=self.input_kwargs),
            "partitions": TaskPartitionData(
                input_kwargs=self.input_kwargs, 
                testing_kwargs=self.testing_kwargs
                )       
        }
    
    def run(self):
        
        from .sk_pipeline_modeler import NumFeaturesPreprocessor
        
        if self.do_split:
            # Load either train or test set
            if self.train:
                data_to_process = self.input().get("partitions").get('train_set').load()   
            if self.test:
                data_to_process = self.input().get("partitions").get('test_set').load()
        else:
            # Else load production housing data
            data_to_process = self.input()["housing_data"].load()
            
        # Apply transformations of numerical features
        preprocessed_numerical_features = NumFeaturesPreprocessor().transform(
            data_to_process, config=self.config
        )
        
        # Save preprocessed numerical features as Task output
        self.save(preprocessed_numerical_features)

        return preprocessed_numerical_features
    
    
class TaskFitScaler(d6tflow.tasks.TaskPickle):
    """
    Fit a Standard Scaler for numerical features
    """
    do_split=luigi.BoolParameter(default=False)
    train=luigi.BoolParameter(default=False)
    test=luigi.BoolParameter(default=False)
    input_kwargs=luigi.DictParameter()
    testing_kwargs=luigi.DictParameter()
    config=luigi.Parameter()

    def requires(self):
        # specify Tasks to inherit from
        return TaskPreprocessNumericalFeatures(
            do_split=self.do_split, 
            input_kwargs=self.input_kwargs,
            testing_kwargs=self.testing_kwargs,
            config=self.config
        )
    
    def run(self):
        # Load preprocessed numerical features
        preprocessed_num_data = self.inputLoad()
        # Fit standard scaler
        scaler = StandardScaler().fit(preprocessed_num_data)  
        # Save scaled numerical features as task output
        self.save(scaler)
        return scaler
    
    
class TaskScaleNumericalFeatures(d6tflow.tasks.TaskCSVPandas):
    """
    Task to scale numerical features
    """
    do_split=luigi.BoolParameter(default=True)
    train=luigi.BoolParameter(default=False)
    test=luigi.BoolParameter(default=False)
    input_kwargs=luigi.DictParameter()
    testing_kwargs=luigi.DictParameter()
    config=luigi.Parameter()
    
    def requires(self):
        # specify Tasks to inherit from
        return {
            "scaler": TaskFitScaler(
                do_split=False, 
                train=self.train,
                test=self.test,
                input_kwargs=self.input_kwargs,
                testing_kwargs=self.testing_kwargs,
                config=self.config
                ), 
            "processed_num_data": TaskPreprocessNumericalFeatures(
                do_split=self.do_split, 
                train=self.train,
                test=self.test,
                input_kwargs=self.input_kwargs,
                testing_kwargs=self.testing_kwargs,
                config=self.config
            )
        }
    
    def run(self):
        # Load fitted scaler
        scaler = self.input()['scaler'].load()
        # Load preprocessed numerical features
        processed_num_data = self.input()['processed_num_data'].load()
        # Scale preprocessed features
        scaled_num_data = pd.DataFrame(
            scaler.transform(processed_num_data), columns=processed_num_data.columns
        )
        # Save scaled numerical features as Task output
        self.save(scaled_num_data)
        return scaled_num_data
    

class TaskPreprocessCategoricalFeatures(d6tflow.tasks.TaskCSVPandas):
    """
    Task to preprocess categorical features
    """
    do_split=luigi.BoolParameter(default=True)
    train=luigi.BoolParameter(default=False)
    test=luigi.BoolParameter(default=False)
    input_kwargs=luigi.DictParameter()
    testing_kwargs=luigi.DictParameter()
    config=luigi.Parameter()

    def requires(self):
        # specify Tasks to inherit from
        return {
            "partitions": TaskPartitionData(
                input_kwargs=self.input_kwargs,
                testing_kwargs=self.testing_kwargs
                ), 
            "housing_data": TaskCollectData(input_kwargs=self.input_kwargs,)
        }
    
    def run(self):
        
        from .sk_pipeline_modeler import CatFeaturesPreprocessor
        
        if self.do_split:
            # Set wheater to preprocess train or test
            if self.train:
                data_to_process = self.input().get("partitions").get('train_set').load()   
            if self.test:
                data_to_process = self.input().get("partitions").get('test_set').load()
        else:
            # Else preprocess production housing data
            data_to_process = self.input()['housing_data'].load()
            
        # Apply transformations of numerical features
        preprocessed_categorical_features = CatFeaturesPreprocessor().transform(
            data_to_process, config=self.config
        )
        
        # Save preprocessed numerical features as Task output
        self.save(preprocessed_categorical_features)

        return preprocessed_categorical_features
    
class TaskMergeAllFeatures(d6tflow.tasks.TaskCSVPandas):
    """
    Task to merge together numerical and categorical transformations.
    """
    do_split=luigi.BoolParameter(default=True)
    train=luigi.BoolParameter(default=False)
    test=luigi.BoolParameter(default=False)
    input_kwargs=luigi.DictParameter()
    testing_kwargs=luigi.DictParameter()
    config=luigi.Parameter()
    
    def requires(self):
        # specify Tasks to inherit from
        return {
            "num_data": TaskScaleNumericalFeatures(
                do_split=self.do_split,
                train=self.train,
                test=self.test,
                input_kwargs=self.input_kwargs,
                testing_kwargs=self.testing_kwargs,
                config=self.config
            ), 
            "cat_data": TaskPreprocessCategoricalFeatures(
                do_split=self.do_split,
                train=self.train,
                test=self.test,
                input_kwargs=self.input_kwargs,
                testing_kwargs=self.testing_kwargs,
                config=self.config
            )
        }
    
    def run(self):
        # Load numerical scaled data
        num_data = self.input()['num_data'].load().reset_index()
        # Load categorical encoded data
        cat_data = self.input()['cat_data'].load().reset_index()
        # Merge all features
        all_processed_data = (
            pd.concat([num_data, cat_data], axis=1)
            .drop("index",axis=1)
        )
        print("All features processed.")
        # save all processed features as Task output
        self.save(all_processed_data)
        return all_processed_data
    
    
class TaskTrain(d6tflow.tasks.TaskPickle):
    """
    Task to train a given estimator with Randomized Search 
    of the best hyperparameters
    """
    do_split=luigi.BoolParameter(default=True)
    train=luigi.BoolParameter(default=False)
    test=luigi.BoolParameter(default=False)
    input_kwargs=luigi.DictParameter()
    testing_kwargs=luigi.DictParameter()
    training_kwargs=luigi.DictParameter()
    estimator=luigi.Parameter()
    config=luigi.Parameter()
    
    def requires(self):
        # specify Tasks to inherit from
        return {
            "all_processed_data": TaskMergeAllFeatures(
                do_split=self.do_split,
                train=self.train,
                test=self.test,
                input_kwargs=self.input_kwargs,
                testing_kwargs=self.testing_kwargs,
                config=self.config
            ),
            "partitions": TaskPartitionData(
                input_kwargs=self.input_kwargs, 
                testing_kwargs=self.testing_kwargs
                )       
        }
    
    def run(self):
        # Load all preprocessed features
        X = self.input()['all_processed_data'].load()

        if self.do_split:
            # Load either the train or the test target
            if self.train:
                y = self.input().get("partitions").get('train_target').load()   
            if self.test:
                y = self.input().get("partitions").get('test_target').load() 

        # Random Search of the best hyperparameter
        from .sk_pipeline_modeler import CustomModelTrainer

        trainer=CustomModelTrainer(
            self.estimator, **self.training_kwargs
        ).train(X, y)

        # Set the best fitted estimator
        final_model = trainer.best_estimator_

        # Save the best fitted estimator as Task output
        self.save(final_model)
        return final_model

    
class TaskPredictTrain(d6tflow.tasks.TaskCSVPandas):
    """
    Task to make predictions after model training.
    """
    do_split=luigi.BoolParameter(default=True)
    train=luigi.BoolParameter(default=False)
    test=luigi.BoolParameter(default=False)
    input_kwargs=luigi.DictParameter()
    testing_kwargs=luigi.DictParameter()
    training_kwargs=luigi.DictParameter()
    config=luigi.Parameter()
    estimator=luigi.Parameter()
    
    def requires(self):
        # specify Tasks to inherit from
        return {
            "model": TaskTrain(
                do_split=self.do_split, 
                train=self.train,
                test=self.test,
                input_kwargs=self.input_kwargs,
                testing_kwargs=self.testing_kwargs,
                training_kwargs=self.training_kwargs,
                estimator=self.estimator,
                config=self.config
            ), 
            "all_processed_features": TaskMergeAllFeatures(
                do_split=self.do_split,
                train=self.train,
                test=self.test,
                input_kwargs=self.input_kwargs,
                testing_kwargs=self.testing_kwargs,
                config=self.config
            )
               }
    
    def run(self):
        
        # Load all processed data
        all_processed_data = self.input()['all_processed_features'].load()
        
        # Load the model with TaskTrain
        model = self.input()['model'].load()
        
        print("Doing model training and prediction.")
            
        # Make predictions
        predictions = pd.DataFrame(model.predict(all_processed_data))
        predictions.columns = ['predictions']
        
        # Save predictions as Task output
        self.save(predictions)
        
        return predictions
    
    
class TaskPredictData(d6tflow.tasks.TaskCSVPandas):
    """
    Task to make predictions with a trained model.
    """
    do_split=luigi.BoolParameter(default=True)
    train=luigi.BoolParameter(default=False)
    test=luigi.BoolParameter(default=False)
    input_kwargs=luigi.DictParameter()
    testing_kwargs=luigi.DictParameter()
    config=luigi.Parameter()
    model_path=luigi.Parameter()
    
    
    def requires(self):
        # specify Tasks to inherit from
        return TaskMergeAllFeatures(
                do_split=self.do_split,
                train=self.train,
                test=self.test,
                input_kwargs=self.input_kwargs,
                testing_kwargs=self.testing_kwargs,
                config=self.config
            )
    
    def run(self):
        
        # Load all processed data
        all_processed_data = self.inputLoad()
        
        # Load the a pre-trained model
        print("Using trained model and doing predictions.")
        from pickle import load
        model = load(open(self.model_path, 'rb'))
            
        # Make predictions
        predictions = pd.DataFrame(model.predict(all_processed_data))
        predictions.columns = ['predictions']
        
        # Save predictions as Task output
        self.save(predictions)
        
        return predictions

class TaskEvaluateTrain(d6tflow.tasks.TaskCSVPandas):
    """
    Task to evaluate the model soon after training.
    """
    do_split=luigi.BoolParameter(default=True)
    train=luigi.BoolParameter(default=False)
    test=luigi.BoolParameter(default=False)
    input_kwargs=luigi.DictParameter()
    testing_kwargs=luigi.DictParameter()
    training_kwargs=luigi.DictParameter()
    evaluation=luigi.BoolParameter(default=False)
    estimator=luigi.Parameter()
    config=luigi.Parameter()
    
    
    def requires(self):
        # specify Tasks to inherit from
        return {
            "predictions": TaskPredictTrain(
                do_split=self.do_split,
                train=self.train,
                test=self.test,
                input_kwargs=self.input_kwargs,
                testing_kwargs=self.testing_kwargs,
                training_kwargs=self.training_kwargs,
                estimator=self.estimator,
                config=self.config
            ),
            "partitions": TaskPartitionData(
                input_kwargs=self.input_kwargs,
                testing_kwargs=self.testing_kwargs
                ) 
                }
    
    def run(self):
        
        # evaluation is optional
        if self.evaluation:
            # Load predictions
            predictions = self.input()['predictions'].load()
            # Load the train_set target
            if self.train:
                target = self.input().get("partitions").get('train_target').load()   
                
            
            # Compute the root mean squared error metric
            final_rmse = np.sqrt(mean_squared_error(target, predictions)) 
            print("Model scored.")
            
            return final_rmse


class TaskEvaluateData(d6tflow.tasks.TaskCSVPandas):
    """
    Task to evaluate predictions from any non-train dataset.
    """
    do_split=luigi.BoolParameter(default=True)
    train=luigi.BoolParameter(default=False)
    test=luigi.BoolParameter(default=False)
    evaluation=luigi.BoolParameter(default=False)
    input_kwargs=luigi.DictParameter()
    testing_kwargs=luigi.DictParameter()
    config=luigi.Parameter()
    model_path=luigi.Parameter()
    
    
    def requires(self):
        # specify Tasks to inherit from
        return {
            "predictions": TaskPredictData(
                do_split=self.do_split,
                train=self.train,
                test=self.test,
                input_kwargs=self.input_kwargs,
                testing_kwargs=self.testing_kwargs,
                model_path=self.model_path,
                config=self.config
            ),
            "partitions": TaskPartitionData(
                input_kwargs=self.input_kwargs,
                testing_kwargs=self.testing_kwargs
                )
                }
    
    def run(self):

        # evaluation is optional
        if self.evaluation:
            from sklearn.metrics import mean_squared_error
            # Load predictions
            predictions = self.input()['predictions'].load()
            # Load the test target (in this case)
            # TO DO in production: add also target for prod data 
            target = self.input().get("partitions").get('test_target').load() 
            
            # Compute the root mean squared error metric
            final_rmse = np.sqrt(mean_squared_error(target, predictions)) 
            print("Model scored.")
            return final_rmse


def databolt_training_dataflow(config, do_split=False, input_kwargs=None, train=False, 
                               test=False, training_kwargs=None, testing_kwargs=None,
                               estimator=None, evaluation=False):
    
    # set the end-to-end training flow with databolt
    return d6tflow.run(
        [
            # 1. Load housing data
            TaskCollectData(input_kwargs=input_kwargs, do_collection=True), 
            
            # 2. Set train/test split
            TaskPartitionData(input_kwargs=input_kwargs, testing_kwargs=testing_kwargs), 
            
            # 3. Get Numerical transformations
            TaskPreprocessNumericalFeatures(
                config=config, input_kwargs=input_kwargs, testing_kwargs=testing_kwargs,
                do_split=do_split, train=train, test=test, 
            ),
            
            # 4. Fit the scaler to transofrmed numerical features
            TaskFitScaler(
                do_split=False, train=False, test=False, config=config,
                input_kwargs=input_kwargs, testing_kwargs=testing_kwargs
            ),
            
            # 5. Scale numerical features
            TaskScaleNumericalFeatures(
                config=config, do_split=do_split, train=train, test=test, 
                input_kwargs=input_kwargs, testing_kwargs=testing_kwargs
            ),
            
            # 6. Get Categorical transformations
            TaskPreprocessCategoricalFeatures(
                config=config, do_split=do_split, train=train, test=test, 
                input_kwargs=input_kwargs, testing_kwargs=testing_kwargs
            ),
            
            # 7. Merge all numerical and categorical processed features
            TaskMergeAllFeatures(
                config=config, do_split=do_split, train=train, test=test, 
                input_kwargs=input_kwargs, testing_kwargs=testing_kwargs
            ),
            
            # 8. Train the model
            TaskTrain(
                config=config, do_split=do_split, train=train, test=test, 
                input_kwargs=input_kwargs, testing_kwargs=testing_kwargs,
                training_kwargs=training_kwargs, estimator=estimator
            ),
            
            # 9. Make predictions
            TaskPredictTrain(
                config=config, do_split=do_split, train=train, test=test, 
                input_kwargs=input_kwargs, testing_kwargs=testing_kwargs,
                training_kwargs=training_kwargs, estimator=estimator
            ),
            
            # 10. Evaluate model error
            TaskEvaluateTrain(
                config=config, do_split=do_split, train=train, test=test,  
                input_kwargs=input_kwargs, testing_kwargs=testing_kwargs,
                training_kwargs=training_kwargs, estimator=estimator,
                evaluation=evaluation
            )
            
        ]
    )


def databolt_prediction_dataflow(config, do_split=False, input_kwargs=None, train=False, 
                                  test=False, training_kwargs=None, testing_kwargs=None,
                                  model_path=None, evaluation=False):

    # set the end-to-end prediction flow with databolt
    return d6tflow.run(
        [
            # 1. Load housing data
            TaskCollectData(input_kwargs=input_kwargs, do_collection=True), 
            
            # 2. Set train/test split
            TaskPartitionData(input_kwargs=input_kwargs, testing_kwargs=testing_kwargs), 
            
            # 3. Get Numerical transformations
            TaskPreprocessNumericalFeatures(
                config=config, input_kwargs=input_kwargs, testing_kwargs=testing_kwargs,
                do_split=do_split, train=train, test=test, 
            ),
            
            # 4. Fit the scaler to transofrmed numerical features
            TaskFitScaler(
                do_split=False, train=False, test=False, config=config,
                input_kwargs=input_kwargs, testing_kwargs=testing_kwargs
            ),
            
            # 5. Scale numerical features
            TaskScaleNumericalFeatures(
                config=config, do_split=do_split, train=train, test=test, 
                input_kwargs=input_kwargs, testing_kwargs=testing_kwargs
            ),
            
            # 6. Get Categorical transformations
            TaskPreprocessCategoricalFeatures(
                config=config, do_split=do_split, train=train, test=test, 
                input_kwargs=input_kwargs, testing_kwargs=testing_kwargs
            ),
            
            # 7. Merge all numerical and categorical processed features
            TaskMergeAllFeatures(
                config=config, do_split=do_split, train=train, test=test, 
                input_kwargs=input_kwargs, testing_kwargs=testing_kwargs
            ),
            
            # 8. Make predictions
            TaskPredictData(
                config=config, do_split=do_split, train=train, test=test, 
                input_kwargs=input_kwargs, testing_kwargs=testing_kwargs,
                model_path=model_path
            ),
            
            # 9. Evaluate model error
            TaskEvaluateData(
                config=config, do_split=do_split, train=train, test=test,  
                input_kwargs=input_kwargs, testing_kwargs=testing_kwargs,
                model_path=model_path, evaluation=evaluation
            )
            
        ]
    )            


def get_model_path(config, estimator, input_kwargs, training_kwargs, 
                   testing_kwargs, do_split=True, train=True):
    model_path = (
        TaskTrain(
            config=config, estimator=estimator, input_kwargs=input_kwargs,
            training_kwargs=training_kwargs, testing_kwargs=testing_kwargs,
            do_split=do_split, train=train
                 )
        .output()
        .path
    )
    return model_path