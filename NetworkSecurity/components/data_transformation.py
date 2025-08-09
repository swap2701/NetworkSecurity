import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from NetworkSecurity.constants.training_pipeline import TARGET_COLUMN
from NetworkSecurity.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from NetworkSecurity.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
)
from NetworkSecurity.entity.config_entity import DataTransformationConfig
from NetworkSecurity.utils.main_utils.utils import save_numpy_array_data,save_object

from NetworkSecurity.logging.logger import logging
from NetworkSecurity.exception.Exception import NetworkSecurityException

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact=data_validation_artifact
            self.data_transformation_config=data_transformation_config
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)


    ## Read function for Train,Test files
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    
    ### Funtion for KNN Imputer
    def get_data_transformer_object(cls)->Pipeline:
        """
        It initialises a KNN Imputer object with the parameters specified in the training_pipeline.py file
        and return a Pipeline object with the KNN Imputer object as the first key

        Args:
            cls:DataTransformer
        Returns:
            A pipeline object
        """
        logging.info("Entered get_data_transformer_object Method of Transformation Class")
        try:
           imputer:KNNImputer=KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
           logging.info(f"Initialised KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}")
           processor:Pipeline=Pipeline([("imputer",imputer)])
           return processor
        except Exception as e:
            raise NetworkSecurityException(e,sys)





    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_data_tranformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            
            ### Training Dataframe
            input_feature_train_df=train_df.drop(TARGET_COLUMN,axis=1)
            target_feature_train_df=train_df[TARGET_COLUMN]
            target_feature_train_df=target_feature_train_df.replace(-1,0)
            
            ### Testing Dataframe
            input_feature_test_df=test_df.drop(TARGET_COLUMN,axis=1)
            target_feature_test_df=test_df[TARGET_COLUMN]
            target_feature_test_df=target_feature_test_df.replace(-1,0)

            preprocessor=self.get_data_transformer_object()
            preprocessor_object=preprocessor.fit(input_feature_train_df)
            transformed_input_train_features=preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_features=preprocessor_object.transform(input_feature_test_df)

            train_arr=np.c_[transformed_input_train_features,np.array(target_feature_train_df)]
            test_arr=np.c_[transformed_input_test_features,np.array(target_feature_test_df)]


            ### Save numpy array Data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,array= test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path,preprocessor_object)

            #### Preparing Artifacts

            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path

            )

            return data_transformation_artifact


        except Exception as e:
            raise NetworkSecurityException(e,sys)