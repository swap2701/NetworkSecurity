from NetworkSecurity.components.data_ingestion import DataIngestion
from NetworkSecurity.components.data_validation import DataValidation
from NetworkSecurity.exception.Exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging
from NetworkSecurity.entity.config_entity import DataIngestionConfig,DataValidationConfig
from NetworkSecurity.entity.config_entity import TrainingPipelineConfig

import sys

if __name__=="__main__":
    try:
        training_pipeline_config=TrainingPipelineConfig()
        data_ingestion_config=DataIngestionConfig(training_pipeline_config)
        data_ingestion=DataIngestion(data_ingestion_config)
        logging.info("Initiate the Data Ingestion")
        data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")
        print(data_ingestion_artifact)

        data_validation_config=DataValidationConfig(training_pipeline_config)
        data_validation=DataValidation(data_ingestion_artifact,data_validation_config)
        logging.info("Initiate the Data Validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("Data Validation Completed")
        print(data_validation_artifact)

        
    except Exception as e:
        raise NetworkSecurityException(e,sys)