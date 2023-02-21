import numpy as np
import os
from config.core import config
from pipeline import price_pipe
from sklearn.model_selection import train_test_split
from regression_model.processings.data_manager import load_dataset,save_pipeline


def run_training()->None:
    """Train model"""
    print(config.app_config)
    data= load_dataset(file_name=config.app_config.training_data_file)
    X_train, X_test,y_train,y_test= train_test_split(
        data[config.model_config.features],
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        ## We are setting random state for reproducibility
        random_state=config.model_config.random_state
        )
    y_train=np.log(y_train)

    ## fit model
    price_pipe.fit(X_train,y_train)

    ## Persist Trained model
    save_pipeline(pipeline_to_persist=price_pipe)






if __name__== "__main__":
    run_training()
