import pandas as pd
import typing as t
from pathlib import Path
from regression_model.config.core import DATASET_DIR, config,TRAINED_MODEL_DIR
from sklearn.pipeline import Pipeline
import joblib

from regression_model import __version__ as _version

    
def load_dataset(*, file_name:str)-> pd.DataFrame:
    dataframe=pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    dataframe['MSSubClass']=dataframe["MSSubClass"].astype("O")

    final_dataframe=dataframe.rename(columns=config.model_config.variables_to_rename)
    return final_dataframe
    

def save_pipeline(*, pipeline_to_persist:Pipeline)->None:
    save_filename=f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path=  TRAINED_MODEL_DIR/save_filename
    remove_old_pipeline(files_to_keep=[save_filename])
    joblib.dump(pipeline_to_persist,save_path)
    


def remove_old_pipeline(*, files_to_keep=t.List[str])-> None:
    donot_delete= files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name  not in donot_delete:
            model_file.unlink()

def load_pipeline(*,file_name:str)-> Pipeline:
    filepath=TRAINED_MODEL_DIR/file_name
    trained_model= joblib.load(filename=filepath)
    return trained_model



