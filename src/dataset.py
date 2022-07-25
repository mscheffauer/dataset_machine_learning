################################################################################
# Author 1:      Martin Scheffauer
# MatNr 1:       51917931
# Author 2:      Dominik Geschwinde
# MatNr 2:       12108977
# File:          dataset.py
# Description: this file implements the Dataset class
# Comments:    general outline was given in the assignment. The class method split_data 
#               was added to split the test data into training set and test set
#               for the classifiers using the sklearn.SingleImputer as well as train_test_split
#               parameters _random_state_for_split,_test_size,_features_to_use_for_classification were added.
################################################################################



import json
from pathlib import Path
from typing import List, Union

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, config_info_dict: dict) -> None:
        self.name = config_info_dict["dataset_name"]
        self._data_folder_path = config_info_dict["data_folder"]
        self._na_characters = config_info_dict["na_characters"]
        self._target_feature_name = config_info_dict["target_feature"]

        with open(Path(self._data_folder_path) / config_info_dict["column_info_file"]) as column_info_file:
            self._column_info = json.load(column_info_file)

        self._load_data()
        if "binarize_threshold" in config_info_dict:
            self._binarize_targets(config_info_dict["binarize_threshold"])
        self._random_state_for_split = config_info_dict["random_state_for_split"]
        self._test_size = config_info_dict["test_size"]
        self._features_to_use_for_classification = config_info_dict["features_to_use_for_classification"]
        

    def __str__(self) -> str:
        num_samples, num_features = self._dataframe.shape
        return (f"Dataset Info:\n"
                f"\tDataset name: {self.name}\n"
                f"\tNumber of features: {num_features}, number of rows: {num_samples}")


    @property
    def name(self) -> str:
        return self._name


    @name.setter
    def name(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError(f"name must be a string")
        self._name = name


    @property
    def column_info(self) -> dict:
        return self._column_info


    @property
    def feature_names(self) -> list:
        return list(self._column_info.keys())


    @property
    def target_feature_name(self) -> str:
        return self._target_feature_name


    def get_feature_values(self, *feature_names: str, remove_missing: bool = False) -> Union[list, List[list]]:
        feature_values = self._dataframe.loc[:, feature_names]
        if remove_missing:
            feature_values = feature_values.dropna()
        return feature_values.values.T.squeeze().tolist()


    def get_target_values(self) -> list:
        return self._dataframe[self._target_feature_name].tolist()


    def write_data_to_file(self, output_file: str) -> None:
        self._dataframe.to_csv(output_file,
                               index=False,
                               header=False,
                               na_rep=self._na_characters[0])


    def _binarize_targets(self, threshold: int) -> None:
        below_threshold = self._dataframe[self._target_feature_name] < threshold
        self._dataframe.loc[below_threshold, self._target_feature_name] = 0
        self._dataframe.loc[~below_threshold, self._target_feature_name] = 1


    def _load_data(self) -> None:
        data_files = Path(self._data_folder_path).glob("*.csv")
        dataframes = []
        for data_file in data_files:
            dataframes.append(pd.read_csv(data_file,\
                                        names=self.feature_names,\
                                        keep_default_na=False,\
                                        na_values=self._na_characters))
        self._dataframe = pd.concat(dataframes)
    def split_data(self,impute_strategy : str = None) -> list: 

   #take full dataframe, drop na if necessary
        _feature_dataframe = self._dataframe

        if (impute_strategy is None):
            _feature_dataframe = _feature_dataframe.dropna()
            
       
        #select all features to use if a list is given. add also target feature
        if (self._features_to_use_for_classification != ['all']):
            _temp_features = list()
            for feat in self._features_to_use_for_classification:
                _temp_features.append(feat)
        
            _temp_features.append(self.target_feature_name)
            _feature_dataframe = _feature_dataframe.loc[:,_temp_features]
            

         #take only required feature columns without target, make new target dataframe
        _target_dataframe = _feature_dataframe.loc[:,self.target_feature_name]
        _feature_dataframe = _feature_dataframe.loc[:, _feature_dataframe.columns != self.target_feature_name]
 

        #then split data
        _x_train, _x_test, _y_train, _y_test = train_test_split(_feature_dataframe,\
           _target_dataframe, test_size=self._test_size, random_state=self._random_state_for_split)
        
        #now impute
        if (impute_strategy is not None):
            _imputer = SimpleImputer(missing_values = np.nan, strategy = impute_strategy)
            _imputer.fit(_x_train)
            _x_test = _imputer.transform(_x_test)
            _x_train = _imputer.fit_transform(_x_train)
     
        if (isinstance(_x_train,np.ndarray) is False):
            _x_train = _x_train.to_numpy()
        if (isinstance(_y_train,np.ndarray) is False):
            _y_train = _y_train.to_numpy()
        if (isinstance(_x_test,np.ndarray) is False):
            _x_test = _x_test.to_numpy()
        if (isinstance(_y_test,np.ndarray) is False):
            _y_test = _y_test.to_numpy()
              
        return _x_train, _y_train, _x_test,  _y_test
        
        

        

        

 

        