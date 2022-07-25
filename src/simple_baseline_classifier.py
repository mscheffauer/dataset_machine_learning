################################################################################
# Author 1:      Martin Scheffauer
# MatNr 1:       51917931
# Author 2:      Dominik Geschwinde
# MatNr 2:       12108977
# File:          simple_baseline_classifier.py
# Description: implements the simple baseline classifier based on scikit.learn.DummyClassifier
# Comments:    this classifier functions similar to scikit.learn.DummyClassifier. it ignores the given
#              X_test values or feature values 
#              and bases prediction entirely on the target values with a specified simple strategy as a rule
################################################################################

from typing import Union
import numpy as np


class SimpleBaselineClassifier:

    def __init__(self, random_state : int = None, constant : Union[int,str] = None,\
        strategy : str = "most_frequent" ) -> None:
        
        self.strategy = strategy
        self.random_state = random_state
        self.constant = constant
        
    def __repr__(self) -> str:
        return f"SimpleBaselineClassifier(strategy={self.strategy},"+ \
            f" random_state={self.random_state}, constant={self.constant})"

    def _init_randomizer(self):
        if self.random_state is None or self.random_state is np.random:
            return np.random.mtrand._rand
        elif isinstance(self.random_state,int):
            return np.random.RandomState(self.random_state)
        else:
            raise ValueError("random_state cannot be used to generate random number.")

        
    def fit(self, X_train : np.ndarray , y_train : np.ndarray)->None:
        #just copy over the training values to internal values here
        self._x_train = X_train
        self._y_train = y_train
    def predict(self, X_test : np.ndarray)->np.ndarray:
     
        #constant: return the constant value given if the constant value given is in the target value set
        if self.strategy == 'constant':
            if self.constant is None:
                raise ValueError("Constant target value has to be specified when the constant strategy is used.")  
            else:    
                if int(self.constant) in self._y_train:
                    return np.full(len(X_test), self.constant)
                else:
                    raise ValueError("The constant target value must be present in the training data.")  
        #uniform: pick at random one target value of the unique set of target values. Init random object 
        elif self.strategy == 'uniform':
            _uniques = np.unique(self._y_train)
            _np_random_obj = self._init_randomizer()

            return np.array(_np_random_obj.choice(_uniques,size=len(X_test)))
            
        #most_frequent: pick the most frequent element in the list and if several are equally frequent choose one
        elif self.strategy == 'most_frequent':
            #randomizer
            _np_random_obj = self._init_randomizer()


            _uniques, _counts = np.unique(self._y_train,return_counts = True)
            #zip uniques and counts into dict and sort the dict descending according to the counts
            _target_val_dict = dict(zip(_uniques, _counts))
            _target_val_sorted_dict = dict(sorted(_target_val_dict.items(), key=lambda item: item[1], reverse = True))
            _random_candidates = list()
            #then the highest count is up front. add every key with count of (key) equal to highest to the candidates 
            for key,val in _target_val_sorted_dict.items():
                if (val == list(_target_val_sorted_dict.items())[0][1]):
                    _random_candidates.append(key)
            #choose at random from candidates
            return np.full(len(X_test), _np_random_obj.choice(_random_candidates))

        else:
            #error if wrong strategy specified
            raise ValueError("strategy must be one of the following: most_frequent, uniform or constant")
           

      
        
    