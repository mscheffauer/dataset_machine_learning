################################################################################
# Author 1:      Martin Scheffauer
# MatNr 1:       51917931
# Author 2:      Dominik Geschwinde
# MatNr 2:       12108977
# File:          experiments.py
# Description:  this file contains code for testing the different classifiers
# Comments:    the experiments were conducted with different classifiers: support vector machine, the own kNN, 
#              gaussian naive bayes, logistic regression and decision tree and for reference the own 
#               SimpleBaselineClassifier, both heart disease and parkinson sound recordings datasets were used
################################################################################


from simple_baseline_classifier import SimpleBaselineClassifier
from knn import kNN
from dataset import Dataset
from config_reader import ConfigReader
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, recall_score
from matplotlib import pyplot
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from typing import Union


#load in datasets
_config_reader_heart = ConfigReader.read_json_config("./config_heart_disease.json")
_config_reader_parkinson = ConfigReader.read_json_config("./config_parkinson_sound_recording.json")
datasets = list()
datasets.append(Dataset(_config_reader_heart))
datasets.append(Dataset(_config_reader_parkinson))

#extend classes with get_params
class Ext_SimpleBaselineClassifier(SimpleBaselineClassifier):
    def __init__(self, random_state : int = None, constant : Union[int,str] = None,\
        strategy : str = "most_frequent" ) -> None:
        self._random_state = random_state
        self._constant = constant
        self._strategy = strategy
        super().__init__(random_state,constant,strategy)
    def get_params(self,deep = False):
        ret = dict()
        ret["random_state"] = self._random_state
        ret["constant"] = self._constant
        ret["strategy"] = self._strategy
        return ret

class Ext_kNN(kNN):
    def __init__(self, n_neighbors:int, metric:str = "cosine"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        super().__init__(n_neighbors,metric)
    def get_params(self,deep = False):
        ret = dict()
        ret["n_neighbors"] = self.n_neighbors
        ret["metric"] = self.metric
      
        return ret


#for every dataset 
for dataset in datasets:
    #split
    x_train, y_train, x_test, y_test = dataset.split_data("mean")

    #now scale
    scaler = MinMaxScaler(feature_range=(0, 1))

    x_test = scaler.fit_transform(x_test)
    x_train = scaler.fit_transform(x_train)

    #classify all data here

    results = []
    recall = []
    names = []
    accuracies = []
    models = []
    confusions = []

    models.append(('Decision \n Tree', DecisionTreeClassifier(random_state=datasets[0]._random_state_for_split)))
    models.append(('Gaussian\n Naive\n Bayes', GaussianNB()))
    models.append(('LIN_SVC', LinearSVC(random_state=datasets[0]._random_state_for_split,dual= False)))
    models.append(('Logistic \n Regression', LogisticRegression(random_state=datasets[0]._random_state_for_split)))
    models.append(('Baseline\n Classifier', \
     Ext_SimpleBaselineClassifier(random_state=datasets[0]._random_state_for_split)))
    models.append(('KNN', Ext_kNN(n_neighbors = 5)))
   
    
    for name,model in models:
       
        names.append(name)
        
        #make cross validation 
        kfold = KFold(n_splits=10, shuffle=True, random_state=datasets[0]._random_state_for_split)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results.mean())
        
        #fit and predict
        model.fit(x_train,y_train)
        predict = model.predict(x_test)
        
        #accuracies  
        accuracies.append(accuracy_score(y_test, predict))
        #confusion matrix
        confusions.append((y_test,predict))
        #recall score
        recall.append(recall_score(y_test,predict))
        
    #plot confusion matrixes
    fig, [ax_l,ax_t] = pyplot.subplots(2,int(len(models)/2))
    cnt = 0

    fig.suptitle(f"{dataset.name} - confusion matrixes:")

    for y_test, predict in confusions:
  
        cm = confusion_matrix(y_test, predict)

        if (cnt <3):

            cm_display = ConfusionMatrixDisplay(cm).plot(ax = ax_l[cnt],colorbar = False )
            ax_l[cnt].set_xlabel(names[cnt])
            ax_l[cnt].set_ylabel("")
        elif (cnt <6):
            cm_display = ConfusionMatrixDisplay(cm).plot(ax = ax_t[cnt-3],colorbar = False )
            ax_t[cnt-3].set_xlabel(names[cnt])
            ax_t[cnt-3].set_ylabel("")
        cnt += 1
    fig.figsize = [18,12]
    fig.tight_layout()
    
    #prepare all bar charts
    charts = []
    charts.append(["cross_val","accuracy",results])
    charts.append(["accuracy","accuracy",accuracies])
    charts.append(["score_recall","accuracy",recall])
    
    
      #plot barcharts
    
    for name, y_label, data in charts:
        
        fig, ax = pyplot.subplots(1,len(models),sharey=True)
        cnt = 0

        fig.suptitle(f"{dataset.name} - {name}:")


        for datapoint in data:
            if cnt == 0:
                ax[cnt].set_ylabel(y_label)
            ax[cnt].set_title("")
            ax[cnt].set_xlabel(names[cnt],rotation = 45)
            ax[cnt].bar(x=cnt, height=datapoint, width=0.4)
            ax[cnt].set_ylim(top = 1.0)
            ax[cnt].set_frame_on(False)
            ax[cnt].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
            cnt += 1
        fig.tight_layout(pad=2.0)
        fig.frameon = False
        fig.figsize = [10,7]

   
