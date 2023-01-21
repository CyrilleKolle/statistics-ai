import pandas as pd
import scipy.stats as scs
import matplotlib.pyplot as plt
import numpy as np

df1 = pd.read_csv('../dataset/iris.csv')
df1 = df1.rename(columns={' sepal_length':'sepal_length', ' sepal_width':'sepal_width', ' petal_length':'petal_length', ' petal_width':'petal_width', ' class':'class'})
setosa = df1[df1['class'] == 'Iris-setosa']
versicolor = df1[df1['class'] == 'Iris-versicolor']
verginica = df1[df1['class'] == 'Iris-virginica']

class Statistics_ai:
    
    def __init__(self, test, alph) -> None:
        self._test = test
        self._alph = alph
        self._df_list = {'setosa':setosa,'verginica': verginica, 'versicolor':versicolor}
    
    def description(self, df):     
        print(f"Description for {df} \n{self._df_list[df].describe()}\n")
            
    def confidence_interval(self):
        self._alpha = 1 - (self._alph/100)
        c_i = []
        for key, value in self._df_list.items():
            c_i.append({key:scs.t.interval(confidence=1-self._alpha, df=len(value[self._test]) - 1, loc=value[self._test].mean(), scale=scs.sem(value[self._test]))})

        print(f'Confidence interval with confidence grade of {self._alph} for {self._test} of the dataframes are: \n ')
        for item in c_i:
            for key, value in item.items():     
                print(f'{key}["{self._test}"] between {round(value[0], 3)} and {round(value[1],3)} \n')


