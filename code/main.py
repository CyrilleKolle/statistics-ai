"""
STATISTICS REPORT
"""

import pandas as pd
import scipy.stats as scs
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f, norm, ttest_ind
from statsmodels.stats.weightstats import ttest_ind as it, DescrStatsW, CompareMeans
import seaborn as sns

df1 = pd.read_csv("../dataset/iris.csv")

# Rename all columns to remove white space at beginning of column name
df1 = df1.rename(
    columns={
        " sepal_length": "sepal_length",
        " sepal_width": "sepal_width",
        " petal_length": "petal_length",
        " petal_width": "petal_width",
        " class": "class",
    }
)

# Dataframes from dataset that will be used in the project
setosa = df1[df1["class"] == "Iris-setosa"]
versicolor = df1[df1["class"] == "Iris-versicolor"]
verginica = df1[df1["class"] == "Iris-virginica"]


class Statistics_ai:
    """
    Statistics_ai will emcompass all the tests, devided into seperate methods that will be used in the project
    """

    def __init__(self, test, alph) -> None:
        self._test = test
        self._alph = alph
        self._df_list = {
            "setosa": setosa,
            "verginica": verginica,
            "versicolor": versicolor,
        }

    def description(self, df):
        """
        The description function returns an overview of the particular dataframe I am interested in. That is, the mean, standard deviation, mode etc.
        These values can alternatively be calculate with numpy or scipy in the following ways for example:

        """
        print(f"Description for {df} \n{self._df_list[df].describe()}\n")

    def confidence_interval(self):
        self._alpha = 1 - (self._alph / 100)
        c_i = []
        for key, value in self._df_list.items():
            c_i.append(
                {
                    key: scs.t.interval(
                        confidence=1 - self._alpha,
                        df=len(value[self._test]) - 1,
                        loc=value[self._test].mean(),
                        scale=scs.sem(value[self._test]),
                    )
                }
            )

        print(
            f"Confidence interval with confidence grade of {self._alph} for {self._test} of the dataframes are: \n "
        )
        for item in c_i:
            for key, value in item.items():
                print(
                    f'{key}["{self._test}"] between {round(value[0], 3)} and {round(value[1],3)} \n'
                )

    def variance_hypothesis_test(self, noll_df, other_df, sub_df):
        alternate_df = self._df_list[other_df][sub_df]
        noll_data = self._df_list[noll_df][sub_df]

        statistic1 = np.var(noll_data, ddof=1) / np.var(alternate_df, ddof=1)
        result = f.ppf(q=0.05, dfn=len(noll_data) - 1, dfd=len(alternate_df) - 1)

        print(statistic1, result)

    def mean_hypothesis_test(self, noll_df, other_df, sub_df):
        """
        h0 : versicolor['sepal_length'] <= verginica['sepal_length']
        ha : versicolor['sepal_length'] > ver['sepal_length']

        if p-value < 0.05 which is the significance level, then h0 is not rejected
        """
        alternate_data = self._df_list[other_df][sub_df]
        noll_data = self._df_list[noll_df][sub_df]

        result = ttest_ind(
            a=noll_data, b=alternate_data, equal_var=False, alternative="two-sided"
        )

        print(
            f"hypothesis rejected because p_value({result.pvalue}) is less than 0.05 which is the significance level"
            if result.pvalue <= 0.05
            else f"hypothesis not rejected since p_value({result.pvalue}) is greater than 0.05 which is the signifance level"
        )
        print(
            f"\n Alternatively, We could compare the result.statistic from the ttest_ind \n"
        )
        print(
            f"hypothesis rejected because t_statistic({result.statistic}) is greater than 0.95 which is the significance level"
            if result.statistic > norm.ppf(0.95)
            else f"hypothesis not rejected since t_statistic({result.statistic}) is less than 0.95 which is the signifance level"
        )

    def mean_hypothesis_statsmodel(self, noll_df, other_df, sub_df):
        alternate_data = self._df_list[other_df][sub_df]
        noll_data = self._df_list[noll_df][sub_df]
        result = it(
            x1=noll_data.sample(10),
            x2=alternate_data.sample(10),
            alternative="larger",
            usevar="unequal",
            value=6,
        )
        print(f"\n\n using the statsmodel we get {result}")

    def compare_means(self, noll_df, other_df, sub_df):
        """
        The confidential interval (CI) returned in this case signifies that there is a 95%
        probability that the difference in sub_df between the noll_df and other_df is found within the CI.
        """

        alternate_data = self._df_list[other_df][sub_df]
        noll_data = self._df_list[noll_df][sub_df]

        result = CompareMeans(d1=DescrStatsW(noll_data), d2=DescrStatsW(alternate_data))
        lower, upper = result.tconfint_diff(
            alpha=0.05, alternative="two-sided", usevar="unequal"
        )

        print(
            f"\n\n using the statsmodel compare means, I can conclude that, using 0.05 significance level, {noll_df}[{sub_df}] is greater than {other_df}[{sub_df}] by a value found with get the lower ci: {lower} and upper ci {upper}"
        )

    def correlation_within_specy(self,df, sub_df_1):
        df = self._df_list[df]
        
        sns.lmplot(data=df, x=self._test, y=sub_df_1)
        plt.show()
        
    def correlation_heat_map(self, df):
        df = self._df_list[df]
        
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(4, 4))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        
    def diagrams(self, df, sub_df):
        df = self._df_list[df]
        g = sns.FacetGrid(df, col=self._test, hue=sub_df)
        g.map(sns.scatterplot, "sepal_width", "sepal_length", alpha=.7)
        g.add_legend()