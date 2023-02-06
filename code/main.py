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
from statsmodels.formula.api import ols

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
# Each dataframe below represents a class in the dataset
setosa = df1[df1["class"] == "Iris-setosa"]
versicolor = df1[df1["class"] == "Iris-versicolor"]
verginica = df1[df1["class"] == "Iris-virginica"]


class Statistics_ai:
    """
    Statistics_ai will emcompass all the tests, devided into seperate methods that will be used in the project.

    The rational for using a class is to reduce the possibility of re-writing thesame piece multiple times.
    """

    def __init__(self, test) -> None:
        self._test = test
        self._df_list = {
            "setosa": setosa,
            "verginica": verginica,
            "versicolor": versicolor,
        }

    def description(self, df):
        """
        The description function returns an overview of the particular dataframe I am interested in. That is, the mean, standard deviation, mode etc.
        These values can alternatively be computed directly using pandas for e.g, setosa.mean() or setosa['petal_width].mean();
        will return the mean for all the columns in the setosa dataframe or for a specific column.
        """
        print(f"Description for {df} \n{self._df_list[df].describe()}\n")
        print(
            f"\nThe above could be gotten by using pandas mean() or meadian() \nfor example to compute the mean for {df}[{self._test}] would be {df}[{self._test}].mean() \n{self._df_list[df].mean(numeric_only=True)}"
        )
        print(f"\nMedian: {self._df_list[df].median(numeric_only=True)}")
        print(f"\nMode: {self._df_list[df].mode(numeric_only=True)}")
        print(f"\nStandard deviation: {self._df_list[df].std(numeric_only=True)}")
        print(f"\nSkewness: {self._df_list[df].skew(numeric_only=True)}")

    def visual_description(self, df):
        """
        Visualise show a specific column in a dataframe and plotting its mean and median
        """
        # df represents the inputing dataframe when function is called
        df = self._df_list[df]

        sns.histplot(data=df, x=self._test, kde=True, bins=6, legend=True)

        # use the axvline to plot the line where the mean or median is in the dataframe
        plt.axvline(x=df[self._test].mean(), color="orange", ls="--", lw=2.5)
        plt.axvline(x=df[self._test].median(), color="red", ls="-.", lw=2.5)
        plt.legend(labels=["kde", "mean", "median"])
        plt.show()

    def confidence_interval(self, alpha_):
        """
        Use Scipy.stats module to compute the confidence interval.
        """

        # calculate the significance level
        alpha = 1 - (alpha_ / 100)

        # empty array for all the confidential intervals for each dataframe in the dataset, i.e setosa, versicolor, virginica.
        c_i = []

        # For every dataframe in the dataset
        # scipy.stats.t is a continuous probability distribution in the SciPy. 
        # It represents the t-distribution, which is used to estimate the population mean of a normally
        # distributed population when the sample size is small and the population standard deviation is unknown. 
        
        # sem() = standard error of the mean is used in place of the standard deviation since the standard deviation is not known
        for key, value in self._df_list.items():
            c_i.append(
                {
                    key: scs.t.interval(
                        confidence=1 - alpha,
                        df=len(value[self._test]) - 1,
                        loc=value[self._test].mean(),
                        scale=scs.sem(value[self._test]),
                    )
                }
            )

        print(
            f"Confidence interval with confidence grade of {alpha_} for {self._test} of the dataframes are: \n "
        )
        for item in c_i:
            for key, value in item.items():
                print(
                    f'{key}["{self._test}"] between {round(value[0], 3)} and {round(value[1],3)} \n'
                )

    def variance_hypothesis_test(self, noll_df, other_df):
        alternate_df = self._df_list[other_df][self._test]
        noll_data = self._df_list[noll_df][self._test]

        statistic1 = np.var(noll_data, ddof=1) / np.var(alternate_df, ddof=1)
        result = f.ppf(q=0.05, dfn=len(noll_data) - 1, dfd=len(alternate_df) - 1)

        print(statistic1, result)

    def mean_hypothesis_test(self, noll_df, other_df):
        """
        An example hypothesis test would be

        h0 : primary['sepal_length'] = secondary['sepal_length']
        ha : primary['sepal_length'] =/ secondary['sepal_length']

        if p-value < 0.05 which is the significance level, then h0 is not rejected
        """
        # noll_data = primary dataframe
        #  alternate_data = secondary dataframe

        alternate_data = self._df_list[other_df][self._test]
        noll_data = self._df_list[noll_df][self._test]

        result = ttest_ind(
            a=noll_data, b=alternate_data, equal_var=False, alternative="two-sided"
        )

        print(
            f"Hypothesis rejected because p_value({result.pvalue}) is less than 0.05 which is the significance level"
            if result.pvalue <= 0.05
            else f"Hypothesis not rejected since p_value({result.pvalue}) is greater than 0.05 which is the signifance level"
        )
        print(
            f"\nAlternatively, We could compare the result.statistic from the ttest_ind \n"
        )
        print(
            f"Hypothesis rejected because t_statistic({result.statistic}) is greater than 0.95 which is the significance level"
            if result.statistic > norm.ppf(0.95)
            else f"hypothesis not rejected since t_statistic({result.statistic}) is less than 0.95 which is the signifance level"
        )

    def mean_hypothesis_statsmodel(self, noll_df, other_df):
        alternate_data = self._df_list[other_df][self._test]
        noll_data = self._df_list[noll_df][self._test]
        
        # To cancel the potential of conflicts, ttest_ind from statsmodels.stats.weightstats was imported as it b
        result = it(
            x1=noll_data.sample(10),
            x2=alternate_data.sample(10),
            alternative="two-sided",
            usevar="unequal",
            value=6,
        )
        print(f"\n\n using the statsmodel we get {result}")

    def compare_means(self, noll_df, other_df):
        """
        The confidential interval (CI) returned in this case signifies that there is a 95%
        probability that the difference in sub_df between the noll_df and other_df is found within the CI.
        """

        alternate_data = self._df_list[other_df][self._test]
        noll_data = self._df_list[noll_df][self._test]

        result = CompareMeans(d1=DescrStatsW(noll_data), d2=DescrStatsW(alternate_data))
        lower, upper = result.tconfint_diff(
            alpha=0.05, alternative="two-sided", usevar="unequal"
        )

        print(
            f"\nUsing the statsmodel compare means, I can conclude that, using 0.05 significance level, {noll_df}[{self._test}] is greater than {other_df}[{self._test}] by a value found within the lower ci: {lower} and upper ci {upper}"
        )

    def correlation_within_class(self, df, sub_df_1):
        """
        Returns the correlation between two properties within a specific dataframe
        """
        df = self._df_list[df]

        sns.lmplot(data=df, x=self._test, y=sub_df_1, legend=True)
        plt.legend(loc="best", labels=["points in dataframe", "OLS prediction"])
        plt.show()

    def correlation_heat_map(self, df):
        df = self._df_list[df]

        # remove pandas warning for corr() by sending in the numeric_only=True, thereby including only float,
        # int or boolean according to documentation
        corr = df.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(4, 4))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmax=0.3,
            center=0,
            annot=True,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
        )

    def correlation_in_all_classes(self, sub_df):
        sns.lmplot(df1, y=sub_df, x=self._test, hue="class")

    def prediction_interval_out_of_our_sample_space(self, df, sub_df):
        
        """"
        Since my dataset is just 50 rows per class, I thought about an out of sample prediction by using 
        statistical models to make new predictions
        """
        
        df = self._df_list[df]
        
        
        # the first prediction model will be used to plot or draw the upper and lower limits
        # of my prediction intervall 
        points = pd.DataFrame(np.linspace(0, 5, 50), columns=[self._test])
        model = ols(f"{sub_df} ~ {self._test}", data=df).fit()
        predictions = model.get_prediction(points)
        frame = predictions.summary_frame(alpha=0.05)
        
        lower = frame["obs_ci_lower"]
        upper = frame["obs_ci_upper"]

        fig, ax = plt.subplots(figsize=(16, 5))

        sns.scatterplot(data=df, x=self._test, y=sub_df)

        
        predictions1 = model.get_prediction(df[self._test])
        frame1 = predictions1.summary_frame(alpha=0.05)
        ypred = frame1["mean"]

        ax.plot(df[self._test], ypred, "r", label="OLS prediction")

        ax.plot(points, lower, "r--", label="lower prediction limit", linewidth=0.75)
        ax.plot(points, upper, "r--", label="upper prediction limit", linewidth=0.95)
        ax.legend(loc="best")
        plt.title('Out of sample regression prediction')
        
        # Use scipy stats to compute pearson correlation coefficient and display on plot
        r, p = scs.pearsonr(x=df[self._test], y=df[sub_df])
        plt.text(.3, 0.8, 'r={:.2f}'.format(r), transform=ax.transAxes)
        plt.show()
