# to execute: Ctrl+R
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
from IPython.core.display_functions import display
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# import matplotlib as mpl
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.preprocessing.lfr import LFR


data = pd.read_csv('../bank/bank-full.csv', sep=';')
# print(f"{data.shape[0]} observations")
# print(f"{data.shape[1]} variables")

outcome_variables = ['balance', 'job']
# Jobs, arranged in order from most likely to least likely to achieve Excellent Credit Risk
jobs = ['management', 'entrepreneur', 'self-employed', 'admin.', 'retired', 'unknown', 'technician', 'services', 'blue-collar', 'housemaid', 'student', 'unemployed']
jobs_values = np.linspace(2, 0.1, 12)
# range of scores from 0-100 where 100 is the maximum value for Excellent Credit Risk
jobs_dict = {}
index = 0
for job in jobs:
    jobs_dict[job] = jobs_values[index]
    index += 1
max_balance = data['balance'].max()
# min = data['balance'].min()

predicted_score = []

# max_score = 0

for row in data.itertuples():
    # balance = row.balance
    # job = row.job
    value = (((row.balance/2) * jobs_dict[row.job]) / max_balance) * 100
    if value < 0:
        value = 0
    # if value > max_score:
    #     max_score = value
    predicted_score.append(value)


newdf = pd.DataFrame(predicted_score, columns=['predicted_score'])
data = pd.concat([data, newdf], axis=1)

data['age_group'] = data['age'].map(lambda v: 0 if v < 40 else 1)
data['y_number'] = data['y'].map(lambda v: 0 if v == 'no' else 1)

sections = [int(.5 * data.shape[0])]
data_train, data_test = np.split(data.sample(frac=1, random_state=0), sections)
dependent = 'y'
x_train = data_train.drop(dependent, axis=1).values
y_train = data_train[dependent].values
x_test = data_test.drop(dependent, axis=1).values
y_test = data_test[dependent].values

# _, ax = plt.subplots()
# # x_vals = data_train['predicted_score'].values
# # Nope, ^that's^ not the right data...
# x_vals = data_train['y_number'].values
# ax.hist(x_vals, 5)
# # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
# # https://www.geeksforgeeks.org/plotting-histogram-in-python-using-matplotlib/
# # https://pythongeeks.org/python-histogram/
# x_mean = x_vals.mean()
# ax.axvline(x_mean, color='#FF0000', label='Mean')
# ax.axvline(.50, color='#00FF00', label='Threshold')
# # Labels don't work as expected
# ax.annotate(r"Mean = {:.3f}".format(x_mean), xy=(1.05 * x_mean, 0.5), rotation=90)
# ax.annotate('Threshold: 0.50', xy=(0.45, 0.5), rotation=90)
# ax.set_title('Creditworthiness')
# ax.grid(True)
# plt.show()

# https://github.gatech.edu/jtriveri3/CS6603/blob/master/Assignment5/CS6603_Assignment5.ipynb

# threshold = 0.2  # -17,843
# threshold = 0.3  # -16,079
# threshold = 0.4  # -15,020
threshold = 0.5  # -14,531
# threshold = 0.6  # -14,264
# threshold = 0.7  # -14,042
# threshold = 0.8  # -13,409
# threshold = 0.9  # -13,136

# dftest["creditworth"] = data_train['y_number']

# dftest["yact"] = data_train["y_number"]
# dftest["ypred"] = data_train['predicted_score']
data_train['profit'] = 0
data_train['profit'] = data_train.apply(lambda record: 10 if record.y_number == 1 and record.predicted_score >= threshold else record.profit, axis=1)
data_train['profit'] = data_train.apply(lambda record: -5 if record.y_number == 1 and record.predicted_score < threshold else record.profit, axis=1)
data_train['profit'] = data_train.apply(lambda record: -3 if record.y_number == 0 and record.predicted_score >= threshold else record.profit, axis=1)
data_train['profit'] = data_train.apply(lambda record: 0 if record.y_number == 0 and record.predicted_score < threshold else record.profit, axis=1)

print(f"Profit with default threshold: {data_train.profit.sum():,.0f}")

data_train['prediction_correct'] = (data_train['predicted_score'] >= threshold) * 1
data_prediction_rate = data_train.groupby('age_group', as_index=False).agg(
    applied=('prediction_correct', 'count'), approved=('prediction_correct', 'sum')
    )

data_prediction_rate['approval_rate'] = data_prediction_rate['approved'] / data_prediction_rate['applied']
display(data_prediction_rate)

privileged_group, unprivileged_group = [{"age_group": 0}], [{"age_group": 1}]

binary_label_dataset_train = BinaryLabelDataset(
    favorable_label=1, unfavorable_label=0, df=data_train, label_names=['y'], protected_attribute_names=['age_group'])

# binary_label_dataset_test = BinaryLabelDataset(
#     favorable_label=1, unfavorable_label=0, df=data_test[data_train.columns.tolist()], label_names=['y'],
#     protected_attribute_names=['age_group'])
#
# binary_label_dataset_metric_train = BinaryLabelDatasetMetric(
#     binary_label_dataset_train, unprivileged_groups=unprivileged_group, privileged_groups=privileged_group)
# binary_label_dataset_metric_test = BinaryLabelDatasetMetric(
#     binary_label_dataset_test, unprivileged_groups=unprivileged_group, privileged_groups=privileged_group)
#
# learn_fair_representation = LFR(
#     unprivileged_group[0], privileged_groups=privileged_group, k=10, Ax=0.1, Ay=1.0, Az=2.0, verbose=False)
# learn_fair_representation = learn_fair_representation.fit(binary_label_dataset_train)
print()
