# to execute: Ctrl+R
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
# https://aif360.mybluemix.net/
# from aif360.datasets.multiclass_label_dataset import MulticlassLabelDataset
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from sklearn.preprocessing import MaxAbsScaler
from IPython.display import Markdown, display
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

# Get the dataset and split into train and test
dataset_orig = pd.read_csv('../bank/bank-full.csv', sep=';')
dataset_orig['age_group'] = dataset_orig['age'].map(lambda v: 0 if v < 40 else 1)
# Make all non-numerical fields numerical, and remove non-numericals:
# 1 - y
dataset_orig['y_number'] = dataset_orig['y'].map(lambda v: 0 if v == 'no' else 1)
non_numeric = 'y'
dataset_orig = dataset_orig.drop(non_numeric, axis=1)

# 2 - job
jobs = ['management', 'entrepreneur', 'self-employed', 'admin.', 'retired', 'unknown',
        'technician', 'services', 'blue-collar', 'housemaid', 'student', 'unemployed']
jobs_values = np.linspace(2, 0.1, 12)
# range of scores from 0-100 where 100 is the maximum value for Excellent Credit Risk
jobs_dict = {}
index = 0
for job in jobs:
    jobs_dict[job] = jobs_values[index]
    index += 1

job_numbers = []

for row in dataset_orig.itertuples():
    job_numbers.append(jobs_dict[row.job])

job_numbers_df = pd.DataFrame(job_numbers, columns=['job_numbers'])
dataset_orig = pd.concat([dataset_orig, job_numbers_df], axis=1)

non_numeric = 'job'
dataset_orig = dataset_orig.drop(non_numeric, axis=1)

# 3 - marital
dataset_orig['marital_number'] = dataset_orig['marital'].map(lambda v: 1 if v == 'married' else 0)
non_numeric = 'marital'
dataset_orig = dataset_orig.drop(non_numeric, axis=1)

# 4 - education
dataset_orig['education_number'] = dataset_orig['education'].map(lambda v: 0 if v == 'unknown' else 1)
non_numeric = 'education'
dataset_orig = dataset_orig.drop(non_numeric, axis=1)

# 5 - default
dataset_orig['default_number'] = dataset_orig['default'].map(lambda v: 0 if v == 'no' else 1)
non_numeric = 'default'
dataset_orig = dataset_orig.drop(non_numeric, axis=1)

# 6 - housing
dataset_orig['housing_number'] = dataset_orig['housing'].map(lambda v: 0 if v == 'no' else 1)
non_numeric = 'housing'
dataset_orig = dataset_orig.drop(non_numeric, axis=1)

# 7 - loan
dataset_orig['loan_number'] = dataset_orig['loan'].map(lambda v: 0 if v == 'no' else 1)
non_numeric = 'loan'
dataset_orig = dataset_orig.drop(non_numeric, axis=1)

# 8 - contact
dataset_orig['contact_number'] = dataset_orig['contact'].map(lambda v: 0 if v == 'unknown' else 1)
non_numeric = 'contact'
dataset_orig = dataset_orig.drop(non_numeric, axis=1)

# 9 - month
# dataset_orig['contact_number'] = dataset_orig['contact'].map(lambda v: 0 if v == 'unknown' else 1)
non_numeric = 'month'
dataset_orig = dataset_orig.drop(non_numeric, axis=1)

# 10 - poutcome
dataset_orig['poutcome_number'] = dataset_orig['poutcome'].map(lambda v: 0 if v == 'unknown' else 1)
non_numeric = 'poutcome'
dataset_orig = dataset_orig.drop(non_numeric, axis=1)

privileged_groups = [{'age_group': 1}]
unprivileged_groups = [{'age_group': 0}]

sections = [int(.5 * dataset_orig.shape[0])]
dataset_orig_train, dataset_orig_test = np.split(dataset_orig.sample(frac=1, random_state=0), sections)


def df_to_BinaryLabelDataset(df, protected_groups, label_names, favorable_label, unfavorable_label):
    # Modified from here:
    # https://programtalk.com/python-more-examples/aif360.datasets.BinaryLabelDataset/
    # label_names = [y.name]
    # https://aif360.readthedocs.io/en/latest/modules/generated/aif360.datasets.BinaryLabelDataset.html
    result = BinaryLabelDataset(
        df=df, protected_attribute_names=protected_groups,
        label_names=label_names, favorable_label=favorable_label,
        unfavorable_label=unfavorable_label
    )
    return result


# def df_to_StructuredDataset(df, protected_groups, label_names):
#     # label_names = [y.name]
#     # https://github.com/Trusted-AI/AIF360/blob/master/aif360/datasets/multiclass_label_dataset.py
#     result = MulticlassLabelDataset(df=df, protected_attribute_names=protected_groups, label_names=label_names)
#     return result


protected_attribute_names = ['age_group']
binary_label_dataset_train = df_to_BinaryLabelDataset(df=dataset_orig_train, protected_groups=protected_attribute_names,
                                                      label_names=['y_number'], favorable_label=1, unfavorable_label=0)
binary_label_dataset_test = df_to_BinaryLabelDataset(df=dataset_orig_test, protected_groups=protected_attribute_names,
                                                     label_names=['y_number'], favorable_label=1, unfavorable_label=0)
# structured_dataset_train = df_to_StructuredDataset(df=dataset_orig_train, protected_groups=protected_attribute_names,
#                                                    label_names=['y_number'])
# structured_dataset_test = df_to_StructuredDataset(df=dataset_orig_test, protected_groups=protected_attribute_names,
#                                                   label_names=['y_number'])

# https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_adversarial_debiasing.ipynb
# Metric for the original dataset
metric_orig_train = BinaryLabelDatasetMetric(binary_label_dataset_train,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
display(Markdown("#### Original training dataset"))
print(
    "Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
metric_orig_test = BinaryLabelDatasetMetric(binary_label_dataset_test,
                                            unprivileged_groups=unprivileged_groups,
                                            privileged_groups=privileged_groups)
print(
    "Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())

min_max_scaler = MaxAbsScaler()
binary_label_dataset_train.features = min_max_scaler.fit_transform(binary_label_dataset_train.features)
binary_label_dataset_test.features = min_max_scaler.transform(binary_label_dataset_test.features)
metric_scaled_train = BinaryLabelDatasetMetric(binary_label_dataset_train,
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)
display(Markdown("#### Scaled dataset - Verify that the scaling does not affect the group label statistics"))
print(
    "Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_scaled_train.mean_difference())
metric_scaled_test = BinaryLabelDatasetMetric(binary_label_dataset_test,
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups)
print(
    "Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_scaled_test.mean_difference())

# Load post-processing algorithm that equalizes the odds
# Learn parameters with debias set to False
sess = tf.Session()
plain_model = AdversarialDebiasing(privileged_groups=privileged_groups,
                                   unprivileged_groups=unprivileged_groups,
                                   scope_name='plain_classifier',
                                   debias=False,
                                   sess=sess)

plain_model.fit(binary_label_dataset_train)

# Apply the plain model to test data
dataset_nodebiasing_train = plain_model.predict(binary_label_dataset_train)
dataset_nodebiasing_test = plain_model.predict(binary_label_dataset_test)

# Metrics for the dataset from plain model (without debiasing)
display(Markdown("#### Plain model - without debiasing - dataset metrics"))
metric_dataset_nodebiasing_train = BinaryLabelDatasetMetric(dataset_nodebiasing_train,
                                                            unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)

print(
    "Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_train.mean_difference())

metric_dataset_nodebiasing_test = BinaryLabelDatasetMetric(dataset_nodebiasing_test,
                                                           unprivileged_groups=unprivileged_groups,
                                                           privileged_groups=privileged_groups)

print(
    "Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_test.mean_difference())

display(Markdown("#### Plain model - without debiasing - classification metrics"))
classified_metric_nodebiasing_test = ClassificationMetric(binary_label_dataset_test,
                                                          dataset_nodebiasing_test,
                                                          unprivileged_groups=unprivileged_groups,
                                                          privileged_groups=privileged_groups)
print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
TPR = classified_metric_nodebiasing_test.true_positive_rate()
TNR = classified_metric_nodebiasing_test.true_negative_rate()
bal_acc_nodebiasing_test = 0.5 * (TPR + TNR)
print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
print("Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
print("Test set: Theil_index = %f" % classified_metric_nodebiasing_test.theil_index())

sess.close()
tf.reset_default_graph()
sess = tf.Session()

# Learn parameters with debias set to True
debiased_model = AdversarialDebiasing(privileged_groups=privileged_groups,
                                      unprivileged_groups=unprivileged_groups,
                                      scope_name='debiased_classifier',
                                      debias=True,
                                      sess=sess)

debiased_model.fit(binary_label_dataset_train)

# Apply the plain model to test data
dataset_debiasing_train = debiased_model.predict(binary_label_dataset_train)
dataset_debiasing_test = debiased_model.predict(binary_label_dataset_test)
'''

# Metrics for the dataset from plain model (without debiasing)
display(Markdown("#### Plain model - without debiasing - dataset metrics"))
print(
    "Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_train.mean_difference())
print(
    "Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_test.mean_difference())

# Metrics for the dataset from model with debiasing
display(Markdown("#### Model - with debiasing - dataset metrics"))
metric_dataset_debiasing_train = BinaryLabelDatasetMetric(dataset_debiasing_train,
                                                          unprivileged_groups=unprivileged_groups,
                                                          privileged_groups=privileged_groups)

print(
    "Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_debiasing_train.mean_difference())

metric_dataset_debiasing_test = BinaryLabelDatasetMetric(dataset_debiasing_test,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups)

print(
    "Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_debiasing_test.mean_difference())

display(Markdown("#### Plain model - without debiasing - classification metrics"))
print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
TPR = classified_metric_nodebiasing_test.true_positive_rate()
TNR = classified_metric_nodebiasing_test.true_negative_rate()
bal_acc_nodebiasing_test = 0.5 * (TPR + TNR)
print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
print("Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
print("Test set: Theil_index = %f" % classified_metric_nodebiasing_test.theil_index())

display(Markdown("#### Model - with debiasing - classification metrics"))
classified_metric_debiasing_test = ClassificationMetric(binary_label_dataset_test,
                                                        dataset_debiasing_test,
                                                        unprivileged_groups=unprivileged_groups,
                                                        privileged_groups=privileged_groups)
print("Test set: Classification accuracy = %f" % classified_metric_debiasing_test.accuracy())
TPR = classified_metric_debiasing_test.true_positive_rate()
TNR = classified_metric_debiasing_test.true_negative_rate()
bal_acc_debiasing_test = 0.5 * (TPR + TNR)
print("Test set: Balanced classification accuracy = %f" % bal_acc_debiasing_test)
print("Test set: Disparate impact = %f" % classified_metric_debiasing_test.disparate_impact())
print("Test set: Equal opportunity difference = %f" % classified_metric_debiasing_test.equal_opportunity_difference())
print("Test set: Average odds difference = %f" % classified_metric_debiasing_test.average_odds_difference())
print("Test set: Theil_index = %f" % classified_metric_debiasing_test.theil_index())
'''
