# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

import pandas as pd
# import numpy as np
import itertools
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

data = pd.read_csv('student/student-por.csv', sep=';')
data.dropna(inplace=True)  # observation before dropna: 649; after = same
dataset_name = 'Student Performance Data Set'
dataset_source = 'https://archive.ics.uci.edu/ml/datasets/Student+Performance'
regulated_domain = 'Education'
num_observations = data.shape[0]
num_variables = data.shape[1]
dependent_variables = ['G1', 'G2']
mean_age = int(data.age.mean())  # 16
data['age_groups'] = data['age'].map(lambda value: 0 if value <= mean_age else 1)  # 'over '+str(mean_age)).astype(str)
protected_classes = ['age_groups', 'sex']
pairs = list(itertools.product(protected_classes, dependent_variables))


def df_to_BinaryLabelDataset(df, protected_groups, label_names, favorable_label, unfavorable_label):
    # From dstrube3's A5 - Fairness and Bias
    result = BinaryLabelDataset(
        df=df, protected_attribute_names=protected_groups,
        label_names=label_names, favorable_label=favorable_label,
        unfavorable_label=unfavorable_label
    )
    return result


def numberfy(data, numeric, non_numeric, value_0, value_1=None):
    # Make all non-numerical fields numerical, and remove non-numericals:
    if value_1 is None:
        data[numeric] = data[non_numeric].map(lambda v: 0 if v == value_0 else 1)
    else:
        data[numeric] = data[non_numeric].map(lambda v: 0 if v == value_0 or v == value_1 else 1)
    data = data.drop(non_numeric, axis=1)
    return data


data = numberfy(data, 'school_number', 'school', 'GP')
data = numberfy(data, 'sex_number', 'sex', 'M')
data = numberfy(data, 'address_number', 'address', 'R')
data = numberfy(data, 'famsize_number', 'famsize', 'GT3')
data = numberfy(data, 'Pstatus_number', 'Pstatus', 'A')
data = numberfy(data, 'Mjob_number', 'Mjob', 'at_home', 'other')
data = numberfy(data, 'Fjob_number', 'Fjob', 'at_home', 'other')
data = numberfy(data, 'reason_number', 'reason', 'other')
data = numberfy(data, 'guardian_number', 'guardian', 'other')
data = numberfy(data, 'schoolsup_number', 'schoolsup', 'no')
data = numberfy(data, 'famsup_number', 'famsup', 'no')
data = numberfy(data, 'paid_number', 'paid', 'no')
data = numberfy(data, 'activities_number', 'activities', 'no')
data = numberfy(data, 'nursery_number', 'nursery', 'no')
data = numberfy(data, 'higher_number', 'higher', 'no')
data = numberfy(data, 'internet_number', 'internet', 'no')
data = numberfy(data, 'romantic_number', 'romantic', 'yes')

# Special numberfy logic for the targets:
data['G1_number'] = data['G1'].map(lambda v: 0 if v < 12 else 1)
data = data.drop('G1', axis=1)

data['G2_number'] = data['G2'].map(lambda v: 0 if v < 12 else 1)
data = data.drop('G2', axis=1)


# Reset protected_classes
protected_classes = ['age_groups', 'sex_number']

favorable_G12 = 1
unfavorable_G12 = 0

binary_label_dataset_G1 = df_to_BinaryLabelDataset(df=data, protected_groups=protected_classes,
                                                label_names=['G1_number'], favorable_label=favorable_G12,
                                                unfavorable_label=unfavorable_G12)

binary_label_dataset_G2 = df_to_BinaryLabelDataset(df=data, protected_groups=protected_classes,
                                                label_names=['G2_number'], favorable_label=favorable_G12,
                                                unfavorable_label=unfavorable_G12)

privileged_groups = [{'age_groups': 1}, {'sex_number': 'F'}]
unprivileged_groups = [{'age_groups': 0}, {'sex_number': 'M'}]

classification_metric_G1 = ClassificationMetric(binary_label_dataset_G1, binary_label_dataset_G1,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

print('Done')
