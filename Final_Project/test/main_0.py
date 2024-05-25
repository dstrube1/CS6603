import time
import functools

import pandas as pd
import numpy as np
import itertools
# import matplotlib.pyplot as plt

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

# from aif360.datasets import BinaryLabelDataset
# from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
import tensorflow as tf  # .compat.v1 as tf
# tf.disable_eager_execution()

print('Setup complete\n')


def get_time(seconds):
    # From dstrube3's A5 - Fairness and Bias
    if int(seconds / 60) == 0:
        if int(seconds) == 0 and round(seconds, 3) == 0.0:
            # Close enough to 0 to call it 0
            return '0 seconds'
        else:
            return f"{round(seconds, 3)} second(s)"
    minutes = int(seconds / 60)
    seconds = int(seconds % 60)
    if int(minutes / 60) == 0:
        return f"{minutes} minute(s), and {seconds} second(s)"
    hours = int(minutes / 60)
    minutes = int(minutes % 60)
    # Assuming this won't be called for any time span greater than 24 hours
    return f"{hours} hour(s), {minutes} minute(s), and {seconds} second(s)"


df = pd.read_csv('student/student-por.csv', sep=';')
df.dropna(inplace=True)  # observation before dropna: 649; after = same
dataset_name = 'Student Performance Data Set'
dataset_source = 'https://archive.ics.uci.edu/ml/datasets/Student+Performance'
regulated_domain = 'Education'
num_observations = df.shape[0]
num_variables = df.shape[1]
dependent_variables = ['G1', 'G2']
protected_classes = ['age', 'sex']

# print('Step 1:\n')
# print('● Which dataset did you select?')
# print(dataset_name + ' - ' + dataset_source)
# print()
#
# print('● Which regulated domain does your dataset belong to?')
# print(regulated_domain)
# print()
#
# print('● How many observations are in the dataset?')
# print(num_observations)
# # print('data length, post dropna: ' + str(len(data.index)))  # still 649
# print()
#
# print('● How many variables are in the dataset?')
# print(num_variables)
# print()
#
# print('● Which variables did you select as your dependent variables?')
# print(dependent_variables)
# print()
#
# print('● How many and which variables in the dataset are associated with a legally recognized protected class?')
# print('\tWhich legal precedence/law (as discussed in the lectures) does each protected class fall under?')
# print(len(protected_classes))
# print(protected_classes[0] + ' - Age Discrimination in Employment Act of 1967')
# print(protected_classes[1] + ' - Equal Pay Act of 1963; Civil Rights Act of 1964, 1991')
# print()

# print('Step 2:\n')
#
# print('● Table documenting the relationship between members and membership categories for each protected class variable')
# print('\t(from Step 2.1)')
# print('age:')
unique_ages = df.age.unique()
unique_ages.sort()
# print(unique_ages)
# print('sex:')
# print(df.sex.unique())
# print()

# print('● Table documenting the relationship between values and discrete categories/numerical values associated with your')
# print('\tdependent variables (from Step 2.2)')
# print('G1:')
unique_G1s = df.G1.unique()
unique_G1s.sort()
# print(unique_G1s)
# print('G2:')
unique_G2s = df.G2.unique()
unique_G2s.sort()
# print(unique_G2s)
# print()

# print('● Table providing the computed frequency values for the membership categories each protected class variable')
# print('\t(from Step 2.3)')
# print('data.age.min():' + str(data.age.min())) # 15
# print('data.age.max():' + str(data.age.max())) # 22
# print('halfway point: ' + str((data.age.max() + data.age.min())/2)) # 18.5
# Setting the age group demarcation at about halfway between the max and min:
# data['age_group'] = data['age'].map(lambda value: '18 and under' if value <= 18 else 'over 18').astype(str)

# Age group histograms look very unbalanced.
# This is because the average age is 16, significantly less than the halfway point between the max and min.
# Resetting age groups to be based around the average, not halfway between min and max.
mean_age = int(df.age.mean())  # 16
# print('mean_age age = ' + str(mean_age))
# median_age = int(data.age.median()) # 17
# print('median_age age = ' + str(median_age))
# mode_age = int(data.age.mode()) # 17
# print('mode_age age = ' + str(mode_age))
# print()

df['age_group'] = df['age'].map(
    lambda value: str(mean_age) + ' and under' if value <= mean_age else 'over ' + str(mean_age))
# Tried groups 'under 16' & '16 and over', but those still looked a little unbalanced

# Reset protected_classes because we added the column age_group
protected_classes = ['age_group', 'sex']

# Pair up protected_classes & dependent_variables
pairs = list(itertools.product(protected_classes, dependent_variables))


# Print frequencies
# for _, (prot_class, dep_var) in enumerate(pairs):
#     # Aggregate dataframe
#     df_aggregate = df.groupby([prot_class, dep_var], as_index=False).size()
#     # Pivot aggregate
#     df_pivot = df_aggregate.pivot(index=prot_class, columns=dep_var)
#     # Remove size column values from legend
#     df_pivot.columns = df_pivot.columns.droplevel(0)
#     # Remove column name from legend
#     df_pivot.columns.name = None
#     # Reset index
#     df_pivot = df_pivot.reset_index(drop=False).rename({"index": prot_class}, axis=1).fillna(0)
#     print('Frequencies of ' + prot_class + ' & ' + dep_var + ':', end='')
#     # display(df_pivot)
#
# print()


# print('● Histograms derived from Step 2.4')
# for _, (prot_class, dep_var) in enumerate(pairs):
#     df_aggregate = df.groupby([prot_class, dep_var], as_index=False).size()
#     df_pivot = df_aggregate.pivot(index=prot_class, columns=dep_var)
#     df_pivot.columns = df_pivot.columns.droplevel(0)
#     df_pivot.columns.name = None
#     df_pivot = df_pivot.reset_index(drop=False).rename({"index": prot_class}, axis=1).fillna(0)
#     # Get subplots
#     _, ax = plt.subplots(1, 1, figsize=(10, 7), tight_layout=False)
#     # Plot the pivot aggregate
#     ax = df_pivot.plot(x=prot_class, kind="bar", stacked=True, title=f"{dep_var} vs. {prot_class}",
#                        colormap="turbo", ax=ax)
#     # Show the graph
#     plt.show()
#     print("\n\n")

# print()

# print('Step 3:\n')
#
# print('● Privileged/unprivileged groups associated with each protected class variable:')
# print('\tPrivileged age: Over 16')
# print('\tUnprivileged age: 16 & under')
# print('\tPrivileged sex: F')
# print('\tUnprivileged sex: M')
# print()


def group_G1_score(row):
    if row['G1'] < 6:
        return 'VeryLow'
    if row['G1'] < 11:
        return 'Low'
    if row['G1'] < 16:
        return 'Medium'
    return 'High'


def group_G2_score(row):
    if row['G2'] < 6:
        return 'VeryLow'
    if row['G2'] < 11:
        return 'Low'
    if row['G2'] < 16:
        return 'Medium'
    return 'High'


def group_G1_score_sex_transform(row):
    if row['sex'] == 'F':
        return row['G1_Group']
    if row['G1'] < 5:
        return 'VeryLow'
    if row['G1'] < 10:
        return 'Low'
    if row['G1'] < 15:
        return 'Medium'
    return 'High'


def group_G2_score_sex_transform(row):
    if row['sex'] == 'F':
        return row['G2_Group']
    if row['G2'] < 5:
        return 'VeryLow'
    if row['G2'] < 10:
        return 'Low'
    if row['G2'] < 15:
        return 'Medium'
    return 'High'


def group_G1_score_age_transform(row):
    if row['age_group'] == str(mean_age) + ' and under':
        return row['G1_Group']
    if row['G1'] < 5:
        return 'VeryLow'
    if row['G1'] < 10:
        return 'Low'
    if row['G1'] < 15:
        return 'Medium'
    return 'High'


def group_G2_score_age_transform(row):
    if row['age_group'] == str(mean_age) + ' and under':
        return row['G2_Group']
    if row['G2'] < 5:
        return 'VeryLow'
    if row['G2'] < 10:
        return 'Low'
    if row['G2'] < 15:
        return 'Medium'
    return 'High'


# Add columns that group G1 and G2 scores into 4 groups, very low - low - medium - high
df['G1_Group'] = df.apply(lambda row: group_G1_score(row), axis=1)
df['G2_Group'] = df.apply(lambda row: group_G2_score(row), axis=1)

# Add columns that group G1 and G2 scores into 4 groups shifted for males and over18 groups to remove bias
df['G1_Group_Sex_Transform'] = df.apply(lambda row: group_G1_score_sex_transform(row), axis=1)
df['G2_Group_Sex_Transform'] = df.apply(lambda row: group_G2_score_sex_transform(row), axis=1)
df['G1_Group_Age_Transform'] = df.apply(lambda row: group_G1_score_age_transform(row), axis=1)
df['G2_Group_Age_Transform'] = df.apply(lambda row: group_G2_score_age_transform(row), axis=1)

# print(df['G1_Group'].value_counts())
# print()
# print(df['G2_Group'].value_counts())
# print()

# Compute frequency values for the membership categories each protected class variable
rowCount = df.shape[0]
# print(f"Female Frequency: {round((df['sex'].value_counts()[0] / rowCount)*100, 3)}%")
# print(f"Male Frequency: {round((df['sex'].value_counts()[1] / rowCount)*100, 3)}%")
# print(f"{mean_age} and under Frequency: {round((df['age_group'].value_counts()[0] / rowCount)*100, 3)}%")
# print(f"Over {mean_age} Frequency: {round((df['age_group'].value_counts()[1] / rowCount)*100, 3)}%")

# Create histogram groups
sex_g1 = df.value_counts(['sex', 'G1_Group']).sort_index()
# print(sex_g1)
# print()
agegroup_g1 = df.value_counts(['age_group', 'G1_Group']).sort_index()
# print(agegroup_g1)
# print()
sex_g2 = df.value_counts(['sex', 'G2_Group']).sort_index()
# print(sex_g2)
# print()
agegroup_g2 = df.value_counts(['age_group', 'G2_Group']).sort_index()
# print(agegroup_g2)
# print()
sex_g1_transformed = df.value_counts(['sex', 'G1_Group_Sex_Transform']).sort_index()
# print(sex_g1_transformed)
# print()
agegroup_g1_transformed = df.value_counts(['age_group', 'G1_Group_Age_Transform']).sort_index()
# print(agegroup_g1_transformed)
# print()
sex_g2_transformed = df.value_counts(['sex', 'G2_Group_Sex_Transform']).sort_index()
# print(sex_g2_transformed)
# print()
agegroup_g2_transformed = df.value_counts(['age_group', 'G2_Group_Age_Transform']).sort_index()
# print(agegroup_g2_transformed)
# print()

# Create histogram for G1
barWidth = 0.25
# fig = plt.subplots(figsize =(12, 8))

VeryLow = [sex_g1[3], sex_g1[7], agegroup_g1[7], agegroup_g1[3]]
Low = [sex_g1[1], sex_g1[5], agegroup_g1[5], agegroup_g1[1]]
Medium = [sex_g1[2], sex_g1[6], agegroup_g1[6], agegroup_g1[2]]
High = [sex_g1[0], sex_g1[4], agegroup_g1[4], agegroup_g1[0]]

# Set position of bar on X axis
br1 = np.arange(len(VeryLow))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]

# Make the plot
# plt.bar(br1, VeryLow, color ='r', width = barWidth,
#         edgecolor ='grey', label ='VeryLow')
# plt.bar(br2, Low, color ='y', width = barWidth,
#         edgecolor ='grey', label ='Low')
# plt.bar(br3, Medium, color ='b', width = barWidth,
#         edgecolor ='grey', label ='Medium')
# plt.bar(br4, High, color ='g', width = barWidth,
#         edgecolor ='grey', label ='High')

# Adding Xticks
# plt.xlabel('Group', fontweight ='bold', fontsize = 15)
# plt.ylabel('Count', fontweight ='bold', fontsize = 15)
# plt.xticks([r + barWidth for r in range(len(VeryLow))],
#         ['Female', 'Male', str(mean_age) + ' and under', 'Over ' + str(mean_age)])

# plt.title('1st Period Grade (G1)', fontweight ='bold', fontsize = 18)
# plt.legend()
# plt.show()
# plt.close()

# Bias Mitigated histogram for G1
barWidth = 0.25
# fig = plt.subplots(figsize =(12, 8))

VeryLow = [sex_g1_transformed[3], sex_g1_transformed[7], agegroup_g1_transformed[7], agegroup_g1_transformed[3]]
Low = [sex_g1_transformed[1], sex_g1_transformed[5], agegroup_g1_transformed[5], agegroup_g1_transformed[1]]
Medium = [sex_g1_transformed[2], sex_g1_transformed[6], agegroup_g1_transformed[6], agegroup_g1_transformed[2]]
High = [sex_g1_transformed[0], sex_g1_transformed[4], agegroup_g1_transformed[4], agegroup_g1_transformed[0]]

# Set position of bar on X axis
br1 = np.arange(len(VeryLow))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]

# Make the plot
# plt.bar(br1, VeryLow, color ='r', width = barWidth,
#         edgecolor ='grey', label ='VeryLow')
# plt.bar(br2, Low, color ='y', width = barWidth,
#         edgecolor ='grey', label ='Low')
# plt.bar(br3, Medium, color ='b', width = barWidth,
#         edgecolor ='grey', label ='Medium')
# plt.bar(br4, High, color ='g', width = barWidth,
#         edgecolor ='grey', label ='High')

# Adding Xticks
# plt.xlabel('Group', fontweight ='bold', fontsize = 15)
# plt.ylabel('Count', fontweight ='bold', fontsize = 15)
# plt.xticks([r + barWidth for r in range(len(VeryLow))],
#         ['Female', 'Male', str(mean_age) + ' and under', 'Over ' + str(mean_age)])

# plt.title('1st Period Grade (G1) Transformed', fontweight ='bold', fontsize = 18)
# plt.legend()
# plt.show()
# plt.close()

# Create histogram for G2
barWidth = 0.25
# fig = plt.subplots(figsize =(12, 8))

VeryLow = [sex_g2[3], sex_g2[7], agegroup_g2[7], agegroup_g2[3]]
Low = [sex_g2[1], sex_g2[5], agegroup_g2[5], agegroup_g2[1]]
Medium = [sex_g2[2], sex_g2[6], agegroup_g2[6], agegroup_g2[2]]
High = [sex_g2[0], sex_g2[4], agegroup_g2[4], agegroup_g2[0]]

# Set position of bar on X axis
br1 = np.arange(len(VeryLow))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]

# Make the plot
# plt.bar(br1, VeryLow, color ='r', width = barWidth,
#         edgecolor ='grey', label ='VeryLow')
# plt.bar(br2, Low, color ='y', width = barWidth,
#         edgecolor ='grey', label ='Low')
# plt.bar(br3, Medium, color ='b', width = barWidth,
#         edgecolor ='grey', label ='Medium')
# plt.bar(br4, High, color ='g', width = barWidth,
#         edgecolor ='grey', label ='High')
#
# # Adding Xticks
# plt.xlabel('Group', fontweight ='bold', fontsize = 15)
# plt.ylabel('Count', fontweight ='bold', fontsize = 15)
# plt.xticks([r + barWidth for r in range(len(VeryLow))],
#         ['Female', 'Male', str(mean_age) + ' and under', 'Over ' + str(mean_age)])
#
# plt.title('2nd Period Grade (G2)', fontweight ='bold', fontsize = 18)
# plt.legend()
# plt.show()
# plt.close()

# Bias Mitigated histogram for G2
barWidth = 0.25
# fig = plt.subplots(figsize =(12, 8))

VeryLow = [sex_g2_transformed[3], sex_g2_transformed[7], agegroup_g2_transformed[7], agegroup_g2_transformed[3]]
Low = [sex_g2_transformed[1], sex_g2_transformed[5], agegroup_g2_transformed[5], agegroup_g2_transformed[1]]
Medium = [sex_g2_transformed[2], sex_g2_transformed[6], agegroup_g2_transformed[6], agegroup_g2_transformed[2]]
High = [sex_g2_transformed[0], sex_g2_transformed[4], agegroup_g2_transformed[4], agegroup_g2_transformed[0]]

# Set position of bar on X axis
br1 = np.arange(len(VeryLow))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]

# Make the plot
# plt.bar(br1, VeryLow, color ='r', width = barWidth,
#         edgecolor ='grey', label ='VeryLow')
# plt.bar(br2, Low, color ='y', width = barWidth,
#         edgecolor ='grey', label ='Low')
# plt.bar(br3, Medium, color ='b', width = barWidth,
#         edgecolor ='grey', label ='Medium')
# plt.bar(br4, High, color ='g', width = barWidth,
#         edgecolor ='grey', label ='High')
#
# # Adding Xticks
# plt.xlabel('Group', fontweight ='bold', fontsize = 15)
# plt.ylabel('Count', fontweight ='bold', fontsize = 15)
# plt.xticks([r + barWidth for r in range(len(VeryLow))],
#         ['Female', 'Male', str(mean_age) + ' and under', 'Over ' + str(mean_age)])
#
# plt.title('2nd Period Grade (G2) Transformed', fontweight ='bold', fontsize = 18)
# plt.legend()
# plt.show()
# plt.close()

# For G1 combine very low and low into negative group, combine medium and high into positive group
female_negative_g1 = sex_g1[3] + sex_g1[1]
female_positive_g1 = sex_g1[2] + sex_g1[0]
male_negative_g1 = sex_g1[7] + sex_g1[5]
male_positive_g1 = sex_g1[6] + sex_g1[4]
mean_age_under_negative_g1 = agegroup_g1[7] + agegroup_g1[5]
mean_age_under_positive_g1 = agegroup_g1[6] + agegroup_g1[4]
over_mean_age_negative_g1 = agegroup_g1[3] + agegroup_g1[1]
over_mean_age_positive_g1 = agegroup_g1[2] + agegroup_g1[0]

female_negative_g1_transformed = sex_g1_transformed[3] + sex_g1_transformed[1]
female_positive_g1_transformed = sex_g1_transformed[2] + sex_g1_transformed[0]
male_negative_g1_transformed = sex_g1_transformed[7] + sex_g1_transformed[5]
male_positive_g1_transformed = sex_g1_transformed[6] + sex_g1_transformed[4]
mean_age_under_negative_g1_transformed = agegroup_g1_transformed[7] + agegroup_g1_transformed[5]
mean_age_under_positive_g1_transformed = agegroup_g1_transformed[6] + agegroup_g1_transformed[4]
over_mean_age_negative_g1_transformed = agegroup_g1_transformed[3] + agegroup_g1_transformed[1]
over_mean_age_positive_g1_transformed = agegroup_g1_transformed[2] + agegroup_g1_transformed[0]

# print(f"Female negative G1: {female_negative_g1}")
# print(f"Female positive G1: {female_positive_g1}")
# print(f"Male negative G1: {male_negative_g1}")
# print(f"Male positive G1: {male_positive_g1}")
# print(f"{mean_age} and under negative G1: {mean_age_under_negative_g1}")
# print(f"{mean_age} and under positive G1: {mean_age_under_positive_g1}")
# print(f"Over {mean_age} negative G1: {over_mean_age_negative_g1}")
# print(f"Over {mean_age} positive G1: {over_mean_age_positive_g1}")
#
# print(f"Female negative G1 transformed: {female_negative_g1_transformed}")
# print(f"Female positive G1 transformed: {female_positive_g1_transformed}")
# print(f"Male negative G1 transformed: {male_negative_g1_transformed}")
# print(f"Male positive G1 transformed: {male_positive_g1_transformed}")
# print(f"{mean_age} and under negative G1 transformed: {mean_age_under_negative_g1_transformed}")
# print(f"{mean_age} and under positive G1 transformed: {mean_age_under_positive_g1_transformed}")
# print(f"Over {mean_age} negative G1 transformed: {over_mean_age_negative_g1_transformed}")
# print(f"Over {mean_age} positive G1 transformed: {over_mean_age_positive_g1_transformed}")

# Statistical parity difference sex g1
stat_par_diff_sex_g1 = (male_positive_g1 / female_positive_g1) - ((male_positive_g1 + male_negative_g1) / \
                                                                  (female_positive_g1 + female_negative_g1))
# print(f"Statistical Parity Difference Sex G1: {round(stat_par_diff_sex_g1, 3)}")

stat_par_diff_list = [stat_par_diff_sex_g1]
# plt.ylabel('Statistical Parity Difference Sex G1')
# plt.bar(1, stat_par_diff_list)
# plt.xticks([1], ['Statistical Parity Difference Sex G1'])
# plt.hlines(y=0.2, xmin=0, xmax=2, color='r', label='bias')
# plt.hlines(y=-0.2, xmin=0, xmax=2, color='r')
# plt.hlines(y=0.0, xmin=0, xmax=2, color='b', label='fair')
# plt.legend()
# # plt.savefig('statpardiffsexg1.png')
# plt.show()
# plt.close()

# Statistical parity difference sex g1 bias mitigated
stat_par_diff_sex_g1_transformed = (male_positive_g1_transformed / female_positive_g1_transformed) - \
                                   ((male_positive_g1_transformed + male_negative_g1_transformed) / (
                                               female_positive_g1_transformed + \
                                               female_negative_g1_transformed))
# print(f"Statistical Parity Difference Sex G1 Transformed: {round(stat_par_diff_sex_g1_transformed, 3)}")

stat_par_diff_list = [stat_par_diff_sex_g1_transformed]
# plt.ylabel('Statistical Parity Difference Sex G1 Transformed')
# plt.bar(1, stat_par_diff_list)
# plt.xticks([1], ['Statistical Parity Difference Sex G1 Transformed'])
# plt.hlines(y=0.2, xmin=0, xmax=2, color='r', label='bias')
# plt.hlines(y=-0.2, xmin=0, xmax=2, color='r')
# plt.hlines(y=0.0, xmin=0, xmax=2, color='b', label='fair')
# plt.legend()
# # plt.savefig('statpardiffsexg1transf.png')
# plt.show()
# plt.close()

# Disparate impact sex g1
disparate_impact_sex_g1 = (male_positive_g1 / (male_positive_g1 + male_negative_g1)) / (female_positive_g1 / \
                                                                                        (
                                                                                                    female_positive_g1 + female_negative_g1))
# print(f"Disparate Impact Sex G1: {round(disparate_impact_sex_g1, 3)}")

disparate_impact_list = [disparate_impact_sex_g1]
# plt.ylabel('Disparate Impact Sex G1')
# plt.bar(1, disparate_impact_list)
# plt.xticks([1], ['Disparate Impact Sex G1'])
# plt.hlines(y=0.8, xmin=0, xmax=2, color='r', label='bias')
# plt.hlines(y=1.2, xmin=0, xmax=2, color='r')
# plt.hlines(y=1.0, xmin=0, xmax=2, color='b', label='fair')
# plt.legend()
# # plt.savefig('disparateimpactsexg1.png')
# plt.show()
# plt.close()

# Disparate impact sex g1 bias mitigated
disparate_impact_sex_g1_transformed = (male_positive_g1_transformed / (male_positive_g1_transformed + \
                                                                       male_negative_g1_transformed)) / (
                                                  female_positive_g1_transformed / (female_positive_g1_transformed + \
                                                                                    female_negative_g1_transformed))
# print(f"Disparate Impact Sex G1 Transformed: {round(disparate_impact_sex_g1_transformed, 3)}")

disparate_impact_list = [disparate_impact_sex_g1_transformed]
# plt.ylabel('Disparate Impact Sex G1 Transformed')
# plt.bar(1, disparate_impact_list)
# plt.xticks([1], ['Disparate Impact Sex G1 Transformed'])
# plt.hlines(y=0.8, xmin=0, xmax=2, color='r', label='bias')
# plt.hlines(y=1.2, xmin=0, xmax=2, color='r')
# plt.hlines(y=1.0, xmin=0, xmax=2, color='b', label='fair')
# plt.legend()
# # plt.savefig('disparateimpactsexg1transformed.png')
# plt.show()
# plt.close()

# Statistical parity difference age g1
stat_par_diff_age_g1 = (over_mean_age_positive_g1 / mean_age_under_positive_g1) - ((over_mean_age_positive_g1 + \
                                                                                    over_mean_age_negative_g1) / (
                                                                                               mean_age_under_positive_g1 + mean_age_under_negative_g1))
# print(f"Statistical Parity Difference Age G1: {round(stat_par_diff_age_g1, 3)}")

stat_par_diff_list = [stat_par_diff_age_g1]
# plt.ylabel('Statistical Parity Difference Age G1')
# plt.bar(1, stat_par_diff_list)
# plt.xticks([1], ['Statistical Parity Difference Age G1'])
# plt.hlines(y=0.2, xmin=0, xmax=2, color='r', label='bias')
# plt.hlines(y=-0.2, xmin=0, xmax=2, color='r')
# plt.hlines(y=0.0, xmin=0, xmax=2, color='b', label='fair')
# plt.legend()
# # plt.savefig('statpardiffageg1.png')
# plt.show()
# plt.close()

# Statistical parity difference age g1 bias mitigated
stat_par_diff_age_g1_transformed = (over_mean_age_positive_g1_transformed / mean_age_under_positive_g1_transformed) \
                                   - ((over_mean_age_positive_g1_transformed + over_mean_age_negative_g1_transformed) / \
                                      (mean_age_under_positive_g1_transformed + mean_age_under_negative_g1_transformed))
# print(f"Statistical Parity Difference Age G1 Transformed: {round(stat_par_diff_age_g1_transformed, 3)}")

stat_par_diff_list = [stat_par_diff_age_g1_transformed]
# plt.ylabel('Statistical Parity Difference Age G1 Transformed')
# plt.bar(1, stat_par_diff_list)
# plt.xticks([1], ['Statistical Parity Difference Age G1 Transformed'])
# plt.hlines(y=0.2, xmin=0, xmax=2, color='r', label='bias')
# plt.hlines(y=-0.2, xmin=0, xmax=2, color='r')
# plt.hlines(y=0.0, xmin=0, xmax=2, color='b', label='fair')
# plt.legend()
# # plt.savefig('statpardiffageg1transformed.png')
# plt.show()
# plt.close()

# Disparate impact age g1
disparate_impact_age_g1 = (over_mean_age_positive_g1 / (over_mean_age_positive_g1 + over_mean_age_negative_g1)) / \
                          (mean_age_under_positive_g1 / (mean_age_under_positive_g1 + mean_age_under_negative_g1))
# print(f"Disparate Impact Age G1: {round(disparate_impact_age_g1, 3)}")

disparate_impact_list = [disparate_impact_age_g1]
# plt.ylabel('Disparate Impact Age G1')
# plt.bar(1, disparate_impact_list)
# plt.xticks([1], ['Disparate Impact Age G1'])
# plt.hlines(y=0.8, xmin=0, xmax=2, color='r', label='bias')
# plt.hlines(y=1.2, xmin=0, xmax=2, color='r')
# plt.hlines(y=1.0, xmin=0, xmax=2, color='b', label='fair')
# plt.legend()
# # plt.savefig('disparateimpactageg1.png')
# plt.show()
# plt.close()

# Disparate impact age g1 bias mitigated
disparate_impact_age_g1_transformed = (over_mean_age_positive_g1_transformed / \
                                       (
                                                   over_mean_age_positive_g1_transformed + over_mean_age_negative_g1_transformed)) / \
                                      (mean_age_under_positive_g1_transformed / (
                                                  mean_age_under_positive_g1_transformed + \
                                                  mean_age_under_negative_g1_transformed))
# print(f"Disparate Impact Age G1 Transformed: {round(disparate_impact_age_g1_transformed, 3)}")

disparate_impact_list = [disparate_impact_age_g1_transformed]
# plt.ylabel('Disparate Impact Age G1 Transformed')
# plt.bar(1, disparate_impact_list)
# plt.xticks([1], ['Disparate Impact Age G1 Transformed'])
# plt.hlines(y=0.8, xmin=0, xmax=2, color='r', label='bias')
# plt.hlines(y=1.2, xmin=0, xmax=2, color='r')
# plt.hlines(y=1.0, xmin=0, xmax=2, color='b', label='fair')
# plt.legend()
# # plt.savefig('disparateimpactageg1transformed.png')
# plt.show()
# plt.close()

# For G2 combine very low and low into negative group, combine medium and high into positive group
female_negative_g2 = sex_g2[3] + sex_g2[1]
female_positive_g2 = sex_g2[2] + sex_g2[0]
male_negative_g2 = sex_g2[7] + sex_g2[5]
male_positive_g2 = sex_g2[6] + sex_g2[4]
mean_age_under_negative_g2 = agegroup_g2[7] + agegroup_g2[5]
mean_age_under_positive_g2 = agegroup_g2[6] + agegroup_g2[4]
over_mean_age_negative_g2 = agegroup_g2[3] + agegroup_g2[1]
over_mean_age_positive_g2 = agegroup_g2[2] + agegroup_g2[0]

female_negative_g2_transformed = sex_g2_transformed[3] + sex_g2_transformed[1]
female_positive_g2_transformed = sex_g2_transformed[2] + sex_g2_transformed[0]
male_negative_g2_transformed = sex_g2_transformed[7] + sex_g2_transformed[5]
male_positive_g2_transformed = sex_g2_transformed[6] + sex_g2_transformed[4]
mean_age_under_negative_g2_transformed = agegroup_g2_transformed[7] + agegroup_g2_transformed[5]
mean_age_under_positive_g2_transformed = agegroup_g2_transformed[6] + agegroup_g2_transformed[4]
over_mean_age_negative_g2_transformed = agegroup_g2_transformed[3] + agegroup_g2_transformed[1]
over_mean_age_positive_g2_transformed = agegroup_g2_transformed[2] + agegroup_g2_transformed[0]

# print(f"Female negative G2: {female_negative_g2}")
# print(f"Female positive G2: {female_positive_g2}")
# print(f"Male negative G2: {male_negative_g2}")
# print(f"Male positive G2: {male_positive_g2}")
# print(f"{mean_age} and under negative G2: {mean_age_under_negative_g2}")
# print(f"{mean_age} and under positive G2: {mean_age_under_positive_g2}")
# print(f"Over {mean_age} negative G2: {over_mean_age_negative_g2}")
# print(f"Over {mean_age} positive G2: {over_mean_age_positive_g2}")
#
# print(f"Female negative G2 transformed: {female_negative_g2_transformed}")
# print(f"Female positive G2 transformed: {female_positive_g2_transformed}")
# print(f"Male negative G2 transformed: {male_negative_g2_transformed}")
# print(f"Male positive G2 transformed: {male_positive_g2_transformed}")
# print(f"{mean_age} and under negative G2 transformed: {mean_age_under_negative_g2_transformed}")
# print(f"{mean_age} and under positive G2 transformed: {mean_age_under_positive_g2_transformed}")
# print(f"Over {mean_age} negative G2 transformed: {over_mean_age_negative_g2_transformed}")
# print(f"Over {mean_age} positive G2 transformed: {over_mean_age_positive_g2_transformed}")

# Statistical parity difference sex g2
stat_par_diff_sex_g2 = (male_positive_g2 / female_positive_g2) - ((male_positive_g2 + male_negative_g2) / \
                                                                  (female_positive_g2 + female_negative_g2))
# print(f"Statistical Parity Difference Sex G2: {round(stat_par_diff_sex_g2, 3)}")

stat_par_diff_list = [stat_par_diff_sex_g2]
# plt.ylabel('Statistical Parity Difference Sex G2')
# plt.bar(1, stat_par_diff_list)
# plt.xticks([1], ['Statistical Parity Difference Sex G2'])
# plt.hlines(y=0.2, xmin=0, xmax=2, color='r', label='bias')
# plt.hlines(y=-0.2, xmin=0, xmax=2, color='r')
# plt.hlines(y=0.0, xmin=0, xmax=2, color='b', label='fair')
# plt.legend()
# # plt.savefig('statpardiffsexg2.png')
# plt.show()
# plt.close()

# Statistical parity difference sex g2 bias mitigated
stat_par_diff_sex_g2_transformed = (male_positive_g2_transformed / female_positive_g2_transformed) - \
                                   ((male_positive_g2_transformed + male_negative_g2_transformed) / (
                                               female_positive_g2_transformed + \
                                               female_negative_g2_transformed))
# print(f"Statistical Parity Difference Sex G2 Transformed: {round(stat_par_diff_sex_g2_transformed, 3)}")

stat_par_diff_list = [stat_par_diff_sex_g2_transformed]
# plt.ylabel('Statistical Parity Difference Sex G2 Transformed')
# plt.bar(1, stat_par_diff_list)
# plt.xticks([1], ['Statistical Parity Difference Sex G2 Transformed'])
# plt.hlines(y=0.2, xmin=0, xmax=2, color='r', label='bias')
# plt.hlines(y=-0.2, xmin=0, xmax=2, color='r')
# plt.hlines(y=0.0, xmin=0, xmax=2, color='b', label='fair')
# plt.legend()
# # plt.savefig('statpardiffsexg2transformed.png')
# plt.show()
# plt.close()

# Disparate impact sex g2
disparate_impact_sex_g2 = (male_positive_g2 / (male_positive_g2 + male_negative_g2)) / (female_positive_g2 / \
                                                                                        (
                                                                                                    female_positive_g2 + female_negative_g2))
# print(f"Disparate Impact Sex G2: {round(disparate_impact_sex_g2, 3)}")

disparate_impact_list = [disparate_impact_sex_g2]
# plt.ylabel('Disparate Impact Sex G2')
# plt.bar(1, disparate_impact_list)
# plt.xticks([1], ['Disparate Impact Sex G2'])
# plt.hlines(y=0.8, xmin=0, xmax=2, color='r', label='bias')
# plt.hlines(y=1.2, xmin=0, xmax=2, color='r')
# plt.hlines(y=1.0, xmin=0, xmax=2, color='b', label='fair')
# plt.legend()
# # plt.savefig('disparateimpactsexg2.png')
# plt.show()
# plt.close()

# Disparate impact sex g2 bias mitigated
disparate_impact_sex_g2_transformed = (male_positive_g2_transformed / (male_positive_g2_transformed + \
                                                                       male_negative_g2_transformed)) / (
                                                  female_positive_g2_transformed / (female_positive_g2_transformed + \
                                                                                    female_negative_g2_transformed))
# print(f"Disparate Impact Sex G2 Transformed: {round(disparate_impact_sex_g2_transformed, 3)}")

disparate_impact_list = [disparate_impact_sex_g2_transformed]
# plt.ylabel('Disparate Impact Sex G2 Transformed')
# plt.bar(1, disparate_impact_list)
# plt.xticks([1], ['Disparate Impact Sex G2 Transformed'])
# plt.hlines(y=0.8, xmin=0, xmax=2, color='r', label='bias')
# plt.hlines(y=1.2, xmin=0, xmax=2, color='r')
# plt.hlines(y=1.0, xmin=0, xmax=2, color='b', label='fair')
# plt.legend()
# # plt.savefig('disparateimpactsexg2transformed.png')
# plt.show()
# plt.close()

# Statistical parity difference age g2
stat_par_diff_age_g2 = (over_mean_age_positive_g2 / mean_age_under_positive_g2) - ((over_mean_age_positive_g2 + \
                                                                                    over_mean_age_negative_g2) / (
                                                                                               mean_age_under_positive_g2 + mean_age_under_negative_g2))
# print(f"Statistical Parity Difference Age G2: {round(stat_par_diff_age_g2, 3)}")

stat_par_diff_list = [stat_par_diff_age_g2]
# plt.ylabel('Statistical Parity Difference Age G2')
# plt.bar(1, stat_par_diff_list)
# plt.xticks([1], ['Statistical Parity Difference Age G2'])
# plt.hlines(y=0.2, xmin=0, xmax=2, color='r', label='bias')
# plt.hlines(y=-0.2, xmin=0, xmax=2, color='r')
# plt.hlines(y=0.0, xmin=0, xmax=2, color='b', label='fair')
# plt.legend()
# # plt.savefig('statpardiffageg2.png')
# plt.show()
# plt.close()

# Statistical parity difference age g2 bias mitigated
stat_par_diff_age_g2_transformed = (over_mean_age_positive_g2_transformed / mean_age_under_positive_g2_transformed) \
                                   - ((over_mean_age_positive_g2_transformed + over_mean_age_negative_g2_transformed) / \
                                      (mean_age_under_positive_g2_transformed + mean_age_under_negative_g2_transformed))
# print(f"Statistical Parity Difference Age G2 Transformed: {round(stat_par_diff_age_g2_transformed, 3)}")

stat_par_diff_list = [stat_par_diff_age_g2_transformed]
# plt.ylabel('Statistical Parity Difference Age G2 Transformed')
# plt.bar(1, stat_par_diff_list)
# plt.xticks([1], ['Statistical Parity Difference Age G2 Transformed'])
# plt.hlines(y=0.2, xmin=0, xmax=2, color='r', label='bias')
# plt.hlines(y=-0.2, xmin=0, xmax=2, color='r')
# plt.hlines(y=0.0, xmin=0, xmax=2, color='b', label='fair')
# plt.legend()
# # plt.savefig('statpardiffageg2transformed.png')
# plt.show()
# plt.close()

# Disparate impact age g2
disparate_impact_age_g2 = (over_mean_age_positive_g2 / (over_mean_age_positive_g2 + over_mean_age_negative_g2)) / \
                          (mean_age_under_positive_g2 / (mean_age_under_positive_g2 + mean_age_under_negative_g2))
# print(f"Disparate Impact Age G2: {round(disparate_impact_age_g2, 3)}")

disparate_impact_list = [disparate_impact_age_g2]
# plt.ylabel('Disparate Impact Age G2')
# plt.bar(1, disparate_impact_list)
# plt.xticks([1], ['Disparate Impact Age G2'])
# plt.hlines(y=0.8, xmin=0, xmax=2, color='r', label='bias')
# plt.hlines(y=1.2, xmin=0, xmax=2, color='r')
# plt.hlines(y=1.0, xmin=0, xmax=2, color='b', label='fair')
# plt.legend()
# # plt.savefig('disparateimpactageg2.png')
# plt.show()
# plt.close()

# Disparate impact age g2 bias mitigated
disparate_impact_age_g2_transformed = (over_mean_age_positive_g2_transformed / \
                                       (
                                                   over_mean_age_positive_g2_transformed + over_mean_age_negative_g2_transformed)) / \
                                      (mean_age_under_positive_g2_transformed / (
                                                  mean_age_under_positive_g2_transformed + \
                                                  mean_age_under_negative_g2_transformed))
# print(f"Disparate Impact Age G2 Transformed: {round(disparate_impact_age_g2_transformed, 3)}")

disparate_impact_list = [disparate_impact_age_g2_transformed]
# plt.ylabel('Disparate Impact Age G2 Transformed')
# plt.bar(1, disparate_impact_list)
# plt.xticks([1], ['Disparate Impact Age G2 Transformed'])
# plt.hlines(y=0.8, xmin=0, xmax=2, color='r', label='bias')
# plt.hlines(y=1.2, xmin=0, xmax=2, color='r')
# plt.hlines(y=1.0, xmin=0, xmax=2, color='b', label='fair')
# plt.legend()
# # plt.savefig('disparateimpactageg2transformed.png')
# plt.show()
# plt.close()

print('Step 4:')
print('Option A\n')
sections = [int(.5 * df.shape[0])]
df_train, df_test = np.split(df.sample(frac=1, random_state=0), sections)

# print(f"df_train.shape[0]: {df_train.shape[0]}")
# print(f"df_test.shape[0]: {df_test.shape[0]}")

sections = [int(.5 * sex_g1_transformed.shape[0])]
sex_g1_transformed_train, sex_g1_transformed_test = np.split(sex_g1_transformed.sample(frac=1, random_state=0),
                                                             sections)


# print(f"sex_g1_transformed_train.shape[0]: {sex_g1_transformed_train.shape[0]}")
# print(f"sex_g1_transformed_test.shape[0]: {sex_g1_transformed_test.shape[0]}")

# sections = [int(.5 * agegroup_g1_transformed.shape[0])]
# agegroup_g1_transformed_train, agegroup_g1_transformed_test = np.split(agegroup_g1_transformed.sample(frac=1, random_state=0), sections)
# # print(f"agegroup_g1_transformed_train.shape[0]: {agegroup_g1_transformed_train.shape[0]}")
# # print(f"agegroup_g1_transformed_test.shape[0]: {agegroup_g1_transformed_test.shape[0]}")

# sections = [int(.5 * sex_g2_transformed.shape[0])]
# sex_g2_transformed_train, sex_g2_transformed_test = np.split(sex_g2_transformed.sample(frac=1, random_state=0), sections)
# # print(f"sex_g2_transformed_train.shape[0]: {sex_g2_transformed_train.shape[0]}")
# # print(f"sex_g2_transformed_test.shape[0]: {sex_g2_transformed_test.shape[0]}")

# sections = [int(.5 * agegroup_g2_transformed.shape[0])]
# agegroup_g2_transformed_train, agegroup_g2_transformed_test = np.split(agegroup_g2_transformed.sample(frac=1, random_state=0), sections)
# # print(f"agegroup_g2_transformed_train.shape[0]: {agegroup_g2_transformed_train.shape[0]}")
# # print(f"agegroup_g2_transformed_test.shape[0]: {agegroup_g2_transformed_test.shape[0]}")

# def df_to_BinaryLabelDataset(df, protected_groups, label_names, favorable_label, unfavorable_label):
#     # Modified from here:
#     # https://programtalk.com/python-more-examples/aif360.datasets.BinaryLabelDataset/
#     # label_names = [y.name], a list of the names of the target column
#     # https://aif360.readthedocs.io/en/latest/modules/generated/aif360.datasets.BinaryLabelDataset.html
#     result = BinaryLabelDataset(
#         df=df, protected_attribute_names=protected_groups,
#         label_names=label_names, favorable_label=favorable_label,
#         unfavorable_label=unfavorable_label
#     )
#     return result


# print(sex_g1_transformed_train.head())
# sex_g1_transformed_train['sex_number'] = sex_g1_transformed_train['sex'].map(lambda v: 0 if v == 'M' else 1)
# non_numeric = 'sex'
# sex_g1_transformed_train = sex_g1_transformed_train.drop(non_numeric, axis=1)
# sex_g1_transformed_train['sex_number'] = np.where(df['sex'] == 'F', 1, 0)

# 4.1: Randomly split original dataset into training and testing datasets
sections = [int(.5 * df.shape[0])]
df_original = pd.read_csv('student/student-por.csv', sep=';')
df_original.dropna(inplace=True)
df_original['age_group'] = df_original['age'].map(lambda value: str(mean_age) + ' and under' if value <= mean_age else 'over ' + str(mean_age))
df_original_train, df_original_test = np.split(df_original.sample(frac=1, random_state=0), sections)

# 4.2: Randomly split transformed dataset into training and testing datasets
df_transformed_train, df_transformed_test = np.split(df.sample(frac=1, random_state=0), sections)


# Prepare for next steps:
def numberfy(df, numeric, original, value_0):
    df[numeric] = df[original].map(lambda v: 0 if v == value_0 else 1)
    df = df.drop(original, axis=1)
    return df


def numberfy_all(df, transformed=False):
    df = numberfy(df, 'school_number', 'school', 'GP')
    df = numberfy(df, 'sex_number', 'sex', 'M')
    df = numberfy(df, 'address_number', 'address', 'R')
    df = numberfy(df, 'famsize_number', 'famsize', 'GT3')
    df = numberfy(df, 'Pstatus_number', 'Pstatus', 'A')
    df = numberfy(df, 'Mjob_number', 'Mjob', 'other')
    df = numberfy(df, 'Fjob_number', 'Fjob', 'other')
    df = numberfy(df, 'reason_number', 'reason', 'other')
    df = numberfy(df, 'guardian_number', 'guardian', 'other')
    df = numberfy(df, 'schoolsup_number', 'schoolsup', 'no')
    df = numberfy(df, 'famsup_number', 'famsup', 'no')
    df = numberfy(df, 'paid_number', 'paid', 'no')
    df = numberfy(df, 'activities_number', 'activities', 'no')
    df = numberfy(df, 'nursery_number', 'nursery', 'no')
    df = numberfy(df, 'higher_number', 'higher', 'no')
    df = numberfy(df, 'internet_number', 'internet', 'no')
    df = numberfy(df, 'romantic_number', 'romantic', 'yes')
    df = numberfy(df, 'age_group_number', 'age_group', 'over 16')
    if transformed:
        df = numberfy(df, 'G1_Group_number', 'G1_Group', 'VeryLow')
        df = numberfy(df, 'G2_Group_number', 'G2_Group', 'VeryLow')
        df = numberfy(df, 'G1_Group_Age_Transform_number', 'G1_Group_Age_Transform', 'VeryLow')
        df = numberfy(df, 'G1_Group_Sex_Transform_number', 'G1_Group_Sex_Transform', 'VeryLow')
        df = numberfy(df, 'G2_Group_Age_Transform_number', 'G2_Group_Age_Transform', 'VeryLow')
        df = numberfy(df, 'G2_Group_Sex_Transform_number', 'G2_Group_Sex_Transform', 'VeryLow')

    return df


df_original_train = numberfy_all(df_original_train)
df_transformed_train = numberfy_all(df_transformed_train, transformed=True)


# Converts a dataframe into a list of tf.Example protos.
def df_to_examples(df, columns=None):
    examples = []
    if columns is None:
        columns = df.columns.values.tolist()
    for index, row in df.iterrows():
        example = tf.train.Example()
        for col in columns:
            if df[col].dtype is np.dtype(np.int64):
                example.features.feature[col].int64_list.value.append(int(row[col]))
            elif df[col].dtype is np.dtype(np.float64):
                example.features.feature[col].float_list.value.append(row[col])
            elif row[col] == row[col]:
                example.features.feature[col].bytes_list.value.append(row[col].encode('utf-8'))
        examples.append(example)
    return examples


# Creates simple numeric and categorical feature columns from a feature spec and a
# list of columns from that spec to use.
#
# NOTE: Models might perform better with some feature engineering such as bucketed
# numeric columns and hash-bucket/embedding columns for categorical features.
def create_feature_columns(data, columns, feature_spec_internal):
    ret = []
    for col in columns:
        if feature_spec_internal[col].dtype is tf.int64 or feature_spec_internal[col].dtype is tf.float32:
            ret.append(tf.feature_column.numeric_column(col))
        else:
            ret.append(tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(col, list(data[col].unique()))))
    return ret


# Creates a Tensorflow feature spec from the dataframe and columns specified.
def create_feature_spec(df, columns=None):
    feature_spec_internal = {}
    if columns is None:
        columns = df.columns.values.tolist()
    for f in columns:
        if df[f].dtype is np.dtype(np.int64):
            feature_spec_internal[f] = tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
        elif df[f].dtype is np.dtype(np.float64):
            feature_spec_internal[f] = tf.io.FixedLenFeature(shape=(), dtype=tf.float32)
        else:
            feature_spec_internal[f] = tf.io.FixedLenFeature(shape=(), dtype=tf.string)
    return feature_spec_internal


# Parses Tensorflow.Example protos into features for the input function, tf_examples_input_fn.
def parse_tf_example(example_proto, label, feature_spec_internal):
    parsed_features = tf.io.parse_example(serialized=example_proto, features=feature_spec_internal)
    target = parsed_features.pop(label)
    return parsed_features, target


# Tensorflow Examples input function
# An input function for providing input to a model from Tensorflow.Examples
def tf_examples_input_fn(examples, feature_spec_internal, label, mode=tf.estimator.ModeKeys.EVAL, num_epochs=None, batch_size=64):
    def ex_generator():
        for j_index in range(len(examples)):
            yield examples[j_index].SerializeToString()
    dataset = tf.data.Dataset.from_generator(
      ex_generator, tf.dtypes.string, tf.TensorShape([]))
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda tf_example: parse_tf_example(tf_example, label, feature_spec_internal))
    dataset = dataset.repeat(num_epochs)
    return dataset


def train_classifier(df, df_name, protected_class):
    # Set list of all columns from the dataset we will use for model input.
    input_features = ['G1', 'G2']  # outcome_variables
    # Turn dataframe into a list of Tensorflow Examples
    test_examples = df_to_examples(df)
    # Create a feature spec for the classifier
    feature_spec = create_feature_spec(df, None)

    # Define and train the classifier
    num_steps = 500
    train_inpf = functools.partial(tf_examples_input_fn, test_examples, feature_spec, protected_class) # 'age_group_number')
    classifier = tf.estimator.LinearClassifier(
        feature_columns=create_feature_columns(df, input_features, feature_spec))

    print(f"Starting training {df_name} for {num_steps} steps...")
    start = time.time()

    classifier.train(train_inpf, steps=num_steps)

    end = time.time()
    print(f"Finished training {df_name} in {get_time(end - start)}\n")
    return classifier


# 4.3: Train a classifier using the original training dataset from Step 4.1
classifier_otd = train_classifier(df_original_train, 'Original training dataset', 'age_group_number')

# 4.4: Train a classifier using the transformed training dataset from Step 4.2;
classifier_ttd = train_classifier(df_transformed_train, 'Transformed training dataset', 'age_group_number')

# classifier_ttd.

print('Done')
