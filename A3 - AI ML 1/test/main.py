import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np


def correlation_category(correlation_value):
    if correlation_value < 0:
        output = ' (negative '
    else:
        output = ' (positive '
    if abs(correlation_value) < 0.2:
        output += 'very weak correlation)'
    elif abs(correlation_value) < 0.4:
        output += 'weak correlation)'
    elif abs(correlation_value) < 0.6:
        output += 'moderate correlation)'
    elif abs(correlation_value) < 0.8:
        output += 'strong correlation)'
    else:
        output += 'very strong correlation)'
    return output


def my_sort(n):
    return abs(n)


data = pd.read_csv("../toxity_per_attribute.csv", low_memory=False)

key_columns = ["Wiki_ID", "TOXICITY"]

combined_group_members = {
    'race_color_nationality': {
        'asian_group': ['asian', 'chinese', 'japanese'],
        'black_group': ['african', 'african american', 'black'],
        'hispanic_group': ['hispanic', 'latino', 'latina', 'latinx', 'mexican'],
        'white_group': ['american', 'canadian', 'european', 'white'],
        'nota': ['indian', 'middle eastern']
    },
    'sex': {
        'not_straight': ['bisexual', 'gay', 'homosexual', 'lesbian', 'lgbt', 'lgbtq', 'queer'],
        'straight_group': ['heterosexual', 'straight'],
        'binary_group': ['female', 'male'],
        'not_binary': ['nonbinary', 'transgender', 'trans'],
    },
    'religion': {
        'christian_group': ['catholic', 'christian', 'protestant'],
        'not_christian': ['buddhist', 'jewish', 'muslim', 'sikh', 'taoist']
    },
    'age': {
        'young_group': ['young', 'younger', 'teenage'],
        'middle': ['millenial', 'middle aged'],
        'old_group': ['elderly',  'older',  'old']
    },
    'disability': {
    }
}

ordering_scheme = {
    'race_color_nationality': ['asian_group', 'black_group', 'hispanic_group', 'white_group', 'nota'],
    'sex': ['not_straight', 'straight_group', 'binary_group', 'not_binary'],
    'religion': ['christian_group', 'not_christian'],
    'age': ['young_group', 'middle', 'old_group'],
    'disability': ['blind', 'deaf', 'paralyzed']
    }

key_cols = ['Wiki_ID', 'TOXICITY']
categories = list(ordering_scheme.keys())

data.dropna(inplace=True)
data_new = data.drop(data.query('lesbian == False and gay == False and bisexual == False and queer == False and lgbt == False and lgbtq == False and homosexual == False and straight == False and heterosexual == False and male == False and female == False and nonbinary == False and transgender == False and trans == False and african == False and `african american` == False and black == False and white == False and european == False and hispanic == False and latino == False and latina == False and latinx == False and mexican == False and canadian == False and american == False and asian == False and indian == False and `middle eastern` == False and chinese == False and japanese == False and christian == False and muslim == False and jewish == False and buddhist == False and catholic == False and protestant == False and sikh == False and taoist == False and old == False and older == False and young == False and younger == False and teenage == False and millenial == False and `middle aged` == False and elderly == False and blind == False and deaf == False and paralyzed == False').index)

for category in ordering_scheme.keys():
    groups_of_category = combined_group_members[category]
    for category_index, category_group in enumerate(ordering_scheme[category], start=1):
        if category_group in groups_of_category.keys():
            category_items = groups_of_category[category_group]
            data_new[category_group] = data_new[category_items].max(axis=1).replace(1, category_index)
        else:
            data_new[category_group] = data_new[category_group].replace(1, category_index)

for protected_class in ordering_scheme:
    data_new[protected_class] = data_new[ordering_scheme[protected_class]].max(axis=1)

data_categorized = data_new[key_cols + categories].reset_index(drop=True)

data_categorized.head(10)

correlation_key = 'TOXICITY'
correlation_columns_labels = {
    'race_color_nationality': 'Race / color / nationality',
    'sex': 'Sex',
    'religion': 'Religion',
    'age': 'Age',
    'disability': 'Disability'}

correlations_dict = {}
correlations_list = []

print('Correlations of toxicity with protected classes:\n')

for correlation_column in correlation_columns_labels.keys():
    correlation = stats.pearsonr(data_categorized[correlation_key], data_categorized[correlation_column])
    correlations_dict[correlation.statistic] = correlation_column
    correlations_list.append(correlation.statistic)
    print(correlation_columns_labels[correlation_column] + ' correlation: ', end='')
    print("%.3f" % correlation.statistic, end='')
    print(correlation_category(correlation.statistic))


correlations_list.sort(key=my_sort, reverse=True)

toxic_mean = data_categorized.TOXICITY.mean()
toxic_std = data_categorized.TOXICITY.std(ddof=0)
print("mean: %.3f" % toxic_mean)
print("std: %.3f" % toxic_std)

values = data_categorized.TOXICITY.values
mean = values.mean()
std = values.std(ddof=0)
samples = np.linspace(0, 1, 100)
lower_value = 0.0
upper_value = 1.0
for index, sample in enumerate(samples, start=1):
    std_index = std * (1 + sample)
    lower_value = np.clip(mean - std_index, min(values), None)
    upper_value = np.clip(mean + std_index, None, max(values))
    array = np.asarray(np.logical_and(values > lower_value, values < upper_value)).nonzero()
    proportion = len(array[0]) / len(values)
    # print(f"[{index}] std_index = {std_index:.4f}, proportion = {proportion:.3f}, range = ({lower_value:.3f}, {upper_value:.3f})")
    if proportion >= 0.95:
        break

print(f"Range of values around the mean that includes 95% of TOXICITY: ({lower_value:.3f}, {upper_value:.3f}).\n")

# the computed population mean/standard deviation
toxic_mean = data_categorized.TOXICITY.mean()
toxic_std = data_categorized.TOXICITY.std(ddof=0)

# the computed mean/standard deviation for the protected class category
category = 'religion'
data_step5 = data_categorized[key_cols + [category]]
data_step5 = data_step5[data_step5[category] != 0].reset_index(drop=True)
data_step5['religion'] = 1
data_step5_grouped = data_step5.groupby('religion', as_index=False).agg(
    mean=('TOXICITY', 'mean'),
    std=('TOXICITY', 'std'),
    count=('TOXICITY', 'count')
    ).drop('religion', axis=1)

# the computed mean/standard deviation for each subgroup of the protected class category
data_step6 = data_categorized[key_cols + [category]]
data_step6 = data_step6[data_step6[category] != 0].reset_index(drop=True)
data_step6_grouped = data_step6.groupby(category, as_index=False).agg(
    mean=('TOXICITY', 'mean'),
    std=('TOXICITY', 'std'),
    count=('TOXICITY', 'count')
    )

compare_dict = {}
for i in range(len(data_step6_grouped['mean'].values)):
    compare_dict.update({data_step6_grouped['mean'].values[i]: ordering_scheme[category][i]})


def which_is_higher(compare):
    high = 0
    for value in compare.keys():
        if value > high:
            high = value
    return compare[high]


def which_is_lower(compare):
    low = 2
    for value in compare.keys():
        if value < low:
            low = value
    return compare[low]


def largest_diff(compare, mean):
    largest = 0
    diff_value = 0
    for value in compare.keys():
        if abs(value) - abs(mean) > largest:
            largest = abs(value) - abs(mean)
            diff_value = value
    return compare[diff_value]


print('Which of the subgroups has the highest TOXICITY value? ' + which_is_higher(compare_dict))
print('Which of the subgroups has the lowest TOXICITY value? ' + which_is_lower(compare_dict))
print('Which of the subgroups has the largest difference in TOXICITY value when compared to the population mean? '
      + largest_diff(compare_dict, toxic_mean))
print()

for i in range(3):
    # print("%.3f" % correlations_list[i] + " : " + correlations_dict[correlations_list[i]])
    data_plot = data_categorized[[correlation_key, correlations_dict[correlations_list[i]]]]
    data_plot_aggregated = data_plot.groupby(correlation_key, as_index=False).mean()
    n_rows = 1
    n_cols = 1
    _, ax = plt.subplots(n_rows, n_cols, tight_layout=True)
    title = f"Mean {correlations_dict[correlations_list[i]]} vs. {correlation_key} (correlation = {correlations_list[i]:.3f})"

    ax = data_plot_aggregated.plot(
        x=correlation_key, y=correlations_dict[correlations_list[i]], kind="scatter",
        title=title,
        ax=ax)
    ax.set_xlabel(correlation_key)
    ax.set_ylabel(correlations_dict[correlations_list[i]])
    plt.show()
