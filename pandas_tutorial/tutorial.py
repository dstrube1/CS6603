import pandas as pd
import numpy as np

"""# from https://data36.com/pandas-tutorial-1-basics-reading-data-files-dataframes-data-selection/

data = pd.read_csv('pandas_tutorial_read.csv', delimiter=';', 
	names = ['my_datetime', 'event', 'country', 'user_id', 'source', 'topic'])
# Alternative:
# url = 'http://46.101.230.157/dilan/pandas_tutorial_read.csv'
# columns = ['my_datetime', 'event', 'country', 'user_id', 'source', 'topic']
# data = pd.read_csv(url, delimiter=';', names = columns)
print("data:")
print(data)
print()
# first 5 rows:
# print(data.head())

# first n rows:
# n = 10
# print(data.head(n))

# last 5 rows:
# print(data.tail())

# n random rows:
# print(data.sample(5))

print('just first 5 country & user_id: ')
print(data.head()[['country', 'user_id']])
print()

# print('just first 5 user_id & country: ')
# print(data.head()[['user_id', 'country']])

print('series instead of dataframe:')
print(data.head()['user_id'])
print()

print('(data.source == SEO).head(3):')
print((data.source == 'SEO').head(3))
print()

print('first 4 of only print where (data.source == SEO) = True:')
print((data[data.source == 'SEO']).head(4))
print()

print('first n of only print where (data.source == SEO) = True && topic = North America:')
print((data[data.topic == 'North America'][data.source == 'SEO']).head())
print()

# test: Select the user_id, the country and the topic columns for the users
# who are from country_2! Print the first five rows only!
print(data[data.country == 'country_2'][['country', 'user_id', 'topic']].head())
print()

# part 2:
# https://data36.com/pandas-tutorial-2-aggregation-and-grouping/

zoo = pd.read_csv('zoo.csv', delimiter = ',')

print("zoo head: ") 
print(zoo.head())
print()
print("zoo count: " + str(zoo.count()))
print()
print("zoo count better: " + str(zoo[['animal']].count()))
print()
print("zoo count even better: " + str(zoo.animal.count()))
print()
print("zoo.water_need.sum(): " + str(zoo.water_need.sum()))
print()
print("zoo.sum(): " )
print(zoo.sum())
print()
print("zoo.water_need.min(): " + str(zoo.water_need.min()))
print("zoo.water_need.max(): " + str(zoo.water_need.max()))
print()
print("zoo.water_need.mean(): " + str(zoo.water_need.mean()))
print("zoo.water_need.median(): " + str(zoo.water_need.median()))

# ???:
# print("zoo.water_need.mode(): " + str(zoo.water_need.mode()))
# :
# 0    220
# 1    410
# 2    500
# 3    600
# Name: water_need, dtype: int64

print()

print("Mean using groupby:")
print(zoo.groupby('animal').mean())
print()
print("The two ways of removing the id column: ")
print(zoo.groupby('animal').mean()[['water_need']])
print("^ return a DataFrame object")
print()

print(zoo.groupby('animal').mean().water_need)
print("^ return a Series object.")
print()

print("Combine groupby and count: ")
print(zoo.groupby('animal').count()[['water_need']])
print()

# Challenges:

print("What’s the most frequent source in the article_read dataframe?: ")
print(data.groupby('source').source.count())
print("Alternatively:")
print(data.groupby('source').count()[['user_id']])
print()

print("For the users of country_2, what was the most frequent topic and source combination?")
print("Or in other words: which topic, from which source, brought the most views from country_2?")
print(data[data.country == 'country_2'].groupby(['source', 'topic']).count())  
"""
