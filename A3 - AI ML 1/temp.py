import pandas as pd

# data = pd.read_csv('toxity_per_attribute.csv') 
data = pd.read_csv('toxity_per_attribute.csv', low_memory=False) 
# https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#basics-dtypes
#dtypes = {'Wiki_ID': 'float32', 'TOXICITY': 'float64', 'lesbian':'boolean', 'gay':'boolean','bisexual':'boolean','transgender':'boolean','trans':'boolean','queer':'boolean','lgbt':'boolean','lgbtq':'boolean','homosexual':'boolean','straight':'boolean','heterosexual':'boolean','male':'boolean','female':'boolean','nonbinary':'boolean','african':'boolean','african american':'boolean','black':'boolean','white':'boolean','european':'boolean','hispanic':'boolean','latino':'boolean','latina':'boolean','latinx':'boolean','mexican':'boolean','canadian':'boolean','american':'boolean','asian':'boolean','indian':'boolean','middle eastern':'boolean','chinese':'boolean','japanese':'boolean','christian':'boolean','muslim':'boolean','jewish':'boolean','buddhist':'boolean','catholic':'boolean','protestant':'boolean','sikh':'boolean','taoist':'boolean','old':'boolean','older':'boolean','young':'boolean','younger':'boolean','teenage':'boolean','millenial':'boolean','middle aged':'boolean','elderly':'boolean','blind':'boolean','deaf':'boolean','paralyzed':'boolean'}
#data = pd.read_csv('toxity_per_attribute.csv', dtype=dtypes)  

# print("data:")
# print(data)
# print()
# first 5 rows:
# print("head: ")
# print(data.head())

print("data count: " + str(data.Wiki_ID.count()))  # 76,563

data.dropna(inplace=True)
print("data count, post dropna: " + str(data.Wiki_ID.count()))  # still 76,563?


# are there any where Wiki_ID or TOXICITY are false?:
# print("data count where Wiki_ID = 'False': " + str((data[data.Wiki_ID == 'False']).Wiki_ID.count()))
# 0, as expected
# print("data count where TOXICITY = 'False': " + str((data[data.TOXICITY == 'False']).TOXICITY.count()))
# 0, as expected
# print("data count where lesbian = 'False': " + str(data[data.lesbian == 'False'].lesbian.count()))
# 0, NOT as expected
# empty dataframe, wtf
# print("data[data.lesbian == 'False'].head():")
# print(data[data.lesbian == 'False'].head())
# Ah, it's because of 'False', not False
# print("data count where Wiki_ID = False: " + str((data[data.Wiki_ID == False]).Wiki_ID.count()))
# not 0, not as expected, must investigate
# print("data count where TOXICITY = False: " + str((data[data.TOXICITY == False]).TOXICITY.count()))
# 0, as expected
# print("data count where lesbian = False: " + str(data[data.lesbian == False].lesbian.count()))
# not 0, as expected: 75,050
# better way - chainable:
print("data count where lesbian = False: " + str(data.query("lesbian == False").lesbian.count()))
# print("data where Wiki_ID = False: ")
# print(data[data.Wiki_ID == False])
# interesting: maybe 0.0 = False?; either way, we can ignore this
# print("count of rows where all non-Wiki_ID & non-TOXICITY rows are False:")

# sexuality
print("data count where sexuality = False: " + str(data.query("lesbian == False and gay == False and bisexual == False and queer == False and lgbt == False and lgbtq == False and homosexual == False and straight == False and heterosexual == False").Wiki_ID.count()))
# 62937

# gender
print("data count where gender = False: " + str(data.query("male == False and female == False and nonbinary == False and transgender == False and trans == False").Wiki_ID.count()))
# 68993

# race / ethnicity / nationality
print("data count where race / ethnicity / nationality = False: " + str(data.query("african == False and `african american` == False and black == False and white == False and european == False and hispanic == False and latino == False and latina == False and latinx == False and mexican == False and canadian == False and american == False and asian == False and indian == False and `middle eastern` == False and chinese == False and japanese == False").Wiki_ID.count()))
# 50825

# religion
print("data count where religion = False: " + str(data.query("christian == False and muslim == False and jewish == False and buddhist == False and catholic == False and protestant == False and sikh == False and taoist == False").Wiki_ID.count()))
# 64452

# age
print("data count where age = False: " + str(data.query("old == False and older == False and young == False and younger == False and teenage == False and millenial == False and `middle aged` == False and elderly == False").Wiki_ID.count()))
# 64451

# disability
print("data count where disability = False: " + str(data.query("blind == False and deaf == False and paralyzed == False").Wiki_ID.count()))
# 72021

# all
print("data count where all = False: " + str(data.query("lesbian == False and gay == False and bisexual == False and queer == False and lgbt == False and lgbtq == False and homosexual == False and straight == False and heterosexual == False and male == False and female == False and nonbinary == False and transgender == False and trans == False and african == False and `african american` == False and black == False and white == False and european == False and hispanic == False and latino == False and latina == False and latinx == False and mexican == False and canadian == False and american == False and asian == False and indian == False and `middle eastern` == False and chinese == False and japanese == False and christian == False and muslim == False and jewish == False and buddhist == False and catholic == False and protestant == False and sikh == False and taoist == False and old == False and older == False and young == False and younger == False and teenage == False and millenial == False and `middle aged` == False and elderly == False and blind == False and deaf == False and paralyzed == False").Wiki_ID.count()))
# 864

# !all
# print("data count where all != False: " + str(data.query("lesbian != False and gay != False and bisexual != False and queer != False and lgbt != False and lgbtq != False and homosexual != False and straight != False and heterosexual != False and male != False and female != False and nonbinary != False and transgender != False and trans != False and african != False and `african american` != False and black != False and white != False and european != False and hispanic != False and latino != False and latina != False and latinx != False and mexican != False and canadian != False and american != False and asian != False and indian != False and `middle eastern` != False and chinese != False and japanese != False and christian != False and muslim != False and jewish != False and buddhist != False and catholic != False and protestant != False and sikh != False and taoist != False and old != False and older != False and young != False and younger != False and teenage != False and millenial != False and `middle aged` != False and elderly != False and blind != False and deaf != False and paralyzed != False").Wiki_ID.count()))
# 76563 - 864 = 75699 ? nope, 0 -_-
print("data count where lesbian != False: " + str(data.query("lesbian != False").Wiki_ID.count()))
# 76563 - 75050 = 1513 (+ 1 for False Wiki_ID ?)
# digging further:
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
data_new = data.drop(data[data.lesbian == False].index)
print("data count where lesbian != False: " + str(data_new.Wiki_ID.count()))
# 1514, as expected

data_new = data.drop(data.query("lesbian == False and gay == False and bisexual == False and queer == False and lgbt == False and lgbtq == False and homosexual == False and straight == False and heterosexual == False and male == False and female == False and nonbinary == False and transgender == False and trans == False and african == False and `african american` == False and black == False and white == False and european == False and hispanic == False and latino == False and latina == False and latinx == False and mexican == False and canadian == False and american == False and asian == False and indian == False and `middle eastern` == False and chinese == False and japanese == False and christian == False and muslim == False and jewish == False and buddhist == False and catholic == False and protestant == False and sikh == False and taoist == False and old == False and older == False and young == False and younger == False and teenage == False and millenial == False and `middle aged` == False and elderly == False and blind == False and deaf == False and paralyzed == False").index)
print("data count where all != False: " + str(data_new.Wiki_ID.count()))
# Yes!