import pandas as pd
import matplotlib.pyplot as plt 
        
df = pd.read_csv("bank-additional.csv",sep=";")

original = df.copy()
original.shape

df.shape
    
df.isnull().sum()

cols = df.columns

for i in cols:
    try:
        plt.figure(figsize=(20,15))
        plot = df[i].value_counts().plot(kind="barh",figsize=(15,10),rot=0)
        fig = plot.get_figure()
        fig.savefig(str(i)+".png")
    except:
        pass
    

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

labelEncoder = LabelEncoder()

Y = df['y']
Y = labelEncoder.fit_transform(Y)

numerical=["age","duration","campaign","pdays","previous","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed"]
categorical=["job","marital","education","default","housing","loan","month","day_of_week","poutcome"]


import numpy as np
numerical = df.select_dtypes(include=np.number)
categorical = df.select_dtypes(exclude=np.number)

labelencoder_X = LabelEncoder()
df['job']      = labelencoder_X.fit_transform(df['job']) 
df['marital']  = labelencoder_X.fit_transform(df['marital']) 
df['education']= labelencoder_X.fit_transform(df['education']) 
df['default']  = labelencoder_X.fit_transform(df['default']) 
df['housing']  = labelencoder_X.fit_transform(df['housing']) 
df['loan']     = labelencoder_X.fit_transform(df['loan'])
df['contact']     = labelencoder_X.fit_transform(df['contact']) 
df['month']       = labelencoder_X.fit_transform(df['month']) 
df['day_of_week'] = labelencoder_X.fit_transform(df['day_of_week'])

df['poutcome'].replace(['nonexistent', 'failure', 'success'], [1,2,3], inplace  = True)

X = df.drop(['y'],1)

plt.figure(figsize=(20,15))
#plot = plt.hist(df['age'], color = 'blue', edgecolor = 'black', bins = int(98/5))
plot = df['age'].plot(kind="hist",figsize=(15,10),rot=0)
fig = plot.get_figure()
fig.savefig("histogram_of_Age.png")


sns_plot = sns.distplot(df['age'])
plt.savefig("Age_density.png")

sns_plot = sns.distplot(df['cons.conf.idx'])
plt.savefig("Consumer_confidence_index_density.png")


df['y'] = labelEncoder.fit_transform(df['y'])

'''sns.relplot(x="cons.conf.idx", y="y",ci=None, kind="line", data=df);
#sns.relplot(x="timepoint", y="signal", ci=None, kind="line", data=fmri);

sns.distplot(df['cons.conf.idx'], rug= True, hist= False, rug_kws = {'shade': True, 'linewidth': 3})

sns.distplot(df['cons.conf.idx'], rug= True, hist= False, rug_kws = {'shade': True, 'linewidth': 3})

plt.savefig("Consumer_confidence_index_density.png")'''




'''
indexes = df['y'].unique()
for index in indexes:
    subset = df[df['y'] == index]
    sns.distplot(subset['emp.var.rate'], hist = False, kde = True,kde_kws = {'shade':False, 'linewidth': 3}, label=index)
  
plt.title('Density Plot for Employment Variation Rate')
plt.xlabel('Employment Variation Rate')
plt.ylabel('Density')
plt.savefig("Employment Variation Rate_density_vs_y.png")

for index in indexes:
    subset = df[df['y'] == index]
    sns.distplot(subset['cons.conf.idx'], hist = False, kde = True,kde_kws = {'shade':False, 'linewidth': 3}, label=index)
  
plt.title('Density Plot for Consumer Confidence Index')
plt.xlabel('Consumer Confidence Index')
plt.ylabel('Density')
plt.savefig("CCI_density_vs_y.png")


for index in indexes:
    subset = df[df['y'] == index]
    sns.distplot(subset['contact'], hist = False, kde = True,kde_kws = {'shade':False, 'linewidth': 3}, label=index)
  
plt.title('Density Plot for Contact')
plt.xlabel('Cotacted via')
plt.ylabel('Density')
plt.savefig("contact_vs_y.png")


for index in indexes:
    subset = df[df['y'] == index]
    sns.distplot(subset['day_of_week'], hist = False, kde = True,kde_kws = {'shade':False, 'linewidth': 3}, label=index)
  
plt.title('Density Plot for day_of_week')
plt.xlabel('day_of_week')
plt.ylabel('Density')
plt.savefig("day_of_week_vs_y.png")


for index in indexes:
    subset = df[df['y'] == index]
    sns.distplot(subset['cons.price.idx'], hist = False, kde = True,kde_kws = {'shade':False, 'linewidth': 3}, label=index)
  
plt.title('Density Plot for Consumer Price Index')
plt.xlabel('Consumer Price Index')
plt.ylabel('Density')
plt.savefig("CPI_density_vs_y.png")


for index in indexes:
    subset = df[df['y'] == index]
    sns.distplot(subset['euribor3m'], hist = False, kde = True,kde_kws = {'shade':False, 'linewidth': 3}, label=index)
  
plt.title('Density Plot for Euribor')
plt.xlabel('Euribor')
plt.ylabel('Density')
plt.savefig("Euribor_density_vs_y.png")



for index in indexes:
    subset = df[df['y'] == index]
    sns.distplot(subset['nr.employed'], hist = False, kde = True,kde_kws = {'shade':False, 'linewidth': 3}, label=index)
  
plt.title('Density Plot for Number of Employees')
plt.xlabel('Number of Employees')
plt.ylabel('Density')
plt.savefig("number Of Employees_density_vs_y.png")




for index in indexes:
    subset = df[df['y'] == index]
    sns.distplot(subset['cons.conf.idx'], hist = False, kde = True,kde_kws = {'shade':False, 'linewidth': 3}, label=index)
  
plt.title('Density Plot for Consumer Confidence Index')
plt.xlabel('Consumer Confidence Index')
plt.ylabel('Density')
plt.savefig("CCI_density_vs_y.png")


for index in indexes:
    subset = df[df['y'] == index]
    sns.distplot(subset['cons.conf.idx'], hist = False, kde = True,kde_kws = {'shade':False, 'linewidth': 3}, label=index)
  
plt.title('Density Plot for Consumer Confidence Index')
plt.xlabel('Consumer Confidence Index')
plt.ylabel('Density')
plt.savefig("CCI_density_vs_y.png")


for index in indexes:
    subset = df[df['y'] == index]
    sns.distplot(subset['cons.conf.idx'], hist = False, kde = True,kde_kws = {'shade':False, 'linewidth': 3}, label=index)
  
plt.title('Density Plot for Consumer Confidence Index')
plt.xlabel('Consumer Confidence Index')
plt.ylabel('Density')
plt.savefig("CCI_density_vs_y.png")



for index in indexes:
    subset = df[df['y'] == index]
    sns.distplot(subset['cons.conf.idx'], hist = False, kde = True,kde_kws = {'shade':False, 'linewidth': 3}, label=index)
  
plt.title('Density Plot for Consumer Confidence Index')
plt.xlabel('Consumer Confidence Index')
plt.ylabel('Density')
plt.savefig("CCI_density_vs_y.png")

''''




'''for i in cols:
    try:
        sns_plot = sns.factorplot(x=i, y="age", hue = 'y', data=df, kind='bar', ci=None)
        sns_plot.fig.suptitle("y_vs_" + str(i))
        sns_plot.savefig("age_vs_" + str(i)+".png")
    except:
        pass

for i in cols:
    try:
        sns_plot = sns.factorplot(x=i, y="campaign", hue = 'y', data=df, kind='bar', ci=None)
        sns_plot.fig.suptitle("campaign_vs_" + str(i))
        sns_plot.savefig("campaign_vs_" + str(i)+".png")
    except:
        pass 

for i in cols:
    try:
        sns_plot = sns.factorplot(x=i, y="duration", hue = 'y', data=df, kind='bar', ci=None)
        sns_plot.fig.suptitle("duration_vs_" + str(i))
        sns_plot.savefig("duration_vs_" + str(i)+".png")
    except:
        pass 

for i in cols:
    try:
        sns_plot = sns.factorplot(x=i, y="pdays", hue = 'y', data=df, kind='bar', ci=None)
        sns_plot.fig.suptitle("pdays_vs_" + str(i))
        sns_plot.savefig("pdays_vs_" + str(i)+".png")
    except:
        pass
    
for i in cols:
    try:
        sns_plot = sns.factorplot(x=i, y="previous", hue = 'y', data=df, kind='bar', ci=None)
        sns_plot.fig.suptitle("previous_vs_" + str(i))
        sns_plot.savefig("previous_vs_" + str(i)+".png")
    except:
        pass
    
for i in cols:
    try:
        sns_plot = sns.factorplot(x=i, y="rate", hue = 'y', data=df, kind='bar', ci=None)
        sns_plot.fig.suptitle("rate_vs_" + str(i))
        sns_plot.savefig("rate_vs_" + str(i)+".png")
    except:
        pass
    
for i in cols:
    try:
        sns_plot = sns.factorplot(x=i, y="CPI", hue = 'y', data=df, kind='bar', ci=None)
        sns_plot.fig.suptitle("CPI_vs_" + str(i))
        sns_plot.savefig("CPI_vs_" + str(i)+".png")
    except:
        pass


for i in cols:
    try:
        sns_plot = sns.factorplot(x=i, y="CCI", hue = 'y', data=df, kind='bar', ci=None)
        sns_plot.fig.suptitle("CCI_vs_" + str(i))
        sns_plot.savefig("CCI_vs_" + str(i)+".png")
    except:
        pass
    
for i in cols:
    try:
        sns_plot = sns.factorplot(x=i, y="euribor", hue = 'y', data=df, kind='bar', ci=None)
        sns_plot.fig.suptitle("euribor_vs_" + str(i))
        sns_plot.savefig("euribor_vs_" + str(i)+".png")
    except:
        pass

for i in cols:
    try:
        sns_plot = sns.factorplot(x=i, y="employed", hue = 'y', data=df, kind='bar', ci=None)
        sns_plot.fig.suptitle("employed_vs_" + str(i))
        sns_plot.savefig("employed_vs_" + str(i)+".png")
    except:
        pass    
    
    
    
for i in cols:
    try:
        sns_plot = sns.factorplot(x=i, y="job", hue = 'y', data=df, kind='bar', ci=None)
        sns_plot.fig.suptitle("job_vs_" + str(i))
        sns_plot.savefig("job_vs_" + str(i)+".png")
    except:
        pass

for i in cols:
    try:
        sns_plot = sns.factorplot(x=i, y="marital", hue = 'y', data=df, kind='bar', ci=None)
        sns_plot.fig.suptitle("marital_status_vs_" + str(i))
        sns_plot.savefig("marital_status_vs_" + str(i)+".png")
    except:
        pass 

for i in cols:
    try:
        sns_plot = sns.factorplot(x=i, y="default", hue = 'y', data=df, kind='bar', ci=None)
        sns_plot.fig.suptitle("default_vs_" + str(i))
        sns_plot.savefig("default_vs_" + str(i)+".png")
    except:
        pass 

for i in cols:
    try:
        sns_plot = sns.factorplot(x=i, y="housing", hue = 'y', data=df, kind='bar', ci=None)
        sns_plot.fig.suptitle("housing_vs_" + str(i))
        sns_plot.savefig("housing_vs_" + str(i)+".png")
    except:
        pass
    
for i in cols:
    try:
        sns_plot = sns.factorplot(x=i, y="loan", hue = 'y', data=df, kind='bar', ci=None)
        sns_plot.fig.suptitle("loan_vs_" + str(i))
        sns_plot.savefig("loan_vs_" + str(i)+".png")
    except:
        pass
    
for i in cols:
    try:
        sns_plot = sns.factorplot(x=i, y="month", hue = 'y', data=df, kind='bar', ci=None)
        sns_plot.fig.suptitle("month_vs_" + str(i))
        sns_plot.savefig("month_vs_" + str(i)+".png")
    except:
        pass
    
for i in cols:
    try:
        sns_plot = sns.factorplot(x=i, y="day_of_week", hue = 'y', data=df, kind='bar', ci=None)
        sns_plot.fig.suptitle("day_of_week_vs_" + str(i))
        sns_plot.savefig("day_of_week_vs_" + str(i)+".png")
    except:
        pass


for i in cols:
    try:
        sns_plot = sns.factorplot(x=i, y="poutcome", hue = 'y', data=df, kind='bar', ci=None)
        sns_plot.fig.suptitle("poutcome_vs_" + str(i))
        sns_plot.savefig("poutcome_vs_" + str(i)+".png")
    except:
        pass
'''


  












