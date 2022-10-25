#!/usr/bin/env python
# coding: utf-8

# In[8]:


#20 inbuilt methods and functions for Pandas 


# In[2]:


#1
#melt function is used to change a DataFrame from wide to long format, optionally leaving identifiers set.
#syntax = pandas.melt(frame[,id_vars, value_vars,vars_name,...])
#example

#creating a simple dataframe

#importing pandas as pd

import pandas as pd

#creating a dataframe

df = pd.DataFrame ({'Name':{1: 'Sommy', 2: 'Naza', 3: "David"},
                    'Course':{1: 'Data Science', 2: 'Backend Development', 3: 'Product Management'},
                    'Age':{1: 28, 2: 26, 3: 23}})
df


# In[17]:


#Name is id_vars and course us value_vars
pd.melt(df, id_vars =['Name'], value_vars =['Course'])


# In[30]:


#2
#Pivot (data[,index,columns,values]). This returns reshaped DataFrame organized by given index/column values

df.pivot(index='Age', columns= 'Name', values = 'Course')


# In[5]:


#3 
#pd.read_csv, pd.read_excel. Used to read an excel csv or excel file respetively, to a pandas DatFrame format.

import pandas as pd

excel_data_df = pd.read_excel (r'C:\Users\hp\Documents\Data Analysis_&_Science\Structured Sales.xlsx', sheet_name = 'Cleaned')
print(excel_data_df)


# In[33]:


#4
#df.head used if you want a specific number of rows
excel_data_df.head(20)


# In[34]:


#5
#df.columns used to print out all the columns of a dataset

excel_data_df.columns


# In[47]:


#6
#df.drop() used to drop some unneccessary columns

excel_data_df = df.drop(columns=['Segment','Order_Date'])
print(excel_data_df)


# In[37]:


#7
#len() provides the length of the DataFrame
len(excel_data_df)


# In[38]:


#8
#df.iloc() takes as a parameter the rows and column indices and gives you the subset of the DataFrame accordingly.

excel_data_df.iloc[:25,1:3]


# In[41]:


#9
#loc
df.loc() is almost similar to the df.iloc() function. However, we can specify exactly which row index we want and also the name of the columns we want in our dataset. 

excel_data_df.loc[[3, 12, 20, 24], ['Segment','Order_Date']]


# In[42]:


#10
#df["].dtypes is used to get the data type of each column.
excel_data_df.dtypes


# In[83]:


#11
#df.select_dtypes used to select the variables or columns of a certain data types
excel_data_df.select_dtypes(include='object')


# In[43]:


#12
#df.sample() is used to take a representative sample from a large dataset yo perfrom the analysis and predictive modelling

excel_data_df.sample(n = 100)


# In[94]:


#13
#df["].unique() is used to find out the unique values of a categorical column. 

excel_data_df.Ship_Mode.unique()


# In[99]:


#14
#df["].nunique() lets one know how many unique values is in a column

excel_data_df.nunique()


# In[44]:


#15
#df["].rank() provides ypu with the rank based on a certain column

excel_data_df.rank()


# In[45]:


#16
#df.groupby is used to group data based on a certain varaible and find out useful information about the group. 
excel_data_df.groupby("Segment")["Sales"].sum()


# In[4]:


#17
#.pct_change is used to get the percenatge change from the prveious value of a variable. The fist row will always be NaN beacuse it has no preceding value
#it further gives the name, lenght and data type of the row
excel_data_df.Sales.pct_change()


# In[19]:


#18
#df.count(0) is used to know the number of data in the DataFrame in the specified direction. 
excel_data_df.count(0)


# In[21]:


#19
#nlargest and nsmallestis gives us the dataset with n number of either the largest values or the smallest values of a specified variable

excel_data_df.nlargest(10, "Sales")


# In[23]:


excel_data_df.nsmallest(10, "Sales")


# In[6]:


#20 
#df.replace() replaces the values of a column. 
excel_data_df.replace("First Class", "VIP")


# In[4]:


#20 inbuilt methods on Numpy

import numpy as np


# In[5]:


#1 
#np.array() - Used to create a Numpy array from a Python list

x = [10, 28, 23, 5, 3, 8, 0, 45, 15]
np.array(x)


# In[16]:


x = [[10, 28, 23, 5, 3, 8, 0, 45, 15], [56, 27, 1, 3, 2, 8, 48, 57, 67]]
np.array(x)


# In[23]:


#2
#np.zeros() is used to create an array of zeros

np.zeros((10))


# In[48]:


#for multidimensional array of zeros

np.zeros((10, 33))


# In[56]:


np.zeros((10, 33, 2))


# In[43]:


#3 
#np.ones() is used to create an array of ones

np.ones(11)


# In[57]:


np.ones((11, 10, 2))


# In[64]:


#4
#np.eye() returns a 2D array with ones on the diagonal and zeros elsewhere

np.eye((5))


# In[66]:


#5 
#np.arange is used to generate equally spaced values within a given interval

np.arange(5)


# In[70]:


np.arange(5, 25)

#generte values from start=5, stop=25


# In[72]:


np.arange(25, 50, 5)

#generate values from start = 25, stop=50 with step=5


# In[80]:


#6
#np.linspace() is used to generate numbers in an interval that are evenly spaced

np.linspace(10, 40, 4)


# In[94]:


#7
#np.random.randint is used to generate a random list of integers
np.random.randint(1, 10, 5)


# In[112]:


#8
#np.random.random is used to generate a random list of floats
np.random.random(10)


# In[124]:


#9
#np.concatenate: concatente mulptiple Numpy arrays to form one Numpy array

x = np.array([[2, 3, 5], [4, 7, 1]])
y = np.array([[1, 6, 11], [5, 2, 9]])
np.concatenate((x, y), axis=1)


# In[131]:


#10
#ndarray.flatten() is used to collapse an entire NumPy array into a single dimension.

naza = np.array([[14, 25, 67, 98], [56, 58, 24, 66]])
naza.flatten()


# In[142]:


#11
#np.unique is used to determine the unique elements in anumpy array

x=np.array([[1, 2, 3], [2, 4, 5], [6, 3, 2]])
np.unique(x)


# In[147]:


x=np.array([[1, 2, 3], [3, 4, 5], [1, 2, 3]])
np.unique(x, axis = 0)


# In[148]:


x=np.array([[1, 2, 3], [2, 4, 5], [6, 3, 2]])
np.unique(x, axis = 1)


# In[153]:


#12
#np.squeeze() is used to remove axis of length one from an array

naza = np.array([[[13], [34], [13], [45]]])
naza.shape


# In[155]:


naza = np.array([[[13], [34], [13], [45]]])
naza.shape
np.squeeze(naza)


# In[157]:


naza = np.array([[[13], [34], [13], [45]]])
naza.shape
np.squeeze(naza).shape              


# In[161]:


#13
#ndarray.tolist() can be used toobatin a python list from a numpy array
m = np.array([[[4, 7, 9], [5, 6, 8], [1, 2, 0], [3, 10, 11]]])
m.tolist()


# In[163]:


#14
#np.sum()

x = np.array([[1, 2, 4], [3, 4, 3]])
np.sum(x)


# In[165]:


x = np.array([[1, 2, 4], [3, 4, 3]])
np.sum(x, axis = 0)


# In[167]:


x = np.array([[1, 2, 4], [3, 4, 3]])
np.sum(x, axis = 1)


# In[169]:


x = np.array([[1, 2, 4], [3, 4, 3]])
np.product(x, axis = 0)


# In[175]:


#15
#np.sqrt is used to find the square root of an array of elements

x = np.array([[4, 16], [25, 9]])
np.sqrt(x)


# In[177]:


#16
#Statistical method
#np.mean is used to find the mean of the values in a numpy array along an axis

a = np.array([[4, 16], [25, 9]])
np.mean(a)


# In[179]:


a = np.array([[4, 16], [25, 9]])
np.median(a)


# In[181]:


a = np.array([[4, 16], [25, 9]])
np.std(a)


# In[196]:


#17
#np.argmax() is used to return the indices of the maximum values along an axis
np.random.randint(2, 8, 2)


# In[232]:


x = np.random.randint(2, 8, 4).reshape(2, 2)
print(x)

np.argmax(x)


# In[236]:


#18
#np.argmin() is used to return indices of the minimum values along an axis

x = np.random.randint(1, 20, 10).reshape(2, 5)
print(x)

np.argmin(x)


# In[16]:


#19
#np.where is used to select betwwen two arrays basd on a condition. 
i = np.random.randint(5, 15, 5)
print(i)

np.where(i<10, 0, i)

#I still need an explanation on this method


# In[250]:


#20
#np.nonzero() is used to determine the indices of non-zero elements in numpy array. 

y = np.array([[[4, 7, 9], [5, 6, 8], [1, 2, 0], [3, 10, 11]]])
np.nonzero(y)

