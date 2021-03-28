#!/usr/bin/env python
# coding: utf-8

# ## Importing the Packages

# In[1]:


import numpy as np


# In[2]:


np.set_printoptions(suppress = True, linewidth = 100, precision = 2)


# ## Importing the Data

# In[3]:


raw_data_np = np.genfromtxt("loan-data.csv", delimiter = ';', skip_header = 1, autostrip = True)
raw_data_np


# ## Checking for Incomplete Data

# In[4]:


np.isnan(raw_data_np).sum()


# In[5]:


temporary_fill = np.nanmax(raw_data_np) + 1
temporary_mean = np.nanmean(raw_data_np, axis = 0)


# In[6]:


temporary_mean


# In[7]:


temporary_stats = np.array([np.nanmin(raw_data_np, axis = 0),
                           temporary_mean,
                           np.nanmax(raw_data_np, axis = 0)])


# In[8]:


temporary_stats


# ## Splitting the Dataset

# ### Splitting the Columns

# In[9]:


columns_strings = np.argwhere(np.isnan(temporary_mean)).squeeze()
columns_strings


# In[10]:


columns_numeric = np.argwhere(np.isnan(temporary_mean) == False).squeeze()
columns_numeric


# ### Re-importing the Dataset

# In[11]:


loan_data_strings = np.genfromtxt("loan-data.csv",
                                  delimiter = ';',
                                  skip_header = 1,
                                  autostrip = True, 
                                  usecols = columns_strings,
                                  dtype = np.str)
loan_data_strings


# In[12]:


loan_data_numeric = np.genfromtxt("loan-data.csv",
                                  delimiter = ';',
                                  autostrip = True,
                                  skip_header = 1,
                                  usecols = columns_numeric,
                                  filling_values = temporary_fill)
loan_data_numeric


# ### The Names of the Columns

# In[13]:


header_full = np.genfromtxt("loan-data.csv",
                            delimiter = ';',
                            autostrip = True,
                            skip_footer = raw_data_np.shape[0],
                            dtype = np.str)
header_full


# In[14]:


header_strings, header_numeric = header_full[columns_strings], header_full[columns_numeric]


# In[15]:


header_strings


# In[16]:


header_numeric


# ## Creating Checkpoints:

# In[17]:


def checkpoint(file_name, checkpoint_header, checkpoint_data):
    np.savez(file_name, header = checkpoint_header, data = checkpoint_data)
    checkpoint_variable = np.load(file_name + ".npz")
    return(checkpoint_variable)


# In[18]:


checkpoint_test = checkpoint("checkpoint-test", header_strings, loan_data_strings)


# In[19]:


checkpoint_test['data']


# In[20]:


np.array_equal(checkpoint_test['data'], loan_data_strings)


# ## Manipulating String Columns

# In[21]:


header_strings


# In[22]:


header_strings[0] = "issue_date"


# In[23]:


loan_data_strings


# ### Issue Date

# In[24]:


np.unique(loan_data_strings[:,0])


# In[25]:


loan_data_strings[:,0] = np.chararray.strip(loan_data_strings[:,0], "-15")


# In[26]:


np.unique(loan_data_strings[:,0])


# In[27]:


months = np.array(['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])


# In[28]:


for i in range(13):
        loan_data_strings[:,0] = np.where(loan_data_strings[:,0] == months[i],
                                          i,
                                          loan_data_strings[:,0])


# In[29]:


np.unique(loan_data_strings[:,0])


# ### Loan Status

# In[30]:


header_strings


# In[31]:


np.unique(loan_data_strings[:,1])


# In[32]:


np.unique(loan_data_strings[:,1]).size


# In[33]:


status_bad = np.array(['','Charged Off','Default','Late (31-120 days)'])


# In[34]:


loan_data_strings[:,1] = np.where(np.isin(loan_data_strings[:,1], status_bad),0,1)


# In[35]:


np.unique(loan_data_strings[:,1])


# ### Term

# In[36]:


header_strings


# In[37]:


np.unique(loan_data_strings[:,2])


# In[38]:


loan_data_strings[:,2] = np.chararray.strip(loan_data_strings[:,2], " months")
loan_data_strings[:,2]


# In[39]:


header_strings[2] = "term_months"


# In[40]:


loan_data_strings[:,2] = np.where(loan_data_strings[:,2] == '', 
                                  '60', 
                                  loan_data_strings[:,2])
loan_data_strings[:,2]


# In[41]:


np.unique(loan_data_strings[:,2])


# ### Grade and Subgrade

# In[42]:


header_strings


# In[43]:


np.unique(loan_data_strings[:,3])


# In[44]:


np.unique(loan_data_strings[:,4])


# #### Filling Sub Grade

# In[45]:


for i in np.unique(loan_data_strings[:,3])[1:]:
    loan_data_strings[:,4] = np.where((loan_data_strings[:,4] == '') & (loan_data_strings[:,3] == i),
                                      i + '5',
                                      loan_data_strings[:,4])


# In[46]:


np.unique(loan_data_strings[:,4], return_counts = True)


# In[47]:


loan_data_strings[:,4] = np.where(loan_data_strings[:,4] == '',
                                  'H1',
                                  loan_data_strings[:,4])


# In[48]:


np.unique(loan_data_strings[:,4])


# #### Removing Grade

# In[49]:


loan_data_strings = np.delete(loan_data_strings, 3, axis = 1)


# In[50]:


loan_data_strings[:,3]


# In[51]:


header_strings = np.delete(header_strings, 3)


# In[52]:


header_strings[3]


# #### Converting Sub Grade

# In[53]:


np.unique(loan_data_strings[:,3])


# In[54]:


keys = list(np.unique(loan_data_strings[:,3]))                         
values = list(range(1, np.unique(loan_data_strings[:,3]).shape[0] + 1)) 
dict_sub_grade = dict(zip(keys, values))


# In[55]:


dict_sub_grade


# In[55]:


for i in np.unique(loan_data_strings[:,3]):
        loan_data_strings[:,3] = np.where(loan_data_strings[:,3] == i, 
                                          dict_sub_grade[i],
                                          loan_data_strings[:,3])


# In[56]:


np.unique(loan_data_strings[:,3])


# ### Verification Status

# In[57]:


header_strings


# In[58]:


np.unique(loan_data_strings[:,4])


# In[59]:


loan_data_strings[:,4] = np.where((loan_data_strings[:,4] == '') | (loan_data_strings[:,4] == 'Not Verified'), 0, 1)


# In[60]:


np.unique(loan_data_strings[:,4])


# ### URL

# In[61]:


loan_data_strings[:,5]


# In[62]:


np.chararray.strip(loan_data_strings[:,5], "https://www.lendingclub.com/browse/loanDetail.action?loan_id=")


# In[63]:


loan_data_strings[:,5] = np.chararray.strip(loan_data_strings[:,5], "https://www.lendingclub.com/browse/loanDetail.action?loan_id=")


# In[64]:


header_full


# In[65]:


loan_data_numeric[:,0].astype(dtype = np.int32)


# In[66]:


loan_data_strings[:,5].astype(dtype = np.int32)


# In[67]:


np.array_equal(loan_data_numeric[:,0].astype(dtype = np.int32), loan_data_strings[:,5].astype(dtype = np.int32))


# In[68]:


loan_data_strings = np.delete(loan_data_strings, 5, axis = 1)
header_strings = np.delete(header_strings, 5)


# In[69]:


loan_data_strings[:,5]


# In[70]:


header_strings


# In[71]:


loan_data_numeric[:,0]


# In[72]:


header_numeric


# ### State Address

# In[73]:


header_strings


# In[74]:


header_strings[5] = "state_address"


# In[75]:


states_names, states_count = np.unique(loan_data_strings[:,5], return_counts = True)
states_count_sorted = np.argsort(-states_count)
states_names[states_count_sorted], states_count[states_count_sorted]


# In[76]:


loan_data_strings[:,5] = np.where(loan_data_strings[:,5] == '', 
                                  0, 
                                  loan_data_strings[:,5])


# In[77]:


states_west = np.array(['WA', 'OR','CA','NV','ID','MT', 'WY','UT','CO', 'AZ','NM','HI','AK'])
states_south = np.array(['TX','OK','AR','LA','MS','AL','TN','KY','FL','GA','SC','NC','VA','WV','MD','DE','DC'])
states_midwest = np.array(['ND','SD','NE','KS','MN','IA','MO','WI','IL','IN','MI','OH'])
states_east = np.array(['PA','NY','NJ','CT','MA','VT','NH','ME','RI'])


# https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf

# In[78]:


loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_west), 1, loan_data_strings[:,5])
loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_south), 2, loan_data_strings[:,5])
loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_midwest), 3, loan_data_strings[:,5])
loan_data_strings[:,5] = np.where(np.isin(loan_data_strings[:,5], states_east), 4, loan_data_strings[:,5])


# In[79]:


np.unique(loan_data_strings[:,5])


# ## Converting to Numbers

# In[80]:


loan_data_strings


# In[81]:


loan_data_strings = loan_data_strings.astype(np.int)


# In[82]:


loan_data_strings


# ### Checkpoint 1: Strings

# In[83]:


checkpoint_strings = checkpoint("Checkpoint-Strings", header_strings, loan_data_strings)


# In[84]:


checkpoint_strings["header"]


# In[85]:


checkpoint_strings["data"]


# In[86]:


np.array_equal(checkpoint_strings['data'], loan_data_strings)


# ## Manipulating Numeric Columns

# In[87]:


loan_data_numeric


# In[88]:


np.isnan(loan_data_numeric).sum()


# ### Substitute "Filler" Values

# In[89]:


header_numeric


# #### ID

# In[90]:


temporary_fill


# In[91]:


np.isin(loan_data_numeric[:,0], temporary_fill)


# In[92]:


np.isin(loan_data_numeric[:,0], temporary_fill).sum()


# In[93]:


header_numeric


# #### Temporary Stats

# In[94]:


temporary_stats[:, columns_numeric]


# #### Funded Amount

# In[95]:


loan_data_numeric[:,2]


# In[96]:


loan_data_numeric[:,2] = np.where(loan_data_numeric[:,2] == temporary_fill, 
                                  temporary_stats[0, columns_numeric[2]],
                                  loan_data_numeric[:,2])
loan_data_numeric[:,2]


# In[97]:


temporary_stats[0,columns_numeric[3]]


# #### Loaned Amount, Interest Rate, Total Payment, Installment

# In[98]:


header_numeric


# In[99]:


for i in [1,3,4,5]:
    loan_data_numeric[:,i] = np.where(loan_data_numeric[:,i] == temporary_fill,
                                      temporary_stats[2, columns_numeric[i]],
                                      loan_data_numeric[:,i])


# In[100]:


loan_data_numeric


# ### Currency Change

# #### The Exchange Rate

# In[101]:


EUR_USD = np.genfromtxt("EUR-USD.csv", delimiter = ',', autostrip = True, skip_header = 1, usecols = 3)
EUR_USD


# In[102]:


loan_data_strings[:,0]


# In[103]:


exchange_rate = loan_data_strings[:,0]

for i in range(1,13):
    exchange_rate = np.where(exchange_rate == i,
                             EUR_USD[i-1],
                             exchange_rate)    

exchange_rate = np.where(exchange_rate == 0,
                         np.mean(EUR_USD),
                         exchange_rate)

exchange_rate


# In[104]:


exchange_rate.shape


# In[105]:


loan_data_numeric.shape


# In[106]:


exchange_rate = np.reshape(exchange_rate, (10000,1))


# In[107]:


loan_data_numeric = np.hstack((loan_data_numeric, exchange_rate))


# In[108]:


header_numeric = np.concatenate((header_numeric, np.array(['exchange_rate'])))
header_numeric


# #### From USD to EUR

# In[109]:


header_numeric


# In[110]:


columns_dollar = np.array([1,2,4,5])


# In[111]:


loan_data_numeric[:,6]


# In[112]:


for i in columns_dollar:
    loan_data_numeric = np.hstack((loan_data_numeric, np.reshape(loan_data_numeric[:,i] / loan_data_numeric[:,6], (10000,1))))


# In[113]:


loan_data_numeric.shape


# In[114]:


loan_data_numeric


# #### Expanding the header

# In[115]:


header_additional = np.array([column_name + '_EUR' for column_name in header_numeric[columns_dollar]])


# In[116]:


header_additional


# In[117]:


header_numeric = np.concatenate((header_numeric, header_additional))


# In[118]:


header_numeric


# In[119]:


header_numeric[columns_dollar] = np.array([column_name + '_USD' for column_name in header_numeric[columns_dollar]])


# In[120]:


header_numeric


# In[121]:


columns_index_order = [0,1,7,2,8,3,4,9,5,10,6]


# In[122]:


header_numeric = header_numeric[columns_index_order]


# In[123]:


loan_data_numeric


# In[124]:


loan_data_numeric = loan_data_numeric[:,columns_index_order]


# ### Interest Rate

# In[125]:


header_numeric


# In[126]:


loan_data_numeric[:,5]


# In[127]:


loan_data_numeric[:,5] = loan_data_numeric[:,5]/100


# In[128]:


loan_data_numeric[:,5]


# ### Checkpoint 2: Numeric

# In[129]:


checkpoint_numeric = checkpoint("Checkpoint-Numeric", header_numeric, loan_data_numeric)


# In[130]:


checkpoint_numeric['header'], checkpoint_numeric['data']


# ## Creating the "Complete" Dataset

# In[131]:


checkpoint_strings['data'].shape


# In[132]:


checkpoint_numeric['data'].shape


# In[133]:


loan_data = np.hstack((checkpoint_numeric['data'], checkpoint_strings['data']))


# In[134]:


loan_data


# In[135]:


np.isnan(loan_data).sum()


# In[136]:


header_full = np.concatenate((checkpoint_numeric['header'], checkpoint_strings['header']))


# ## Sorting the New Dataset

# In[137]:


loan_data = loan_data[np.argsort(loan_data[:,0])]


# In[138]:


loan_data


# In[139]:


np.argsort(loan_data[:,0])


# ## Storing the New Dataset

# In[140]:


loan_data = np.vstack((header_full, loan_data))


# In[141]:


np.savetxt("loan-data-preprocessed.csv", 
           loan_data, 
           fmt = '%s',
           delimiter = ',')


# In[ ]:




