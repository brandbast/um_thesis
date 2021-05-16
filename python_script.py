#!/usr/bin/env python
# coding: utf-8

# # Script thesis Brandon Bastiaans

# # 1 Loading dataset

# In[1]:


# Import basic packages.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Importing neighbourhood statistics file.
rawdf = pd.read_excel (r'C:\Users\brand\Desktop\School\WUR\thesis\data\neighbourhood_statistics_thesis.xlsx')
print(rawdf.shape)
print(rawdf.info())


# # 2 Cleaning Dataset

# In[3]:


# Import seaborn.
import seaborn 


# In[4]:


# replace NULL values in the file with numpy NaN values.
rawdf.replace('NULL', np.NaN)


# In[5]:


# Count up all the NaN values.
rawdf.shape[0] - rawdf.dropna().shape[0]


# In[6]:


# counting neighbourhoods with density under 1500 per km2 for df with NaN values
only_na1500 = rawdf.loc[rawdf['bev_dich'] < 1500]
only_na1500.shape[0]


# In[7]:


# Dropping all NaN values.
df1 = rawdf.dropna()


# In[8]:


# counting neighbourhoods with density under 1500 per km2 for df without NaN values
only_na1500_2 = df1.loc[df1['bev_dich'] < 1500]
only_na1500_2.shape[0]


# In[9]:


# Count up all the NaN values to see if NaN rows were dropped.
print(df1.shape[0])
df1.shape[0] - df1.dropna().shape[0]


# In[10]:


# Resetting the index of df1.
df1.reset_index(drop=True, inplace=True)


# In[11]:


# Identifying the minimum value for the population density, so that only urban areas are included in the study. 
# This is because the degree of urbanity variable (ste_mvs) is based on the density of addresses,
# which also includes unhabited buildings, like businesses.
# ste_mvs <= 3 were already selected in excel.
# First the distribution of bev_dich is checked.
df1['bev_dich'].describe()


# In[12]:


# European Comission minimum population density for urban centres is 1500 per km2.
# This seems like a high treshold, but check first to see how many neighbourhoods will be excluded,
# because 75% of the neighbourhoods have more than 3857 inhabitants per km2.
df1['gwb_code_10'][(df1['bev_dich']<1500)].count()
# Only 232 neighbourhoods will be removed with a treshold of 1500, so this is a good minimum density for urban areas.


# In[13]:


# Drop neighbourhoods with a population density < 1500.
print(df1.shape[0])
cleandf = df1.drop(df1[df1['bev_dich'] < 1500].index)


# In[14]:


# Check if the neighbourhoods were successfully dropped.
print(cleandf.shape[0])
cleandf['gwb_code_10'][(cleandf['bev_dich']<1500)].count()


# In[15]:


# Resetting the index of cleandf.
cleandf.reset_index(drop=True, inplace=True)


# In[16]:


# Check variables (which aren't percentual) for a possible presence of outliers.
# average gas use:
seaborn.boxplot(x=cleandf['g_gas'])


# In[17]:


# avarage electricity use:
seaborn.boxplot(x=cleandf['g_ele'])


# In[18]:


# Removing outliers.
# Every value above the 99% threshold and below the 1% threshold will be removed.
# g_gas has a lot of extreme outliers so thresholds were set to 5% and 95%.
for col in cleandf[['g_gas', 'g_ele']]:
    if(col == 'g_gas'):
        percentiles = cleandf[col].quantile([0.05,0.95]).values
        cleandf = cleandf[cleandf[col] >= percentiles[0]]
        cleandf = cleandf[cleandf[col] <= percentiles[1]]
    else:
        percentiles = cleandf[col].quantile([0.01,0.99]).values
        cleandf = cleandf[cleandf[col] >= percentiles[0]]
        cleandf = cleandf[cleandf[col] <= percentiles[1]]


# In[19]:


# Check the amount of rows that are left over. 
cleandf.shape


# In[20]:


# Check variables to see if outliers were removed.
seaborn.boxplot(x=cleandf['g_gas'])


# In[21]:


seaborn.boxplot(x=cleandf['g_ele'])


# In[22]:


# Resetting the index of cleandf.
cleandf.reset_index(drop=True, inplace=True)


# # 3 Collinearity analysis

# In[23]:


# Make a selection of all independent variables
indep_var = cleandf.iloc[:, 6:33]

# Look at correlation table
corr = indep_var.corr()
plt.figure(figsize=(60,10))
ax = seaborn.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=seaborn.diverging_palette(10, 240, n=9),
    square=True
)


# In[24]:


# Show VIF values
df_cor = indep_var.corr()
vifs = pd.Series(np.linalg.inv(indep_var.corr().values).diagonal(), index=df_cor.index)
print(vifs)


# In[25]:


# First remove collinear features.
def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset
            elif (corr_matrix.iloc[i, j] <= (threshold*-1)) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset
    print(dataset)


# In[26]:


# Remove collinear features with a threshold of 0.7
correlation(indep_var,0.7)


# In[27]:


# Check remaining VIF values.
# Only high values are left in the 'age' categories.
df_cor = indep_var.corr()
vifs = pd.Series(np.linalg.inv(indep_var.corr().values).diagonal(), index=df_cor.index)
print(vifs)


# In[28]:


# Remove p_65_oo, because it has the highest VIF value.
indep_var = indep_var.drop(['p_65_oo'], axis=1)


# In[29]:


# Check VIF values again.
df_cor = indep_var.corr()
vifs = pd.Series(np.linalg.inv(indep_var.corr().values).diagonal(), index=df_cor.index)
print(vifs)


# In[30]:


# Make a dataset for RF.
# dataset = cleaned df, remove = columns not to be removed.  
def finalData (dataset, notremove):
    notremoveset = notremove.columns.tolist()
    notremoveset.insert(0,'g_gas')
    notremoveset.insert(0,'g_ele')
    notremoveset.insert(0,'type_stad')
    notremoveset.insert(0,'gm_naam')
    finaldf = dataset[notremoveset]
    return finaldf

# Use finalData function to prepare dataset for RF Model.
finalDF = finalData(cleandf, indep_var)
finalDF.columns.tolist()


# # 4 Analyse Hyperparameters with Cross Validation

# In[31]:


# Import necessary packages.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint


# In[32]:


# Set parameter ranges for random grid.
# Number of trees in random forest.
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 2000, num = 100)]
# Number of features to consider at every split.
max_features = ['auto']
# Maximum number of levels in tree.
max_depth = [int(x) for x in np.linspace(5, 60, num = 13)]
max_depth.append(None)
# Minimum number of samples required to split a node.
min_samples_split = [int(x) for x in np.linspace(2, 10, num = 5)]
# Minimum number of samples required at each leaf node.
min_samples_leaf = [int(x) for x in np.linspace(2, 10, num = 5)]
# Method of selecting samples for training each tree.
bootstrap = [True]

# Create the random grid.
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)


# In[33]:


# Function for selecting best hyperparameters.
def hyperparam (dataset, random_grid):
    
    # Divide data into attributes and labels.
    X = dataset.iloc[:, 4:].values
    y_gas = np.array(dataset.iloc[:, 3].values)
    y_ele = np.array(dataset.iloc[:, 2].values)

    # Split the data into training and testing sets.
    train_features_gas, test_features_gas, train_labels_gas, test_labels_gas = train_test_split(X, y_gas, test_size = 0.25, random_state = 42)
    train_features_ele, test_features_ele, train_labels_ele, test_labels_ele = train_test_split(X, y_ele, test_size = 0.25, random_state = 42)

    # Set random_state.
    rf = RandomForestRegressor(random_state = 42)

    # Random search of parameters, using 3 fold cross validation. 
    # Search across 100 different combinations, and use all available cores.
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions =random_grid, n_iter = 100, cv = 3, verbose=2, n_jobs = -1)

    # Fit the random search model
    rf_random.fit(train_features_gas, train_labels_gas);
    gas_params = rf_random.best_params_
    
    rf_random.fit(train_features_ele, train_labels_ele);
    ele_params = rf_random.best_params_
    
    # Instantiate model with best parameters
    rf_gas = RandomForestRegressor(n_estimators = gas_params['n_estimators'], max_depth = gas_params['max_depth'], min_samples_split = gas_params['min_samples_split'], min_samples_leaf = gas_params['min_samples_leaf'], max_features = 0.33, bootstrap = gas_params['bootstrap'], oob_score = True, random_state = 42)
    rf_ele = RandomForestRegressor(n_estimators = ele_params['n_estimators'], max_depth = ele_params['max_depth'], min_samples_split = ele_params['min_samples_split'], min_samples_leaf = ele_params['min_samples_leaf'], max_features = 0.33, bootstrap = ele_params['bootstrap'], oob_score = True, random_state = 42)
    
    return train_features_gas, test_features_gas, train_labels_gas, test_labels_gas, rf_gas, train_features_ele, test_features_ele, train_labels_ele, test_labels_ele, rf_ele


# # 5 Train and validate the model for full dataset

# In[34]:


# Import necessary packages.
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr, pearsonr
from sklearn.inspection import permutation_importance


# In[35]:


# Use function for dataset splitting and finding best hyperparameters.
train_features_gas, test_features_gas, train_labels_gas, test_labels_gas, rf_gas, train_features_ele, test_features_ele, train_labels_ele, test_labels_ele, rf_ele = hyperparam(finalDF, random_grid)


# In[36]:


# Check the distribution between test and train data. 
print('Training Features Shape (Gas):', train_features_gas.shape)
print('Training Labels Shape (Gas):', train_labels_gas.shape)
print('Testing Features Shape (Gas):', test_features_gas.shape)
print('Testing Labels Shape (Gas):', test_labels_gas.shape)

print('Training Features Shape (Electricity):', train_features_ele.shape)
print('Training Labels Shape (Electricity):', train_labels_ele.shape)
print('Testing Features Shape (Electricity):', test_features_ele.shape)
print('Testing Labels Shape (Electricity):', test_labels_ele.shape)


# In[37]:


# Train the models using the best hyperparameters.
rf_gas.fit(train_features_gas, train_labels_gas);
rf_ele.fit(train_features_ele, train_labels_ele);


# In[38]:


# Use the forest's predict method on the test data.
gas_predictions = rf_gas.predict(test_features_gas)
ele_predictions = rf_ele.predict(test_features_ele)


# In[39]:


# Import packages.
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# Function for validating the model.
def validate (rf, prediction, test_labels, name):
    
    print(name)
    # Print out the mean absolute error (MAE).
    mae = mean_absolute_error(test_labels, prediction)
    # Calculate mean absolute percentage error (MAPE).
    mape = mean_absolute_percentage_error(test_labels, prediction)*100
    # Calculate and display accuracy.
    accuracy = 100 - mape
    
    avrg = np.mean(test_labels)

    r2_val = r2_score(test_labels, prediction)
    #spearman = spearmanr(test_labels, prediction)
    #pearson = pearsonr(test_labels, prediction)
    rmse = mean_squared_error(test_labels, prediction, squared=False)
    oob = rf.oob_score_
    
    
    output = name,'MEAN:',round(avrg,2),'MAE:',round(mae,2),'MAPE:',round(mape,2),'RMSE:',round(rmse,2),'R2 validation set:',round(r2_val,2),'OOB R2:',round(oob,2)
    
    return output


# In[40]:


# Getting accuracy scores and correlation coefficients for gas
gas_validate = validate(rf_gas, gas_predictions, test_labels_gas, 'Validation scores for total dataset (gas)')
gas_validate


# In[41]:


# Getting accuracy scores and correlation coefficients for electricity
ele_validate = validate(rf_ele, ele_predictions, test_labels_ele, 'Validation scores for total dataset (electricity)')
ele_validate


# In[42]:


# Permutation importance fuction.
def permutation (df, rf, train_features, train_labels, name):
    permutation = permutation_importance(rf, train_features, train_labels, n_repeats=10, random_state=0)
    data = df.iloc[:, 4:]
    for i in permutation.importances_mean.argsort()[::-1]:
        print(f"{data.columns[i]:<8}"
                  f"{permutation.importances_mean[i]:.3f}"
                  f" +/- {permutation.importances_std[i]:.3f}")
    
    perm_dict = {"name": name}
    
    for i in permutation.importances_mean.argsort()[::-1]:
            perm_dict[str(data.columns[i])] = permutation.importances_mean[i]
    
    df2 = pd.DataFrame(perm_dict, index=[0])
    df2 = df2.set_index('name') 
                  
    return df2


# In[43]:


# Calculate permutation importance scores for gas.
perm_gas = permutation(finalDF, rf_gas, train_features_gas, train_labels_gas, 'total_gas')


# In[44]:


# Calculate permutation importance scores for electricity.
perm_ele = permutation(finalDF, rf_ele, train_features_ele, train_labels_ele, 'total_ele')


# In[45]:


# Function for plotting R2 accuracy scatter plots.
def plot_accuracy (predictions, test_labels, title, r2):
    seaborn.regplot(x = predictions, y = test_labels, label = r2,line_kws={"color": "red"}).set_title(title)
    plt.xlabel("Predictions")
    plt.ylabel("Real Values")
    plt.legend(loc="best")
    plt.show()


# In[46]:


# Plot accuracy scatter plots for entire dataset.
# R2 is the string of the r2 score of the model.
plot_accuracy(gas_predictions,test_labels_gas,'Scatter plot gas test data (total)', 'R2 = 0.68')
plot_accuracy(ele_predictions,test_labels_ele,'Scatter plot electricity test data (total)', 'R2 = 0.76')


# In[47]:


# Function for plotting R2 scatter plots for independent variables.
def plot_variables (predictions, test_labels, title, r2, x, y):
    seaborn.regplot(x = predictions, y = test_labels, label = r2,line_kws={"color": "red"}).set_title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(loc="best")
    plt.show()


# In[48]:


# Looking at some scatter plots to see if there are linear correlations between gas consumption and important variables.
# Entire dataset g_gas with independent variables:
# Plot accuracy scatter plots for entire dataset.
# R2 is the string of the r2 score of the model.
# R2 scores between g_gas and independent variables.
# R2 score for electricity visscher.
column_1 = finalDF["g_gas"]
column_2 = finalDF["p_00_14"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('A')
plot_variables(column_1,column_2,'Scatter plot gas vs p_00_14', str(correlation), 'g_gas', 'p_00_14')

column_2 = finalDF["p_15_24"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('B')
plot_variables(column_1,column_2,'Scatter plot gas vs p_15_24', str(correlation), 'g_gas', 'p_15_24')

column_2 = finalDF["p_25_44"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('C')
plot_variables(column_1,column_2,'Scatter plot gas vs p_25_44', str(correlation), 'g_gas', "p_25_44")

column_2 = finalDF["p_45_64"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('D')
plot_variables(column_1,column_2,'Scatter plot gas vs p_45_64', str(correlation), 'g_gas', "p_45_64")

column_2 = finalDF["p_w_all"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('E')
plot_variables(column_1,column_2,'Scatter plot gas vs p_w_all', str(correlation), 'g_gas', "p_w_all")

column_2 = finalDF["p_nw_all"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('F')
plot_variables(column_1,column_2,'Scatter plot gas vs p_nw_all', str(correlation), 'g_gas', "p_nw_all")

column_2 = finalDF["bev_dich"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('G')
plot_variables(column_1,column_2,'Scatter plot gas vs bev_dich', str(correlation), 'g_gas', "bev_dich")

column_2 = finalDF["g_woz"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('H')
plot_variables(column_1,column_2,'Scatter plot gas vs g_woz', str(correlation), 'g_gas', "g_woz")

column_2 = finalDF["p_mgezw"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('I')
plot_variables(column_1,column_2,'Scatter plot gas vs p_mgezw', str(correlation), 'g_gas', "p_mgezw")

column_2 = finalDF["p_leegsw"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('J')
plot_variables(column_1,column_2,'Scatter plot gas vs p_leegsw', str(correlation), 'g_gas', "p_leegsw")

column_2 = finalDF["p_koopw"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('K')
plot_variables(column_1,column_2,'Scatter plot gas vs p_koopw', str(correlation), 'g_gas', "p_koopw")

column_2 = finalDF["p_soz_ww"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('L')
plot_variables(column_1,column_2,'Scatter plot gas vs p_soz_ww', str(correlation), 'g_gas', "p_soz_ww")

column_2 = finalDF["p_wat"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('M')
plot_variables(column_1,column_2,'Scatter plot gas vs p_wat', str(correlation), 'g_gas', "p_wat")

column_2 = finalDF["g_bjaar"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('N')
plot_variables(column_1,column_2,'Scatter plot gas vs g_bjaar', str(correlation), 'g_gas', "g_bjaar")

column_2 = finalDF["p_label_a"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('O')
plot_variables(column_1,column_2,'Scatter plot gas vs p_label_a', str(correlation), 'g_gas', "p_label_a")

column_2 = finalDF["p_tussw"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('P')
plot_variables(column_1,column_2,'Scatter plot gas vs p_tussw', str(correlation), 'g_gas', "p_tussw")

column_2 = finalDF["p_2o1kap"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('Q')
plot_variables(column_1,column_2,'Scatter plot gas vs p_2o1kap', str(correlation), 'g_gas', "p_2o1kap")

column_2 = finalDF["p_hoek"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('R')
plot_variables(column_1,column_2,'Scatter plot gas vs p_hoek', str(correlation), 'g_gas', "p_hoek")

column_2 = finalDF["p_vrijstd"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('S')
plot_variables(column_1,column_2,'Scatter plot gas vs p_vrijstd', str(correlation), 'g_gas', "p_vrijstd")

column_2 = finalDF["p_slim_m"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('T')
plot_variables(column_1,column_2,'Scatter plot gas vs p_slim_m', str(correlation), 'g_gas', "p_slim_m")

column_2 = finalDF["FSI/FAR"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('U')
plot_variables(column_1,column_2,'Scatter plot gas vs FSI/FAR', str(correlation), 'g_gas', "FSI/FAR")


# In[49]:


# Looking at some scatter plots to see if there are linear correlations between gas consumption and important variables.
# Entire dataset g_gas with independent variables:
# Plot accuracy scatter plots for entire dataset.
# R2 is the string of the r2 score of the model.
# R2 scores between g_gas and independent variables.
# R2 score for electricity visscher.
column_1 = finalDF["g_ele"]
column_2 = finalDF["p_00_14"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('A')
plot_variables(column_1,column_2,'Scatter plot g_ele vs p_00_14', str(correlation), 'g_ele', 'p_00_14')

column_2 = finalDF["p_15_24"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('B')
plot_variables(column_1,column_2,'Scatter plot g_ele vs p_15_24', str(correlation), 'g_ele', 'p_15_24')

column_2 = finalDF["p_25_44"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('C')
plot_variables(column_1,column_2,'Scatter plot g_ele vs p_25_44', str(correlation), 'g_ele', "p_25_44")

column_2 = finalDF["p_45_64"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('D')
plot_variables(column_1,column_2,'Scatter plot g_ele vs p_45_64', str(correlation), 'g_ele', "p_45_64")

column_2 = finalDF["p_w_all"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('E')
plot_variables(column_1,column_2,'Scatter plot g_ele vs p_w_all', str(correlation), 'g_ele', "p_w_all")

column_2 = finalDF["p_nw_all"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('F')
plot_variables(column_1,column_2,'Scatter plot g_ele vs p_nw_all', str(correlation), 'g_ele', "p_nw_all")

column_2 = finalDF["bev_dich"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('G')
plot_variables(column_1,column_2,'Scatter plot g_ele vs bev_dich', str(correlation), 'g_ele', "bev_dich")

column_2 = finalDF["g_woz"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('H')
plot_variables(column_1,column_2,'Scatter plot g_ele vs g_woz', str(correlation), 'g_ele', "g_woz")

column_2 = finalDF["p_mgezw"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('I')
plot_variables(column_1,column_2,'Scatter plot g_ele vs p_mgezw', str(correlation), 'g_ele', "p_mgezw")

column_2 = finalDF["p_leegsw"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('J')
plot_variables(column_1,column_2,'Scatter plot g_ele vs p_leegsw', str(correlation), 'g_ele', "p_leegsw")

column_2 = finalDF["p_koopw"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('K')
plot_variables(column_1,column_2,'Scatter plot g_ele vs p_koopw', str(correlation), 'g_ele', "p_koopw")

column_2 = finalDF["p_soz_ww"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('L')
plot_variables(column_1,column_2,'Scatter plot g_ele vs p_soz_ww', str(correlation), 'g_ele', "p_soz_ww")

column_2 = finalDF["p_wat"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('M')
plot_variables(column_1,column_2,'Scatter plot g_ele vs p_wat', str(correlation), 'g_ele', "p_wat")

column_2 = finalDF["g_bjaar"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('N')
plot_variables(column_1,column_2,'Scatter plot g_ele vs g_bjaar', str(correlation), 'g_ele', "g_bjaar")

column_2 = finalDF["p_label_a"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('O')
plot_variables(column_1,column_2,'Scatter plot g_ele vs p_label_a', str(correlation), 'g_ele', "p_label_a")

column_2 = finalDF["p_tussw"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('P')
plot_variables(column_1,column_2,'Scatter plot g_ele vs p_tussw', str(correlation), 'g_ele', "p_tussw")

column_2 = finalDF["p_2o1kap"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('Q')
plot_variables(column_1,column_2,'Scatter plot g_ele vs p_2o1kap', str(correlation), 'g_ele', "p_2o1kap")

column_2 = finalDF["p_hoek"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('R')
plot_variables(column_1,column_2,'Scatter plot g_ele vs p_hoek', str(correlation), 'g_ele', "p_hoek")

column_2 = finalDF["p_vrijstd"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('S')
plot_variables(column_1,column_2,'Scatter plot g_ele vs p_vrijstd', str(correlation), 'g_ele', "p_vrijstd")

column_2 = finalDF["p_slim_m"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('T')
plot_variables(column_1,column_2,'Scatter plot g_ele vs p_slim_m', str(correlation), 'g_ele', "p_slim_m")

column_2 = finalDF["FSI/FAR"]
correlation = column_1.astype('float64').corr(column_2.astype('float64'))
print('U')
plot_variables(column_1,column_2,'Scatter plot g_ele vs FSI/FAR', str(correlation), 'g_ele', "FSI/FAR")


# # 6 Train RF models for various city sizes

# # 6.1 Small cities

# In[50]:


# Select all the neighbourhoods which are in small cities.
small_cities = finalDF.loc[finalDF['type_stad'] == 1]


# In[51]:


# Check the dataframe.
small_cities.head


# In[52]:


# Use function for dataset splitting and finding best hyperparameters.
train_features_gas_sc, test_features_gas_sc, train_labels_gas_sc, test_labels_gas_sc, rf_gas_sc, train_features_ele_sc, test_features_ele_sc, train_labels_ele_sc, test_labels_ele_sc, rf_ele_sc = hyperparam(small_cities, random_grid)


# In[53]:


# Train the models using the best hyperparameters.
rf_gas_sc.fit(train_features_gas_sc, train_labels_gas_sc);
rf_ele_sc.fit(train_features_ele_sc, train_labels_ele_sc);


# In[54]:


# Use the forest's predict method on the test data.
gas_predictions_sc = rf_gas_sc.predict(test_features_gas_sc)
ele_predictions_sc = rf_ele_sc.predict(test_features_ele_sc)


# In[55]:


# Getting accuracy scores and correlation coefficients for gas.
gas_validate_sc = validate(rf_gas_sc, gas_predictions_sc, test_labels_gas_sc, 'Validation scores for small cities (gas)')
gas_validate_sc


# In[56]:


# Getting accuracy scores and correlation coefficients for electricity.
ele_validate_sc = validate(rf_ele_sc, ele_predictions_sc, test_labels_ele_sc, 'Validation scores for small cities (electricity)')
ele_validate_sc


# In[57]:


# Plot accuracy scatter plots for small cities.
# R2 is the string of the r2 score of the model.
plot_accuracy(gas_predictions_sc ,test_labels_gas_sc,'Scatter plot gas test data (small cities)', 'R2 = 0.631')
plot_accuracy(ele_predictions_sc,test_labels_ele_sc,'Scatter plot electricity test data (small cities)', 'R2 = 0.759')


# In[58]:


# Calculate permutation importance scores for gas.
perm_gas_sc = permutation(small_cities, rf_gas_sc, train_features_gas_sc, train_labels_gas_sc, 'gas_small')


# In[59]:


# Calculate permutation importance scores for electricity.
perm_ele_sc = permutation(small_cities, rf_ele_sc, train_features_ele_sc, train_labels_ele_sc, 'ele_small')


# # 6.2 Middle sized cities

# In[60]:


# Select all the neighbourhoods which are in middle sized cities.
medium_cities = finalDF.loc[finalDF['type_stad'] == 2]


# In[61]:


# Check the dataframe.
medium_cities.info()


# In[62]:


# Use function for dataset splitting and finding best hyperparameters.
train_features_gas_mc, test_features_gas_mc, train_labels_gas_mc, test_labels_gas_mc, rf_gas_mc, train_features_ele_mc, test_features_ele_mc, train_labels_ele_mc, test_labels_ele_mc, rf_ele_mc = hyperparam(medium_cities, random_grid)


# In[63]:


# Train the models using the best hyperparameters.
rf_gas_mc.fit(train_features_gas_mc, train_labels_gas_mc);
rf_ele_mc.fit(train_features_ele_mc, train_labels_ele_mc);


# In[64]:


# Use the forest's predict method on the test data.
gas_predictions_mc = rf_gas_mc.predict(test_features_gas_mc)
ele_predictions_mc = rf_ele_mc.predict(test_features_ele_mc)


# In[65]:


# Getting accuracy scores and correlation coefficients for gas.
gas_validate_mc = validate(rf_gas_mc, gas_predictions_mc, test_labels_gas_mc, 'Validation scores for middle sized cities (gas)')
gas_validate_mc


# In[66]:


# Getting accuracy scores and correlation coefficients for electricity.
ele_validate_mc = validate(rf_ele_mc, ele_predictions_mc, test_labels_ele_mc, 'Validation scores for middle sized cities (electricity)')
ele_validate_mc


# In[67]:


# Plot accuracy scatter plots for middle sized cities.
# R2 is the string of the r2 score of the model.
plot_accuracy(gas_predictions_mc ,test_labels_gas_mc,'Scatter plot gas test data (middle sized cities)', 'R2 = 0.604')
plot_accuracy(ele_predictions_mc,test_labels_ele_mc,'Scatter plot electricity test data (middle sized cities)', 'R2 = 0.683')


# In[68]:


# Calculate permutation importance scores for gas.
perm_gas_mc = permutation(medium_cities, rf_gas_mc, train_features_gas_mc, train_labels_gas_mc, 'gas_medium')


# In[69]:


# Calculate permutation importance scores for electricity.
perm_ele_mc = permutation(medium_cities, rf_ele_mc, train_features_ele_mc, train_labels_ele_mc, 'ele_medium')


# # 6.3 Big cities

# In[70]:


# Select all the neighbourhoods which are in big cities.
big_cities = finalDF.loc[finalDF['type_stad'] == 3]


# In[71]:


# Check dataframe.
big_cities


# In[72]:


# Use function for dataset splitting and finding best hyperparameters.
train_features_gas_bc, test_features_gas_bc, train_labels_gas_bc, test_labels_gas_bc, rf_gas_bc, train_features_ele_bc, test_features_ele_bc, train_labels_ele_bc, test_labels_ele_bc, rf_ele_bc = hyperparam(big_cities, random_grid)


# In[73]:


# Train the models using the best hyperparameters.
rf_gas_bc.fit(train_features_gas_bc, train_labels_gas_bc);
rf_ele_bc.fit(train_features_ele_bc, train_labels_ele_bc);


# In[74]:


# Use the forest's predict method on the test data.
gas_predictions_bc = rf_gas_bc.predict(test_features_gas_bc)
ele_predictions_bc = rf_ele_bc.predict(test_features_ele_bc)


# In[75]:


# Getting accuracy scores and correlation coefficients for gas.
gas_validate_bc = validate(rf_gas_bc, gas_predictions_bc, test_labels_gas_bc, 'Validation scores for large cities (gas)')
gas_validate_bc


# In[76]:


# Getting accuracy scores and correlation coefficients for electricity.
ele_validate_bc = validate(rf_ele_bc, ele_predictions_bc, test_labels_ele_bc, 'Validation scores for large cities (electricity)')
ele_validate_bc


# In[77]:


# Calculate permutation importance scores for gas.
perm_gas_bc = permutation(big_cities, rf_gas_bc, train_features_gas_bc, train_labels_gas_bc, 'gas_big')


# In[78]:


# Calculate permutation importance scores for gas.
perm_ele_bc = permutation(big_cities, rf_ele_bc, train_features_ele_bc, train_labels_ele_bc, 'ele_big')


# In[79]:


# Plot accuracy scatter plots for big cities.
# R2 is the string of the r2 score of the model.
plot_accuracy(gas_predictions_bc ,test_labels_gas_bc,'Scatter plot gas test data (big cities)', 'R2 = 0.691')
plot_accuracy(ele_predictions_bc,test_labels_ele_bc,'Scatter plot electricity test data (big cities)', 'R2 = 0.83')


# # 8 Additional analysis amsterdam

# In[82]:


# A function for 10-fold cross-validation.
def cross_validation (dataset): 
    from sklearn.model_selection import KFold, cross_val_score
    
    X_bc = dataset.iloc[:, 4:].values
    y_gas_bc = np.array(dataset.iloc[:, 3].values)
    y_ele_bc = np.array(dataset.iloc[:, 2].values)
    
    # Split the data into training and testing sets.
    train_features_gas, test_features_gas, train_labels_gas, test_labels_gas = train_test_split(X_bc, y_gas_bc, test_size = 0.25, random_state = 42)
    train_features_ele, test_features_ele, train_labels_ele, test_labels_ele = train_test_split(X_bc, y_ele_bc, test_size = 0.25, random_state = 42)

    # Set random_state.
    rf = RandomForestRegressor(random_state = 42)

    # Random search of parameters, using 3 fold cross validation. 
    # Search across 100 different combinations, and use all available cores.
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions =random_grid, n_iter = 30, cv = 3, verbose=2, n_jobs = -1)

    # Fit the random search model
    rf_random.fit(train_features_gas, train_labels_gas);
    gas_params = rf_random.best_params_
    
    rf_random.fit(train_features_ele, train_labels_ele);
    ele_params = rf_random.best_params_
    
    # Instantiate model with best parameters
    rf_gas_bc2 = RandomForestRegressor(n_estimators = gas_params['n_estimators'], max_depth = gas_params['max_depth'], min_samples_split = gas_params['min_samples_split'], min_samples_leaf = gas_params['min_samples_leaf'], max_features = 0.33, bootstrap = gas_params['bootstrap'], oob_score = True, random_state = 42)
    rf_ele_bc2 = RandomForestRegressor(n_estimators = ele_params['n_estimators'], max_depth = ele_params['max_depth'], min_samples_split = ele_params['min_samples_split'], min_samples_leaf = ele_params['min_samples_leaf'], max_features = 0.33, bootstrap = ele_params['bootstrap'], oob_score = True, random_state = 42)
    
    # make KFold 
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Create empty arrays for storing values
    municipality = np.empty
    predictions_ele = np.empty
    actual_ele = np.empty
    
    
    predictions_gas = np.empty
    actual_gas = np.empty
    
    #Create a for loop which splits data into 10 folds, stores the municipality names, actual values and 
    # predicitons in lists and puts them into a dataframe afterwards.
    # Because this is done in order, the right municipality names will be with the right actual values and predicitons.
    for train_index, test_index in kf.split(X_bc):
    
        municipality = np.append(municipality, dataset.iloc[:, 0].values[test_index])
        actual_ele = np.append(actual_ele, dataset.iloc[:, 2].values[test_index])
        actual_gas = np.append(actual_gas, dataset.iloc[:, 3].values[test_index])
    
        X_train, X_test = X_bc[train_index], X_bc[test_index]
        y_train_ele, y_test_ele = y_ele_bc[train_index], y_ele_bc[test_index] 
        y_train_gas, y_test_gas = y_gas_bc[train_index], y_gas_bc[test_index]
    
        rf_ele_bc2.fit(X_train, y_train_ele)
        rf_gas_bc2.fit(X_train, y_train_gas)
    
        ele_predictions_bc2 = rf_ele_bc2.predict(X_test)
        gas_predictions_bc2 = rf_gas_bc2.predict(X_test)
    
        predictions_ele = np.append(predictions_ele, ele_predictions_bc2)
        predictions_gas = np.append(predictions_gas, gas_predictions_bc2)
    
    df_ele = pd.DataFrame({'municipality':municipality[1:], 'prediction':predictions_ele[1:], 'actual':actual_ele[1:]})
    df_gas = pd.DataFrame({'municipality':municipality[1:], 'prediction':predictions_gas[1:], 'actual':actual_gas[1:]})
    
    return df_ele, df_gas, rf_ele_bc2, rf_gas_bc2, train_features_ele, train_labels_ele, train_features_gas, train_labels_gas, test_features_ele, test_labels_ele, test_features_gas, test_labels_gas
    


# In[83]:


# Initiate cross_validation function
bc2_df_ele, bc2_df_gas, rf_ele_bc2, rf_gas_bc2, train_features_ele_bc2, train_labels_ele_bc2, train_features_gas_bc2, train_labels_gas_bc2, test_features_ele_bc2, test_labels_ele_bc2, test_features_gas_bc2, test_labels_gas_bc2 = cross_validation(big_cities)


# In[84]:


# check output
bc2_df_gas


# In[85]:


# check output
bc2_df_ele


# In[86]:


# Make subset of amsterdam
bc2_ams_ele = bc2_df_ele.loc[bc2_df_ele['municipality'] == 'Amsterdam']
bc2_ams_gas = bc2_df_gas.loc[bc2_df_gas['municipality'] == 'Amsterdam']


# In[92]:


# accuracy for amsterdam gas
ams_val_gas = validate(rf_gas_bc2, bc2_ams_gas["prediction"],bc2_ams_gas["actual"], 'Validation scores for Amsterdam (gas)')
ams_val_gas


# In[93]:


# accuracy for amsterdam ele
ams_val_ele = validate(rf_ele_bc2, bc2_ams_ele["prediction"],bc2_ams_ele["actual"], 'Validation scores for Amsterdam (electricity)')
ams_val_ele


# In[95]:


# permutation importance electricity
perm_ele_bc2 = permutation(big_cities, rf_ele_bc2, train_features_ele_bc2, train_labels_ele_bc2, 'ams_ele')


# In[96]:


# permutation importance gas
perm_gas_b2c = permutation(big_cities, rf_gas_bc2, train_features_gas_bc2, train_labels_gas_bc2, 'ams_gas')


# In[97]:


# select Visscher's variables
visscher_select = cleandf.iloc[:, :35]
visscher = visscher_select[['gm_naam','type_stad','g_ele','g_gas','bev_dich','FSI/FAR', 'p_mgezw', 'g_bjaar', 'p_koopw', 'p_wcorpw', 'p_25_44', 'g_hhgro', 'p_nw_all', 'g_woz']]


# In[98]:


# select big cities
visscher_big_cities = visscher.loc[visscher['type_stad'] == 3]


# In[99]:


# run cross validation for Visscher's variables
visscher_ele, visscher_gas, rf_ele_visscher, rf_gas_visscher, train_features_ele_visscher, train_labels_ele_visscher, train_features_gas_visscher, train_labels_gas_visscher, test_features_ele_visscher, test_labels_ele_visscher, test_features_gas_visscher, test_labels_gas_visscher = cross_validation(visscher_big_cities)


# In[100]:


# Make Amsterdam subsets
visscher_ams_ele = visscher_ele.loc[visscher_ele['municipality'] == 'Amsterdam']
visscher_ams_gas = visscher_gas.loc[visscher_gas['municipality'] == 'Amsterdam']


# In[101]:


# accuracy for visscher's variables gas
visscher_val_gas = validate(rf_gas_visscher, visscher_ams_gas["prediction"],visscher_ams_gas["actual"], 'Validation scores for Amsterdam Visscher (gas)')
visscher_val_gas


# In[102]:


# accuracy for visscher's variables electricity
visscher_val_ele = validate(rf_ele_visscher, visscher_ams_ele["prediction"],visscher_ams_ele["actual"], 'Validation scores for Amsterdam Visscher (electricity)')
visscher_val_ele


# In[103]:


# permutation scores for electricity.
perm_ele_visscher = permutation(visscher, rf_ele_visscher, train_features_ele_visscher, train_labels_ele_visscher, 'ams_ele_visscher')


# In[104]:


# permutation scores for gas.
perm_gas_visscher = permutation(visscher, rf_gas_visscher, train_features_gas_visscher, train_labels_gas_visscher, 'ams_gas_visscher')


# In[105]:


# plotting scatter plot gas
column1 = np.float64(bc2_ams_gas["prediction"])
column2 = np.float64(bc2_ams_gas["actual"])
plot_accuracy(column1,column2,'Scatter plot gas test data (Amsterdam based on large cities)', 'R2 = 0.66')


# In[106]:


# plot scatter plot ele
plot_accuracy(np.float64(bc2_ams_ele["prediction"]),np.float64(bc2_ams_ele["actual"]),'Scatter plot ele test data (Amsterdam based on large cities)', 'R2 = 0.84')


# # 9 Visualising data

# In[113]:


# Making a dataframe with permutation importances of small-, medium- and big cities for gas use.
city_types_gas = perm_gas_sc.append([perm_gas_mc, perm_gas_bc], sort=True)


# In[114]:


# Making a dataframe with permutation importances of small-, medium- and big cities for electricity use.
city_types_ele = perm_ele_sc.append([perm_ele_mc, perm_ele_bc], sort=True)


# In[115]:


# Changing order of the columns so that they are grouped properly.
def change_order (df):
    df = df[['p_wat','bev_dich','FSI/FAR','p_leegsw','g_bjaar','p_label_a','p_tussw','p_hoek', 'p_2o1kap', 'p_vrijstd','p_mgezw', 'p_koopw','g_woz','p_soz_ww','p_nw_all', 'p_w_all', 'p_00_14', 'p_15_24', 'p_25_44', 'p_45_64','p_slim_m']]
    return df

# Change order of training permutation importance features for all city types combined in 1 radar chart (1 for gas 1 for electricity).
city_types_gas = change_order(city_types_gas)
city_types_ele = change_order(city_types_ele)

# Change order of training permutation importance features for all city types seperately (3 for gas and 3 for electricity).
perm_gas_sc = change_order(perm_gas_sc)
perm_gas_mc = change_order(perm_gas_mc)
perm_gas_bc = change_order(perm_gas_bc)
perm_ele_sc = change_order(perm_ele_sc)
perm_ele_mc = change_order(perm_ele_mc)
perm_ele_bc = change_order(perm_ele_bc)



# In[116]:


# Creating a radar chart function.
# validate_list is a list of strings.
def radar_chart (df, plot_title, validate_list):
    
    categories = df.columns.tolist()
    # Adding the first column again at the back so that the chart has a start and ending.
    categories = [*categories, categories[0]]
    
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=(len(df.iloc[0].tolist())+1))

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(polar=True)
    
    for row in df.itertuples():
        info = row[1:]
        info = [*info, info[0]]
        # Adding the value of the first column again at the back so that the chart has a start and ending.
        plt.plot(label_loc, info, label=row[0])
    
    plt.title(plot_title, size=20)
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
    plt.legend()
    
    counter = 0.2
    for i in range(len(validate_list)):
        plt.figtext(1, counter, validate_list[i])
        counter += 0.18
    plt.show()
    return fig


# In[117]:


# Creating lists of validation info.
city_types_val_gas = [gas_validate_sc, gas_validate_mc, gas_validate_bc]
city_types_val_ele = [ele_validate_sc, ele_validate_mc, ele_validate_bc]
sc_val_gas = [gas_validate_sc]
mc_val_gas = [gas_validate_mc]
bc_val_gas = [gas_validate_bc]
sc_val_ele = [ele_validate_sc]
mc_val_ele = [ele_validate_mc]
bc_val_ele = [ele_validate_bc]


# In[118]:


# Using the radar chart function to create a chart for comparing the city types' permutation importances for gas.
# Training set.
city_types_gas_plt = radar_chart(city_types_gas, 'Permutation importance for three city types (gas training dataset)', city_types_val_gas)
city_types_gas_plt.savefig(r'C:\Users\brand\Desktop\School\WUR\thesis\city_types_gas.png', transparent=True)

sc_gas_plt = radar_chart(perm_gas_sc, 'Permutation importance for small cities (gas training dataset)', sc_val_gas)
sc_gas_plt.savefig(r'C:\Users\brand\Desktop\School\WUR\thesis\sc_gas.png', transparent=True)

mc_gas_plt = radar_chart(perm_gas_mc, 'Permutation importance for middle sized cities (gas training dataset)', mc_val_gas)
mc_gas_plt.savefig(r'C:\Users\brand\Desktop\School\WUR\thesis\mc_gas.png', transparent=True)

bc_gas_plt = radar_chart(perm_gas_bc, 'Permutation importance for large cities (gas training dataset)', bc_val_gas)
bc_gas_plt.savefig(r'C:\Users\brand\Desktop\School\WUR\thesis\bc_gas.png', transparent=True)




# In[119]:


# Using the radar chart function to create a chart for comparing the city types' permutation importances for electricity.
# Training set.
city_types_ele_plt = radar_chart(city_types_ele, 'Permutation importance for three city types (electricity training dataset)', city_types_val_ele)
city_types_ele_plt.savefig(r'C:\Users\brand\Desktop\School\WUR\thesis\city_types_ele.png', transparent=True)

sc_ele_plt = radar_chart(perm_ele_sc, 'Permutation importance for small cities (electricity training dataset)', sc_val_ele)
sc_ele_plt.savefig(r'C:\Users\brand\Desktop\School\WUR\thesis\sc_ele.png', transparent=True)

mc_ele_plt = radar_chart(perm_ele_mc, 'Permutation importance for middle sized cities (electricity training dataset)', mc_val_ele)
mc_ele_plt.savefig(r'C:\Users\brand\Desktop\School\WUR\thesis\mc_ele.png', transparent=True)

bc_ele_plt = radar_chart(perm_ele_bc, 'Permutation importance for large cities (electricity training dataset)', bc_val_ele)
bc_ele_plt.savefig(r'C:\Users\brand\Desktop\School\WUR\thesis\bc_ele.png', transparent=True)


# In[122]:


# For entire dataset
# Change order of features.
perm_gas = change_order(perm_gas)
perm_ele = change_order(perm_ele)

# Creating lists of validation info.
val_gas = [gas_validate]
# Creating lists of validation info.
val_ele = [ele_validate]
# Using the radar chart function to create a chart for comparing the city types' permutation importances for gas and electricity.

all_gas_plt = radar_chart(perm_gas, 'Permutation importance for entire dataset (gas training dataset)', val_gas)
all_gas_plt.savefig(r'C:\Users\brand\Desktop\School\WUR\thesis\gas_train_all.png', transparent=True)



all_ele_plt = radar_chart(perm_ele, 'Permutation importance for entire dataset (electricity training dataset)', val_ele)
all_ele_plt.savefig(r'C:\Users\brand\Desktop\School\WUR\thesis\ele_train_all.png', transparent=True)


# In[123]:


# radar charts for Amsterdam permutation
perm_gas_b2c = change_order(perm_gas_b2c)
perm_ele_bc2 = change_order(perm_ele_bc2)


bc2_gas_plt = radar_chart(perm_gas_b2c, 'Permutation importance for Amsterdam (gas training dataset)', bc_val_gas)
bc2_gas_plt.savefig(r'C:\Users\brand\Desktop\School\WUR\thesis\bc2_gas.png', transparent=True)



bc2_ele_plt = radar_chart(perm_ele_bc2, 'Permutation importance for Amsterdam (electricity training dataset)', bc_val_gas)
bc2_ele_plt.savefig(r'C:\Users\brand\Desktop\School\WUR\thesis\bc2_ele.png', transparent=True)


# In[124]:


# radar charts for Amsterdam permutation
perm_gas_visscher = perm_gas_visscher[['bev_dich','FSI/FAR', 'g_bjaar','p_mgezw', 'g_hhgro', 'p_25_44','p_nw_all','p_koopw', 'p_wcorpw', 'g_woz']]
perm_ele_visscher = perm_ele_visscher[['bev_dich','FSI/FAR', 'g_bjaar','p_mgezw', 'g_hhgro', 'p_25_44','p_nw_all','p_koopw', 'p_wcorpw', 'g_woz']]


visscher_gas_plt = radar_chart(perm_gas_visscher, 'Permutation importance for Amsterdam (Visscher) (gas training dataset)', bc_val_gas)
visscher_gas_plt.savefig(r'C:\Users\brand\Desktop\School\WUR\thesis\vis_gas.png', transparent=True)



visscher_ele_plt = radar_chart(perm_ele_visscher, 'Permutation importance for Amsterdam (Visscher) (electricity training dataset)', bc_val_gas)
visscher_ele_plt.savefig(r'C:\Users\brand\Desktop\School\WUR\thesis\vis_ele.png', transparent=True)


# In[ ]:




