#!/usr/bin/env python
# coding: utf-8

# In[33]:

import matplotlib.pyplot as plt
import xarray as xr
import os
import scipy.stats as stats
import numpy as np
from datetime import datetime
import time
from scipy.stats import skew
import glob

# In[11]:


outvarlist=["state_q0002"]
yearlist=["0001","0002","0003","0004","0005","0006","0007","0008","0009"]
monlist=["01","02","03","04","05","06","07","08","09","10","11","12"]


# In[6]:


DATAPATH1="/pscratch/sd/s/sungduk/var_concat/E3SM-MMF_ne30/mlo/split_level/"
DATAPATH2="/pscratch/sd/s/sungduk/var_concat/E3SM-MMF_ne30/mli/split_level/"
ilev='44'
# In[32]:
missing_check = 0
for variable in outvarlist:
    start = time.time()
    # Convert the current time to a datetime object
    dt_object = datetime.fromtimestamp(start)
    # Print the datetime object in a readable format
    print("Current time:", dt_object.strftime("%Y-%m-%d %H:%M:%S"))
    sfile_path = 'mlo_'+variable+'_lev_44'+'.txt'
    sfile_path = 'mli_'+variable+'_lev_44'+'.txt'
    file = open(sfile_path, 'w')
    file_path1=DATAPATH1+"ne30.mlo."+variable+".lev-44*.nc"
    file_path2=DATAPATH2+"ne30.mli."+variable+".lev-44*.nc"
    flist1 = glob.glob(file_path1)
    flist2 = glob.glob(file_path2)
    dataset1 = xr.open_mfdataset(flist1, concat_dim='record', combine='nested', parallel=True) 
    dataset2 = xr.open_mfdataset(flist2, concat_dim='record', combine='nested', parallel=True) 
    dataset1 = dataset1.chunk({'record': 'auto'})
    dataset2 = dataset2.chunk({'record': 'auto'})
    var1 = dataset1[variable]
    var2 = dataset2[variable]
    var_flattened1 = var1.data.flatten()
    var_flattened2 = var2.data.flatten()
    var_flattened = (var_flattened2-var_flattened1)/1200
    var_flattened = var_flattened.compute()
    print("====>"+variable+" level:44")
    kurtosis_value = stats.kurtosis(var_flattened, nan_policy='omit')
    mean_value = np.nanmean(var_flattened)
    std_dev = np.nanstd(var_flattened)
    skewness = skew(var_flattened, nan_policy='omit')
    median_value = np.nanmedian(var_flattened)
    deciles = np.nanpercentile(var_flattened, [25, 75])
    minimum = np.nanmin(var_flattened)
    maximum = np.nanmax(var_flattened)
    mode_result = stats.mode(var_flattened)
    mode_result = stats.mode(var_flattened)
    print("Mean:", mean_value)
    print("Standard Deviation:", std_dev)
    print("Skewness:", skewness)
    print("Kurtosis:", kurtosis_value)
    print("Median:", median_value)
    print("Deciles (25% 75%):", deciles)
    print("Minimum:", minimum)
    print("Maximum:", maximum)
    print("Mode:", mode_result)
     # Write the remaining statistics to the file
    file.write("Level:"+(ilev)+"\n")
    file.write("Standard Deviation: " + str(std_dev) + "\n")
    file.write("Skewness: " + str(skewness) + "\n")
    file.write("Kurtosis: " + str(kurtosis_value) + "\n")
    file.write("Median: " + str(median_value) + "\n")
    file.write("Mean: " + str(mean_value) + "\n")
    file.write("Deciles (25% 75%): " + str(deciles) + "\n")
    file.write("Minimum: " + str(minimum) + "\n")
    file.write("Maximum: " + str(maximum) + "\n")
    # Create the histogram
    hist, bin_edges,dd = plt.hist(var_flattened, bins=100, edgecolor='black')
    # Add labels and title
    plt.xlabel(variable+" level:44")
    plt.ylabel('Frequency')
    plt.title('Histogram')
    # Save the histogram to a file
    plt.savefig('tend'+variable+'_lev_44_tend.png')
    # Display the histogram
    plt.show()
    # Convert bin edges and frequencies to strings
    bin_edges_str = "Bin Edges: " + str(bin_edges)
    frequencies_str = "Frequencies: " + str(hist)
    # Write the bin edges and frequencies to the file
    file.write(bin_edges_str + "\n")
    file.write(frequencies_str + "\n")
    end = time.time()
    print(end - start)
    # Convert the current time to a datetime object
    dt_object = datetime.fromtimestamp(end)
    # Print the datetime object in a readable format
    print("End time:", dt_object.strftime("%Y-%m-%d %H:%M:%S"))
    
    

