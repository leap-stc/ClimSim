Dataset statistics

Here, we present distribution statistics to aid in understanding the dataset. 

    .
    ├── code                    # Code for calculating statistics
    ├── input2D                 # Statistics for 2D input variables
    ├── input3D                 # Statistics for 3D input variables
    ├── output2D                # Statistics for 2D output variables
    ├── output3D                # Statistics for 3D output variables
    └── README.md

The statistics for the 3D variables (in `input3D` and `output3D`) are calculated for each vertical level individually. For each variable, a histogram is provided to visualize the distribution using 100 bins. Additionally, a text file accompanies each histogram, containing key statistical measures such as the mean, standard deviation, skewness, kurtosis, median, deciles, quartiles, minimum, maximum, and mode. The text file also includes the bin edges and the corresponding frequency values used to generate the histogram figures. 

