Dataset statistics

Here, we present distribution statistics to aid in understanding the dataset. 

.
├── code                    # Code for calculating statistics
├── mli2DVar                # Statistics for 2D input variables
├── mli3DVar                # Statistics for 3D input variables
├── mlo2DVar                # Statistics for 2D output variables
├── mlo3DVar                # Statistics for 3D output variables
├── tendency                # ?
└── README.md

The statistics for the 3D variables (in `mli3DVar` and `mlo3DVar`) are calculated for each vertical level individually. For each variable, a histogram is provided to visualize the distribution using 100 bins. Additionally, a text file accompanies each histogram, containing key statistical measures such as the mean, standard deviation, skewness, kurtosis, median, deciles, quartiles, minimum, maximum, and mode. The text file also includes the bin edges and the corresponding frequency values used to generate the histogram figures. 

