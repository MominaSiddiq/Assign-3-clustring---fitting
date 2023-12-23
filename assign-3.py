#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 03:04:29 2023

@author: macbook
"""

import seaborn as sns
import pandas as pd
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import cluster_tools as ct
from matplotlib.colors import ListedColormap

# giving the filenames path of choosen indicators
elec_access = "electricity_access.csv"
elec_power = "electric_power.csv"
energy_use = "energy_use.csv"
co2_emission = "CO2_emission.csv"
energy_consumption = "renewable_energy_consumption.csv"
popu_growth = "population_growth.csv"
#greenhouse_emission = "greenhouse_gas_emission.csv"

# Creating global variables
selected_countries = {}
start_year = 2000
end_year = 2014
selected_years = {}
#grouped_data = {}
selected_data = {}


def read_data(filename):
    """
    This function reads data into a pandas dataframe take transpose and 
    clean the data.

    Parameters
    ----------
    filename : csv
       take the filenames as an argument.

    Returns
    -------
    two dataframes.

    """

    # read the data from the file
    data = pd.read_csv(filename, skiprows = 4, index_col = False)

    # cleaning the data(removing empty rows etc.)
    data.dropna(axis = 0, how = 'all', inplace = True)

    # remove emty columns
    data.dropna(axis = 1, how = 'all', inplace = True)

    # Drop the unnecessary columns in the data
    data.drop(columns = ['Country Code', 'Indicator Name',
              'Indicator Code'], inplace = True)
    country_col_df = data

    # taking the transpose
    years_col_df = data.T

    years_col_df.columns = years_col_df.iloc[0]
    years_col_df = years_col_df[1:]
    years_col_df.index.name = 'Year'  # Set the index name as 'Year' after transpose
    years_col_df = years_col_df.reset_index()

    # removing empty rows
    years_col_df.dropna(axis = 0, how = 'all', inplace = True)

    # removing empty columns
    years_col_df.dropna(axis = 1, how = 'all', inplace = True)

    # Removeing any unnecessary columns in the transpose of data
    years_col_df = years_col_df.loc[:, ~years_col_df.columns.duplicated()]

    # Removing any duplicated rows
    years_col_df = years_col_df[~years_col_df.index.duplicated(keep='first')]

    print(years_col_df)
    years_col_df.to_csv("transpose.csv")
    print("=" * 50)
    
    country_col_df = years_col_df.T
    #country_col_df.index.name = 'Country Name'
    
    country_col_df.columns = country_col_df.iloc[0]
    country_col_df = country_col_df[1:]
    country_col_df.index.name = 'Country'  # Set the index name as 'Country' after transpose
    country_col_df.to_csv("re-trans.csv")
    
    print(country_col_df)


    return country_col_df, years_col_df


# Filtering all the indicators data for selected data 
def filtered_data(data):
    """
    filtering data on selective years and countries for all the indicators 

    Parameters
    ----------
    data : python dataframe

    Returns
    -------
    filtered data.

    """
    filtered_dataframes = {}

    for key, df in data.items():
        # Filter by selected years and countries
        filtered_df = df.loc[selected_countries, selected_years]
        filtered_dataframes[key] = filtered_df
    
    return filtered_dataframes


# Obtaining the summary statistics of data by the describe method
def summary_statistics(data):
    """
    applies describe method on different indicators.

    Parameters
    ----------
    data : pandas dataframe
       The numerical data to analyze

    Returns
    -------
    summary_stats
        summary of selected data.

    """
    summary = {}
    for key, df in data.items():
        summary[key] = df.describe()
    
    return summary  


def corr_heatmap(dataframes):
    """
    produce a correlation map and creates scatter plots

    Parameters
    ----------
    dataframes : python dataframes
        contain all the indicaters df

    Returns
    -------
    None.

    """
    # Calculate mean of each dataframe
    summary_data = {key: df.mean() for key, df in dataframes.items()}
    
    # Concatenate summary data into a single dataframe
    summary_df = pd.DataFrame(summary_data)
    
    title = "Correaltion Heatmap between Indicators"
    
    # heatmap
    ct.map_corr(summary_df, title, 9)
    
    # Ensure all data can be converted to numeric, otherwise set as NaN
    summary_numeric = summary_df.apply(pd.to_numeric, errors='coerce')
    summary_numeric.dropna(axis=1, how='all', inplace=True)
    
    if len(summary_numeric.columns) > 1:  # Checking if there are more than one numeric columns
        pd.plotting.scatter_matrix(summary_numeric, figsize=(9.0, 9.0)) 
        plt.tight_layout()
        plt.show()
    else:
        print("No numeric data available to plot.")


def copy_dataframes(dfs, selected_dfs):
    """
    copy the dataframes selected for clutring

    Parameters
    ----------
    dfs : python dataframes
        
    selected_dfs : python dataframes

    Returns
    -------
    selected indicators.

    """
    selected_indicator_data = {}

    for indicator in selected_dfs:
        for key, df in dfs.items():
            if df.equals(indicator):
                selected_indicator_data[key] = df.copy()

    return selected_indicator_data


def normalize_df(df_dict):
    """
    normalize the dataframes which are stored in a dictionary 

    Parameters
    ----------
    df_dict : python dataframes dictionary

    Returns
    -------
    normalized dataframes and min, max values.

    """
    normalized_dataframes = {}
    min_max_values = {}

    for key, df in df_dict.items():
        # Normalize each dataframe separately
        normalized_df, min_vals, max_vals = ct.scaler(df)
        normalized_dataframes[key] = normalized_df
        min_max_values[key] = (min_vals, max_vals)

    return normalized_dataframes, min_max_values


def clustring(normalized_dataframes, cluster_number):
    """
    making cluters between the different dataframes by normalizing data

    Parameters
    ----------
    dataframe : python dataframe
    
    cluster_number : integer
        number of cluters make.

    Returns
    -------
    dataframe.

    """
    cluster_labels = {}
    cluster_centers = {}

    for key, df in normalized_dataframes.items():
        kmeans = cluster.KMeans(n_clusters=cluster_number)
        kmeans.fit(df)
        
        cluster_labels[key] = kmeans.labels_
        cluster_centers[key] = kmeans.cluster_centers_
    
    return cluster_labels, cluster_centers


def scatter_plot_clustring(normalized_dataframes, x_indicator, y_indicator, cluster_labels, cluster_centers):
    """
    plot a scatter plot for showing clusters

    Parameters
    ----------
    dataframe : python dataframes for clustring
        
    cluster_labels : array-like
        Array of cluster labels assigned to each data point.
        
    cluster_centers : array-like
        Coordinates of cluster centers.

    Returns
    -------
    None.

    """
    x_data = normalized_dataframes[x_indicator]
    y_data = normalized_dataframes[y_indicator]
    labels = cluster_labels[y_indicator]  # Assuming cluster labels are for the y-axis indicator
    
    centers = cluster_centers[y_indicator]  # Centers for the y-axis indicator
    
    # Reshape the data for clustering
    x_values = x_data.values.flatten()
    y_values = y_data.values.flatten()

    # Plotting the entire dataset with clustering information
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, c=labels, cmap='viridis')

    plt.title("Scatter Plot for All Years with Clusters")
    plt.xlabel("X-axis Label")
    plt.ylabel("Y-axis Label")
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()
    
def main():
    """
    A main function calling other functions.

    Returns
    -------
    None.

    """
    # Callig the read function and saving each dataframes into variables
    co2_emission_data, co2_emission_trans = read_data(co2_emission)
    energy_consumption_data, energy_consumption_trans = read_data(energy_consumption)
    popu_growth_data, popu_growth_trans = read_data(popu_growth)
    elec_access_data, elec_access_trans = read_data(elec_access)
    energy_use_data, energy_use_trans = read_data(energy_use)

    # selecting countries
    global selected_countries
    selected_countries = ['Qatar', 'China', 'United Kingdom', 'Portugal', 
                          'United States', 'South Asia', 'South Africa']

    # Selecting years
    global selected_years
    selected_years = [str(year) for year in range(start_year, end_year + 1, 2)]
    
    # Creating a list of all the transformed dataframes
    dataframes = {
        "popu_growth" : popu_growth_data,
        "energy_consume": energy_consumption_data,
        "electricity_access": elec_access_data,
        "energy_use" : energy_use_data,
        "CO2_emission": co2_emission_data
    }
    
    # Calling the function to filter data for all the indicators
    selected_data = filtered_data(dataframes)
    
    # Printing indicators for each filtered dataframe
    for key, df in selected_data.items():
        print(f"Indicators for {key}:")
        print(df.head())
        print("=" * 50)
        
    # Getting a summary of each filtered indicator
    summary_stats = summary_statistics(selected_data)

    # Printing summary statistics for each filtered indicator
    for key, stats in summary_stats.items():
        print(f"Summary statistics for {key}:")
        print(stats)
        print("=" * 50)
   
    # Calling the correlation heat map function
    corr_heatmap(selected_data)
    
    # Selecting the indicators for clustring
    selected_indicators = [selected_data["CO2_emission"], selected_data["electricity_access"]]  
    
    # Calling the function to copy the selected indicators
    selected_indicator_data = copy_dataframes(selected_data, selected_indicators)
    #print("data: ", selected_indicator_data)
    
    #normalizing selected dataframes 
    normalized_data, min_max_values = normalize_df(selected_indicator_data)
    #print("Normalized data: ", normalized_data)
    
    # Getting a summary of each normalized indicator
    summary_stats = summary_statistics(normalized_data)

    # Printing summary statistics for each filtered indicator
    for key, stats in summary_stats.items():
        print(f"Summary statistics for {key}:")
        print(stats)
        print("=" * 50)
    
    # Clustring the selected data
    # Clustring with 3
    labels, center = clustring(normalized_data, 3)
    
    print("lengths: ", len(normalized_data["CO2_emission"]), len(normalized_data["electricity_access"]), len(labels["CO2_emission"]))
    
    # Ensure alignment
    print("x_data:", normalized_data["CO2_emission"][:5])  # Check the first few elements
    print("y_data:", normalized_data["electricity_access"][:5])  # Check the first few elements
    print("labels:", labels["CO2_emission"][:5])  # Check the first few elements
      
  
    # Scatter plot of clusteerin with 3
    scatter_plot_clustring(normalized_data, "CO2_emission", "electricity_access", labels, center)
    
    
    


if __name__ == "__main__":
    main()
    