#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 03:04:29 2023

@author: macbook
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from scipy.stats import t


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
    
    # taking the transpose
    years_col_df = data.T

    # setting the header
    years_col_df.columns = years_col_df.iloc[0]
    years_col_df = years_col_df[1:]

    # reset index for making years as columns
    years_col_df = years_col_df.reset_index()
    years_col_df = years_col_df.rename(columns = {'index': 'Year'})

    # setting years as index
    years_col_df.set_index('Year', inplace = True)
    
    # removing empty rows
    years_col_df.dropna(axis = 0, how = 'all', inplace = True)

    # removing empty columns
    years_col_df.dropna(axis = 1, how = 'all', inplace = True)

    # Removeing any unnecessary columns in the transpose of data
    years_col_df = years_col_df.loc[:, ~years_col_df.columns.duplicated()]

    # Removing any duplicated rows
    years_col_df = years_col_df[~years_col_df.index.duplicated(keep='first')]
    
    # taking the transpose again for country column 
    country_col_df = years_col_df.T
    
    # Reset index for making countries as columns
    country_col_df = country_col_df.reset_index().rename(columns={'index': 'Country'})
    
    return country_col_df, years_col_df


# Filtering all the indicators data for selected data 
def filtered_data(df, start_year, end_year):
    """
    filtering data on selective years and countries for all the indicators 

    Parameters
    ----------
    data : python dataframe

    Returns
    -------
    filtered data.

    """
    # Ensure the DataFrame has an index named 'Country' or reset it if necessary
    if df.index.name != 'Country Name':
        if 'Country Name' in df.columns:
            df.set_index('Country Name', inplace=True)
        else:
            print("Country Name column not found.")
            return None
        
    # Convert years to string if necessary
    start_year, end_year = str(start_year), str(end_year)
    
    # Generate a list of year columns as strings
    years = [str(year) for year in range(int(start_year), int(end_year) + 1)]
    
    # Ensure that all years are present in the dataframe columns
    missing_years = [year for year in years if year not in df.columns]
    if missing_years:
        print(f"Missing year columns in dataframe: {missing_years}")
        return None

    # Filter for the range of years
    filtered_df = df.loc[:, years]

    return filtered_df


def corr_heatmap(dataframes, df_names):
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
   # Calculate mean of each dataframe and apply name mapping
    summary_data = {df_names[key]: df.mean() for key, df in dataframes.items() if key in df_names}
    
    # Concatenate summary data into a single dataframe
    summary_df = pd.DataFrame(summary_data)
    
    # Calculate correlation matrix for the summary data
    correlation_matrix = summary_df.corr()
    
    # Plotting heatmap for dataset correlations
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap between Datasets')
    plt.show()
    
    # Ensure all data can be converted to numeric, otherwise set as NaN
    summary_numeric = summary_df.apply(pd.to_numeric, errors='coerce')
    summary_numeric.dropna(axis=1, how='all', inplace=True)
    
    if len(summary_numeric.columns) > 1:  # Checking if there are more than one numeric columns
        pd.plotting.scatter_matrix(summary_numeric, figsize=(9.0, 9.0)) 
        plt.tight_layout()
        plt.show()
    else:
        print("No numeric data available to plot.")


def extract_data(selected_data, indicators, countries, years):
    """
    Extract data for specific indicators, countries, and years

    Parameters
    ----------
    selected_data : Python dictionary
        Dictionary containing filtered indicators.

    indicators : List
        List of indicators to consider.

    countries : List
        List of countries to consider.

    years : List
        List of years to consider.

    Returns
    -------
    Pandas DataFrame containing data for the specified indicators, countries, and years.
    """
    extracted_data = {}
    for indicator in indicators:
        data = selected_data[indicator].loc[countries, years]
        extracted_data[indicator] = data
        
    # Combine selected data
    combine_data = pd.concat(extracted_data, axis=1)
    print("combined data: ", combine_data)
    
    return combine_data


def normalize_data(df_dict):
    """
    Normalize the DataFrames stored in a dictionary.

    Parameters
    ----------
    df_dict : dict
        Dictionary containing DataFrames for each indicator.

    Returns
    -------
    normalized_dict : dict
        Dictionary containing normalized DataFrames for each indicator.
    min_vals : dict
        Dictionary containing minimum values for each indicator.
    max_vals : dict
        Dictionary containing maximum values for each indicator.
    """
    normalized_dict = {}
    min_vals = {}
    max_vals = {}

    for indicator, df in df_dict.items():
        min_val = df.values.min()
        max_val = df.values.max()

        # Normalize the DataFrame
        normalized_df = (df - min_val) / (max_val - min_val)

        normalized_dict[indicator] = normalized_df
        min_vals[indicator] = min_val
        max_vals[indicator] = max_val

    return normalized_dict, min_vals, max_vals
    
        
def clustering(extracted_df, cluster_number):
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
    normalized_df = {}
    min_vals = {}
    max_vals = {}
    
    # Normalize the data
    normalized_df, min_vals, max_vals = ct.scaler(extracted_df)
    
    # Applying k-means clustering on the data
    kmeans = cluster.KMeans(n_clusters=cluster_number)
    labels = kmeans.fit_predict(normalized_df)
    
    extracted_df['Cluster'] = labels
    
    #return labels, kmeans.cluster_centers_
    return extracted_df


def scatter_plot_clustering(data, x1, y1, x2, y2, title, xlabel, ylabel):
    """
    Plot a scatter plot for showing clusters of combined data from two indicators

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame containing the combined data of two indicators.

    x1 : str
        Name of the column for x-axis of indicator 1.

    y1 : str
        Name of the column for y-axis of indicator 1.

    x2 : str
        Name of the column for x-axis of indicator 2.

    y2 : str
        Name of the column for y-axis of indicator 2.

    title : str
        Title of the plot.

    xlabel : str
        Label for x-axis.

    ylabel : str
        Label for y-axis.

    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(data[x1],
                data[y1],
                c=data['Cluster'],
                cmap='viridis',
                label='Indicator 1')  # Plot for indicator 1

    plt.scatter(data[x2],
                data[y2],
                c=data['Cluster'],
                cmap='plasma',
                marker='x',
                label='Indicator 2')  # Plot for indicator 2

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.colorbar(label='Cluster')
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
    greenhouse_emission_data, greenhouse_emission_trans = read_data(greenhouse_emission)
    
    # selecting countries
    global selected_countries
    selected_countries = ['Qatar', 'China', 'United Kingdom', 'Portugal', 
                          'United States', 'South Asia', 'South Africa']

    # Selecting years
    global selected_years
    selected_years = [str(year) for year in range(start_year, end_year + 1)]
    
    # Creating a list of all the transformed dataframes
    dataframes = {
        "popu_growth" : popu_growth_data,
        "renewable_Econsume": energy_consumption_data,
        "electricity_access": elec_access_data,
        "greenhouse_emission" : greenhouse_emission_data,
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
    
    selected_indicators = ['CO2_emission', 'renewable_Econsume']

    selected_indicators_data = {indicator: selected_data[indicator] for indicator in selected_indicators}
    
    #normalizing selected dataframes 
    normalized_data, min_values, max_values = normalize_data(selected_indicators_data)
    print("normalize data: ", normalized_data)
    
    # Getting a summary of each normalized indicator
    summary_stats = summary_statistics(normalized_data)

    # Printing summary statistics for each filtered indicator
    for key, stats in summary_stats.items():
        print(f"Summary statistics for {key}:")
        print(stats)
        print("=" * 50)
    
    # Clustring the selected data
    # Clustring with 3
    clustered_data = clustering(normalized_data, 3)

    x1 = 'CO2_emission'
    y1 = 'renewable_Econsume'

    scatter_plot_clustering(clustered_data, x1, y1,
                            x1, y1,  
                            "CO2_emission vs Renewable_Energy_Consume", 
                            "Renewable_Energy_Consume", "CO2_Emission")
        
    

if __name__ == "__main__":
    main()
    
    
