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

# Creating global variables 
start_year = 1990
end_year = 2020
selected_countries = {}


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
    years_col_df = years_col_df[~years_col_df.index.duplicated(keep = 'first')]

    # taking the transpose again for country column
    country_col_df = years_col_df.T

    # Reset index for making countries as columns
    country_col_df = country_col_df.reset_index().rename(
        columns = {'index': 'Country'})

    return country_col_df, years_col_df


# Filtering all the indicators data for selected data
def filtered_data(df):
    """
    filtering data on selective years for all the indicators. 

    Parameters
    ----------
    df : python dataframe

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
    produce a correlation map and creates scatter plots.

    Parameters
    ----------
    dataframes : 
        python dataframes contain all the indicaters df.
        
    df_names : 
        names of the dataframes.

    Returns
    -------
    None.

    """
   # Calculate mean of each dataframe and apply name mapping
    summary_data = {df_names[key]: df.mean()
                    for key, df in dataframes.items() if key in df_names}

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


def population_growth_pie(filtered_data, selected_countries):
    """
    Plots a pie chart of the total population growth for selected countries 
    over the years 1990 to 2020.

    Parameters
    ----------
    filtered_data (pd.DataFrame): 
        The filtered DataFrame containing population growth data.
        
    selected_countries (list): 
        List of countries to include in the pie chart.

    Returns
    -------
    None: Displays a pie chart.
    """

    # Assuming the country names are the index of the DataFrame
    population_data = filtered_data.loc[selected_countries]

    # Calculate the total population growth over the years 1990 to 2020 for selected countries
    total_population_growth = population_data.sum(axis=1)

    # Plotting the pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(total_population_growth, labels=total_population_growth.index,
            autopct='%1.1f%%', startangle=140)
    plt.show()


def merge_datasets(df1, df2, countries, df1_column_name, df2_column_name):
    """
    Merge two datasets for specified countries.

    Parameters:
    df1 (pd.DataFrame): First DataFrame.
    df2 (pd.DataFrame): Second DataFrame.
    countries (list): List of countries to include in the merge.
    df1_name (str): Column name for values from the first DataFrame.
    df2_name (str): Column name for values from the second DataFrame.

    Returns:
    pd.DataFrame: Merged data for the specified countries.
    """
    merged_data_list = []

    for country in countries:
        for year in df1.columns:  # Assuming years are columns in df1
            df1_value = df1.loc[country,
                                year] if country in df1.index else None
            df2_value = df2.loc[country,
                                year] if country in df2.index else None

            row_data = {
                'Country': country,
                'Year': year,
                df1_column_name: df1_value,
                df2_column_name: df2_value
            }
            merged_data_list.append(row_data)

    return pd.DataFrame(merged_data_list)


def perform_clustering_and_find_centers(data, num_clusters, features):
    """
    Perform clustering on the provided dataset and find the cluster centers.

    Parameters:
    data (pd.DataFrame): The dataset to cluster.
    num_clusters (int): The number of clusters to form.
    features (list): List of column names to use for clustering.

    Returns:
    tuple: 
        - The original dataset with an additional 'Cluster' column.
        - The cluster centers as a numpy array.
    """
    # Normalizing the data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data[features])

    # Applying K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(data_normalized)
    clusters = kmeans.predict(data_normalized)

    # Adding the cluster labels to the original data
    data['Cluster'] = clusters

    # Getting the cluster centers
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

    return data, cluster_centers


def visualize_clusters(data, cluster_column, feature_columns, centers):
    """
    Visualize the clustering results.

    Parameters:
    data (pd.DataFrame): The clustered dataset.
    cluster_column (str): The name of the column containing cluster labels.
    feature_columns (list): List of column names to use for visualization.

    Returns:
    None: The function will output plots directly.
    """
   # Create a pairplot colored by cluster labels
    pairplot_fig = sns.pairplot(
        data, vars=feature_columns, hue=cluster_column, palette='bright')
    plt.suptitle('Pairplot of Features by Cluster', y=1.02)

    # Extract axes from pairplot to plot centers
    axes = pairplot_fig.axes
    num_features = len(feature_columns)
    for i in range(num_features):
        for j in range(num_features):
            if i != j:
                ax = axes[i][j]
                ax.scatter(centers[:, j], centers[:, i],
                           c='red', s=100, marker='X')  # Plot centers

    plt.show()

    # Create individual bar plots for each feature by cluster
    for feature in feature_columns:
        plt.figure(figsize=(8, 4))
        sns.barplot(x=cluster_column, y=feature, data=data)
        plt.title(f'Average {feature} by Cluster')
        plt.show()


# Polynomial model function
def poly_model(x, a, b, c):
    """
    Defines a second-order polynomial (quadratic) model for curve fitting.

    Parameters
    ----------
    x : array_like
        The independent variable where the data is measured, typically representing time or space.

    a : float
        Coefficient for the quadratic term in the polynomial equation.

    b : float
        Coefficient for the linear term in the polynomial equation.

    c : float
        Constant term in the polynomial equation.

    Returns
    -------
    array_like
        The values of the polynomial at x, given the specified coefficients.

    """
    return a * x**2 + b * x + c


# Function to fit the model to the data
def fit_model(years, emissions):
    """
    Fitting the curve fit model to the data.

    Parameters
    ----------
    years : ndarray or list
        The independent data — typically years — over which the model is to be fitted.

    emissions : ndarray or list
        The dependent data — typically emissions — which we are trying to fit with the model.

    Returns
    -------
    popt : ndarray
        Optimal values for the parameters so that the sum of the squared residuals of the model fit is minimized.

    pcov : 2d ndarray
        The covariance matrix of the parameters. The diagonal elements represent the variance of the fitted parameters.

    """
    popt, pcov = curve_fit(poly_model, years, emissions)
    return popt, pcov


# Error range calculation function
def err_ranges(x, popt, pcov, confidence=0.95):
    """


    Calculates the confidence intervals for the fitted model predictions.

    Parameters
    ----------
    x : array_like
        The independent variable values where the predictions are to be made.

    popt : array_like
        Optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized.

    pcov : 2d array_like
        The estimated covariance of popt. The diagonals provide the variance of the parameter estimate.

    confidence : float, optional
        The confidence level for the interval calculation. The default is 0.95 for a 95% confidence interval.

    Returns
    -------
    y_upper : array_like
        Upper prediction boundary of the confidence interval.

    y_lower : array_like
        Lower prediction boundary of the confidence interval.

    """
    # Predictions from the model
    y_model = poly_model(x, *popt)

    # Calculate the variance at each point from parameter covariance
    var_model = np.array([x**2, x, np.ones_like(x)]
                         ).T @ pcov @ np.array([x**2, x, np.ones_like(x)])

    sigma = np.sqrt(np.diag(var_model))  # Standard deviation at each point

    # The t value for confidence interval
    t_val = t.ppf((1+confidence)/2., len(x)-len(popt))

    # Upper and lower bounds
    y_upper = y_model + t_val * sigma
    y_lower = y_model - t_val * sigma

    return y_upper, y_lower


# Function to plot the historical data and future predictions with confidence intervals
def plot_emissions(country, years, emissions, future_years, popt, pcov, title):
    """
   Plots historical emissions data and future predictions with confidence intervals for a given country.

    Parameters:
    ----------
    country : str
        The name of the country for which emissions data is plotted.
    years : array_like
        Array of years for the historical data.
    emissions : array_like
        Array of observed emission values corresponding to the years.
    future_years : array_like
        Array of future years for which predictions are made.
    popt : array_like
        Optimal parameters obtained from the curve fitting process.
    pcov : 2d array_like
        Covariance matrix of the optimal parameters.
    title : str
        The title for the plot.

    Returns:
    -------
    None: Displays a plot showing historical emissions, fitted model, and future predictions with confidence intervals.

    """
    plt.figure(figsize=(10, 5))

    # Predictions for historical and future years using the model
    y_model = poly_model(years, *popt)
    y_future = poly_model(future_years, *popt)
    y_future_upper, y_future_lower = err_ranges(future_years, popt, pcov)

    # Plot historical data
    plt.scatter(years, emissions, label='Historical Data', color='green')
    plt.plot(years, y_model, label='Fitted Model', color='blue')

    # Plot future predictions with confidence intervals
    plt.plot(future_years, y_future, label='Future Predictions', color='red')
    plt.fill_between(future_years, y_future_lower, y_future_upper,
                     color='lightgray', alpha=0.3, label='Confidence Interval')

    plt.title(f"{title} {country}")
    plt.xlabel('Year')
    plt.ylabel('Emissions')
    plt.legend()
    plt.show()


# Main analysis function
def run_analysis(df, countries, future_year_span, title):
    """
    Plots historical emissions data and future predictions with confidence 
    intervals for a given country.

    Parameters:
    ----------
    country : str
        The name of the country for which emissions data is plotted.
    
    future_years : array_like
        Array of future years for which predictions are made.
        
    title : str
        The title for the plot.

    Returns:
    -------
    None: 
        Displays a plot showing historical emissions, fitted model, 
        and future predictions with confidence intervals.

    """
    # Ensure the index is integer-based for proper slicing
    df.index = pd.to_numeric(df.index)

    for country in countries:
        # Extract data for the country
        emissions = df.loc[start_year:end_year, country].values
        years = np.arange(start_year, end_year + 1)

        # Fit the model and predict future values
        popt, pcov = fit_model(years, emissions)

        future_years = np.arange(end_year + 1, end_year + future_year_span + 1)
        plot_emissions(country, years, emissions,
                       future_years, popt, pcov, title)


def main():
    """
    A main function calling other functions.

    Returns
    -------
    None.

    """
    # List of files
    filename = ["CO2_emission.csv", "renewable_energy_consumption.csv",
                "population_growth.csv", "greenhouse_gas_emission.csv",
                "GDP_per_capita.csv"]

    # Process each file and save the transposed data
    transposed_files = []  # To keep track of the new transposed files

    for file in filename:
        # Process the data
        country_col_df, years_col_df = read_data(file)

        # Create a new filename for the transposed data
        transposed_filename = file.replace('.csv', '_trans.csv')
        transposed_files.append(transposed_filename)

        # Save the transposed data
        country_col_df.to_csv(transposed_filename, index=False)

    # List to store filtered DataFrames
    filtered_dfs = {}

    # Read and filter each transposed file
    for transposed_file in transposed_files:
        # Read the transposed data
        df = pd.read_csv(transposed_file)

        # Filter the data
        filtered_df = filtered_data(df)

        # Add the filtered DataFrame to the list
        filtered_dfs[transposed_file] = filtered_df

        # Add the filtered DataFrame to the dictionary
        if filtered_df is not None:
            filtered_dfs[transposed_file] = filtered_df
            print(f"Filtered data from {transposed_file} added to the list")
        else:
            print(
                f"Skipped {transposed_file} due to missing 'Country Name' column.")

    # Print the filtered data for each file in the dictionary
    for filename, filtered_df in filtered_dfs.items():
        print(f"Filtered data from {filename}:")
        print(filtered_df)
        print("\n")

    # Mapping of long file names to short labels
    df_short_names = {
        'CO2_emission_trans.csv': 'CO2_emission',
        'renewable_energy_consumption_trans.csv': 'Renewable Energy',
        'population_growth_trans.csv': 'Population Growth',
        'GDP_per_capita_trans.csv': 'GDP_per_capita',
        'greenhouse_gas_emission_trans.csv': 'Greenhouse Gas'
    }

    # Calling the correlation heat map function
    corr_heatmap(filtered_dfs, df_short_names)

    # Extract data from the dictionary
    pop_df = filtered_dfs['population_growth_trans.csv']
    co2_df = filtered_dfs['CO2_emission_trans.csv']
    renewable_df = filtered_dfs['renewable_energy_consumption_trans.csv']

    # Selected countries for visualization
    selected_countries = ['United States', 'India', 'Kenya', 'Germany',
                          'Brazil', 'China', 'Australia', 'South Africa', 'Japan', 'Canada']

    # Population growth pie chart
    population_growth_pie(pop_df, selected_countries)

    # Merging datasets for clusters
    merged_data = merge_datasets(
        co2_df, renewable_df, selected_countries, "CO2_Emission", "Renewable_Energy")

    # Assuming 'merged_data' is your merged dataset and you want to cluster based on CO2 emissions and renewable energy
    features_to_cluster = ['CO2_Emission', 'Renewable_Energy']
    num_clusters = 3  # You can adjust the number of clusters as needed

    # Performing clustering on the merged data
    clustered_data, cluster_centers = perform_clustering_and_find_centers(
        merged_data, num_clusters, features_to_cluster)

    # Visualizing clusters
    cluster_column = 'Cluster'
    features_for_visualization = ['CO2_Emission', 'Renewable_Energy']

    visualize_clusters(clustered_data, cluster_column,
                       features_for_visualization, cluster_centers)

    greenhouse_df = filtered_dfs['greenhouse_gas_emission_trans.csv']
    gdp_df = filtered_dfs['GDP_per_capita_trans.csv']

    # Transpose the DataFrame so that each row is a year and each column is a country
    greenhouse_df_transposed = greenhouse_df.transpose()
    gdp_df_transposed = gdp_df.transpose()

    # List of selected countries and the range of years for prediction
    selected_countries = ['India', 'Germany', 'Kenya']
    title_ghg = "Greenhouse Gases Emission"
    title_gdp = "GDP_per_capita"
    
    # Running the fitting visualization 
    run_analysis(greenhouse_df_transposed, selected_countries, 20, title_ghg)
    run_analysis(gdp_df_transposed, selected_countries, 20, title_gdp)


if __name__ == "__main__":
    main()
