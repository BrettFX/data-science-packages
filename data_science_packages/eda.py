__author__ = "Brett Allen (brettallen777@gmail.com"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def print_null_values_report(df: pd.DataFrame):
    """
    Generate and print null values report for provided dataframe. Displays the count of null records
    for each column as well as the relative null records percentage to the entire dataset.

    Args:
        df (pd.DataFrame): Dataframe to analyze nulls on.
    """
    # Inspect percent of null values for each column across the dataset
    missing_count = df.isna().sum()  # Count of missing values
    total_count = len(df)  # Total number of rows in the DataFrame
    percentage_missing = (missing_count / total_count) * 100  # Calculate percentage
    
    # Combine count and percentage into a new DataFrame
    missing_info = pd.DataFrame({'Missing Count': missing_count, 'Percentage Missing': percentage_missing})
    
    # Display the result
    print(missing_info)

def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a dataframe to represent the missing values in the original dataframe.

    Args:
        df (pd.DataFrame): Original dataframe to identify missing values.

    Returns:
        pd.DataFrame: New dataframe representing a report of missing values in original dataframe.
    """
    total_missing = df.isna().sum()
    total_available = len(df) - total_missing
    ratio_missing = total_missing / len(df)
    percent_missing = ratio_missing*100
    return pd.DataFrame({
        'total_missing': total_missing,
        'total_available': total_available,
        'ratio_missing': ratio_missing,
        'percent_missing': percent_missing.map(lambda p: f'{p:.1f}%')
    })

def build_lookup(data: pd.DataFrame, key: str) -> dict:
    """
    Build encoded lookup map for target column in dataframe.

    Args:
        data (pd.DataFrame): Dataframe containing column to build lookup on unique values.
        key (str): Target column to build lookup for.

    Returns:
        dict: Encoded dictionary with keys representing the unique values for the respective column and the values as the encoded id.
    """
    return { val: i+1 for i, val in enumerate(data[key].unique()) }

def get_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Identify outliers using interquartile range (IQR) and extract
    outliers from provided dataframe, returning the extracted
    outliers. Does not modify the original dataframe.

    Args:
        df (pd.DataFrame): Original dataset containing outliers.
        col (str): Target column to identify and extract outliers for.

    Returns:
        pd.DataFrame: Extracted outliers as a dataframe. None if original dataframe is empty or the target column doesn't exist.
    """
    if df is None or df.empty:
        print('[ERROR] Cannot process an empty dataframe.')
        return None

    if col not in df:
        print(f'[ERROR] "{col}" does not exist in dataframe.')
        return None

    # Identify outliers in age by using interquartile range (IQR)
    # See: https://saturncloud.io/blog/how-to-detect-and-exclude-outliers-in-a-pandas-dataframe/
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate lower bound and upper bound based on IQR
    # Any data point outside 1.5 times the IQR below the 25th percentile or above the 75th percentile can be considered an outlier
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    # Return the outliers dataframe by using lower bound and upper bound filters
    return df[(df[col] < lower_bound) | (df[col] > upper_bound)]

def generate_countplot(data: pd.DataFrame, y: str, **kwargs):
    """
    Generate countplot on data for given column (y). Example keyword args include 'title', 'base_color', 'figsize', 'limit', 'xlabel', and 'ylabel'

    Args:
        data (pd.DataFrame): Data to generate a countplot for.
        y (str): Column to within dataset to generate countplot for.
    """
    title = kwargs.get('title', '')
    base_color = kwargs.get('base_color', '#3274A1')
    figsize = kwargs.get('figsize', (16, 8))
    limit = kwargs.get('limit', 30)
    xlabel = kwargs.get('xlabel')
    ylabel = kwargs.get('ylabel')
    
    s = data[y].value_counts()
    
    if len(s) > limit:
        print(f'Warning: "{y}" field has {len(s)} unique values and configured limit is {limit}. Only showing the top {limit} unique values.')
        
    s = s.head(limit)
    plt.figure(figsize=figsize)
    sns.countplot(data[data[y].isin(s.index)], y=y, color=base_color)
    plt.title((title + ' Count Plot').strip())
    
    if xlabel:
        plt.xlabel(xlabel)
        
    if ylabel:
        plt.ylabel(ylabel)
    
    plt.show()

def generate_pieplot(data: pd.DataFrame, y: str, **kwargs):
    """
    Generate pie plot on dataset for provided column.

    Args:
        data (pd.DataFrame): Dataset to generate pie plot for.
        y (str): Target column in dataset to generate pie plot for.
    """
    # Pie Plot
    # https://stackoverflow.com/a/71515035/2901002
    def autopct_format(values):
        def apply_format(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{v:d}\n({p:.1f}%)'.format(v=val, p=pct)
        return apply_format

    s = data[y].value_counts()
    plt.pie(s, labels=s.index, autopct=autopct_format(s))
    plt.title(kwargs.get('title', 'Pie Plot'))
    plt.show()

def plot_corr_matrix(data: pd.DataFrame):
    """
    Generate and plot a correlation matrix for the numeric columns in the provided dataset.

    Args:
        data (pd.DataFrame): Dataset to create correlation plot for.
    """
    corr_matrix = data.select_dtypes('number').corr()
    plt.figure(figsize=(16, 10))
    sns.heatmap(corr_matrix, annot=True, vmin=-1, vmax=1, cmap='coolwarm', square=True)
    plt.show()

def generate_boxplots(data: pd.DataFrame, ncols: int=4, figsize: tuple=(20, 15), **kwargs):
    """
    Generate box plot for dataset to determine outliers.

    Args:
        data (pd.DataFrame): Dataset to generate box plot for.
        ncols (int, optional): Number of column in the produced figure. Defaults to 4.
        figsize (tuple, optional): Size of the figure. Defaults to (20, 15).
    """
    if data is None or data.empty:
        return
    
    # Dynamically determine number of columns and rows for subplots
    ncols = min(ncols, len(data.columns))
    nrows = (len(data.columns) + ncols - 1) // ncols

    # Create subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Flatten 2D axes array to 1D
    if max(nrows, ncols) > 1:
        axes = axes.ravel()
    else:
        axes = np.array([axes])

    for i, col in enumerate(data.columns):
        sns.boxplot(data=data[col], ax=axes[i])
        axes[i].set_title(col)
        axes[i].set(xticklabels=[], xticks=[])

    # Remove unused plots
    for i in range(len(data.columns), nrows * ncols):
        fig.delaxes(axes[i])
    
    if 'title' in kwargs:
        fig.suptitle(kwargs.get('title'), weight='bold')
        
    plt.tight_layout()
    plt.show()

def generate_pairplot(data: pd.DataFrame, legend_loc: str='upper left', legend_ncol: int=1, **kwargs):
    """
    Generate pair plot for all numeric columns in the provided dataset.
    For enhanced visualization/eda, pass in "hue" keyword argument with
    a categorical column (column with unique and finite set of values).

    Args:
        data (pd.DataFrame): Dataset to create pair plot for.
        legend_loc (str, optional): Location to render the legend if hue is provided (e.g., "upper left", "lower center", etc.). Honored if "hue" kwarg is provided. Default "upper left".
        legend_ncol (int, optional): Number of columns in legend to control whether items in legend are displayed horizontally vs. vertically. Honored if "hue" kwarg is provided. Default 1.
    """
    numeric_cols = list(data.select_dtypes('number'))
    g = sns.pairplot(data[numeric_cols], **kwargs)

    # Place legend in different location as desired
    # Inspired by https://stackoverflow.com/a/40910102
    if kwargs.get('hue'):
        handles = g._legend_data.values()
        labels = g._legend_data.keys()
        # g.figure.legend(
        #     title=kwargs.get('hue'),
        #     handles=handles, 
        #     labels=labels, 
        #     loc=legend_loc, 
        #     ncol=legend_ncol
        # )
        sns.move_legend(g, legend_loc, labels=labels, ncol=legend_ncol, title=kwargs.get('hue'), frameon=False)
        g.figure.subplots_adjust(top=0.92, bottom=0.08)

    # Generate columns string (e.g., "a, b, c, and d")
    cols_str = ', '.join(numeric_cols[:-1] + [f'and {numeric_cols[-1]}'])

    plt.suptitle(f'Pairwise Plot Between {cols_str}')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
