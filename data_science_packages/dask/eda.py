__author__ = 'Brett Allen (brettallen777@gmail.com)'

import dask.dataframe as dd
from sklearn.utils import resample

def apply_resampling(df: dd.DataFrame, col: str, do_upsample=True) -> dd.DataFrame:
    """
    Resample a Dask DataFrame to balance the classes in the specified column.

    Parameters:
        df (dd.DataFrame): Input Dask DataFrame.
        col (str): Column name to balance.
        do_upsample (bool): If True, upsample the minority classes. Otherwise, downsample the majority classes.

    Returns:
        dd.DataFrame: Resampled Dask DataFrame.
    """
    # Compute value counts (requires explicit computation in Dask)
    value_counts = df[col].value_counts().compute()
    sample_size = value_counts.max() if do_upsample else value_counts.min()
    print('sample_size = {} ({})'.format(sample_size, 'Upsampling' if do_upsample else 'Downsampling'))

    # Create a new Dask DataFrame to store resampled data
    resampled_dfs = []

    for label, count in value_counts.items():
        data = df[df[col] == label]
        n = count

        # Determine if resampling is needed
        if (do_upsample and n < sample_size) or (not do_upsample and n > sample_size):
            # Resample the data (convert to pandas temporarily for resampling)
            data_pd = data.compute()
            resampled_label = resample(data_pd, replace=do_upsample, n_samples=sample_size, random_state=42)
            resampled_dfs.append(dd.from_pandas(resampled_label, npartitions=1))
        else:
            resampled_dfs.append(data)

    # Concatenate all the resampled data
    resampled_data = dd.concat(resampled_dfs)

    return resampled_data
