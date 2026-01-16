import pandas as pd
import numpy as np
from datetime import timedelta

# הגדרת נתיב לקובץ - שים לב לשנות את זה לנתיב במחשב שלך
FILE_PATH = 'tomato_raw_data_v2.parquet'


def set_outliers_to_nan(data, column, thresholds):
    """
    Sets the values of outliers to NaN based on thresholds.
    """
    if len(thresholds) == 2:
        lower_threshold, upper_threshold = thresholds
        outlier_condition = (data[column] < lower_threshold) | (data[column] > upper_threshold)
    elif len(thresholds) == 1:
        outlier_condition = data[column] > thresholds[0]
    else:
        return data

    data.loc[outlier_condition, column] = float('nan')
    return data


def interpolate_missing_values(data, columns_to_interpolate, plant_id_column='unique_id', timestamp_column='timestamp',
                               threshold=40):
    """
    Interpolates missing values within each unique plant's data using cubic interpolation,
    avoiding gaps larger than the threshold.
    """
    # Ensure timestamp column is the index
    if timestamp_column in data.columns:
        data = data.set_index(timestamp_column)

    interpolated_data_list = []

    # Loop through each unique plant ID
    for uid in data[plant_id_column].unique():
        plant_data = data.loc[data[plant_id_column] == uid].copy()

        for col in columns_to_interpolate:
            if plant_data[col].isna().sum() > 0:
                # Identify gaps larger than threshold
                # Logic: create groups of consecutive NaNs, sum them up
                gaps = plant_data[col].isnull().astype(int).groupby(
                    (plant_data[col].isnull() != plant_data[col].shift().isnull()).cumsum()
                ).sum()

                large_gaps = gaps[gaps > threshold].index

                # Print info about gaps (Optional - can be commented out for cleaner run)
                gap_counts = gaps.value_counts().to_dict()
                print(f"Plant {uid} - {col}: Processing gaps...")

                # Interpolate only where gaps are NOT large
                mask = ~plant_data.index.isin(large_gaps)
                plant_data.loc[mask, col] = plant_data.loc[mask, col].interpolate(method='cubic')

        interpolated_data_list.append(plant_data)

    # Concatenate back to one DataFrame
    final_data = pd.concat(interpolated_data_list)

    # Reset index to bring timestamp back as a column
    final_data = final_data.reset_index()

    return final_data


def main():
    # 1. Load Data
    print("Loading data...")
    try:
        data = pd.read_parquet(FILE_PATH)
    except FileNotFoundError:
        print(f"Error: The file '{FILE_PATH}' was not found. Please check the path.")
        return

    # 2. Define Thresholds for Outliers
    threshold_dict = {
        's4': [300, 10000],
        'wsrh': [],
        'wstemp': [1, 47],
        'wspar': [],
        'vpd': [0, 8]
    }

    # 3. Clean Outliers (Set to NaN)
    print("Cleaning outliers...")
    for col in ['s4', 'wsrh', 'wstemp', 'wspar', 'vpd']:
        if col in data.columns:
            data = set_outliers_to_nan(data, col, threshold_dict[col])

    # 4. Interpolate Missing Values
    print("Interpolating missing values...")
    cols_to_inter = ['s4', 'wsrh', 'wstemp', 'wspar', 'vpd', 'Weight_change']

    # Check if all columns exist before interpolating
    existing_cols_to_inter = [c for c in cols_to_inter if c in data.columns]

    interpolated_data = interpolate_missing_values(
        data,
        columns_to_interpolate=existing_cols_to_inter,
        threshold=40
    )

    # 5. Output / Save
    print("Process complete!")
    print(interpolated_data.head())
    print("\nInfo summary of the interpolated data:")
    interpolated_data.info()

    interpolated_data.to_parquet('tomato_processed_data.parquet')


if __name__ == "__main__":
    main()