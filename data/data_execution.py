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


def analyze_experiments(df):
    # 1. קיבוץ לפי צמח ייחודי כדי לקבל סטטיסטיקות לצמח
    plant_stats = df.groupby('unique_id').agg({
        'exp_ID': 'first',
        'timestamp': ['min', 'max'],
        's4': 'count',  # מספר דגימות המשקל
        'soil_sand': 'first'  # סוג האדמה
    })

    # סידור שמות העמודות
    plant_stats.columns = ['exp_ID', 'start_time', 'end_time', 'sample_count', 'soil_type']

    # חישוב משך הזמן בימים לכל צמח
    plant_stats['duration_days'] = (plant_stats['end_time'] - plant_stats['start_time']).dt.total_seconds() / (
                3600 * 24)

    # 2. עכשיו, בוא נקבץ את זה לפי ניסויים (Experiments) כדי לקבל תמונת על
    experiment_summary = plant_stats.groupby('exp_ID').agg({
        'exp_ID': 'count',  # כמה צמחים בניסוי
        'duration_days': ['mean', 'min', 'max'],  # משך הניסוי הממוצע
        'sample_count': 'sum',  # סה"כ דגימות בניסוי
        'start_time': 'min',  # מתי הניסוי התחיל (הצמח הראשון)
        'end_time': 'max'  # מתי הניסוי נגמר
    })

    # סידור שמות לעמודות של הניסויים
    experiment_summary.columns = [
        'num_plants',
        'avg_duration_days', 'min_duration', 'max_duration',
        'total_samples',
        'exp_start', 'exp_end'
    ]

    return plant_stats, experiment_summary


def create_daily_summary(df):
    print("Processing daily summary...")

    # 1. יצירת עמודת 'תאריך' (ללא שעה) כדי לקבץ לפי ימים
    df['date'] = df['timestamp'].dt.date

    # 2. קיבוץ לפי צמח ויום
    # אנחנו רוצים את המשקל הראשון והאחרון בכל יום
    daily_group = df.groupby(['unique_id', 'date'])

    daily_df = daily_group.agg({
        'exp_ID': 'first',  # מזהה ניסוי
        's4': ['first', 'last'],  # משקל התחלה וסוף
        'soil_sand': 'first',  # סוג אדמה
        'timestamp': 'min'  # זמן התחלת היום (בשביל סידור)
    }).reset_index()

    # 3. סידור שמות העמודות (הפעולה למעלה יוצרת MultiIndex)
    daily_df.columns = ['unique_id', 'date', 'exp_ID', 'start_weight', 'end_weight', 'soil_type', 'start_timestamp']

    # 4. חישוב "מספר היום בניסוי" לכל צמח
    # אנחנו ממיינים לפי צמח וזמן, ואז נותנים מספור רץ לכל תאריך של אותו צמח
    daily_df = daily_df.sort_values(['unique_id', 'date'])
    daily_df['day_num'] = daily_df.groupby('unique_id').cumcount() + 1

    # 5. ניקוי וסידור סופי
    # נוריד את עמודת העזר של timestamp
    final_df = daily_df[['unique_id', 'exp_ID', 'day_num', 'date', 'start_weight', 'end_weight', 'soil_type']]

    return final_df

###############################
def print_experiment_sample(filename, target_exp_id):
    print(f"Loading {filename}...")
    try:
        df = pd.read_parquet(filename)
    except FileNotFoundError:
        print("File not found. Please run the previous step first.")
        return

    # סינון לפי הניסוי המבוקש
    exp_data = df[df['exp_ID'] == target_exp_id].copy()

    if exp_data.empty:
        print(f"Experiment {target_exp_id} not found in the file.")
        # ננסה להציע ניסוי אחר שקיים
        available = df['exp_ID'].unique()[:5]
        print(f"Try one of these IDs instead: {available}")
        return

    # מיון כדי שההדפסה תהיה קריאה: קודם לפי צמח, ואז לפי יום
    exp_data = exp_data.sort_values(['unique_id', 'day_num'])

    print(f"\n=== Full Data for Experiment {target_exp_id} ===")
    print(f"Total records: {len(exp_data)}")
    print(f"Number of plants in experiment: {exp_data['unique_id'].nunique()}")

    # הדפסת הנתונים עצמם.
    # to_string() מכריח את פייתון להדפיס את כל השורות בלי לקצר
    print("-" * 80)
    print(exp_data.to_string(index=False))
    print("-" * 80)


def inspect_specific_day(plant_uid, date_str):
    print(f"Loading full processed data...")
    # טוענים את הקובץ המפורט (לא היומי)
    try:
        df = pd.read_parquet('tomato_processed_data.parquet')
    except FileNotFoundError:
        print("Error: 'tomato_processed_data.parquet' not found.")
        return

    # המרת זמן
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"Filtering for plant {plant_uid} on {date_str}...")

    # סינון לפי ID ולפי התאריך
    # אנחנו ממירים את ה-timestamp לתאריך בלבד (string) כדי להשוות
    mask = (df['unique_id'] == plant_uid) & \
           (df['timestamp'].dt.date.astype(str) == date_str)

    specific_day_data = df[mask].sort_values('timestamp')

    if specific_day_data.empty:
        print("No data found for this specific plant and date.")
        return

    # בחירת עמודות רלוונטיות להדפסה
    cols_to_show = ['timestamp', 'unique_id', 's4']  # s4 זה המשקל

    print(f"\n=== Detailed Measurements for {plant_uid} on {date_str} ===")
    print(f"Total measurements found: {len(specific_day_data)}")

    print("\n--- FIRST 5 MEASUREMENTS (Check Start Weight) ---")
    print(specific_day_data[cols_to_show].head(5).to_string(index=False))

    print("\n...\n(Middle measurements hidden)\n...")

    print("\n--- LAST 5 MEASUREMENTS (Check End Weight) ---")
    print(specific_day_data[cols_to_show].tail(5).to_string(index=False))

    # וידוא מול הערכים שראינו בטבלה הקודמת
    first_weight = specific_day_data['s4'].iloc[0]
    last_weight = specific_day_data['s4'].iloc[-1]

    print("\n=== VERIFICATION ===")
    print(f"Actual First Weight found: {first_weight}")
    print(f"Actual Last Weight found:  {last_weight}")
###############################

if __name__ == "__main__":
    # הפרמטרים שביקשת לבדוק
    target_id = "1002_13_3"
    target_date = "2018-07-06"

    inspect_specific_day(target_id, target_date)

# if __name__ == "__main__":
#     # טעינה
#     print("Loading data...")
#     try:
#         df = pd.read_parquet('tomato_processed_data.parquet')
#     except FileNotFoundError:
#         print("Error: File not found.")
#         exit()
#
#     # המרה לזמן
#     if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
#         df['timestamp'] = pd.to_datetime(df['timestamp'])
#
#     # יצירת הטבלה היומית
#     daily_summary = create_daily_summary(df)
#
#     # הדפסה לבדיקה
#     print("\n=== Daily Summary Table Head ===")
#     print(daily_summary.head(10))
#
#     print(f"\nTotal daily records: {len(daily_summary)}")
#
#     # שמירה לקובץ חדש כפי שביקשת
#     output_filename = 'tomato_daily_summary_per_plant.parquet'
#     daily_summary.to_parquet(output_filename)
#     print(f"Saved daily summary to: {output_filename}")