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


def inspect_weather_sampling(plant_uid=None, date_str=None, filename='tomato_processed_data.parquet'):
    print(f"Loading data from {filename}...")
    try:
        df = pd.read_parquet(filename)
    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        return

    # המרת זמן אם צריך
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # --- מנגנון בחירה אוטומטי (Fallback) ---
    # אם לא סופקו פרמטרים, או שהפרמטרים שסופקו לא קיימים - נבחר דיפולט

    # בדיקה האם ה-ID קיים (אם סופק)
    if plant_uid and plant_uid not in df['unique_id'].values:
        print(f"Warning: Plant ID '{plant_uid}' not found.")
        plant_uid = None  # איפוס כדי שיבחר אוטומטית למטה

    # אם אין ID (או שהקודם לא היה תקין), ניקח את הראשון שקיים בקובץ
    if plant_uid is None:
        plant_uid = df['unique_id'].iloc[0]
        print(f"--> Auto-selected Plant ID: {plant_uid}")

    # בדיקה האם התאריך קיים עבור הצמח הזה (אם סופק)
    plant_dates = df[df['unique_id'] == plant_uid]['timestamp'].dt.date.astype(str).unique()

    if date_str and date_str not in plant_dates:
        print(f"Warning: Date '{date_str}' not found for this plant.")
        date_str = None  # איפוס כדי שיבחר אוטומטית

    # אם אין תאריך, ניקח את התאריך הראשון שמופיע עבור הצמח הזה
    if date_str is None:
        date_str = plant_dates[0]  # לוקח את היום הראשון ברשימה
        print(f"--> Auto-selected Date: {date_str}")

    # --- מכאן ממשיך הקוד הרגיל שלך עם הערכים שנבחרו ---

    print(f"\nAnalyzing Plant: {plant_uid} | Date: {date_str}")

    # סינון לפי ה-ID והתאריך (שעכשיו בטוח קיימים)
    mask = (df['unique_id'] == plant_uid) & \
           (df['timestamp'].dt.date.astype(str) == date_str)

    day_data = df[mask].sort_values('timestamp')

    # --- חישוב הפרשי זמנים ---
    day_data['time_diff'] = day_data['timestamp'].diff()

    # בחירת עמודות: הוספתי את s4 (משקל) כדי שיהיה מעניין, בנוסף למזג האוויר
    cols_to_show = ['timestamp', 'wstemp', 'wsrh', 's4', 'time_diff']

    print(f"\n=== Weather & Sampling Analysis ===")

    # 1. בדיקת כמות דגימות
    total_samples = len(day_data)
    print(f"Total samples found: {total_samples}")
    print(f"Expected samples (if 3 min intervals): ~480")

    # 2. בדיקת המרווח הנפוץ ביותר
    if total_samples > 1:
        common_interval = day_data['time_diff'].mode()[0]
        print(f"Most common sampling interval: {common_interval}")

    print("\n--- FIRST 10 SAMPLES ---")
    print(day_data[cols_to_show].head(10).to_string(index=False))

    print("\n...\n(Middle measurements hidden)\n...")

    # סטטיסטיקה יומית
    print(f"\n--- Daily Stats ---")
    print(f"Temp Range: {day_data['wstemp'].min()} - {day_data['wstemp'].max()} C")
    print(f"Avg Humidity: {day_data['wsrh'].mean():.1f} %")

###############################
def verify_and_update_dt(source_file='tomato_processed_data.parquet',
                         summary_file='tomato_daily_summary_per_plant_with_dt.parquet'):
    print(f"--- Loading source data from {source_file} ---")
    try:
        df = pd.read_parquet(source_file)
    except FileNotFoundError:
        print(f"Error: Source file '{source_file}' not found.")
        return

    # 1. בדיקת קיום העמודה dt
    if 'dt' not in df.columns:
        print("CRITICAL ERROR: The column 'dt' (Daily Transpiration) does NOT exist in the source file!")
        print(f"Available columns are: {df.columns.tolist()}")
        return
    else:
        print("V Column 'dt' found in source data.")

    # 2. בדיקת ערכים חסרים ב-dt ברמת הדאטה הגולמי
    missing_dt = df['dt'].isna().sum()
    total_rows = len(df)
    print(
        f"Checking for missing 'dt' values in raw data: {missing_dt} missing out of {total_rows} rows ({(missing_dt / total_rows) * 100:.2f}%)")

    # 3. יצירת הסיכום היומי מחדש - כולל dt
    print("\n--- Re-generating Daily Summary with 'dt' ---")

    # המרת זמן לתאריך
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    # קיבוץ לפי צמח ויום
    # dt מוגדר במסמך כ-"one value per day". לכן ניקח את הערך הראשון (first) או המקסימלי (max) באותו יום.
    # מכיוון שזה ערך יומי, הם אמורים להיות זהים לכל השורות של אותו יום.
    daily_group = df.groupby(['unique_id', 'date'])

    daily_df = daily_group.agg({
        'exp_ID': 'first',
        's4': ['first', 'last'],  # משקל התחלה וסוף
        'dt': 'first',  # <--- הוספנו את ה-dt כאן!
        'soil_sand': 'first',
        'timestamp': 'min'
    }).reset_index()

    # סידור שמות העמודות
    daily_df.columns = ['unique_id', 'date', 'exp_ID', 'start_weight', 'end_weight', 'dt', 'soil_type',
                        'start_timestamp']

    # חישוב מספרי ימים
    daily_df = daily_df.sort_values(['unique_id', 'date'])
    daily_df['day_num'] = daily_df.groupby('unique_id').cumcount() + 1

    # בחירת העמודות הסופיות
    final_df = daily_df[['unique_id', 'exp_ID', 'day_num', 'date', 'start_weight', 'end_weight', 'dt', 'soil_type']]

    # 4. בדיקת איכות לדאטה המעובד
    print("\n--- Validating New Summary Data ---")
    missing_dt_summary = final_df['dt'].isna().sum()
    zeros_dt = (final_df['dt'] == 0).sum()

    print(f"Total days processed: {len(final_df)}")
    print(f"Days with missing 'dt': {missing_dt_summary}")
    print(f"Days with 'dt' = 0: {zeros_dt}")

    if missing_dt_summary > 0:
        print("Warning: Some days have NaN in 'dt'. You might need to drop them before Q-Learning.")
        # אופציונלי: מחיקת שורות ללא dt
        # final_df = final_df.dropna(subset=['dt'])
        # print("Dropped rows with NaN dt.")

    # 5. שמירה לקובץ
    print(f"\nSaving updated summary to: {summary_file}")
    final_df.to_parquet(summary_file)

    # הדפסת דוגמה
    print("\n--- Sample of Updated Data ---")
    print(final_df[['unique_id', 'date', 'dt', 'start_weight']].head())


def finalize_data_for_mdp():
    # 1. טעינת הקובץ שיצרת בשלב הקודם
    filename = "tomato_daily_summary_per_plant_with_dt.parquet"
    print(f"Loading {filename}...")
    df = pd.read_parquet(filename)

    initial_count = len(df)

    # 2. מחיקת השורות הבודדות שאין בהן dt
    # (ה-32 שורות שראינו בפלט)
    df_clean = df.dropna(subset=['dt'])

    dropped_count = initial_count - len(df_clean)
    print(f"Dropped {dropped_count} rows with missing 'dt'.")
    print(f"Final dataset size: {len(df_clean)} daily records.")

    # 3. שמירה בשם הסופי שבו נשתמש ב-MDP
    final_name = "tomato_mdp_ready.parquet"
    df_clean.to_parquet(final_name)
    print(f"Saved clean data to: {final_name}")

    return final_name
###############################

if __name__ == "__main__":
    finalize_data_for_mdp()