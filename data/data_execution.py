import pandas as pd
import numpy as np
from datetime import timedelta
import os

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


def compare_data_versions():
    # הגדרת נתיבים
    path_old = os.path.join("tomato_daily_summary_per_plant_with_dt.parquet")
    path_new = os.path.join("tomato_mdp_ready_with_temp_humidity.parquet")

    # טעינה
    print(f"Loading OLD file: {path_old}")
    try:
        df_old = pd.read_parquet(path_old)
    except FileNotFoundError:
        print("Old file not found. Make sure paths are correct.")
        return

    print(f"Loading NEW file: {path_new}")
    try:
        df_new = pd.read_parquet(path_new)
    except FileNotFoundError:
        print("New file not found. Did you run the generation script?")
        return

    # בחירת שורה אקראית מהקובץ החדש כדי לבדוק אותה
    # אנחנו משתמשים ב-sample כדי לא לקבל תמיד את השורה הראשונה
    sample_row = df_new.sample(1).iloc[0]

    target_id = sample_row['unique_id']
    target_date = sample_row['date']

    print(f"\n=== Comparing Record for Plant: {target_id} | Date: {target_date} ===")

    # שליפת השורה המתאימה מהקובץ הישן
    row_old = df_old[(df_old['unique_id'] == target_id) & (df_old['date'] == target_date)]

    # שליפת השורה (כשורה בודדת) מהקובץ החדש
    row_new = df_new[(df_new['unique_id'] == target_id) & (df_new['date'] == target_date)]

    if row_old.empty:
        print("Error: This row exists in NEW file but not in OLD file (maybe data cleaning dropped it?)")
        return

    # בחירת עמודות משותפות להשוואה + עמודות חדשות
    common_cols = ['unique_id', 'date', 'start_weight', 'end_weight', 'dt']
    new_cols = ['avg_temp', 'avg_humidity']

    print("\n--- [OLD FILE] Data ---")
    print(row_old[common_cols].to_string(index=False))

    print("\n--- [NEW FILE] Data ---")
    print(row_new[common_cols + new_cols].to_string(index=False))

    # בדיקה לוגית
    val_old = row_old.iloc[0]['start_weight']
    val_new = row_new.iloc[0]['start_weight']

    print("\n--- Verification Result ---")
    if val_old == val_new:  # אפשר להוסיף np.isclose למספרים ממשמשיים אם יש בעיות דיוק
        print("✅ SUCCESS: Weights match perfectly.")
    else:
        print(f"❌ FAILURE: Weights do not match! ({val_old} vs {val_new})")


def analyze_raw_intraday_data():
    # שים לב: אם אתה מריץ מתוך תיקיית הקוד הראשית, והקובץ בתוך data:
    input_file = os.path.join("tomato_processed_data.parquet")

    print(f"Loading raw data from {input_file}...")
    try:
        df = pd.read_parquet(input_file)
    except FileNotFoundError:
        print(f"File not found at {input_file}. Check your path.")
        return

    # המרת זמן
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 1. בחירת ניסוי וצמח מייצג
    # נבחר את הניסוי עם הכי הרבה דאטה
    exp_counts = df['exp_ID'].value_counts()
    target_exp = exp_counts.idxmax()

    # נבחר צמח אחד מתוך הניסוי הזה
    plant_id = df[df['exp_ID'] == target_exp]['unique_id'].iloc[0]

    print(f"\nAnalyzing Raw Data for Plant: {plant_id} (Experiment: {target_exp})")

    # סינון לדאטה של הצמח הזה בלבד
    plant_df = df[df['unique_id'] == plant_id].copy()
    plant_df['date'] = plant_df['timestamp'].dt.date

    # ================================================================
    # 2. מבט על: תנודתיות יומית (Min/Max vs Mean)
    # ================================================================
    print("\n=== Daily Volatility Summary (First 10 days) ===")
    print("This shows how much temp/humidity changes within a single day")
    print("-" * 85)
    print(f"{'Date':<12} | {'Temp (Min-Max)':<15} | {'Temp Avg':<10} | {'Humid (Min-Max)':<15} | {'Humid Avg':<10}")
    print("-" * 85)

    daily_stats = plant_df.groupby('date').agg({
        'wstemp': ['min', 'max', 'mean'],
        'wsrh': ['min', 'max', 'mean']
    })

    # הדפסת 15 הימים הראשונים
    for date, row in daily_stats.head(15).iterrows():
        t_min = row[('wstemp', 'min')]
        t_max = row[('wstemp', 'max')]
        t_mean = row[('wstemp', 'mean')]

        h_min = row[('wsrh', 'min')]
        h_max = row[('wsrh', 'max')]
        h_mean = row[('wsrh', 'mean')]

        print(
            f"{str(date):<12} | {t_min:.1f} - {t_max:.1f}     | {t_mean:.1f}      | {h_min:.1f} - {h_max:.1f}     | {h_mean:.1f}")

    # ================================================================
    # 3. זום-אין: יום אחד במלואו (מהלך היום)
    # ================================================================
    # נבחר יום מהאמצע
    unique_dates = plant_df['date'].unique()
    target_date = unique_dates[len(unique_dates) // 2]

    print(f"\n\n=== Full Day Breakdown: {target_date} ===")
    print("Displaying measurements every ~30 minutes to understand the trend")
    print("-" * 60)
    print(f"{'Time':<10} | {'Temp (°C)':<12} | {'Humidity (%)':<12}")
    print("-" * 60)

    day_data = plant_df[plant_df['date'] == target_date].sort_values('timestamp')

    # כדי לא להציף את המסך באלפי שורות, נדפיס שורה אחת כל 10 שורות (כל כ-30 דקות)
    # אבל נשאיר את הרזולוציה המקורית בזיכרון אם תרצה
    subset = day_data.iloc[::10]

    for _, row in subset.iterrows():
        time_str = row['timestamp'].strftime('%H:%M')
        print(f"{time_str:<10} | {row['wstemp']:<12.1f} | {row['wsrh']:<12.1f}")

    print("-" * 60)
    print(f"Total raw measurements in this day: {len(day_data)}")
    print(f"Temp Spread: {day_data['wstemp'].max() - day_data['wstemp'].min():.1f}°C")
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
def generate_daily_summary_with_temp(source_file='tomato_processed_data.parquet',
                                     output_file='tomato_mdp_ready_with_temp_humidity_light.parquet'):
    print(f"--- Loading source data from {source_file} ---")
    try:
        df = pd.read_parquet(source_file)
    except FileNotFoundError:
        print(f"Error: Source file '{source_file}' not found.")
        return

    # ==========================================
    # 1. בדיקת איכות נתונים עבור הטמפרטורה
    # ==========================================
    required_cols = ['wstemp', 'wspar', 'wsrh', 's4', 'dt']
    for col in required_cols:
        if col not in df.columns:
            print(f"CRITICAL ERROR: Column '{col}' missing!")
            return

    # ==============================================================================
    # 2. ניקוי קרינה (PAR) לפי הנחיית הדוקטורנטית
    # ==============================================================================
    print("\n--- Cleaning PAR (wspar) Data ---")
    initial_rows = len(df)

    # א. טיפול ב"מינוס אפס": ערכים מזעריים שנובעים מחישוב נקודה צפה יהפכו ל-0
    # נניח שכל מה שבין -0.001 ל-0 הוא בעצם 0
    mask_neg_zero = (df['wspar'] > -0.001) & (df['wspar'] < 0)
    corrected_zeros = mask_neg_zero.sum()
    df.loc[mask_neg_zero, 'wspar'] = 0.0
    print(f"Corrected {corrected_zeros} values from '-0' to '0'.")

    # ב. מחיקת שליליים אמיתיים (שגיאות אינטרפולציה)
    # משאירים רק מה שגדול או שווה לאפס
    df = df[df['wspar'] >= 0].copy()

    dropped_neg_par = initial_rows - len(df)
    print(f"Dropped {dropped_neg_par} rows containing negative PAR (interpolation errors).")
    print(f"Remaining rows: {len(df)}")

    # בדיקת ערכים חסרים
    missing_temp = df['wstemp'].isna().sum()
    total_rows = len(df)
    print(
        f"Missing 'wstemp' values in raw data: {missing_temp} / {total_rows} ({(missing_temp / total_rows) * 100:.2f}%)")

    # בדיקת טווחים (לראות שאין ערכים משוגעים כמו 200 מעלות או מינוס 50)
    print(
        f"Temp Range in raw data: Min={df['wstemp'].min():.2f}, Max={df['wstemp'].max():.2f}, Mean={df['wstemp'].mean():.2f}")

    # ==========================================
    # 2. יצירת הסיכום היומי (כולל ממוצע טמפ')
    # ==========================================
    print("\n--- Aggregating Daily Statistics ---")

    # המרת זמן
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    # קיבוץ
    daily_group = df.groupby(['unique_id', 'date'])

    daily_df = daily_group.agg({
        'exp_ID': 'first',
        's4': ['first', 'last'],
        'dt': 'first',
        'wstemp': 'mean',  # ממוצע טמפ'
        'wsrh': 'mean',  # ממוצע לחות
        'wspar': 'mean',  # ממוצע קרינה (אחרי הסינון)
        'soil_sand': 'first',
        'timestamp': 'min'
    }).reset_index()

    # סידור שמות העמודות
    daily_df.columns = ['unique_id', 'date', 'exp_ID', 'start_weight', 'end_weight',
                        'dt', 'avg_temp', 'avg_humidity', 'avg_par',
                        'soil_type', 'start_timestamp']

    # חישוב יום בניסוי
    daily_df = daily_df.sort_values(['unique_id', 'date'])
    daily_df['day_num'] = daily_df.groupby('unique_id').cumcount() + 1

    # בחירת עמודות סופיות
    final_df = daily_df[['unique_id', 'exp_ID', 'day_num', 'date',
                         'start_weight', 'end_weight', 'dt',
                         'avg_temp', 'avg_humidity', 'avg_par',
                         'soil_type']]

    # daily_df = daily_group.agg({
    #     'exp_ID': 'first',
    #     's4': ['first', 'last'],  # משקל התחלה וסוף
    #     'dt': 'first',  # איבוד מים יומי (כבר נתון יומי)
    #     'wstemp': 'mean',  # <--- חישוב ממוצע יומי!
    #     'wsrh': 'mean',  # <--- על הדרך, בוא ניקח גם לחות ממוצעת (יעזור בהמשך)
    #     'soil_sand': 'first',
    #     'timestamp': 'min'
    # }).reset_index()

    # # סידור שמות העמודות (Flattening MultiIndex)
    # daily_df.columns = ['unique_id', 'date', 'exp_ID', 'start_weight', 'end_weight',
    #                     'dt', 'avg_temp', 'avg_humidity', 'soil_type', 'start_timestamp']
    #
    # # חישוב יום בניסוי
    # daily_df = daily_df.sort_values(['unique_id', 'date'])
    # daily_df['day_num'] = daily_df.groupby('unique_id').cumcount() + 1
    #
    # # בחירת עמודות סופיות
    # final_df = daily_df[['unique_id', 'exp_ID', 'day_num', 'date',
    #                      'start_weight', 'end_weight', 'dt',
    #                      'avg_temp', 'avg_humidity', 'soil_type']]

    # ==========================================
    # 3. ניקוי ושמירה
    # ==========================================
    print("\n--- Cleaning Missing Data (Daily Level) ---")
    initial_len = len(final_df)

    # מחיקת שורות שיש בהן NaN באחד הפרמטרים החשובים
    final_df = final_df.dropna(subset=['dt', 'start_weight', 'end_weight', 'avg_temp', 'avg_par'])

    dropped = initial_len - len(final_df)
    print(f"Dropped {dropped} daily records due to missing data.")
    print(f"Final dataset size: {len(final_df)} daily records.")

    print(f"\nSaving to {output_file}...")
    final_df.to_parquet(output_file)
    print("Done.")

    # print("\n--- Cleaning Data ---")
    # initial_len = len(final_df)
    #
    # # מחיקת שורות שיש בהן NaN באחת מהעמודות הקריטיות
    # # עכשיו גם טמפרטורה היא קריטית
    # final_df = final_df.dropna(subset=['dt', 'start_weight', 'end_weight', 'avg_temp'])
    #
    # dropped = initial_len - len(final_df)
    # print(f"Dropped {dropped} rows due to missing data (dt, weight, or temp).")
    # print(f"Final dataset size: {len(final_df)} daily records.")
    #
    # # הצצה לנתונים החדשים
    # print("\n--- Sample of New Data (with Temperature) ---")
    # print(final_df[['date', 'start_weight', 'dt', 'avg_temp', 'avg_humidity']].head(10))
    #
    # # שמירה
    # print(f"\nSaving to {output_file}...")
    # final_df.to_parquet(output_file)
    # print("Done.")

###############################
def create_filtered_mdp_file(source_file, output_file, threshold=800):
    """
    Reads the source file, filters out rows with 'dt' >= threshold,
    and saves the result to a NEW output file.
    """
    print(f"\n=== Creating Filtered Copy (dt < {threshold}) ===")
    print(f"Source: {source_file}")
    print(f"Target: {output_file}")

    # 1. בדיקת קיום קובץ המקור
    if not os.path.exists(source_file):
        print(f"Error: Source file '{source_file}' not found.")
        return

    # 2. טעינת הנתונים
    df = pd.read_parquet(source_file)

    # 3. בדיקת תקינות עמודות
    if 'dt' not in df.columns:
        print("Error: 'dt' column missing in source file.")
        return

    initial_count = len(df)

    # 4. ביצוע הסינון
    # שומרים רק שורות שבהן dt קטן מהסף (קטן ממש, לא שווה)
    df_clean = df[df['dt'] < threshold].copy()

    # 5. חישוב סטטיסטיקות והדפסה
    dropped_count = initial_count - len(df_clean)
    if initial_count > 0:
        percent_dropped = (dropped_count / initial_count) * 100
    else:
        percent_dropped = 0

    print(f"Original rows: {initial_count}")
    print(f"Rows dropped: {dropped_count} ({percent_dropped:.2f}%)")
    print(f"Final rows: {len(df_clean)}")

    if not df_clean.empty:
        print(f"Max dt in new file: {df_clean['dt'].max()}")

    # 6. שמירה לקובץ החדש
    df_clean.to_parquet(output_file)
    print(f"✅ Successfully saved filtered data to: {output_file}")
###############################בדיקות עם פרמטר wspar

def analyze_wspar_behavior(filename='tomato_processed_data.parquet'):
    print(f"\n=== Analyzing 'wspar' (Light/PAR) Behavior in {filename} ===")

    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        return

    df = pd.read_parquet(filename)

    if 'wspar' not in df.columns:
        print("Error: 'wspar' column not found.")
        return

    # מסננים NaN רק לצורך הבדיקה
    wspar_data = df['wspar'].dropna()

    # 1. סטטיסטיקה בסיסית
    print("\n1. Basic Statistics:")
    print(f"Min: {wspar_data.min():.2f}")
    print(f"Max: {wspar_data.max():.2f}")
    print(f"Mean: {wspar_data.mean():.2f}")
    print(f"Median: {wspar_data.median():.2f}")

    # 2. בדיקת לילה (כמה אפסים יש?)
    zeros = (wspar_data == 0).sum()
    total = len(wspar_data)
    print(f"\n2. Night/Day Distribution:")
    print(f"Zero values (Night): {zeros} ({zeros / total:.1%})")
    print(f"Active values (>0): {total - zeros} ({(total - zeros) / total:.1%})")

    # בדיקת התפלגות הערכים הפעילים (ביום)
    day_data = wspar_data[wspar_data > 10]  # מסננים רעש קטן
    if not day_data.empty:
        print(f"Avg value during DAY (>10): {day_data.mean():.2f}")
        print(f"25th Percentile (Day): {day_data.quantile(0.25):.2f}")
        print(f"75th Percentile (Day): {day_data.quantile(0.75):.2f}")

    # 3. בדיקת "קפיצות" (Rate of Change)
    # זה קריטי כדי להחליט על ה-Granularity של ה-MDP
    # אנחנו בודקים מה ההפרש הממוצע בין דגימה לדגימה (כל 3 דקות)
    diffs = day_data.diff().abs()
    print("\n3. Volatility (Changes between 3-min samples during Day):")
    print(f"Average jump: {diffs.mean():.2f}")
    print(f"Max jump: {diffs.max():.2f}")
    print("Recommendation: Your MDP granularity should be larger than the Average Jump.")

    # 4. התנהגות לפי שעה (כדי לראות את הפעמון)
    print("\n4. Average wspar by Hour of Day:")
    # המרת זמן
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    hourly = df.groupby(df['timestamp'].dt.hour)['wspar'].mean()

    # הדפסה יפה של טבלה
    print(f"{'Hour':<5} | {'Avg PAR':<10}")
    print("-" * 18)
    for hour, val in hourly.items():
        # מדפיסים רק שעות זוגיות כדי לחסוך מקום, או הכל
        print(f"{hour:<5} | {val:<10.1f}")



if __name__ == "__main__":
    intermediate_file = 'tomato_mdp_ready_with_temp_humidity_light.parquet'

    print(">>> STEP 1: Generating Daily Summary with PAR...")
    generate_daily_summary_with_temp(
        source_file='tomato_processed_data.parquet',
        output_file=intermediate_file
    )

    # --- שלב 2: סינון ערכי טרנספירציה גבוהים (dt < 800) ---
    # אנחנו לוקחים את הקובץ שיצרנו בשלב 1, ומייצרים ממנו את הקובץ הסופי לאימון
    final_training_file = 'tomato_mdp_final_filtered.parquet'

    print("\n>>> STEP 2: Filtering High Transpiration (dt < 800)...")
    create_filtered_mdp_file(
        source_file=intermediate_file,
        output_file=final_training_file,
        threshold=800
    )

    print(f"\n✅ PIPELINE COMPLETE.")
    print(f"The file ready for the Agent is: {final_training_file}")

    # analyze_wspar_behavior('tomato_processed_data.parquet')
    # original_file = os.path.join("tomato_mdp_ready_with_temp_humidity.parquet")
    # filtered_file = os.path.join("tomato_mdp_filtered_dt800_ready.parquet")
    # create_filtered_mdp_file(original_file, filtered_file, threshold=800)