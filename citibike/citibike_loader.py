"""
Citibike Data Loader - TheCommons XR Homework

Loads NYC Citibike data into PostgreSQL Star Schema

Data: July 2024, November 2024

"""



import pandas as pd

import zipfile

import io

import urllib.request

from datetime import datetime

import psycopg2

from psycopg2.extras import execute_values

import os

from dotenv import load_dotenv



# Load environment variables

load_dotenv()



# Database connection

DATABASE_URL = os.getenv("DATABASE_URL")



# Citibike data URLs (July, November 2024)

DATA_URLS = [

    "https://s3.amazonaws.com/tripdata/202407-citibike-tripdata.zip",

    "https://s3.amazonaws.com/tripdata/202411-citibike-tripdata.zip"

]



# Time of Day definitions (from Britannica)

# Morning: 5am-12pm, Afternoon: 12pm-5pm, Evening: 5pm-9pm, Night: 9pm-4am

TIME_OF_DAY = {

    1: {"name": "Morning", "start": 5, "end": 12},

    2: {"name": "Afternoon", "start": 12, "end": 17},

    3: {"name": "Evening", "start": 17, "end": 21},

    4: {"name": "Night", "start": 21, "end": 5}

}



# Bike type encoding

BIKE_TYPES = {

    "classic_bike": 1,

    "electric_bike": 2,

    "docked_bike": 3

}



# Member type encoding

MEMBER_TYPES = {

    "member": 1,

    "casual": 2

}



def get_db_connection():

    """Create database connection"""

    # Fix URL format for psycopg2

    url = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")

    url = url.split("?")[0]  # Remove query params

    return psycopg2.connect(url)



def get_time_of_day_id(hour):

    """Return time_of_day_id based on hour (0-23)"""

    if 5 <= hour < 12:

        return 1  # Morning

    elif 12 <= hour < 17:

        return 2  # Afternoon

    elif 17 <= hour < 21:

        return 3  # Evening

    else:

        return 4  # Night (21-4)



def get_quarter(month):

    """Return quarter number from month"""

    if month <= 3:

        return 1

    elif month <= 6:

        return 2

    elif month <= 9:

        return 3

    else:

        return 4



def get_day_name(date):

    """Return day name from date"""

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    return days[date.weekday()]



def get_month_name(month):

    """Return month name from month number"""

    months = ["", "January", "February", "March", "April", "May", "June",

              "July", "August", "September", "October", "November", "December"]

    return months[month]



def populate_dim_time_of_day(conn):

    """Insert time of day dimension data"""

    print("Populating dim_time_of_day...")

    cur = conn.cursor()

    

    # Clear existing data

    cur.execute("DELETE FROM citibike.dim_time_of_day")

    

    for id, data in TIME_OF_DAY.items():

        cur.execute("""

            INSERT INTO citibike.dim_time_of_day (time_of_day_id, time_of_day_name, start_hour, end_hour)

            VALUES (%s, %s, %s, %s)

            ON CONFLICT (time_of_day_id) DO NOTHING

        """, (id, data["name"], data["start"], data["end"]))

    

    conn.commit()

    print("✅ dim_time_of_day populated")



def populate_dim_bike_type(conn):

    """Insert bike type dimension data"""

    print("Populating dim_bike_type...")

    cur = conn.cursor()

    

    # Clear existing data

    cur.execute("DELETE FROM citibike.dim_bike_type")

    

    for name, id in BIKE_TYPES.items():

        cur.execute("""

            INSERT INTO citibike.dim_bike_type (bike_type_id, bike_type_name)

            VALUES (%s, %s)

            ON CONFLICT (bike_type_id) DO NOTHING

        """, (id, name))

    

    conn.commit()

    print("✅ dim_bike_type populated")



def populate_dim_member_type(conn):

    """Insert member type dimension data"""

    print("Populating dim_member_type...")

    cur = conn.cursor()

    

    # Clear existing data

    cur.execute("DELETE FROM citibike.dim_member_type")

    

    for name, id in MEMBER_TYPES.items():

        cur.execute("""

            INSERT INTO citibike.dim_member_type (member_type_id, member_type_name)

            VALUES (%s, %s)

            ON CONFLICT (member_type_id) DO NOTHING

        """, (id, name))

    

    conn.commit()

    print("✅ dim_member_type populated")



def populate_dim_date(conn, dates):

    """Insert date dimension data from unique dates"""

    print("Populating dim_date...")

    cur = conn.cursor()

    

    # US Holidays 2024 (simplified)

    holidays = {

        "2024-01-01": "New Year's Day",

        "2024-07-04": "Independence Day",

        "2024-11-28": "Thanksgiving",

        "2024-12-25": "Christmas"

    }

    

    for date in dates:

        date_id = int(date.strftime("%Y%m%d"))

        date_str = date.strftime("%Y-%m-%d")

        holiday_name = holidays.get(date_str, None)

        

        cur.execute("""

            INSERT INTO citibike.dim_date (date_id, full_date, day_num, day_name, month_num, month_name, year, quarter_num, holiday_name)

            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)

            ON CONFLICT (date_id) DO NOTHING

        """, (

            date_id,

            date,

            date.day,

            get_day_name(date),

            date.month,

            get_month_name(date.month),

            date.year,

            get_quarter(date.month),

            holiday_name

        ))

    

    conn.commit()

    print(f"✅ dim_date populated with {len(dates)} dates")



def load_csv_from_url(url):

    """Load CSV from ZIP URL without downloading locally - FIRST WEEK ONLY"""

    print(f"Loading data from: {url}")

    

    response = urllib.request.urlopen(url)

    zf = zipfile.ZipFile(io.BytesIO(response.read()))

    

    # Get all CSV files in ZIP

    csv_files = [f for f in zf.namelist() if f.endswith('.csv')]

    

    # Load and combine all CSVs

    dfs = []

    for csv_file in csv_files:

        print(f"  Reading: {csv_file}")

        df = pd.read_csv(zf.open(csv_file), low_memory=False)

        dfs.append(df)

    

    combined_df = pd.concat(dfs, ignore_index=True)

    print(f"  Loaded {len(combined_df)} total rows")

    

    # Filter for FIRST WEEK ONLY (days 1-7)

    combined_df['started_at'] = pd.to_datetime(combined_df['started_at'])

    combined_df = combined_df[combined_df['started_at'].dt.day <= 7]

    print(f"  Filtered to first week: {len(combined_df)} rows")

    

    # Filter for MORNING ONLY (5am-12pm)

    combined_df = combined_df[(combined_df['started_at'].dt.hour >= 5) & (combined_df['started_at'].dt.hour < 12)]

    print(f"  Filtered to morning only: {len(combined_df)} rows")

    

    return combined_df



def convert_ride_id_to_int(ride_id):

    """Convert ride_id string to integer"""

    # Remove any non-numeric characters and convert

    try:

        # Try direct conversion first

        return int(ride_id)

    except ValueError:

        # If it has letters, hash it to get integer

        return abs(hash(ride_id)) % (10**18)



def process_and_insert_rides(conn, df):

    """Process dataframe and insert into fact_rides"""

    print("Processing and inserting rides...")

    cur = conn.cursor()

    

    # Collect unique dates for dim_date

    df['started_at'] = pd.to_datetime(df['started_at'])

    df['ended_at'] = pd.to_datetime(df['ended_at'])

    

    unique_dates = df['started_at'].dt.date.unique()

    unique_dates = [pd.Timestamp(d).to_pydatetime() for d in unique_dates]

    

    # Populate dim_date with unique dates

    populate_dim_date(conn, unique_dates)

    

    # Process in batches

    batch_size = 10000

    total_rows = len(df)

    inserted = 0

    

    for start_idx in range(0, total_rows, batch_size):

        end_idx = min(start_idx + batch_size, total_rows)

        batch = df.iloc[start_idx:end_idx]

        

        rows = []

        for _, row in batch.iterrows():

            try:

                # Convert ride_id to integer

                ride_id = convert_ride_id_to_int(row['ride_id'])

                

                # Get foreign keys

                bike_type_id = BIKE_TYPES.get(row['rideable_type'], 1)

                member_type_id = MEMBER_TYPES.get(row['member_casual'], 2)

                

                # Get date_id (YYYYMMDD format)

                started_at = row['started_at']

                date_id = int(started_at.strftime("%Y%m%d"))

                

                # Get time_of_day_id

                time_of_day_id = get_time_of_day_id(started_at.hour)

                

                rows.append((

                    ride_id,

                    row['start_station_name'] if pd.notna(row['start_station_name']) else None,

                    row['end_station_name'] if pd.notna(row['end_station_name']) else None,

                    bike_type_id,

                    member_type_id,

                    date_id,

                    time_of_day_id,

                    row['started_at'],

                    row['ended_at']

                ))

            except Exception as e:

                continue  # Skip problematic rows

        

        # Batch insert

        if rows:

            execute_values(cur, """

                INSERT INTO citibike.fact_rides 

                (ride_id, start_station_name, end_station_name, bike_type_id, member_type_id, date_id, time_of_day_id, started_at, ended_at)

                VALUES %s

                ON CONFLICT (ride_id) DO NOTHING

            """, rows)

            conn.commit()

        

        inserted += len(rows)

        print(f"  Inserted {inserted}/{total_rows} rows...")

    

    print(f"✅ Inserted {inserted} rides into fact_rides")



def main():

    """Main function to load all Citibike data"""

    print("=" * 50)

    print("CITIBIKE DATA LOADER")

    print("TheCommons XR Homework")

    print("Data: July 2024, November 2024")

    print("=" * 50)

    

    # Connect to database

    conn = get_db_connection()

    print("✅ Connected to PostgreSQL")

    

    # Populate dimension tables

    populate_dim_time_of_day(conn)

    populate_dim_bike_type(conn)

    populate_dim_member_type(conn)

    

    # Clear fact table before loading

    cur = conn.cursor()

    cur.execute("DELETE FROM citibike.fact_rides")

    conn.commit()

    print("Cleared fact_rides table")

    

    # Load and process each month

    for url in DATA_URLS:

        print("\n" + "-" * 50)

        df = load_csv_from_url(url)

        process_and_insert_rides(conn, df)

    

    # Print summary

    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM citibike.fact_rides")

    total = cur.fetchone()[0]

    

    print("\n" + "=" * 50)

    print("SUMMARY")

    print(f"Total rides loaded: {total}")

    print("Months: July 2024, November 2024")

    print("=" * 50)

    

    conn.close()

    print("✅ Done!")



if __name__ == "__main__":

    main()

