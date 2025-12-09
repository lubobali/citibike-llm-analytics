#!/usr/bin/env python3
"""
Simplified SQL Tools - Multi-Tenant Big Tech AGI Implementation
================================================================
Netflix/Spotify Pattern: Each user has their own data universe.

Changes from original:
1. get_user_schema() - Check if user has uploaded data
2. should_use_user_data() - Decide which tables to query
3. run_sql_query() now accepts user_id for security
4. get_database_schema() now accepts user_id for personalized schema
"""

import os
import re
import json
from typing import Dict, Any, Optional
from sqlalchemy import create_engine, text
from adalflow.utils import setup_env

# Load environment variables
setup_env()

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)

# Security patterns - block dangerous operations
# Big Tech: Word boundaries prevent false positives (e.g., "uploaded" ≠ "UPDATE")
DANGEROUS = re.compile(
    r"\b(DROP|DELETE|TRUNCATE|ALTER|INSERT|UPDATE|CREATE|GRANT|REVOKE|FLUSH|REPAIR|OPTIMIZE|RENAME|RESTORE|BACKUP|REPLACE|SET|EXEC|CALL|LOAD|MERGE|ATTACH|DETACH|REINDEX|VACUUM)\b",
    re.IGNORECASE,
)

# Allowed tables for read-only queries (base tables)
ALLOWED_TABLES = [
    # LuBot analytics tables
    "click_logs", 
    "daily_click_summary", 
    "user_uploaded_metrics", 
    "user_daily_summary",
    # Citibike star schema (TheCommons XR Homework - Task 4)
    "citibike.fact_rides",
    "citibike.dim_date",
    "citibike.dim_time_of_day",
    "citibike.dim_bike_type",
    "citibike.dim_member_type",
]




def _is_select(query: str) -> bool:
    """Check if the query is a SELECT statement."""
    return query.strip().upper().startswith("SELECT")




def _extract_table_name(query: str) -> str:
    """Extract the table name from a SELECT query (supports schema.table format)."""
    match = re.search(r"FROM\s+([\w.]+)", query, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""




# =============================================================================
# 🏢 MULTI-TENANT: User Schema Detection (Big Tech AGI)
# =============================================================================

def get_user_schema(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get user's data schema from their uploads.
    
    Big Tech Pattern: Netflix/Spotify - personalized data universe
    
    Args:
        user_id: User identifier
        
    Returns:
        dict with user's metric info, or None if no data
    """
    if not user_id or user_id == 'anonymous':
        return None
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    u.primary_metric_name,
                    u.dimension_columns,
                    COUNT(m.id) as row_count,
                    MIN(m.date) as date_start,
                    MAX(m.date) as date_end
                FROM users u
                LEFT JOIN user_uploaded_metrics m ON u.id = m.user_id
                WHERE u.id = :user_id
                GROUP BY u.id, u.primary_metric_name, u.dimension_columns
            """), {"user_id": user_id})
            
            row = result.fetchone()
            
            if not row or not row.primary_metric_name or row.row_count == 0:
                return None
            
            # Parse dimensions (stored as JSON array in users table)
            # PostgreSQL JSONB returns Python list directly, not JSON string
            dimensions_raw = row.dimension_columns
            if dimensions_raw:
                if isinstance(dimensions_raw, list):
                    # Already a Python list (PostgreSQL JSONB auto-converts)
                    dimensions = dimensions_raw
                elif isinstance(dimensions_raw, str):
                    try:
                        dimensions = json.loads(dimensions_raw)
                    except:
                        dimensions = []
                else:
                    dimensions = []
            else:
                dimensions = []
            
            return {
                'has_data': True,
                'metric_name': row.primary_metric_name,
                'dimensions': dimensions,
                'date_range': {
                    'start': str(row.date_start) if row.date_start else None,
                    'end': str(row.date_end) if row.date_end else None
                },
                'row_count': row.row_count
            }
            
    except Exception as e:
        print(f"⚠️ User schema lookup failed: {e}")
        return None




def should_use_user_data(user_id: str) -> bool:
    """
    Check user's data_mode setting (respects manual toggle).
    
    Big Tech Pattern: Netflix profile switcher - user controls their view
    
    Args:
        user_id: User identifier
        
    Returns:
        bool: True if data_mode = 'user', False if 'demo'
    """
    if not user_id or user_id == 'anonymous':
        return False
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT data_mode FROM users WHERE id = :user_id
            """), {"user_id": user_id})
            row = result.fetchone()
            return row and row.data_mode == 'user'
    except:
        return False




def run_sql_query(query: str, time_filter: str = "all_time", user_id: str = None) -> Dict[str, Any]:
    """
    Execute a read-only SQL SELECT query on the PostgreSQL database.
    
    Multi-tenant: Automatically injects user_id filter for user tables.
    
    Args:
        query (str): The SQL SELECT query to execute
        time_filter (str): Time filter to apply (all_time, yesterday, last_week, etc.)
        user_id (str): User identifier for multi-tenant security
    
    Returns:
        Dict with data, columns, row_count, query_executed, error
    """
    try:
        # Safety checks
        if DANGEROUS.search(query or ""):
            return {
                "data": [],
                "columns": [],
                "row_count": 0,
                "query_executed": query,
                "error": "Unsafe SQL detected. Only read-only queries are allowed."
            }
        
        if not _is_select(query):
            return {
                "data": [],
                "columns": [],
                "row_count": 0,
                "query_executed": query,
                "error": "Only SELECT queries are permitted."
            }

        # Check table is allowed
        table_name = _extract_table_name(query)
        if table_name and table_name.lower() not in ALLOWED_TABLES:
            return {
                "data": [],
                "columns": [],
                "row_count": 0,
                "query_executed": query,
                "error": f"Query must only reference allowed tables: {ALLOWED_TABLES}"
            }
        
        # =================================================================
        # 🔐 SECURITY: Inject user_id filter for user tables
        # =================================================================
        filtered_query = query
        
        if user_id and ('user_uploaded_metrics' in query.lower() or 'user_daily_summary' in query.lower()):
            # Check if user_id filter already exists
            if 'user_id' not in query.lower():
                # Inject user_id filter
                if 'WHERE' in query.upper():
                    # Add to existing WHERE
                    filtered_query = re.sub(
                        r'\bWHERE\b',
                        f"WHERE user_id = '{user_id}' AND",
                        query,
                        count=1,
                        flags=re.IGNORECASE
                    )
                else:
                    # Add WHERE clause after FROM table
                    filtered_query = re.sub(
                        r'(FROM\s+\w+)',
                        f"\\1 WHERE user_id = '{user_id}'",
                        query,
                        count=1,
                        flags=re.IGNORECASE
                    )
                print(f"🔐 Injected user_id filter: {user_id[:8]}...")
        
        # Apply time filter if not "all_time"
        query_lower = filtered_query.lower()
        
        # Check if the incoming query already contains time filtering in WHERE clause only
        where_match = re.search(r'\bWHERE\b(.+?)(?:\bGROUP BY\b|\bORDER BY\b|\bLIMIT\b|$)', query_lower, re.IGNORECASE | re.DOTALL)
        where_clause = where_match.group(1) if where_match else ""
        already_time_filtered = ("timestamp" in where_clause or "date" in where_clause)



        # FIX GROQ'S BAD DATE SYNTAX

        if already_time_filtered:
            bad_pattern = r"DATE\(timestamp\)\s*=\s*CURRENT_DATE\s*-\s*INTERVAL\s*'(\d+)\s*day'"
            if re.search(bad_pattern, filtered_query, re.IGNORECASE):
                filtered_query = re.sub(
                    bad_pattern,
                    r"timestamp >= CURRENT_DATE - INTERVAL '\1 days'",
                    filtered_query,
                    flags=re.IGNORECASE
                )
                print(f"🔧 Fixed Groq date filter to use range instead of exact match")



        if time_filter != "all_time" and not already_time_filtered:
            # Determine date column based on table
            if "user_uploaded_metrics" in filtered_query.lower() or "user_daily_summary" in filtered_query.lower():
                date_column = "date"
            elif "daily_click_summary" in filtered_query.lower():
                date_column = "date"
            else:
                date_column = "timestamp"
            
            # Build time condition
            time_condition = ""
            if time_filter == "today":
                time_condition = f"{date_column} >= CURRENT_DATE"
            elif time_filter == "yesterday":
                time_condition = f"DATE({date_column}) = CURRENT_DATE - INTERVAL '1 day'"
            elif time_filter == "last_week":
                time_condition = f"{date_column} >= CURRENT_DATE - INTERVAL '7 days'"
            elif time_filter == "last_month":
                time_condition = f"{date_column} >= date_trunc('month', CURRENT_DATE - INTERVAL '1 month') AND {date_column} < date_trunc('month', CURRENT_DATE)"
            elif time_filter == "last_2_months":
                time_condition = f"{date_column} >= CURRENT_DATE - INTERVAL '60 days'"
            elif time_filter == "last_3_months":
                time_condition = f"{date_column} >= CURRENT_DATE - INTERVAL '90 days'"
            elif time_filter == "last_6_months":
                time_condition = f"{date_column} >= CURRENT_DATE - INTERVAL '180 days'"
            elif time_filter == "last_year":
                time_condition = f"{date_column} >= CURRENT_DATE - INTERVAL '365 days'"
            
            # Add time condition to query if specified
            if time_condition:
                query_upper = filtered_query.upper()
                if "WHERE" in query_upper:
                    where_match = re.search(r'\bWHERE\b', filtered_query, re.IGNORECASE)
                    if where_match:
                        where_pos = where_match.end()
                        filtered_query = filtered_query[:where_pos] + f" {time_condition} AND" + filtered_query[where_pos:]
                else:
                    from_match = re.search(r'\bFROM\s+(\w+)', filtered_query, re.IGNORECASE)
                    if from_match:
                        from_end = from_match.end()
                        filtered_query = filtered_query[:from_end] + f" WHERE {time_condition}" + filtered_query[from_end:]
        
        # Debug logging
        print(f"🐛 DEBUG: time_filter = {time_filter}")
        print(f"🐛 DEBUG: Filtered query = {filtered_query[:200]}...")
        
        # Execute query
        with engine.connect() as connection:
            result = connection.execute(text(filtered_query))
            columns = list(result.keys())
            data = [list(row) for row in result.fetchall()]
            row_count = len(data)



        return {
            "data": data,
            "columns": columns,
            "row_count": row_count,
            "query_executed": filtered_query,
            "error": None
        }



    except Exception as e:
        return {
            "data": [],
            "columns": [],
            "row_count": 0,
            "query_executed": query,
            "error": f"Database error: {str(e)}"
        }




def get_database_schema(user_id: str = None) -> str:
    """
    Returns the database schema information.
    
    Big Tech Pattern: Netflix - personalized schema for users with data
    
    Args:
        user_id: Optional user identifier for personalized schema
    
    Returns:
        str: Markdown-formatted schema description
    """
    # Check if user has their own data
    user_schema = get_user_schema(user_id) if user_id else None
    
    if should_use_user_data(user_id) and user_schema and user_schema.get('has_data'):
        # =================================================================
        # 🏢 USER HAS UPLOADED DATA - Show THEIR schema
        # =================================================================
        metric = user_schema['metric_name']
        dimensions = user_schema['dimensions']
        date_range = user_schema['date_range']
        
        dims_list = "\n".join([f"   - {d}" for d in dimensions]) if dimensions else "   - (no dimensions)"
        
        return f"""
DATABASE SCHEMA FOR YOUR UPLOADED DATA:



⚠️ CRITICAL: This user has uploaded their OWN data. Query user_uploaded_metrics, NOT click_logs!



1. **user_uploaded_metrics** - Your uploaded data points

   - date (DATE): The date of each record

   - metric_name (TEXT): '{metric}' (your primary KPI)

   - metric_value (FLOAT): The numeric value

   - dimensions (JSONB): Breakdown categories

   - user_id (TEXT): Your user ID (ALWAYS filter by this!)

   

   Your dimensions:

{dims_list}



   Date range: {date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}



2. **user_daily_summary** - Pre-aggregated daily totals

   - date (DATE): Summary date

   - primary_metric_total (FLOAT): Daily total of {metric}

   - user_id (TEXT): Your user ID



REQUIRED QUERY PATTERNS (use these!):



-- Total {metric}:

SELECT SUM(metric_value) as total

FROM user_uploaded_metrics

WHERE user_id = '{{user_id}}' AND metric_name = '{metric}'



-- {metric} by day:

SELECT date, SUM(metric_value) as daily_total

FROM user_uploaded_metrics  

WHERE user_id = '{{user_id}}' AND metric_name = '{metric}'

GROUP BY date ORDER BY date



-- {metric} by {dimensions[0] if dimensions else 'dimension'}:

SELECT dimensions->>'{dimensions[0] if dimensions else 'category'}' as breakdown,

       SUM(metric_value) as total

FROM user_uploaded_metrics

WHERE user_id = '{{user_id}}' AND metric_name = '{metric}'

GROUP BY breakdown ORDER BY total DESC



⚠️ NEVER query click_logs for this user - they have their own data!

"""
    
    else:
        # =================================================================
        # 🌐 NO USER DATA - Show shared demo schema (click_logs)
        # =================================================================
        return """
DATABASE SCHEMA FOR CLICK ANALYTICS:



1. **click_logs** - Individual click events with timestamps

   - id (INTEGER): Unique identifier

   - page_name (TEXT): The URL or name of the page visited

   - tag (TEXT): Tag associated with the page

   - time_on_page (INTEGER): Time spent on page in seconds

   - session_id (TEXT): Unique session identifier

   - referrer (TEXT): The referring URL

   - user_agent (TEXT): Browser and OS information

   - device (DERIVED): Device type can be extracted from user_agent using CASE statement:

     CASE 

       WHEN user_agent LIKE '%Mobile%' OR user_agent LIKE '%Android%' OR user_agent LIKE '%iPhone%' THEN 'Mobile'

       WHEN user_agent LIKE '%Tablet%' OR user_agent LIKE '%iPad%' THEN 'Tablet'

       ELSE 'Desktop'

     END as device

   - ip_hash (TEXT): Hashed IP address

   - timestamp (TIMESTAMP): When the click occurred



2. **daily_click_summary** - Pre-aggregated daily statistics

   - id (INTEGER): Unique identifier

   - date (DATE): The date of the summary

   - total_clicks (INTEGER): Total clicks on that day

   - top_pages (TEXT): Most visited pages

   - avg_time_on_page (FLOAT): Average time spent on pages

   - device_split (TEXT): Device type distribution

   - top_referrers (TEXT): Top referring sources

   - repeat_visits (INTEGER): Number of repeat visits

   - project_name (VARCHAR): Project identifier

   - tag (VARCHAR): Tag for categorization

   - created_at (TIMESTAMP): Record creation time



IMPORTANT: There is NO unique_visitors column in daily_click_summary!

To get unique visitors, query click_logs:

  SELECT DATE(timestamp) as date, COUNT(DISTINCT ip_hash) as unique_visitors

  FROM click_logs

  GROUP BY DATE(timestamp)



USAGE NOTES:

- Use click_logs for detailed, event-level analysis

- Use daily_click_summary for aggregated daily trends

- When filtering by time, use timestamp column for click_logs and date column for daily_click_summary

- For "yesterday" queries: SELECT COUNT(*) FROM click_logs WHERE DATE(timestamp) = CURRENT_DATE - INTERVAL '1 day'

- For device type queries: SELECT CASE WHEN user_agent LIKE '%Mobile%' OR user_agent LIKE '%Android%' OR user_agent LIKE '%iPhone%' THEN 'Mobile' WHEN user_agent LIKE '%Tablet%' OR user_agent LIKE '%iPad%' THEN 'Tablet' ELSE 'Desktop' END as device, COUNT(*) as count FROM click_logs GROUP BY device ORDER BY count DESC



# =============================================================================
# 🚲 CITIBIKE STAR SCHEMA (TheCommons XR Homework - Task 4)
# =============================================================================
# Data: July & November 2024, First 7 days, Morning rides only (539K rides)
# Question: "How many bikes of each type should be redistributed to each 
#            station every morning in the first week?"
# =============================================================================

CITIBIKE TABLES:

1. **citibike.fact_rides** - Individual bike trips (539,883 morning rides)

   - ride_id (BIGINT): Unique trip identifier

   - start_station_name (TEXT): Pickup station name

   - end_station_name (TEXT): Dropoff station name

   - bike_type_id (INT): FK to dim_bike_type (1=classic, 2=electric)

   - member_type_id (INT): FK to dim_member_type (1=member, 2=casual)

   - date_id (INT): FK to dim_date (format: YYYYMMDD, e.g., 20240701)

   - time_of_day_id (INT): FK to dim_time_of_day (1=Morning for this data)

   - started_at (TIMESTAMP): Trip start time

   - ended_at (TIMESTAMP): Trip end time

2. **citibike.dim_date** - Date dimension

   - date_id (INT): Primary key (YYYYMMDD format)

   - full_date (DATE): The actual date

   - day_num (INT): Day of month (1-7 in this dataset)

   - day_name (TEXT): Monday, Tuesday, etc.

   - month_num (INT): Month number (7=July, 11=November)

   - month_name (TEXT): July, November

   - year (INT): 2024

   - quarter_num (INT): Quarter (3 for July, 4 for November)

   - holiday_name (TEXT): Holiday name if applicable, NULL otherwise

3. **citibike.dim_bike_type** - Bike type dimension

   - bike_type_id (INT): Primary key

   - bike_type_name (TEXT): classic_bike, electric_bike

4. **citibike.dim_member_type** - Member type dimension

   - member_type_id (INT): Primary key

   - member_type_name (TEXT): member, casual

5. **citibike.dim_time_of_day** - Time period dimension

   - time_of_day_id (INT): Primary key

   - time_of_day_name (TEXT): Morning, Afternoon, Evening, Night

   - start_hour (INT): Period start hour

   - end_hour (INT): Period end hour

CITIBIKE QUERY PATTERNS:

-- Count total rides:

SELECT COUNT(*) FROM citibike.fact_rides

-- Rides by month:

SELECT d.month_name, COUNT(*) as rides

FROM citibike.fact_rides f

JOIN citibike.dim_date d ON f.date_id = d.date_id

GROUP BY d.month_name

-- Rides by bike type:

SELECT b.bike_type_name, COUNT(*) as rides

FROM citibike.fact_rides f

JOIN citibike.dim_bike_type b ON f.bike_type_id = b.bike_type_id

GROUP BY b.bike_type_name

-- REDISTRIBUTION CALCULATION (Homework Question):

-- For each station: bikes_needed = arrivals - departures

-- Positive = station needs bikes, Negative = station has excess bikes

-- FAST QUERY (uses UNION ALL, runs in seconds not minutes!)

SELECT 

    d.month_name,

    d.day_name,

    d.day_num,

    b.bike_type_name,

    station_name,

    SUM(CASE WHEN direction = 'arrival' THEN 1 ELSE 0 END) as arrivals,

    SUM(CASE WHEN direction = 'departure' THEN 1 ELSE 0 END) as departures,

    SUM(CASE WHEN direction = 'arrival' THEN 1 ELSE -1 END) as bikes_needed

FROM (

    SELECT date_id, bike_type_id, end_station_name as station_name, 'arrival' as direction

    FROM citibike.fact_rides

    UNION ALL

    SELECT date_id, bike_type_id, start_station_name as station_name, 'departure' as direction

    FROM citibike.fact_rides

) movements

JOIN citibike.dim_date d ON movements.date_id = d.date_id

JOIN citibike.dim_bike_type b ON movements.bike_type_id = b.bike_type_id

WHERE d.month_name IN ('July', 'November')

GROUP BY d.month_name, d.day_name, d.day_num, b.bike_type_name, station_name

ORDER BY d.month_name, d.day_num, b.bike_type_name, bikes_needed DESC

ABOUT:

- Lubo is the creator of this AI analytics agent

- This agent is designed to help analyze website analytics data

- Ask questions about page views, clicks, referrers, and traffic patterns

- For Citibike questions, use the citibike schema tables above

"""
