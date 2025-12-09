"""
Citibike SQL Template System - TheCommons XR Homework
=====================================================
Pre-validated SQL templates for Citibike bike redistribution analysis.
No LLM generation needed - just pattern matching and parameter filling!

Big Tech Pattern: Production-ready template system (Google/Netflix style)

Templates:
1. stations_need_bikes - Top N stations that NEED bikes (highest positive bikes_needed)
2. stations_excess_bikes - Top N stations with EXCESS bikes (most negative bikes_needed)
3. bike_type_comparison - Classic vs Electric bike shortage totals
4. day_of_week_shortage - Which day has biggest total shortage
5. total_rides_summary - Basic ride counts (total, by month, by bike type)
6. busiest_stations - Top N busiest stations by total rides
7. redistribution_by_station_and_type - Bikes needed per station per bike type (morning rush)

Database Schema (citibike star schema):
- citibike.fact_rides: ride_id, start_station_name, end_station_name, bike_type_id, date_id
- citibike.dim_date: date_id, month_name, month_num, day_name, day_num (1-7)
- citibike.dim_bike_type: bike_type_id, bike_type_name (classic_bike, electric_bike)
"""

import re
from typing import Optional, Dict, Any

# =============================================================================
# SQL TEMPLATES
# =============================================================================

STATIONS_NEED_BIKES_TEMPLATE = """
SELECT station_name, SUM(bikes_needed) as total_bikes_needed
FROM (
    SELECT end_station_name as station_name,
           SUM(CASE WHEN direction = 'arrival' THEN 1 ELSE -1 END) as bikes_needed
    FROM (
        SELECT date_id, end_station_name, 'arrival' as direction
        FROM citibike.fact_rides
        UNION ALL
        SELECT date_id, start_station_name, 'departure' as direction
        FROM citibike.fact_rides
    ) movements
    JOIN citibike.dim_date d ON movements.date_id = d.date_id
    WHERE d.month_name = '{month}' AND d.day_num BETWEEN 1 AND 7
    GROUP BY end_station_name
) station_totals
WHERE bikes_needed > 0 AND station_name IS NOT NULL
GROUP BY station_name
ORDER BY total_bikes_needed DESC
LIMIT {limit}
"""

STATIONS_EXCESS_BIKES_TEMPLATE = """
SELECT station_name, ABS(SUM(bikes_needed)) as excess_bikes
FROM (
    SELECT end_station_name as station_name,
           SUM(CASE WHEN direction = 'arrival' THEN 1 ELSE -1 END) as bikes_needed
    FROM (
        SELECT date_id, end_station_name, 'arrival' as direction
        FROM citibike.fact_rides
        UNION ALL
        SELECT date_id, start_station_name, 'departure' as direction
        FROM citibike.fact_rides
    ) movements
    JOIN citibike.dim_date d ON movements.date_id = d.date_id
    WHERE d.month_name = '{month}' AND d.day_num BETWEEN 1 AND 7
    GROUP BY end_station_name
) station_totals
WHERE bikes_needed < 0 AND station_name IS NOT NULL
GROUP BY station_name
ORDER BY excess_bikes DESC
LIMIT {limit}
"""

BIKE_TYPE_COMPARISON_TEMPLATE = """
SELECT 
    b.bike_type_name,
    COUNT(*) as total_rides,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage
FROM citibike.fact_rides f
JOIN citibike.dim_date d ON f.date_id = d.date_id
JOIN citibike.dim_bike_type b ON f.bike_type_id = b.bike_type_id
WHERE d.month_name = '{month}' AND d.day_num BETWEEN 1 AND 7
GROUP BY b.bike_type_name
ORDER BY total_rides DESC
"""

DAY_OF_WEEK_SHORTAGE_TEMPLATE = """
SELECT 
    d.day_name,
    d.day_num,
    SUM(CASE WHEN bikes_needed > 0 THEN bikes_needed ELSE 0 END) as total_shortage
FROM (
    SELECT 
        movements.date_id,
        end_station_name as station_name,
        SUM(CASE WHEN direction = 'arrival' THEN 1 ELSE -1 END) as bikes_needed
    FROM (
        SELECT date_id, end_station_name, 'arrival' as direction
        FROM citibike.fact_rides
        UNION ALL
        SELECT date_id, start_station_name, 'departure' as direction
        FROM citibike.fact_rides
    ) movements
    GROUP BY movements.date_id, end_station_name
) station_daily
JOIN citibike.dim_date d ON station_daily.date_id = d.date_id
WHERE d.month_name = '{month}' AND d.day_num BETWEEN 1 AND 7
GROUP BY d.day_name, d.day_num
ORDER BY total_shortage DESC
"""

TOTAL_RIDES_SUMMARY_TEMPLATE = """
SELECT 
    d.month_name,
    b.bike_type_name,
    COUNT(*) as total_rides
FROM citibike.fact_rides f
JOIN citibike.dim_date d ON f.date_id = d.date_id
JOIN citibike.dim_bike_type b ON f.bike_type_id = b.bike_type_id
WHERE d.month_name = '{month}' AND d.day_num BETWEEN 1 AND 7
GROUP BY d.month_name, b.bike_type_name
ORDER BY d.month_name, b.bike_type_name
"""

BUSIEST_STATIONS_TEMPLATE = """
SELECT 
    station_name,
    SUM(total_rides) as total_rides
FROM (
    SELECT start_station_name as station_name, COUNT(*) as total_rides
    FROM citibike.fact_rides f
    JOIN citibike.dim_date d ON f.date_id = d.date_id
    WHERE d.month_name = '{month}' AND d.day_num BETWEEN 1 AND 7
    GROUP BY start_station_name
    
    UNION ALL
    
    SELECT end_station_name as station_name, COUNT(*) as total_rides
    FROM citibike.fact_rides f
    JOIN citibike.dim_date d ON f.date_id = d.date_id
    WHERE d.month_name = '{month}' AND d.day_num BETWEEN 1 AND 7
    GROUP BY end_station_name
) combined
GROUP BY station_name
ORDER BY total_rides DESC
LIMIT {limit}
"""

# Big Tech Pattern: Multi-month support with TOP N per month (window function)
REDISTRIBUTION_BY_STATION_AND_TYPE_TEMPLATE = """
SELECT month_name, station_name, bike_type_name, total_bikes_needed
FROM (
    SELECT 
        d.month_name,
        station_name,
        bike_type_name,
        SUM(bikes_needed) as total_bikes_needed,
        ROW_NUMBER() OVER (PARTITION BY d.month_name ORDER BY SUM(bikes_needed) DESC) as row_num
    FROM (
        SELECT 
            movements.date_id,
            end_station_name as station_name,
            b.bike_type_name,
            SUM(CASE WHEN direction = 'arrival' THEN 1 ELSE -1 END) as bikes_needed
        FROM (
            SELECT date_id, time_of_day_id, bike_type_id, end_station_name, 'arrival' as direction
            FROM citibike.fact_rides
            UNION ALL
            SELECT date_id, time_of_day_id, bike_type_id, start_station_name, 'departure' as direction
            FROM citibike.fact_rides
        ) movements
        JOIN citibike.dim_date d ON movements.date_id = d.date_id
        JOIN citibike.dim_bike_type b ON movements.bike_type_id = b.bike_type_id
        JOIN citibike.dim_time_of_day t ON movements.time_of_day_id = t.time_of_day_id
        WHERE d.month_name IN ({months}) 
          AND d.day_num BETWEEN 1 AND 7
          AND t.time_of_day_name = 'Morning'
        GROUP BY movements.date_id, end_station_name, b.bike_type_name
    ) station_type_totals
    JOIN citibike.dim_date d ON station_type_totals.date_id = d.date_id
    WHERE bikes_needed > 0 AND station_name IS NOT NULL
    GROUP BY d.month_name, station_name, bike_type_name
) ranked
WHERE row_num <= {limit_per_month}
ORDER BY month_name, row_num
"""

# =============================================================================
# KEYWORD PATTERNS FOR TEMPLATE MATCHING
# =============================================================================

CITIBIKE_TEMPLATE_PATTERNS = {
    'stations_need_bikes': {
        'keywords': ['need bikes', 'needs bikes', 'redistribute', 'stations need', 'bikes needed', 'more bikes', 'station shortage', 'which stations need'],
        'required': None,
        'exclude': ['excess', 'surplus', 'too many', 'take from'],
        'priority': 10
    },
    
    'stations_excess_bikes': {
        'keywords': ['excess', 'surplus', 'too many', 'take from', 'redistribute from', 'stations with excess', 'stations that have too many'],
        'required': None,
        'exclude': ['need', 'require', 'short'],
        'priority': 9
    },
    
    'bike_type_comparison': {
        'keywords': ['classic', 'electric', 'bike type', 'compare', 'breakdown', 'by bike type', 'classic vs electric', 'electric vs classic'],
        'required': None,
        'exclude': ['station', 'stations', 'day', 'total rides'],
        'priority': 8
    },
    
    'day_of_week_shortage': {
        'keywords': ['day of week', 'weekday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'which day', 'busiest day', 'worst day', 'best day', 'day shortage', 'biggest shortage day', 'most shortage', 'shortage by day'],
        'required': None,
        'exclude': ['station', 'stations', 'bike type'],
        'priority': 7
    },
    
    'total_rides_summary': {
        'keywords': ['total rides', 'how many rides', 'ride count', 'summary', 'total', 'count'],
        'required': None,
        'exclude': ['station', 'stations', 'by', 'breakdown'],
        'priority': 6
    },
    
    'busiest_stations': {
        'keywords': ['busiest', 'most popular', 'highest traffic', 'most used', 'most activity', 'top stations by rides', 'popular stations'],
        'required': None,
        'exclude': ['need', 'excess', 'shortage', 'redistribute'],
        'priority': 8
    },
    
    'redistribution_by_station_and_type': {
        'keywords': ['redistribute', 'redistribution', 'each type', 'by type', 'bike type', 
                     'morning', 'rush hour', 'rush hours', 'minimize shortage', 
                     'each station', 'per station', 'every morning', 'first week'],
        'required': None,
        'exclude': [],
        'priority': 11  # Highest priority - most specific template
    }
}

# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================

CITIBIKE_SQL_TEMPLATES = {
    'stations_need_bikes': {
        'sql': STATIONS_NEED_BIKES_TEMPLATE,
        'params': ['month', 'limit'],
        'description': 'Top N stations that need bikes (aggregated total)'
    },
    
    'stations_excess_bikes': {
        'sql': STATIONS_EXCESS_BIKES_TEMPLATE,
        'params': ['month', 'limit'],
        'description': 'Top N stations with excess bikes to redistribute FROM'
    },
    
    'bike_type_comparison': {
        'sql': BIKE_TYPE_COMPARISON_TEMPLATE,
        'params': ['month'],
        'description': 'Classic vs Electric bike shortage comparison'
    },
    
    'day_of_week_shortage': {
        'sql': DAY_OF_WEEK_SHORTAGE_TEMPLATE,
        'params': ['month'],
        'description': 'Which day of week has biggest shortage'
    },
    
    'total_rides_summary': {
        'sql': TOTAL_RIDES_SUMMARY_TEMPLATE,
        'params': ['month'],
        'description': 'Total ride counts by month and bike type'
    },
    
    'busiest_stations': {
        'sql': BUSIEST_STATIONS_TEMPLATE,
        'params': ['month', 'limit'],
        'description': 'Top N busiest stations by total rides (arrivals + departures)'
    },
    
    'redistribution_by_station_and_type': {
        'sql': REDISTRIBUTION_BY_STATION_AND_TYPE_TEMPLATE,
        'params': ['months', 'limit_per_month'],
        'description': 'Bikes needed per station per bike type (morning rush, first week, multi-month support)'
    }
}

# =============================================================================
# TEMPLATE MATCHER
# =============================================================================

def match_citibike_template(user_question: str) -> Optional[Dict[str, Any]]:
    """
    Match user query to best Citibike template.
    
    Big Tech Pattern: Fast pattern matching before LLM fallback
    
    Args:
        user_question: User's natural language query
        
    Returns:
        dict: {'name': str, 'confidence': float} or None
    """
    query_lower = user_question.lower()
    matches = {}
    
    for template_name, pattern in CITIBIKE_TEMPLATE_PATTERNS.items():
        score = 0
        
        # Check excludes first
        if 'exclude' in pattern:
            if any(excl in query_lower for excl in pattern['exclude']):
                continue
        
        # Check required keywords
        if 'required' in pattern and pattern['required']:
            if not any(req in query_lower for req in pattern['required']):
                continue
        
        # Score by keyword matches
        for keyword in pattern['keywords']:
            if keyword in query_lower:
                score += 1
        
        if score > 0:
            matches[template_name] = {
                'score': score,
                'priority': pattern['priority'],
                'confidence': min(0.95, 0.7 + (score * 0.05))
            }
    
    if not matches:
        return None
    
    # Sort by priority then score
    sorted_matches = sorted(
        matches.items(),
        key=lambda x: (x[1]['priority'], x[1]['score']),
        reverse=True
    )
    
    best = sorted_matches[0]
    print(f"🚲 Citibike template matched: {best[0]} (confidence: {best[1]['confidence']:.2f})")
    
    return {
        'name': best[0],
        'confidence': best[1]['confidence']
    }

# =============================================================================
# PARAMETER EXTRACTOR
# =============================================================================

def extract_citibike_parameters(user_question: str, template_name: str) -> Dict[str, Any]:
    """
    Extract parameters from user query for Citibike SQL template.
    
    Args:
        user_question: User's natural language query
        template_name: Matched template name
        
    Returns:
        dict: Parameters for template filling
    """
    params = {}
    query_lower = user_question.lower()
    
    # Big Tech Pattern: Extract ALL mentioned months (supports "July AND November")
    months_found = []
    if 'july' in query_lower or 'jul' in query_lower:
        months_found.append('July')
    if 'november' in query_lower or 'nov' in query_lower:
        months_found.append('November')
    
    # Default to July if no month specified
    if not months_found:
        months_found = ['July']
    
    # Format for SQL IN clause: 'July', 'November' 
    params['months'] = "'" + "', '".join(months_found) + "'"
    params['month'] = months_found[0]  # Backward compatibility for other templates
    
    # Extract LIMIT (for templates that need it)
    if template_name in ['stations_need_bikes', 'stations_excess_bikes', 'busiest_stations', 'redistribution_by_station_and_type']:
        limit_patterns = [
            r'\b(?:top|first|show)\s+(\d+)\b',
            r'\b(\d+)\s+(?:stations?|top)\b'
        ]
        
        limit_found = False
        for pattern in limit_patterns:
            limit_match = re.search(pattern, query_lower)
            if limit_match:
                limit_value = int(limit_match.group(1))
                if 1 <= limit_value <= 100:
                    if template_name == 'redistribution_by_station_and_type':
                        params['limit_per_month'] = limit_value
                    else:
                        params['limit'] = limit_value
                    limit_found = True
                    break
        
        if not limit_found:
            if template_name == 'redistribution_by_station_and_type':
                # Big Tech Pattern: Top 10 per month (window function handles multi-month)
                params['limit_per_month'] = 10
            else:
                # Big Tech Pattern: Scale limit by month count (10 per month)
                params['limit'] = 10 * len(months_found)
    else:
        # Other templates don't need limit
        params['limit'] = None
    
    return params

# =============================================================================
# MAIN FUNCTION (Called from adalflow_agent.py)
# =============================================================================

def get_citibike_sql_from_template(user_question: str) -> Optional[Dict[str, Any]]:
    """
    Generate SQL from Citibike template if pattern matches.
    
    Big Tech Pattern: Fast template matching before LLM fallback
    
    Args:
        user_question: User's natural language query
        
    Returns:
        dict: {'sql': str, 'template_name': str, 'confidence': float} or None
    """
    try:
        # Match template
        match = match_citibike_template(user_question)
        
        if not match:
            print("🚲 No Citibike template matched - using LLM fallback")
            return None
        
        template_name = match['name']
        confidence = match['confidence']
        
        # Minimum confidence threshold
        MIN_CONFIDENCE_THRESHOLD = 0.6
        if confidence < MIN_CONFIDENCE_THRESHOLD:
            print(f"⚠️ Citibike template '{template_name}' confidence {confidence:.2f} below threshold")
            return None
        
        # Check template exists
        if template_name not in CITIBIKE_SQL_TEMPLATES:
            print(f"⚠️ Citibike template '{template_name}' not found in registry")
            return None
        
        # Extract parameters
        params = extract_citibike_parameters(user_question, template_name)
        
        # Get template SQL
        template_info = CITIBIKE_SQL_TEMPLATES[template_name]
        sql_template = template_info['sql']
        
        # Fill parameters
        sql_query = sql_template.strip()
        for param_name, param_value in params.items():
            if param_value is None:
                continue
            placeholder = '{' + param_name + '}'
            sql_query = sql_query.replace(placeholder, str(param_value))
        
        print(f"✅ Generated Citibike SQL from template: {template_name}")
        print(f"   Month: {params.get('month')}")
        if params.get('limit'):
            print(f"   Limit: {params.get('limit')}")
        print(f"   SQL: {sql_query[:100]}...")
        
        return {
            'sql': sql_query,
            'template_name': template_name,
            'confidence': confidence,
            'is_citibike': True
        }
        
    except Exception as e:
        # Big Tech Pattern: Graceful error handling - never crash
        print(f"⚠️ Citibike template generation failed: {e}")
        return None

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_citibike_templates():
    """List all available Citibike SQL templates."""
    print("\n=== CITIBIKE SQL TEMPLATES ===\n")
    for name, info in CITIBIKE_SQL_TEMPLATES.items():
        print(f"{name}: {info['description']}")

def test_citibike_template(user_question: str) -> None:
    """Test Citibike SQL template matching and generation."""
    print(f"\n🧪 Testing: {user_question}")
    print("=" * 60)
    
    result = get_citibike_sql_from_template(user_question)
    
    if result:
        print(f"\n✅ Generated SQL ({result.get('template_name')}):")
        print(result['sql'])
        print(f"\nConfidence: {result.get('confidence', 0):.2f}")
    else:
        print("\n❌ No template match - would use LLM")

# =============================================================================
# TEST CASES
# =============================================================================

if __name__ == "__main__":
    test_cases = [
        # Template 1: stations_need_bikes
        "Top 10 stations that need bikes in July",
        "Which stations need the most bikes in November?",
        "Stations requiring bike redistribution in July",
        
        # Template 2: stations_excess_bikes  
        "Which stations have excess bikes in November?",
        "Top 5 stations with surplus bikes in July",
        "Stations to take bikes from in November",
        
        # Template 3: bike_type_comparison
        "Compare classic vs electric bikes in July",
        "Electric vs classic bike breakdown November",
        "What's the split between bike types in July?",
        
        # Template 4: day_of_week_shortage
        "Which day has the biggest shortage?",
        "What day of the week needs most bikes in July?",
        "Worst day for bike availability November",
        
        # Template 5: total_rides_summary
        "How many total rides in November?",
        "Total ride count for July first week",
        "Summary of rides in November",
        
        # Template 6: busiest_stations
        "What are the busiest stations in July?",
        "Top 10 most popular stations November",
        "Highest traffic stations in July",
        
        # Template 7: redistribution_by_station_and_type
        "For July and November, how many bikes of each type should be redistributed to each station every morning in the first week?",
        "Redistribution by station and bike type for July morning rush",
        "How many classic and electric bikes needed per station in November?",
    ]
    
    for query in test_cases:
        test_citibike_template(query)
        print()

