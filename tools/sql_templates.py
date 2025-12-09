"""
SQL Template System - Production-Ready SQL Queries (Multi-Tenant)
==================================================================
Pre-validated SQL templates for common analytics patterns.
No LLM generation needed - just pattern matching and parameter filling!

Big Tech AGI: Now supports user-uploaded data (Netflix/Spotify pattern)

Templates:
1. top_n_pages_over_time - Multi-page line chart
2. total_clicks - Simple click count
3. device_breakdown - Clicks by device type
4. top_pages - Page ranking
5. single_page_trend - One page over time
6. referrer_breakdown - Traffic sources
7. compare_two_pages - Two specific pages over time
8. time_on_page_by_page - Engagement metrics
9. unique_visitors - Unique user count
10. clicks_by_hour - Hourly pattern

USER TEMPLATES (for uploaded data):
11. user_total_metric - Total of user's metric
12. user_metric_over_time - User metric trend
13. user_metric_by_dimension - Breakdown by dimension
14. user_top_dimension - Top values in dimension
15. user_daily_summary - Recent daily totals
"""

import re
from typing import Optional, Dict, Any
from datetime import datetime

# Import template learning (safe - returns 1.0 if fails)
try:
    from tools.template_learning import get_template_weight
except ImportError:
    def get_template_weight(template_name: str) -> float:
        return 1.0

# Import user schema detection (safe - returns None if fails)
try:
    from tools.sql_tools import get_user_schema, should_use_user_data
except ImportError:
    def get_user_schema(user_id: str):
        return None
    def should_use_user_data(user_id: str):
        return False

# =============================================================================
# 🧠 AGI PARAMETER TRACKING (Big Tech Pattern - Netflix/Spotify)
# =============================================================================

def track_parameter_usage(param_name: str, param_value: Any, source: str, confidence: float = None) -> dict:
    """
    🧠 AGI LEARNING: Track parameter usage for preference learning.
    """
    return {
        'name': param_name,
        'value': str(param_value),
        'source': source,
        'confidence': confidence,
        'timestamp': datetime.now().isoformat()
    }

# Import few-shot prompt builder (safe - returns "" if fails)
try:
    from tools.template_learning import build_few_shot_prompt
except ImportError:
    def build_few_shot_prompt(template_name: str, limit: int = 3) -> str:
        return ""

# =============================================================================
# DATABASE SCHEMA CONSTANTS
# =============================================================================

DATABASE_INFO = """
TABLES:
- click_logs: timestamp, page_name, ip_hash, user_agent, referrer, time_on_page
- daily_click_summary: date, total_clicks, avg_time_on_page, device_split, top_referrers
- user_uploaded_metrics: date, metric_name, metric_value, dimensions, user_id
- user_daily_summary: date, primary_metric_total, user_id
"""

# =============================================================================
# STANDARD TEMPLATES (for click_logs - demo data)
# =============================================================================

TOP_N_PAGES_TEMPLATE = """
SELECT DATE(timestamp) as date, page_name, COUNT(*) as count
FROM click_logs
WHERE page_name IN (
    SELECT page_name 
    FROM click_logs 
    GROUP BY page_name 
    ORDER BY COUNT(*) DESC 
    LIMIT {limit}
)
GROUP BY date, page_name
ORDER BY date, page_name
"""

TOTAL_CLICKS_TEMPLATE = """
SELECT COUNT(*) as total_clicks
FROM click_logs
"""

DEVICE_BREAKDOWN_TEMPLATE = """
SELECT 
    CASE 
        WHEN user_agent LIKE '%Mobile%' OR user_agent LIKE '%Android%' OR user_agent LIKE '%iPhone%' THEN 'Mobile'
        WHEN user_agent LIKE '%Tablet%' OR user_agent LIKE '%iPad%' THEN 'Tablet'
        ELSE 'Desktop'
    END as device,
    COUNT(*) as count
FROM click_logs
GROUP BY device
ORDER BY count DESC
"""

TOP_PAGES_TEMPLATE = """
SELECT page_name, COUNT(*) as clicks
FROM click_logs
GROUP BY page_name
ORDER BY clicks DESC
LIMIT {limit}
"""

SINGLE_PAGE_TREND_TEMPLATE = """
SELECT DATE(timestamp) as date, COUNT(*) as count
FROM click_logs
WHERE page_name = '{page_name}'
GROUP BY date
ORDER BY date
"""

REFERRER_BREAKDOWN_TEMPLATE = """
SELECT referrer, COUNT(*) as visits
FROM click_logs
WHERE referrer IS NOT NULL AND referrer != ''
GROUP BY referrer
ORDER BY visits DESC
LIMIT {limit}
"""

COMPARE_TWO_PAGES_TEMPLATE = """
SELECT DATE(timestamp) as date, page_name, COUNT(*) as count
FROM click_logs
WHERE page_name IN ('{page1}', '{page2}')
GROUP BY date, page_name
ORDER BY date, page_name
"""

TIME_ON_PAGE_TEMPLATE = """
SELECT page_name, AVG(time_on_page) as avg_time, COUNT(*) as visits
FROM click_logs
WHERE time_on_page IS NOT NULL AND time_on_page > 0
GROUP BY page_name
ORDER BY avg_time DESC
LIMIT {limit}
"""

UNIQUE_VISITORS_TEMPLATE = """
SELECT DATE(timestamp) as date, COUNT(DISTINCT ip_hash) as unique_visitors
FROM click_logs
GROUP BY date
ORDER BY date
"""

CLICKS_BY_HOUR_TEMPLATE = """
SELECT EXTRACT(HOUR FROM timestamp) as hour, COUNT(*) as clicks
FROM click_logs
GROUP BY hour
ORDER BY hour
"""

# =============================================================================
# 🏢 USER TEMPLATES (for user_uploaded_metrics - their data)
# =============================================================================

USER_TOTAL_METRIC_TEMPLATE = """
SELECT SUM(metric_value) as total_{metric_name}
FROM user_uploaded_metrics
WHERE user_id = '{user_id}' AND metric_name = '{metric_name}'
"""

USER_METRIC_OVER_TIME_TEMPLATE = """
SELECT date, SUM(metric_value) as daily_{metric_name}
FROM user_uploaded_metrics
WHERE user_id = '{user_id}' AND metric_name = '{metric_name}'
GROUP BY date
ORDER BY date
"""

USER_METRIC_BY_DIMENSION_TEMPLATE = """
SELECT dimensions->>'{dimension}' as {dimension}, SUM(metric_value) as total
FROM user_uploaded_metrics
WHERE user_id = '{user_id}' 
  AND metric_name = '{metric_name}'
  AND dimensions->>'{dimension}' IS NOT NULL
GROUP BY dimensions->>'{dimension}'
ORDER BY total DESC
LIMIT {limit}
"""

USER_TOP_DIMENSION_TEMPLATE = """
SELECT dimensions->>'{dimension}' as {dimension}, SUM(metric_value) as total_{metric_name}
FROM user_uploaded_metrics
WHERE user_id = '{user_id}' 
  AND metric_name = '{metric_name}'
  AND dimensions->>'{dimension}' IS NOT NULL
GROUP BY dimensions->>'{dimension}'
ORDER BY total_{metric_name} DESC
LIMIT {limit}
"""

USER_DAILY_SUMMARY_TEMPLATE = """
SELECT date, primary_metric_total as {metric_name}
FROM user_daily_summary
WHERE user_id = '{user_id}'
ORDER BY date DESC
LIMIT {limit}
"""

USER_COMPARE_DIMENSIONS_TEMPLATE = """
SELECT date, dimensions->>'{dimension}' as {dimension}, SUM(metric_value) as value
FROM user_uploaded_metrics
WHERE user_id = '{user_id}' 
  AND metric_name = '{metric_name}'
  AND dimensions->>'{dimension}' IN ('{value1}', '{value2}')
GROUP BY date, dimensions->>'{dimension}'
ORDER BY date, {dimension}
"""

# =============================================================================
# KEYWORD PATTERNS FOR TEMPLATE MATCHING (Standard)
# =============================================================================

TEMPLATE_PATTERNS = {
    'top_n_pages_over_time': {
        'keywords': ['top', 'pages', 'over time', 'most visited', 'line chart', 'line graph', 'trend', 'popular pages'],
        'required': None,
        'multi_keywords': ['top', 'pages', 'most', 'visited', 'popular'],
        'require_one_of': [['over time'], ['line chart'], ['line graph'], ['trend']],
        'priority': 10
    },
    
    'compare_two_pages': {
        'keywords': ['compare', 'versus', 'vs', 'and', 'pages over time'],
        'required': ['compare'],
        'priority': 9
    },
    
    'single_page_trend': {
        'keywords': ['page', 'over time', 'trend', 'timeline'],
        'required': ['over time'],
        'exclude': ['top', 'pages', 'all', 'compare', 'unique'],
        'priority': 8
    },
    
    'device_breakdown': {
        'keywords': ['device', 'mobile', 'desktop', 'tablet', 'by device'],
        'required': ['device'],
        'priority': 7
    },
    
    'referrer_breakdown': {
        'keywords': ['referrer', 'traffic source', 'where', 'from where', 'source'],
        'priority': 6
    },
    
    'time_on_page': {
        'keywords': ['time on page', 'time spent', 'engagement', 'duration'],
        'required': ['time'],
        'priority': 5
    },
    
    'unique_visitors': {
        'keywords': ['unique', 'visitors', 'users', 'distinct', 'over time', 'trend'],
        'required': ['unique'],
        'priority': 4
    },
    
    'clicks_by_hour': {
        'keywords': ['hour', 'hourly', 'by hour', 'time of day'],
        'required': ['hour'],
        'priority': 3
    },
    
    'top_pages': {
        'keywords': ['top', 'pages', 'most', 'popular', 'ranking'],
        'exclude': ['over time', 'trend', 'line chart', 'line graph', 'timeline'],
        'priority': 2
    },
    
    'total_clicks': {
        'keywords': ['total', 'how many', 'count', 'clicks'],
        'exclude': ['by', 'per', 'each', 'page', 'device', 'referrer'],
        'priority': 1
    }
}

# =============================================================================
# 🏢 USER TEMPLATE PATTERNS (for user-uploaded data)
# =============================================================================

USER_TEMPLATE_PATTERNS = {
    'user_total_metric': {
        'keywords': ['total', 'how many', 'sum', 'count', 'overall', 'all time'],
        'exclude': ['by', 'per', 'breakdown', 'over time', 'trend', 'daily'],
        'priority': 10
    },
    
    'user_metric_over_time': {
        'keywords': ['over time', 'trend', 'daily', 'by day', 'by date', 'timeline', 'line chart', 'history'],
        'exclude': ['by category', 'by product', 'breakdown by'],
        'priority': 9
    },
    
    'user_metric_by_dimension': {
        'keywords': ['by', 'per', 'breakdown', 'split', 'grouped', 'category', 'product', 'segment'],
        'exclude': ['over time', 'trend', 'daily'],
        'priority': 8
    },
    
    'user_top_dimension': {
        'keywords': ['top', 'best', 'highest', 'most', 'ranking', 'leading'],
        'exclude': ['over time', 'trend'],
        'priority': 7
    },
    
    'user_daily_summary': {
        'keywords': ['recent', 'latest', 'last few days', 'summary', 'last week'],
        'priority': 6
    },
    
    'user_compare_dimensions': {
        'keywords': ['compare', 'versus', 'vs', 'between'],
        'priority': 5
    }
}

# =============================================================================
# TEMPLATE REGISTRIES
# =============================================================================

SQL_TEMPLATES = {
    'top_n_pages_over_time': {
        'sql': TOP_N_PAGES_TEMPLATE,
        'params': ['limit'],
        'description': 'Top N pages over time (multi-series line chart)'
    },
    'total_clicks': {
        'sql': TOTAL_CLICKS_TEMPLATE,
        'params': [],
        'description': 'Total click count'
    },
    'device_breakdown': {
        'sql': DEVICE_BREAKDOWN_TEMPLATE,
        'params': [],
        'description': 'Clicks by device type'
    },
    'top_pages': {
        'sql': TOP_PAGES_TEMPLATE,
        'params': ['limit'],
        'description': 'Top N pages by clicks'
    },
    'single_page_trend': {
        'sql': SINGLE_PAGE_TREND_TEMPLATE,
        'params': ['page_name'],
        'description': 'Single page trend over time'
    },
    'referrer_breakdown': {
        'sql': REFERRER_BREAKDOWN_TEMPLATE,
        'params': ['limit'],
        'description': 'Clicks by referrer'
    },
    'compare_two_pages': {
        'sql': COMPARE_TWO_PAGES_TEMPLATE,
        'params': ['page1', 'page2'],
        'description': 'Compare two pages over time'
    },
    'time_on_page': {
        'sql': TIME_ON_PAGE_TEMPLATE,
        'params': ['limit'],
        'description': 'Average time on page by page'
    },
    'unique_visitors': {
        'sql': UNIQUE_VISITORS_TEMPLATE,
        'params': [],
        'description': 'Unique visitors over time'
    },
    'clicks_by_hour': {
        'sql': CLICKS_BY_HOUR_TEMPLATE,
        'params': [],
        'description': 'Clicks by hour of day'
    }
}

USER_SQL_TEMPLATES = {
    'user_total_metric': {
        'sql': USER_TOTAL_METRIC_TEMPLATE,
        'params': ['user_id', 'metric_name'],
        'description': 'Total sum of user metric'
    },
    'user_metric_over_time': {
        'sql': USER_METRIC_OVER_TIME_TEMPLATE,
        'params': ['user_id', 'metric_name'],
        'description': 'User metric trend over time'
    },
    'user_metric_by_dimension': {
        'sql': USER_METRIC_BY_DIMENSION_TEMPLATE,
        'params': ['user_id', 'metric_name', 'dimension', 'limit'],
        'description': 'User metric broken down by dimension'
    },
    'user_top_dimension': {
        'sql': USER_TOP_DIMENSION_TEMPLATE,
        'params': ['user_id', 'metric_name', 'dimension', 'limit'],
        'description': 'Top dimension values by user metric'
    },
    'user_daily_summary': {
        'sql': USER_DAILY_SUMMARY_TEMPLATE,
        'params': ['user_id', 'metric_name', 'limit'],
        'description': 'Recent daily summaries'
    },
    'user_compare_dimensions': {
        'sql': USER_COMPARE_DIMENSIONS_TEMPLATE,
        'params': ['user_id', 'metric_name', 'dimension', 'value1', 'value2'],
        'description': 'Compare two dimension values over time'
    }
}

# =============================================================================
# STANDARD TEMPLATE MATCHER
# =============================================================================

def match_sql_template(user_question: str) -> Optional[Dict[str, Any]]:
    """
    Match user query to best SQL template using keywords.
    """
    query_lower = user_question.lower()
    matches = {}
    
    for template_name, pattern in TEMPLATE_PATTERNS.items():
        score = 0
        
        if 'required' in pattern and pattern['required'] is not None:
            has_required = all(req in query_lower for req in pattern['required'])
            if not has_required:
                continue
        
        if 'exclude' in pattern:
            has_excluded = any(excl in query_lower for excl in pattern['exclude'])
            if has_excluded:
                continue
        
        if 'multi_keywords' in pattern:
            has_multi = any(kw in query_lower for kw in pattern['multi_keywords'])
            if not has_multi:
                continue
        
        if 'require_one_of' in pattern:
            has_required_group = any(
                all(kw in query_lower for kw in group)
                for group in pattern['require_one_of']
            )
            if not has_required_group:
                continue
        
        for keyword in pattern['keywords']:
            if keyword in query_lower:
                score += 1
        
        if score > 0:
            matches[template_name] = {
                'score': score,
                'priority': pattern['priority'],
                'confidence': min(0.95, 0.7 + (score * 0.05)) * get_template_weight(template_name)
            }
    
    if not matches:
        return None
    
    sorted_matches = sorted(
        matches.items(),
        key=lambda x: (x[1]['priority'], x[1]['score']),
        reverse=True
    )
    
    best_match = sorted_matches[0]
    template_name = best_match[0]
    confidence = best_match[1]['confidence']
    
    print(f"🎯 SQL Template matched: {template_name} (confidence: {confidence:.2f})")
    return {
        'name': template_name,
        'confidence': confidence
    }

# =============================================================================
# 🏢 USER TEMPLATE MATCHER
# =============================================================================

def match_user_template(user_question: str, user_schema: dict) -> Optional[Dict[str, Any]]:
    """
    Match user query to best USER template.
    
    Big Tech Pattern: Netflix - personalized query understanding
    """
    if not user_schema or not user_schema.get('has_data'):
        return None
    
    query_lower = user_question.lower()
    matches = {}
    
    for template_name, pattern in USER_TEMPLATE_PATTERNS.items():
        score = 0
        
        if 'exclude' in pattern:
            if any(excl in query_lower for excl in pattern['exclude']):
                continue
        
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
    
    sorted_matches = sorted(
        matches.items(),
        key=lambda x: (x[1]['priority'], x[1]['score']),
        reverse=True
    )
    
    best = sorted_matches[0]
    print(f"🏢 User template matched: {best[0]} (confidence: {best[1]['confidence']:.2f})")
    
    return {
        'name': best[0],
        'confidence': best[1]['confidence']
    }

# =============================================================================
# PARAMETER EXTRACTORS
# =============================================================================

def extract_sql_parameters(user_question: str, template_name: str, user_prefs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract parameters from user query for SQL template.
    """
    params = {}
    parameters_used = []
    query_lower = user_question.lower()
    
    # Extract LIMIT
    limit_pattern = r'\b(?:top|first|show|compare)\s+(\d+)\b'
    limit_match = re.search(limit_pattern, query_lower)
    
    if limit_match:
        limit_value = int(limit_match.group(1))
        params['limit'] = limit_value
        parameters_used.append(track_parameter_usage('limit', limit_value, 'user_explicit', None))
    else:
        try:
            if user_prefs and isinstance(user_prefs, dict) and user_prefs.get('default_limit'):
                learned_limit = int(user_prefs['default_limit'])
                if 1 <= learned_limit <= 100:
                    params['limit'] = learned_limit
                    parameters_used.append(track_parameter_usage('limit', learned_limit, 'learned', user_prefs.get('overall_confidence', 0.5)))
                else:
                    raise ValueError(f"Invalid learned limit: {learned_limit}")
            else:
                raise ValueError("No valid user_prefs")
        except Exception:
            params['limit'] = 10
            parameters_used.append(track_parameter_usage('limit', 10, 'default', None))
    
    # Extract page names for compare_two_pages
    if template_name == 'compare_two_pages':
        compare_patterns = [
            r'compare\s+([^\s]+)\s+(?:and|vs|versus)\s+([^\s]+)',
            r'([^\s]+)\s+(?:vs|versus)\s+([^\s]+)',
        ]
        pages_found = False
        for pattern in compare_patterns:
            match = re.search(pattern, query_lower)
            if match:
                params['page1'] = match.group(1).strip()
                params['page2'] = match.group(2).strip()
                parameters_used.append(track_parameter_usage('page1', params['page1'], 'user_explicit', None))
                parameters_used.append(track_parameter_usage('page2', params['page2'], 'user_explicit', None))
                pages_found = True
                break
        
        if not pages_found:
            params['page1'] = 'home'
            params['page2'] = 'portfolio'
            parameters_used.append(track_parameter_usage('page1', 'home', 'default', None))
            parameters_used.append(track_parameter_usage('page2', 'portfolio', 'default', None))
    
    # Extract single page name
    if template_name == 'single_page_trend':
        page_patterns = [
            r'(?:show|for|of)\s+([a-z0-9/_-]+)\s+(?:page|over)',
            r'([a-z0-9/_-]+)\s+(?:page\s+)?over\s+time',
        ]
        page_found = False
        for pattern in page_patterns:
            match = re.search(pattern, query_lower)
            if match:
                params['page_name'] = match.group(1).strip()
                parameters_used.append(track_parameter_usage('page_name', params['page_name'], 'user_explicit', None))
                page_found = True
                break
        
        if not page_found:
            params['page_name'] = 'home'
            parameters_used.append(track_parameter_usage('page_name', 'home', 'default', None))
    
    parameters_used.append(track_parameter_usage('template', template_name, 'matched', None))
    params['_meta'] = {'parameters_used': parameters_used}
    
    return params





def extract_user_parameters(user_question: str, template_name: str, user_schema: dict) -> Dict[str, Any]:
    """
    Extract parameters for USER template.
    """
    params = {
        'metric_name': user_schema.get('metric_name', 'value'),
        'limit': 10
    }
    
    query_lower = user_question.lower()
    
    # Extract limit
    limit_match = re.search(r'\b(?:top|first|show)\s+(\d+)\b', query_lower)
    if limit_match:
        params['limit'] = int(limit_match.group(1))
    
    # Extract dimension (try to match from user's available dimensions)
    dimensions = user_schema.get('dimensions', [])
    for dim in dimensions:
        dim_lower = dim.lower().replace('_', ' ')
        if dim_lower in query_lower or dim.lower() in query_lower:
            params['dimension'] = dim.lower().replace(' ', '_')
            break
    
    # Default to first dimension if not found
    if 'dimension' not in params and dimensions:
        params['dimension'] = dimensions[0].lower().replace(' ', '_')
    
    # For compare template, try to extract values
    if template_name == 'user_compare_dimensions':
        compare_match = re.search(r'compare\s+([^\s]+)\s+(?:and|vs|versus)\s+([^\s]+)', query_lower)
        if compare_match:
            params['value1'] = compare_match.group(1).strip()
            params['value2'] = compare_match.group(2).strip()
        else:
            params['value1'] = 'value1'
            params['value2'] = 'value2'
    
    return params

# =============================================================================
# VALIDATION CACHE
# =============================================================================

_validation_cache = {}

def _get_cache_key(template_name: str, user_question: str) -> str:
    normalized_query = user_question.lower().strip()
    return f"{template_name}:{normalized_query}"

def clear_validation_cache():
    global _validation_cache
    _validation_cache.clear()
    print("🧹 Validation cache cleared")

# =============================================================================
# SEMANTIC VALIDATION
# =============================================================================

def validate_template_for_request(template_name: str, user_question: str, model_provider: str = "ollama") -> bool:
    """
    LLM validates: Does this template make sense for the user's request?
    """
    cache_key = _get_cache_key(template_name, user_question)
    if cache_key in _validation_cache:
        cached_result = _validation_cache[cache_key]
        print(f"⚡ Validation cache HIT: {template_name} → {cached_result}")
        return cached_result
    
    print(f"💾 Validation cache MISS: Running LLM validation...")
    
    if template_name not in SQL_TEMPLATES:
        _validation_cache[cache_key] = False
        return False
    
    template_info = SQL_TEMPLATES[template_name]
    template_description = template_info['description']
    
    few_shot_examples = build_few_shot_prompt(template_name, limit=3)
    
    validation_prompt = f"""Validate template output structure matches user request requirements.

REQUEST: {user_question}

TEMPLATE: {template_name} - {template_description}

{few_shot_examples}

VALIDATION LOGIC:

1. Line chart detection:
   Keywords: "line chart", "line graph", "over time", "trend"
   Required output: time-series with date dimension
   
2. Template output analysis:
   - Time-series templates: output (date, value) or (date, category, value)
   - Aggregate templates: output (category, total)
   
3. Compatibility check:
   - Line chart request + aggregate template = INVALID
   - Line chart request + time-series template = VALID
   - Bar/pie chart request + any template = VALID

OUTPUT (one word only): VALID or INVALID"""

    try:
        from adalflow.core.generator import Generator
        from adalflow.components.model_client import OllamaClient, OpenAIClient, GroqAPIClient
        
        if model_provider == "ollama":
            client = OllamaClient()
            model_kwargs = {"model": "qwen2.5:14b"}
        elif model_provider == "openai":
            client = OpenAIClient()
            model_kwargs = {"model": "gpt-4o-mini"}
        elif model_provider == "groq":
            client = GroqAPIClient()
            model_kwargs = {"model": "llama-3.1-8b-instant"}
        else:
            client = OllamaClient()
            model_kwargs = {"model": "qwen2.5:14b"}
        
        generator = Generator(
            model_client=client,
            model_kwargs=model_kwargs,
            use_cache=False
        )
        
        result = generator.call(prompt_kwargs={"input_str": validation_prompt})
        
        if result is None:
            _validation_cache[cache_key] = False
            return False
        
        validation_result = result.data if hasattr(result, 'data') else str(result)
        validation_result = validation_result.strip().upper()
        
        is_valid = "VALID" in validation_result
        _validation_cache[cache_key] = is_valid
        
        if is_valid:
            print(f"✅ Template validation passed: {template_name}")
        else:
            print(f"❌ Template validation failed: {template_name}")
        
        return is_valid
        
    except Exception as e:
        print(f"⚠️ Validation error: {e}")
        _validation_cache[cache_key] = False
        return False

# =============================================================================
# 🏢 USER SQL GENERATOR
# =============================================================================

def get_user_sql_from_template(
    user_question: str, 
    user_id: str,
    user_schema: dict
) -> Optional[Dict[str, Any]]:
    """
    Generate SQL from USER template.
    
    Big Tech Pattern: Netflix - personalized SQL generation
    """
    if not user_schema or not user_schema.get('has_data'):
        return None
    
    match = match_user_template(user_question, user_schema)
    
    if not match:
        print("📊 No user template matched")
        return None
    
    template_name = match['name']
    confidence = match['confidence']
    
    if template_name not in USER_SQL_TEMPLATES:
        return None
    
    template_info = USER_SQL_TEMPLATES[template_name]
    sql_template = template_info['sql']
    
    params = extract_user_parameters(user_question, template_name, user_schema)
    params['user_id'] = user_id
    
    sql_query = sql_template.strip()
    for param_name, param_value in params.items():
        if param_name.startswith('_'):
            continue
        placeholder = '{' + param_name + '}'
        sql_query = sql_query.replace(placeholder, str(param_value))
    
    print(f"✅ Generated USER SQL: {template_name}")
    print(f"   Metric: {params.get('metric_name')}")
    print(f"   SQL: {sql_query[:100]}...")
    
    return {
        'sql': sql_query,
        'template_name': template_name,
        'confidence': confidence,
        'is_user_data': True,
        '_meta': {'parameters_used': []}
    }

# =============================================================================
# MAIN FUNCTION (Called from adalflow_agent.py)
# =============================================================================

def get_sql_from_template(
    user_question: str, 
    model_provider: str = "ollama", 
    user_prefs: Optional[Dict[str, Any]] = None,
    user_id: str = None
) -> Optional[Dict[str, Any]]:
    """
    Generate SQL from template if pattern matches.
    
    Big Tech AGI: Now checks user data FIRST, then falls back to standard templates.
    
    Args:
        user_question: User's natural language query
        model_provider: LLM provider for semantic validation
        user_prefs: User preferences for personalization
        user_id: User identifier for multi-tenant data
    
    Returns:
        dict: {'sql': str, 'template_name': str, 'confidence': float, 'is_user_data': bool}
    """
    
    # =================================================================
    # 🏢 PHASE 9 STEP 4: Multi-Tenant Data Mode Routing (Big Tech AGI)
    # =================================================================
    # Netflix/Spotify Pattern: User controls their data universe
    # - data_mode = 'user' → Query THEIR uploaded data
    # - data_mode = 'demo' → Query shared demo data (lubobali.com)
    # 
    # CRITICAL: Check should_use_user_data() NOT just has_data
    # User may have data but explicitly switched to demo mode
    # =================================================================
    if user_id and user_id != 'anonymous':
        # 🛡️ PHASE 9: Check data_mode FIRST (respects user's toggle)
        _wants_user_data = should_use_user_data(user_id)
        
        if _wants_user_data:
            user_schema = get_user_schema(user_id)
            
            if user_schema and user_schema.get('has_data'):
                print(f"🏢 User {user_id[:8]}... in USER mode - querying their data")
                
                user_result = get_user_sql_from_template(
                    user_question=user_question,
                    user_id=user_id,
                    user_schema=user_schema
                )
                
                if user_result:
                    print(f"✅ Using USER template: {user_result['template_name']}")
                    return user_result
                else:
                    print(f"🔄 No user template matched - falling back to standard")
            else:
                print(f"⚠️ User in USER mode but no data found - falling back to demo")
        else:
            print(f"🌐 User {user_id[:8]}... in DEMO mode - using lubobali.com data")
    
    # =================================================================
    # 🌐 STEP 2: Standard templates (click_logs - demo data)
    # =================================================================
    match = match_sql_template(user_question)
    
    if not match:
        print("📊 No SQL template matched - using LLM fallback")
        return None
    
    template_name = match['name']
    confidence = match['confidence']
    
    # Minimum confidence threshold
    MIN_CONFIDENCE_THRESHOLD = 0.6
    if confidence < MIN_CONFIDENCE_THRESHOLD:
        print(f"⚠️ Template '{template_name}' confidence {confidence:.2f} below threshold")
        return None
    
    # Semantic validation
    is_valid = validate_template_for_request(template_name, user_question, model_provider)
    
    if not is_valid:
        print(f"🔄 Template '{template_name}' failed validation")
        return None
    
    # Extract parameters
    params = extract_sql_parameters(user_question, template_name, user_prefs)
    
    if template_name not in SQL_TEMPLATES:
        return None
    
    template_info = SQL_TEMPLATES[template_name]
    sql_template = template_info['sql']
    
    # Fill parameters
    sql_query = sql_template.strip()
    for param_name, param_value in params.items():
        if param_name == '_meta':
            continue
        placeholder = '{' + param_name + '}'
        sql_query = sql_query.replace(placeholder, str(param_value))
    
    print(f"✅ Generated SQL from template: {template_name}")
    
    parameters_used = params.get('_meta', {}).get('parameters_used', [])
    
    return {
        'sql': sql_query,
        'template_name': template_name,
        'confidence': confidence,
        'is_user_data': False,
        '_meta': {'parameters_used': parameters_used}
    }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_sql_templates():
    """List all available SQL templates."""
    print("\n=== STANDARD SQL TEMPLATES ===\n")
    for name, info in SQL_TEMPLATES.items():
        print(f"{name}: {info['description']}")
    
    print("\n=== USER DATA TEMPLATES ===\n")
    for name, info in USER_SQL_TEMPLATES.items():
        print(f"{name}: {info['description']}")

def test_sql_template(user_question: str, user_id: str = None) -> None:
    """Test SQL template matching and generation."""
    print(f"\n🧪 Testing: {user_question}")
    print(f"   User ID: {user_id}")
    print("=" * 60)
    
    sql = get_sql_from_template(user_question, user_id=user_id)
    
    if sql:
        print(f"\n✅ Generated SQL ({sql.get('template_name')}):")
        print(sql['sql'])
        print(f"\nIs user data: {sql.get('is_user_data', False)}")
    else:
        print("\n❌ No template match - would use LLM")

if __name__ == "__main__":
    test_cases = [
        "show top 3 pages over time",
        "total clicks yesterday",
        "clicks by device",
    ]
    
    for query in test_cases:
        test_sql_template(query)
