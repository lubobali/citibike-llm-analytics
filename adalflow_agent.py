import os
from typing import Optional
from datetime import datetime
import uuid
from adalflow.core.generator import Generator
from adalflow.components.model_client import GroqAPIClient, OpenAIClient, OllamaClient
from adalflow.utils import setup_env

from tools.sql_tools import run_sql_query, get_database_schema, get_user_schema, should_use_user_data
from tools.chart_generator import generate_chart_from_data
from tools.sql_templates import get_sql_from_template
# 🚲 CITIBIKE: Pre-validated SQL templates (TheCommons XR Homework)
try:
    from tools.citibike_templates import get_citibike_sql_from_template
    CITIBIKE_TEMPLATES_AVAILABLE = True
    print("✅ Citibike templates loaded")
except ImportError:
    CITIBIKE_TEMPLATES_AVAILABLE = False
    print("⚠️ Citibike templates not available - using LLM fallback")

# =============================================================================
# 🚲 CITIBIKE KEYWORDS - Single Source of Truth (Big Tech: DRY Principle)
# =============================================================================
# Big Tech Pattern: Include singular + plural + noun forms (prevents routing misses)
CITIBIKE_KEYWORDS = [
    'citibike', 
    'bike', 'bikes',
    'station', 'stations',
    'redistribute', 'redistribution',
    'rides', 'ride',
    'electric_bike', 'classic_bike',
    'shortage', 'shortages',
    'excess', 'busiest',
    'morning rush', 'rush hour'
]

# Import route learning (safe - returns 1.0 if fails)
try:
    from tools.route_learning import get_route_weight
except ImportError:
    def get_route_weight(route_name: str) -> float:
        return 1.0  # Fallback if import fails

# ==========================================================================
# 🆕 PHASE 5.2: Chart Preference Learning (Big Tech: Netflix/Spotify pattern)
# ==========================================================================
# Reads learned chart preferences from user_chart_preferences table
# Populated by chart_preferences_worker.py batch job
# ==========================================================================
try:
    from workers.chart_preferences_worker import ChartPreferencesWorker
    CHART_PREFS_AVAILABLE = True
    print("✅ Chart preferences worker loaded")
except ImportError:
    CHART_PREFS_AVAILABLE = False
    print("⚠️ Chart preferences worker not available - using defaults")

# ==========================================================================
# 🆕 PHASE 7: Interaction RAG - Few-Shot Learning (Big Tech: Netflix/Spotify)
# ==========================================================================
# Uses thumbs_up Q&A pairs as few-shot examples to improve LLM responses
# Pattern: "Questions like yours were answered successfully like this"
# Safety: Graceful fallback (never blocks, returns empty if fails)
# ==========================================================================
try:
    from workers.interaction_rag_worker import get_interaction_rag
    INTERACTION_RAG_AVAILABLE = True
    print("✅ Interaction RAG worker loaded")
except ImportError:
    INTERACTION_RAG_AVAILABLE = False
    print("⚠️ Interaction RAG worker not available - using defaults")

# ============================================================================
# 🆕 PHASE 7.1: User Preference Profiles (Big Tech: Netflix/Spotify AGI)
# ============================================================================
# Loads pre-computed user profiles from user_preference_profiles table
# Pattern: "Based on your 50 past interactions, showing as bar chart..."
# Safety: Graceful fallback (never blocks, returns empty dict if fails)
# ============================================================================
try:
    from workers.user_profile_worker import UserProfileWorker
    USER_PROFILE_AVAILABLE = True
    print("✅ User Profile Worker loaded")
except ImportError:
    USER_PROFILE_AVAILABLE = False
    print("⚠️ User Profile Worker not available - using defaults")

# ============================================================================
# 🆕 STEP 8: Data Profile Worker (Big Tech: Fitbit/Spotify baselines)
# ============================================================================
# Loads pre-computed data baselines from user_data_profiles table
# Pattern: "Your average is X" (Fitbit), "Mondays are your best day" (Spotify)
# Safety: Graceful fallback (never blocks, returns empty dict if fails)
# ============================================================================
try:
    from workers.data_profile_worker import DataProfileWorker
    DATA_PROFILE_AVAILABLE = True
    print("✅ Data Profile Worker loaded")
except ImportError:
    DATA_PROFILE_AVAILABLE = False
    print("⚠️ Data Profile Worker not available - using defaults")

# ============================================================================
# 🆕 PHASE 6: UNIVERSAL LEARNER - PER-USER PREFERENCES (Big Tech AGI)
# ============================================================================
# Netflix/Spotify/YouTube pattern: Personalize EVERY interaction
# Learns: limits, chart types, verbosity, time filters, routes
# ============================================================================
try:
    from learning.universal_learner import UniversalLearner
    UNIVERSAL_LEARNER_AVAILABLE = True
    print("✅ Universal learner loaded")
except ImportError:
    UNIVERSAL_LEARNER_AVAILABLE = False
    print("⚠️ Universal learner not available - using defaults")
from plotly_tools import generate_chart_tool
from tools.rag_tools import search_knowledge_base  # Knowledge base search
from tools.prediction_tools import (
    forecast_traffic, 
    detect_traffic_anomalies, 
    analyze_page_trends,
    generate_forecast_csv_report,
    generate_forecast_excel_report,
    generate_anomaly_csv_report,
    generate_anomaly_excel_report
)
from tools.document_generation_tools import (
    generate_pdf_report,
    generate_excel_export,
    generate_docx_report
)
from tools.code_interpreter_tool import run_python_code, get_code_interpreter_info
from memory.session_memory import ConversationMemory
from memory.db_conversation_memory import DBConversationMemory
from document_processing.simple_faiss_vector_store import SimpleFAISSVectorStore
from document_processing.text_chunker import TextChunker

# ============= CHART ARCHITECTURE =============
# This agent uses TWO chart systems (intentional design):
# 1. Web UI Charts: Plotly (interactive) via generate_chart_from_data()
# 2. Document Charts: Chart.js (static) via chart_tools.py
# See CHART_ARCHITECTURE.md for details.
# ==============================================

setup_env()

class AnalyticsReActAgent:
    def __init__(self, model_provider: str = "openai"):
        self.model_provider = model_provider
        self.model_client, self.model_kwargs = self._get_model_config(model_provider)
        
        self.generator = Generator(
            model_client=self.model_client,
            model_kwargs=self.model_kwargs,
            use_cache=False
        )
        
        # Initialize conversation memory
        self.memory = ConversationMemory(max_turns=10)
        
        # Track last route used for source icon detection
        self.last_route_used = None
        
        # Store images from code interpreter for main.py access
        self.last_code_interpreter_images = None
        self.last_code_metadata = None  # Store code execution metadata
        self.last_web_search_metadata = None  # Store web search metadata
        self.last_template_metadata = None  # 🆕 Store template matching metadata
        self.last_route_confidence = None  # 🆕 Store route confidence weight
        self.last_route_execution_time_ms = None  # 🆕 PHASE 4: Route execution time (milliseconds)
        self.last_route_error_occurred = False  # 🆕 PHASE 4: Track if route had an error
        self.last_route_alternative_exists = False  # 🆕 PHASE 4: Track if alternative route existed
        self.last_parameters_used = None  # 🆕 PHASE 5: AGI parameter tracking for learning
        self.last_parameters_extracted = None  # 🆕 PHASE 7.1: Real-time extracted params
        
        # Cache current route to avoid duplicate LLM calls
        self.current_route = None
        
        # Initialize document storage
        self.uploaded_documents = []
        self.doc_chunks = {}
        self.doc_meta = {}
        
        # Initialize vector store for semantic search
        self.vector_store = SimpleFAISSVectorStore()
        self.text_chunker = TextChunker(chunk_size=200, overlap_size=30)
        print("✅ Vector store initialized")
        
        print(f"✅ Agent initialized with {model_provider.upper()}")
    
    def _get_model_config(self, provider: str):
        if provider == "groq":
            return GroqAPIClient(), {"model": "llama-3.1-8b-instant"}
        elif provider == "openai":
            return OpenAIClient(), {"model": "gpt-4o-mini"}
        elif provider == "ollama":
            return OllamaClient(), {"model": "qwen2.5:14b"}
        else:
            raise ValueError(f"Unsupported: {provider}")
    
    def _select_tool(self, user_question: str, observations: list = None) -> str:
        """
        Use LLM to decide which tool to use for the user's question.
        
        Args:
            user_question: The user's question
            observations: List of previous observations from tool executions
        
        Returns:
            str: One of "SQL", "CHART", or "FINISH"
        """
        # Debug logging
        print(f"🐛 DEBUG _select_tool: observations = {observations}")
        print(f"🐛 DEBUG _select_tool: formatted observations = {chr(10).join(observations) if observations else 'None'}")
        
        # Detect prediction requests FIRST (before LLM call)
        user_lower = user_question.lower()
        
        # Report generation requests (check BEFORE predictions)
        # CRITICAL: Check for "report" keyword FIRST
        report_keywords = ['report', 'export', 'download', 'save']
        if any(keyword in user_lower for keyword in report_keywords):
            if any(word in user_lower for word in ['forecast', 'predict', 'traffic', 'future']):
                print("📄 Report generation detected: FORECAST_REPORT")
                return "GENERATE_FORECAST_REPORT"
            elif any(word in user_lower for word in ['anomaly', 'anomalies', 'unusual', 'spike']):
                print("📄 Report generation detected: ANOMALY_REPORT")
                return "GENERATE_ANOMALY_REPORT"
        
        # Smart LLM-based document generation detection (check BEFORE route check)
        doc_gen_prompt = f"""You are a query classifier. Analyze this user question and determine if it's requesting document generation.

USER QUESTION: {user_question}

DOCUMENT TYPES:

1. PDF_REPORT - Professional PDF reports (keywords: pdf, strategic report, business report)

2. EXCEL_EXPORT - Excel spreadsheets with data (keywords: excel, export, spreadsheet, export data, download data)

3. DOCX_REPORT - Editable Word documents (keywords: word, docx, editable report)

4. NONE - Not a document generation request

CRITICAL EXCLUSIONS (Check FIRST - these are NEVER document exports):
- "chart", "graph", "visualize", "plot", "bar chart", "line chart", "pie chart" → NONE
- "show me", "display", "what are", "how many" (without export keywords) → NONE

INSTRUCTIONS:

- CRITICAL: If query contains ANY visualization keyword (chart, graph, plot, visualize) → NONE
- If the query asks to "export", "download", "generate", or "create" data in Excel format → EXCEL_EXPORT

- If the query asks for a PDF, strategic report, or business report → PDF_REPORT

- If the query asks for a Word document or editable report → DOCX_REPORT

- Otherwise → NONE

Examples:

"Export last quarter's data to Excel" → EXCEL_EXPORT

"Create a strategic PDF report" → PDF_REPORT

"Generate a Word document" → DOCX_REPORT

"Download data as spreadsheet" → EXCEL_EXPORT

"How many clicks?" → NONE

Respond with ONLY ONE WORD: PDF_REPORT, EXCEL_EXPORT, DOCX_REPORT, or NONE"""

        doc_result = self.generator.call(prompt_kwargs={"input_str": doc_gen_prompt})
        if doc_result:
            doc_choice = doc_result.data if hasattr(doc_result, 'data') else str(doc_result)
            doc_choice = doc_choice.strip().upper()
            
            if doc_choice == "EXCEL_EXPORT":
                print("📄 Smart detection: EXCEL_EXPORT")
                return "GENERATE_EXCEL_EXPORT"
            elif doc_choice == "PDF_REPORT":
                print("📄 Smart detection: PDF_REPORT")
                return "GENERATE_PDF_REPORT"
            elif doc_choice == "DOCX_REPORT":
                print("📄 Smart detection: DOCX_REPORT")
                return "GENERATE_DOCX_REPORT"
        
        # ========== CODE INTERPRETER DETECTION (LLM-Based Routing) ==========
        # Use intelligent LLM routing to detect code interpreter needs
        # This handles all variations: "per", "visits", "each", "breakdown", etc.
        # Pattern: ChatGPT Advanced Data Analysis, GitHub Copilot, Claude approach
        
        # Check cached route from query() method (avoids duplicate LLM call)
        route = self.current_route
        
        if route == "CODE_INTERPRETER":
            print(f"🐍 Code interpreter detected via LLM routing")
            return "RUN_PYTHON_CODE"
        
        # If route is DATABASE, let it continue to SQL logic below
        # (We'll check again after prediction/report detection)
        # ========== END CODE INTERPRETER DETECTION ==========
        
        # Forecast requests
        if any(word in user_lower for word in ['predict', 'forecast', 'next week', 'next month', 'next quarter', 'future traffic', 'what will']):
            print("🔮 Prediction detected: FORECAST")
            return "PREDICT_TRAFFIC"
        
        # Anomaly detection requests  
        if any(word in user_lower for word in ['anomaly', 'anomalies', 'unusual', 'spike', 'drop', 'outlier', 'strange pattern']):
            print("🔍 Prediction detected: ANOMALIES")
            return "DETECT_ANOMALIES"
        
        # Page trend requests
        if any(word in user_lower for word in ['trending', 'trend', 'trending up', 'trending down', 'page performance', 'which pages']):
            print("📈 Prediction detected: TRENDS")
            return "ANALYZE_TRENDS"
        
        # If no prediction detected, continue with existing LLM logic
        prompt = f"""You are an analytics assistant. Based on the user's question and previous observations, decide which tool to use.

USER QUESTION: {user_question}

PREVIOUS OBSERVATIONS:
{('\n'.join(observations) if observations else "None - this is the first step")}

CRITICAL RULES:
- STOP: If observations contain "SQL executed successfully", respond with ONLY the word "FINISH" - do NOT choose SQL again!
- NEVER run SQL twice - if observations show SQL was executed, you MUST choose "FINISH"
- Only choose "CHART" if user explicitly asks for a chart/graph/visualization
- Only choose "SQL" if observations is empty OR if previous SQL failed with an error

OPTIONS:
- "SQL": ONLY if no data has been retrieved yet
- "CHART": ONLY if user explicitly requests a visualization  
- "FINISH": If observations contain an answer, or for greetings/general questions

Respond with ONLY one word: SQL, CHART, or FINISH"""
        
        result = self.generator.call(prompt_kwargs={"input_str": prompt})
        
        if result is None:
            return "SQL"  # Default to SQL
        
        tool_choice = result.data if hasattr(result, 'data') else str(result)
        tool_choice = tool_choice.strip().upper()
        
        # Validate response
        if tool_choice in ["SQL", "CHART", "FINISH"]:
            return tool_choice
        
        # Default to SQL if invalid response
        return "SQL"
    
    def _detect_time_period(self, user_question: str) -> str:
        """
        Detect time period keywords in the user's question.
        
        Returns one of: "yesterday", "last_week", "last_month", "last_2_months", 
                        "last_3_months", "last_6_months", "last_year", "all_time"
        """
        import re
        question_lower = re.sub(r'\s+', ' ', user_question.lower())
        
        # Check for specific time periods (order matters - check specific before general)
        if "today" in question_lower or "clicks today" in question_lower:
            return "today"
        elif "yesterday" in question_lower or "last day" in question_lower:
            return "yesterday"
        elif any(pattern in question_lower for pattern in [
            "last week", "past week", "last 7 days", "past 7 days",
            "for the last 7 days", "in the last 7 days", "7 days"
        ]):
            return "last_week"
        elif any(pattern in question_lower for pattern in [
            "last month", "past month", "last 30 days", "past 30 days",
            "for the last 30 days", "in the last 30 days", "30 days"
        ]):
            return "last_month"
        elif any(pattern in question_lower for pattern in [
            "last 2 month", "last 2 months", "past 2 months",
            "last two months", "past two months", "last two month", "past two month"
        ]):
            return "last_2_months"
        elif "last 3 months" in question_lower or "past 3 months" in question_lower:
            return "last_3_months"
        elif "last 6 months" in question_lower or "past 6 months" in question_lower:
            return "last_6_months"
        elif "last year" in question_lower or "past year" in question_lower:
            return "last_year"
        else:
            return "all_time"
    
    @staticmethod
    def _parse_upload_dt(md: dict):
        """Parse upload_date from metadata for sorting."""
        v = md.get("upload_date", "")
        try:
            return datetime.fromisoformat(v)
        except Exception:
            return datetime.min
    
    def _detect_chart_request(self, question: str) -> dict:
        """
        Detect if user wants a chart and what type.
        
        Returns:
            dict: {'wants_chart': bool, 'chart_type': str}
        """
        question_lower = question.lower()
        
        # Chart keywords
        chart_keywords = ['chart', 'graph', 'visualize', 'plot', 'show me a', 'over time', 'trend']
        wants_chart = any(keyword in question_lower for keyword in chart_keywords)
        
        if not wants_chart:
            return {'wants_chart': False, 'chart_type': None}
        
        # CRITICAL: Check for explicit chart type FIRST (user's exact request)
        if 'bar chart' in question_lower or 'bar graph' in question_lower:
            chart_type = 'bar'
        elif 'line chart' in question_lower or 'line graph' in question_lower:
            chart_type = 'line'
        elif any(word in question_lower for word in ['pie', 'breakdown', 'distribution']):
            chart_type = 'pie'
        # THEN check for implicit chart type (only if no explicit type mentioned)
        elif any(word in question_lower for word in ['trend', 'over time', 'timeline']):
            chart_type = 'line'
        else:
            chart_type = 'bar'  # Default to bar chart
        
        print(f"📊 Chart detected: type={chart_type}")
        return {'wants_chart': True, 'chart_type': chart_type}

    def _fuzzy_match_pages(self, requested_page: str) -> dict:
        """
        Find actual database pages that match the requested page name.
        
        Returns:
            dict: {
                'exact_match': str or None,
                'fuzzy_matches': list of tuples [(page_name, click_count)],
                'no_match': bool
            }
        """
        try:
            # Query database for page names
            query = "SELECT page_name, COUNT(*) as clicks FROM click_logs GROUP BY page_name ORDER BY clicks DESC"
            from tools.sql_tools import run_sql_query
            result = run_sql_query(query, "all_time")
            
            if result.get('error') or not result.get('data'):
                return {'exact_match': None, 'fuzzy_matches': [], 'no_match': True}
            
            all_pages = [(row[0], row[1]) for row in result['data']]
            requested_lower = requested_page.lower().strip().strip('/').replace('-', ' ').replace('_', ' ')
            
            # 1. Check exact match
            for page, clicks in all_pages:
                page_normalized = page.lower().strip('/').replace('-', ' ').replace('_', ' ')
                if page_normalized == requested_lower:
                    return {'exact_match': page, 'fuzzy_matches': [(page, clicks)], 'no_match': False}
            
            # 2. Check fuzzy matches (contains)
            fuzzy_matches = []
            for page, clicks in all_pages:
                page_normalized = page.lower().strip('/').replace('-', ' ').replace('_', ' ')
                # Exact match after normalization
                if page_normalized == requested_lower:
                    fuzzy_matches.append((page, clicks))
                # Partial match only if query is long enough (avoids matching "me" in "home")
                elif len(requested_lower) > 3 and (requested_lower in page_normalized or page_normalized in requested_lower):
                    fuzzy_matches.append((page, clicks))
            
            if fuzzy_matches:
                # Sort by clicks (descending)
                fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
                return {'exact_match': None, 'fuzzy_matches': fuzzy_matches[:3], 'no_match': False}
            
            # 3. No match - return main navigation pages as suggestions
            # Hardcoded list of main pages for better UX
            main_nav_pages = ['home', '/portfolio', '/about-me', '/contact', '/resume']
            nav_suggestions = []
            for nav_page in main_nav_pages:
                # Find this page in all_pages to get click count
                for page, clicks in all_pages:
                    page_normalized = page.lower().strip('/').replace('-', ' ').replace('_', ' ')
                    nav_normalized = nav_page.lower().strip('/').replace('-', ' ').replace('_', ' ')
                    if page_normalized == nav_normalized:
                        nav_suggestions.append((page, clicks))
                        break
            # If we found nav pages, return them; otherwise fall back to top pages
            if nav_suggestions:
                return {'exact_match': None, 'fuzzy_matches': nav_suggestions, 'no_match': True}
            else:
                return {'exact_match': None, 'fuzzy_matches': all_pages[:5], 'no_match': True}
            
        except Exception as e:
            print(f"⚠️ Fuzzy match error: {e}")
            return {'exact_match': None, 'fuzzy_matches': [], 'no_match': True}
    
    def _route_query(self, user_question: str) -> str:
        """
        Use LLM to intelligently route queries to the right tool.
        
        Args:
            user_question: The user's query
            
        Returns:
            str: One of "CODE_INTERPRETER", "WEB_SEARCH", "DATABASE", "KNOWLEDGE_BASE", or "CONTINUE"
        """
        # =================================================================
        # 🚲 CITIBIKE FAST PATH (TheCommons XR Homework - Task 4)
        # =================================================================
        # Cold-start routing: No learned data yet, force DATABASE for citibike
        # Big Tech: Early-exit pattern (Google/Meta style)
        # Safe: Only affects citibike queries, LLM routing continues for rest
        # =================================================================
        _citibike_keywords = CITIBIKE_KEYWORDS
        if any(kw in user_question.lower() for kw in _citibike_keywords):
            print("🚲 Citibike query detected - forcing DATABASE route")
            self.last_route_confidence = 1.0
            return "DATABASE"
        
        prompt = f"""You are a query router for an analytics AI agent. Analyze the user's question and decide which tool should handle it.

USER QUESTION: {user_question}

AVAILABLE TOOLS:

1. CODE_INTERPRETER - Complex data analysis requiring Python execution
   - Aggregations: "group by", "grouped by", "per", "breakdown by", "split by", "aggregate"
   - Comparisons: "compare", "top X", "rank", "versus", "vs"
   - Statistical analysis: "correlation", "distribution", "average by", "mean", "median"
   - Advanced visualizations: "heatmap", "over time", "trend", "histogram"
   - Machine learning: "predict", "forecast", "regression", "clustering"
   - Use when query needs: grouping, statistical analysis, or complex calculations
   
2. DATABASE - Simple SQL queries (basic counts, filters, single table queries)
   - Use when query is simple: "how many clicks", "show me pages", "list referrers"
   - Simple filters: "clicks yesterday", "pages last week"
   - No grouping/aggregation required
   
3. WEB_SEARCH - Current information from the internet (news, external facts, latest updates)
   - Use when query asks about: external topics, current events, latest trends
   - Examples: "latest Next.js features", "weather today", "current AI news"
   
4. KNOWLEDGE_BASE - Documentation about the agent's capabilities
   - Use when query asks: "what can you do", "how do I", "help with"
   
5. CONTINUE - None of the above (greetings, follow-up to documents, or general conversation)

CRITICAL ROUTING RULES:

CODE_INTERPRETER vs DATABASE:

CRITICAL - Check these DATABASE patterns FIRST (before CODE_INTERPRETER):

1. SIMPLE QUERIES WITH TEMPLATES (Highest Priority):
   - "by device" OR "device type" OR "device breakdown" → DATABASE (CASE statement template exists)
   - "by referrer" OR "traffic source" OR "referrer breakdown" → DATABASE (GROUP BY template exists)
   - "by hour" OR "hourly" OR "by time of day" → DATABASE (EXTRACT HOUR template exists)
   - "by page" (WITHOUT "per page" or "grouped by page") → DATABASE (simple GROUP BY template exists)
   - "top N pages/items" + any keywords → DATABASE (ranking template exists)
   - "total" OR "how many" (simple counts) → DATABASE (COUNT template exists)
   - Pattern: "by X" where X has SQL template → DATABASE FIRST, then fallback to CODE_INTERPRETER if fails

2. VISUALIZATION REQUESTS:
   - "chart/graph/visualize" + simple metrics → DATABASE (has chart generation)
   - "pie/bar/line chart" + ranking/counting → DATABASE (simple query + chart type)
   - "show X as chart" → DATABASE (query + visualization)

3. TIME SERIES & COMPARISONS:
   - "unique visitors/users" (WITHOUT "grouped by" or "per page") → DATABASE (template exists)
   - "N pages over time" OR "top pages over time" → DATABASE (multi-series template)
   - "compare [page1] and [page2]" → DATABASE (comparison template)
   - "X over time" (single metric) → DATABASE (time series template)

4. GENERAL RULE (Critical Decision Logic):
   DATABASE if:
   - Simple aggregation (count, sum, avg) with GROUP BY on single dimension
   - Has explicit SQL template in system
   - No complex multi-dimensional analysis
   
   CODE_INTERPRETER if:
   - "per X BY Y" (multi-dimensional grouping)
   - "grouped by X and Y" (multiple GROUP BY dimensions)
   - "correlation", "regression", "distribution" (statistical analysis)
   - "breakdown by X per Y" (nested aggregation)
   - No SQL template exists AND query needs custom logic

Then check CODE_INTERPRETER patterns (ONLY after DATABASE check fails):
- "per X BY Y" OR "X per Y per Z" → CODE_INTERPRETER (multi-dimensional grouping)
- "grouped by X and Y" → CODE_INTERPRETER (multiple dimensions)
- "unique visitors PER page" OR "visits PER referrer PER day" → CODE_INTERPRETER (nested aggregation)
- "correlation", "regression", "distribution", "heatmap" → CODE_INTERPRETER (statistics)
- "breakdown by X split by Y" → CODE_INTERPRETER (multi-dimensional analysis)
- "compare X and Y" (NOT pages over time) → CODE_INTERPRETER  
- "statistics", "clustering" → CODE_INTERPRETER
- Pattern: Complex aggregation that SQL templates can't handle → CODE_INTERPRETER

Finally check DATABASE patterns:
- "how many", "show", "list" (no aggregation) → DATABASE
- Simple counts, filters, single table queries → DATABASE
- If query is simple SQL (count/list/filter) → DATABASE

Decision logic:
1. Does query ask for "N pages/top pages + over time/line chart"? → DATABASE
2. Does query need Python/statistics/complex grouping? → CODE_INTERPRETER
3. Is query simple SQL? → DATABASE

Examples:

DATABASE Routing (Simple queries with SQL templates):
"show clicks by device type" → DATABASE (device_breakdown template)
"show traffic by referrer" → DATABASE (referrer_breakdown template)
"show clicks by hour" → DATABASE (clicks_by_hour template)
"make me a pie chart of top 3 pages" → DATABASE (top_pages template + pie chart)
"show top 5 pages as bar chart" → DATABASE (top_pages template + bar chart)
"visualize top pages from last month" → DATABASE (top_pages template + auto-chart)
"show unique visitors" → DATABASE (unique_visitors template)
"unique visitors over time" → DATABASE (unique_visitors template + time series)
"show top 3 pages over time" → DATABASE (top_n_pages_over_time template)
"compare portfolio and home page over time" → DATABASE (compare_two_pages template)
"how many clicks today?" → DATABASE (total_clicks template)
"which pages get most traffic?" → DATABASE (top_pages template)

CODE_INTERPRETER Routing (Complex analysis without templates):
"unique visitors per page" → CODE_INTERPRETER (nested aggregation, no template)
"clicks per hour grouped by device" → CODE_INTERPRETER (multi-dimensional, no template)
"correlation between clicks and time" → CODE_INTERPRETER (statistical analysis)
"show clicks by hour and day heatmap" → CODE_INTERPRETER (2D visualization)
"visits per referrer per page" → CODE_INTERPRETER (nested grouping, no template)
"breakdown by device split by referrer" → CODE_INTERPRETER (multi-dimensional)
"distribution of time on page" → CODE_INTERPRETER (statistical distribution)
"What's the weather today?" → WEB_SEARCH (external + time)
"What can you do?" → KNOWLEDGE_BASE (agent capabilities)

Respond with ONLY ONE WORD: CODE_INTERPRETER, DATABASE, WEB_SEARCH, KNOWLEDGE_BASE, or CONTINUE"""
        
        try:
            result = self.generator.call(prompt_kwargs={"input_str": prompt})
            
            if result is None:
                return "CONTINUE"
            
            route = result.data if hasattr(result, 'data') else str(result)
            route = route.strip().upper()
            
            # Validate response
            valid_routes = ["CODE_INTERPRETER", "DATABASE", "WEB_SEARCH", "KNOWLEDGE_BASE", "CONTINUE"]
            if route in valid_routes:
                weight = get_route_weight(route)
                
                # ========== 🆕 PHASE 5: Apply learned route preferences ==========
                # Big Tech Pattern: Netflix/Spotify/YouTube preference learning
                # Extract keywords → lookup preferences → adjust confidence
                try:
                    # Step 1: Extract keywords from query (simple word extraction)
                    import re
                    query_words = re.findall(r'\b[a-zA-Z]{3,}\b', user_question.lower())
                    # Filter to meaningful keywords (remove common words)
                    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'would', 'could', 'there', 'their', 'what', 'when', 'where', 'which', 'how', 'who', 'show', 'give', 'get', 'make', 'from', 'this', 'that', 'with', 'they', 'will', 'would', 'about', 'into', 'than', 'then', 'them', 'these', 'some', 'such', 'only', 'just', 'also', 'more', 'other', 'most', 'very', 'after', 'before', 'being', 'between', 'both', 'each', 'because', 'does', 'doing', 'during', 'having', 'here', 'itself', 'made', 'many', 'might', 'much', 'must', 'need', 'never', 'now', 'over', 'same', 'should', 'since', 'still', 'such', 'through', 'under', 'until', 'upon', 'used', 'using', 'wants', 'well', 'were', 'while', 'within', 'without', 'please', 'thanks', 'thank', 'hello', 'yesterday', 'today', 'last', 'week', 'month', 'year'}
                    keywords = [w for w in query_words if w not in stop_words][:5]  # Max 5 keywords
                    
                    if keywords:
                        # Step 2: Get learned preferences from database
                        preferences = self._get_route_preference(keywords)
                        
                        # Step 3: Apply preferences to adjust confidence (safe, capped)
                        if preferences:
                            adjusted_weight = self._apply_route_preference(route, weight, preferences)
                            weight = adjusted_weight
                            print(f"🧠 Preference learning applied: {keywords}")
                        else:
                            print(f"🧠 No learned preferences for: {keywords}")
                    else:
                        print(f"🧠 No keywords to learn from")
                        
                except Exception as e:
                    # Big Tech Pattern: Never block on preference failure
                    print(f"⚠️ Preference learning skipped: {e}")
                # ========== END PHASE 5 ==========
                
                self.last_route_confidence = weight
                
                # 🆕 PHASE 4: Detect if alternative route exists (Big Tech: failover awareness)
                query_lower = user_question.lower()
                if route == "DATABASE":
                    # Could CODE_INTERPRETER also handle this? Check for grouping/stats keywords
                    alt_keywords = ['per', 'by', 'group', 'breakdown', 'split', 'correlation', 'distribution', 'statistics']
                    if any(kw in query_lower for kw in alt_keywords):
                        self.last_route_alternative_exists = True
                        print(f"🔀 Alternative exists: CODE_INTERPRETER could also handle this")
                elif route == "CODE_INTERPRETER":
                    # Could DATABASE also handle this? Check for simple query patterns
                    simple_keywords = ['how many', 'total', 'count', 'show', 'list', 'top']
                    if any(kw in query_lower for kw in simple_keywords):
                        self.last_route_alternative_exists = True
                        print(f"🔀 Alternative exists: DATABASE could also handle this")
                
                print(f"🧭 Query routed to: {route} (weight: {weight:.2f})")
                return route
            
            # Default to CONTINUE if invalid
            print(f"⚠️ Invalid route '{route}', defaulting to CONTINUE")
            return "CONTINUE"
            
        except Exception as e:
            print(f"❌ Routing error: {e}")
            return "CONTINUE"
    
    def _get_route_preference(self, keywords: list) -> dict:
        """
        🆕 PHASE 5: Query database for learned route preferences.
        Big Tech Pattern: Netflix/Spotify preference learning.
        
        Args:
            keywords: List of query keywords to match
            
        Returns:
            dict: Route preference scores {"DATABASE": 0.85, "CODE_INTERPRETER": 0.45}
        """
        if not keywords:
            return {}
            
        try:
            from sqlalchemy import create_engine, text
            import os
            
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                return {}
            
            engine = create_engine(database_url)
            
            # Query: Find routes with best feedback for overlapping keywords
            # Uses PostgreSQL array overlap operator (&&)
            query = text("""
                SELECT route_used,
                       COUNT(*) FILTER (WHERE feedback = 'thumbs_up') as likes,
                       COUNT(*) FILTER (WHERE feedback = 'thumbs_down') as dislikes,
                       COUNT(*) as total
                FROM template_learning_data
                WHERE query_keywords && :keywords
                  AND feedback IS NOT NULL
                GROUP BY route_used
                HAVING COUNT(*) >= 3
            """)
            
            with engine.connect() as conn:
                result = conn.execute(query, {"keywords": keywords})
                rows = result.fetchall()
            
            # Calculate preference scores (likes / total)
            preferences = {}
            for row in rows:
                route, likes, dislikes, total = row
                if total > 0:
                    score = likes / total
                    preferences[route] = round(score, 2)
                    print(f"📊 Route preference: {route} = {score:.2f} ({likes}👍 / {total} total)")
            
            return preferences
            
        except Exception as e:
            print(f"⚠️ Route preference lookup failed: {e}")
            return {}
    
    def _apply_route_preference(self, route: str, base_confidence: float, preferences: dict) -> float:
        """
        🆕 PHASE 5: Apply learned preferences to adjust route confidence.
        Big Tech Pattern: Netflix/Spotify preference boosting with safety caps.
        
        Args:
            route: The route being considered (e.g., "DATABASE")
            base_confidence: Original confidence score (0.0 - 1.0)
            preferences: Dict from _get_route_preference {"DATABASE": 0.85, ...}
            
        Returns:
            float: Adjusted confidence score (safely capped at 0.0 - 1.0)
        """
        # ========== SAFETY CONSTANTS (Big Tech: prevent runaway preferences) ==========
        MAX_BOOST = 0.15       # Maximum 15% boost from preferences
        MAX_PENALTY = 0.10     # Maximum 10% penalty from negative preferences
        MIN_CONFIDENCE = 0.10  # Never go below 10% (always allow route)
        MAX_CONFIDENCE = 0.95  # Never go above 95% (always allow exploration)
        STRONG_PREFERENCE = 0.65  # Only boost if preference > 65%
        WEAK_PREFERENCE = 0.35    # Only penalize if preference < 35%
        
        # ========== GRACEFUL FALLBACK (Big Tech: never block on failure) ==========
        if not preferences or route not in preferences:
            print(f"📊 No preference data for {route}, using base confidence: {base_confidence:.2f}")
            return base_confidence
        
        try:
            preference_score = preferences[route]
            adjusted = base_confidence
            
            # ========== BOOST LOGIC (Big Tech: reward good routes) ==========
            if preference_score >= STRONG_PREFERENCE:
                # Strong positive preference → boost confidence
                # Scale: 0.65 → +0%, 1.0 → +15%
                boost_factor = (preference_score - STRONG_PREFERENCE) / (1.0 - STRONG_PREFERENCE)
                boost = boost_factor * MAX_BOOST
                adjusted = base_confidence + boost
                print(f"📈 Preference boost: {route} +{boost:.2f} (pref: {preference_score:.2f})")
            
            # ========== PENALTY LOGIC (Big Tech: discourage bad routes) ==========
            elif preference_score <= WEAK_PREFERENCE:
                # Weak preference → reduce confidence
                # Scale: 0.35 → -0%, 0.0 → -10%
                penalty_factor = (WEAK_PREFERENCE - preference_score) / WEAK_PREFERENCE
                penalty = penalty_factor * MAX_PENALTY
                adjusted = base_confidence - penalty
                print(f"📉 Preference penalty: {route} -{penalty:.2f} (pref: {preference_score:.2f})")
            
            else:
                # Neutral preference → no adjustment
                print(f"📊 Neutral preference for {route}: {preference_score:.2f}, no adjustment")
            
            # ========== SAFETY CAPS (Big Tech: prevent extreme values) ==========
            adjusted = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, adjusted))
            
            if adjusted != base_confidence:
                print(f"🎯 Adjusted confidence: {route} {base_confidence:.2f} → {adjusted:.2f}")
            
            return round(adjusted, 2)
            
        except Exception as e:
            print(f"⚠️ Preference application failed: {e}")
            return base_confidence  # Safe fallback
    
    def _get_user_profile(self, user_id: str) -> dict:
        """
        🆕 PHASE 7.1: Get user preference profile from user_preference_profiles table.
        Big Tech Pattern: Netflix/Spotify/YouTube personalization.
        
        Args:
            user_id: User identifier (persists across sessions, unlike session_id)
            
        Returns:
            dict: User profile with preferences, or empty dict if not found
            
        Profile Structure:
            {
                "chart_preferences": {"bar": 0.82, "line": 0.12},
                "limit_preferences": {"10": 0.6, "5": 0.3},
                "time_range_preferences": {"last_7_days": 0.5},
                "verbosity_preferences": {"short": 0.7, "long": 0.3},
                "format_preferences": {"percentage": 0.4},
                "route_success_rates": {"DATABASE": 0.85, "WEB_SEARCH": 0.6},
                "topic_interests": ["traffic", "revenue", "pages"],
                "total_interactions": 25,
                "confidence": 0.5
            }
        """
        # ========== GUARD: Feature not available ==========
        if not USER_PROFILE_AVAILABLE:
            return {}
        
        # ========== GUARD: No user_id ==========
        if not user_id:
            return {}
        
        # ========== GRACEFUL EXECUTION (Big Tech: Never block) ==========
        try:
            worker = UserProfileWorker()
            profile = worker.get_user_profile(user_id)
            
            if profile and profile.get('total_interactions', 0) > 0:
                confidence = profile.get('confidence', 0)
                total = profile.get('total_interactions', 0)
                print(f"🧠 Phase 7.1: Loaded profile for {user_id[:8]}... ({total} interactions, confidence={confidence:.2f})")
                return profile
            else:
                print(f"🧠 Phase 7.1: No profile found for {user_id[:8]}...")
                return {}
                
        except Exception as e:
            # Big Tech Pattern: NEVER block on personalization failure
            print(f"⚠️ Phase 7.1: Profile lookup failed (non-critical): {e}")
            return {}
    
    def _get_data_profile(self, user_id: str = 'default') -> dict:
        """
        🆕 STEP 8.4 + PHASE 9: Get data profile respecting data_mode.
        Big Tech Pattern: Fitbit/Spotify/Bank App data baselines.
        
        🏢 PHASE 9 INTEGRATION: Respects user's data_mode toggle
        - data_mode = 'user' → Load user's uploaded data profile
        - data_mode = 'demo' → Load 'default' profile (lubobali.com)
        
        🆕 AGI FALLBACK: If no user-specific profile, use site-wide 'default'.
        - lubobali.com traffic = SITE data (shared baseline)
        - Netflix pattern: "This show is above average" (global metric)
        
        Args:
            user_id: User identifier ('default' for site-wide baseline)
            
        Returns:
            dict: Data profile with baselines, or empty dict if not found
            
        Profile Structure:
            {
                "baseline_daily_avg": 14.8,
                "baseline_std_dev": 4.5,
                "percentile_25": 10.0,
                "percentile_50": 14.0,
                "percentile_75": 18.0,
                "percentile_95": 25.0,
                "typical_range_low": 10.3,
                "typical_range_high": 19.3,
                "best_day": "monday",
                "worst_day": "sunday",
                "trend_direction": "up",
                "trend_percentage": 8.5,
                "confidence_score": 1.0,
                "total_days_analyzed": 120
            }
        """
        # ========== GUARD: Feature not available ==========
        if not DATA_PROFILE_AVAILABLE:
            return {}
        
        # ========== GUARD: No user_id ==========
        if not user_id:
            user_id = 'default'
        
        # ========== GRACEFUL EXECUTION (Big Tech: Never block) ==========
        try:
            worker = DataProfileWorker()
            
            # =================================================================
            # 🏢 PHASE 9 STEP 4: Multi-Tenant Data Profile Routing (Big Tech AGI)
            # =================================================================
            # Netflix/Spotify Pattern: User controls their data universe
            # - data_mode = 'user' → Load THEIR data profile (uploaded metrics)
            # - data_mode = 'demo' → Load 'default' profile (lubobali.com)
            # 
            # CRITICAL: Check should_use_user_data() to respect user's toggle
            # User may have uploaded data but switched to demo mode
            # =================================================================
            _profile_user_id = 'default'  # Start with demo (safe default)
            
            if user_id and user_id != 'default' and user_id != 'anonymous':
                # Check user's data_mode preference
                _wants_user_data = should_use_user_data(user_id)
                
                if _wants_user_data:
                    # User is in USER mode → use their profile
                    _profile_user_id = user_id
                    print(f"📊 Step 8.4: User {user_id[:8]}... in USER mode - loading their profile")
                else:
                    # User is in DEMO mode → use site-wide baseline
                    _profile_user_id = 'default'
                    print(f"📊 Step 8.4: User {user_id[:8]}... in DEMO mode - loading lubobali.com profile")
            # =================================================================
            # END PHASE 9 STEP 4
            # =================================================================
            
            profile = worker.get_data_profile(_profile_user_id)
            
            # ========== 🆕 BIG TECH AGI: SITE-WIDE BASELINE FALLBACK ==========
            # Netflix/Google Analytics pattern:
            # - No personal profile? Use site-wide baseline ('default')
            # - lubobali.com traffic is SITE data, not personal data
            # - Every user should see: "Your site's typical is 10.7 clicks/day"
            # =================================================================
            if (not profile or profile.get('total_days_analyzed', 0) == 0) and _profile_user_id != 'default':
                print(f"📊 Step 8.4: No profile for {_profile_user_id[:8]}..., using site-wide baseline")
                profile = worker.get_data_profile('default')
            # ========== END BIG TECH AGI FALLBACK ==========
            
            if profile and profile.get('total_days_analyzed', 0) > 0:
                confidence = profile.get('confidence_score', 0)
                total_days = profile.get('total_days_analyzed', 0)
                baseline = profile.get('baseline_daily_avg', 0)
                print(f"📊 Step 8.4: Loaded data profile ({total_days} days, baseline={baseline:.1f}, confidence={confidence:.2f})")
                return profile
            else:
                print(f"📊 Step 8.4: No data profile available")
                return {}
                
        except Exception as e:
            # Big Tech Pattern: NEVER block on personalization failure
            print(f"⚠️ Step 8.4: Data profile lookup failed (non-critical): {e}")
            return {}
    
    def _get_user_data_context(self, user_id: str) -> dict:
        """
        🏢 BIG TECH AGI: Get user's uploaded data metadata.
        Netflix/Spotify Pattern: Know user's data schema for personalized queries.
        
        Args:
            user_id: User identifier
            
        Returns:
            dict: User data context or None
            {
                'metric_column': 'Total Amount',
                'date_column': 'Date', 
                'dimensions': ['Gender', 'Product Category'],
                'table_name': 'user_uploaded_metrics'
            }
        """
        if not user_id:
            return None
        
        try:
            from sqlalchemy import create_engine, text
            import os
            
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                return None
            
            engine = create_engine(database_url)
            
            # Query user_uploads table for user's schema info (most recent upload)
            query = text("""
                SELECT 
                    metric_column,
                    date_column,
                    dimension_columns,
                    filename
                FROM user_uploads
                WHERE user_id = :user_id
                  AND status = 'completed'
                ORDER BY created_at DESC
                LIMIT 1
            """)
            
            with engine.connect() as conn:
                result = conn.execute(query, {"user_id": user_id})
                row = result.fetchone()
                
                if row:
                    # Parse dimensions (handle JSONB auto-conversion)
                    dimensions_raw = row[2]
                    if dimensions_raw:
                        if isinstance(dimensions_raw, list):
                            dimensions = dimensions_raw
                        elif isinstance(dimensions_raw, str):
                            import json
                            try:
                                dimensions = json.loads(dimensions_raw)
                            except:
                                dimensions = []
                        else:
                            dimensions = []
                    else:
                        dimensions = []
                    
                    return {
                        'metric_column': row[0],
                        'date_column': row[1],
                        'dimensions': dimensions,
                        'original_filename': row[3],
                        'table_name': 'user_uploaded_metrics'
                    }
                    
            return None
            
        except Exception as e:
            # Big Tech: NEVER block on metadata lookup failure
            print(f"⚠️ AGI: User data context lookup failed (non-critical): {e}")
            return None
    
    def _contextualize_result(self, value: float, metric: str, data_profile: dict) -> str:
        """
        🆕 STEP 8.4: Transform raw numbers into personalized insights.
        Big Tech Pattern: Fitbit/Spotify/Bank App contextual answers.
        
        Args:
            value: The raw metric value (e.g., 14.80)
            metric: The metric name (e.g., "daily_clicks", "avg_time")
            data_profile: User's data profile from _get_data_profile()
            
        Returns:
            str: Contextualized insight string
            
        Examples:
            Input:  value=14.80, metric="daily_clicks"
            Output: "14.80 is +12% above YOUR typical (13.2). Mondays best. Trending UP."
            
            Input:  value=5.0, metric="daily_clicks" (anomaly low)
            Output: "5.0 is unusually low (-62% below your typical 13.2). Check for issues."
        """
        # ========== GUARD: No profile = return raw value ==========
        if not data_profile or data_profile.get('confidence_score', 0) < 0.3:
            return str(value)
        
        # ========== EXTRACT BASELINES ==========
        try:
            baseline_avg = data_profile.get('baseline_daily_avg', 0)
            baseline_std = data_profile.get('baseline_std_dev', 0)
            typical_low = data_profile.get('typical_range_low', 0)
            typical_high = data_profile.get('typical_range_high', 0)
            anomaly_low = data_profile.get('anomaly_low', 0)
            anomaly_high = data_profile.get('anomaly_high', 0)
            best_day = data_profile.get('best_day', '')
            trend_direction = data_profile.get('trend_direction', 'stable')
            trend_pct = data_profile.get('trend_percentage', 0)
            
            # ========== GUARD: Invalid baseline ==========
            if baseline_avg <= 0:
                return str(value)
            
            # ========== CALCULATE COMPARISON ==========
            diff_pct = ((value - baseline_avg) / baseline_avg) * 100
            
            # ========== BUILD CONTEXT STRING (Big Tech: Fitbit/Spotify style) ==========
            parts = []
            
            # Part 1: Value vs baseline
            if diff_pct > 10:
                parts.append(f"**{value:.1f}** is +{diff_pct:.0f}% above YOUR typical ({baseline_avg:.1f})")
            elif diff_pct < -10:
                parts.append(f"**{value:.1f}** is {diff_pct:.0f}% below YOUR typical ({baseline_avg:.1f})")
            else:
                parts.append(f"**{value:.1f}** is right around YOUR typical ({baseline_avg:.1f})")
            
            # Part 2: Anomaly detection (Bank App pattern)
            if value < anomaly_low:
                parts.append("⚠️ **Unusually low** - check for issues")
            elif value > anomaly_high:
                parts.append("🚀 **Unusually high** - great performance!")
            
            # Part 3: Best day insight (Spotify Wrapped pattern)
            if best_day:
                parts.append(f"📅 {best_day.capitalize()}s are your best day")
            
            # Part 4: Trend insight (Finance App pattern)
            if trend_direction == 'up' and trend_pct > 5:
                parts.append(f"📈 Trending UP (+{trend_pct:.0f}%)")
            elif trend_direction == 'down' and trend_pct < -5:
                parts.append(f"📉 Trending DOWN ({trend_pct:.0f}%)")
            else:
                parts.append("➡️ Stable trend")
            
            # ========== COMBINE INTO SINGLE INSIGHT ==========
            return ". ".join(parts) + "."
            
        except Exception as e:
            # Big Tech Pattern: Never fail on contextualization
            print(f"⚠️ Step 8.4: Contextualization failed (non-critical): {e}")
            return str(value)
    
    def _format_data_profile_for_prompt(self, data_profile: dict) -> str:
        """
        🆕 STEP 8.4: Format data profile for system prompt injection.
        Big Tech Pattern: Give LLM context about user's data patterns.
        
        Args:
            data_profile: Data profile dict from _get_data_profile()
            
        Returns:
            str: Formatted prompt section, or empty string if low confidence
        """
        # ========== GUARD: Low confidence profiles ==========
        confidence = data_profile.get('confidence_score', 0)
        total_days = data_profile.get('total_days_analyzed', 0)
        
        if confidence < 0.3 or total_days < 7:
            return ""
        
        # ========== BUILD PROMPT SECTION ==========
        try:
            baseline_avg = data_profile.get('baseline_daily_avg', 0)
            typical_low = data_profile.get('typical_range_low', 0)
            typical_high = data_profile.get('typical_range_high', 0)
            best_day = data_profile.get('best_day', 'N/A')
            worst_day = data_profile.get('worst_day', 'N/A')
            trend_direction = data_profile.get('trend_direction', 'stable')
            trend_pct = data_profile.get('trend_percentage', 0)
            
            prompt_section = f"""


USER'S DATA BASELINES (learned from {total_days} days of their data, confidence: {confidence:.0%}):


- Daily average: {baseline_avg:.1f} clicks
- Typical range: {typical_low:.1f} - {typical_high:.1f} clicks
- Best day: {best_day.capitalize() if best_day else 'N/A'}
- Worst day: {worst_day.capitalize() if worst_day else 'N/A'}
- Current trend: {trend_direction.upper()} ({trend_pct:+.1f}%)


When answering questions about their data, ALWAYS contextualize numbers:

- Compare values to THEIR baseline (not generic benchmarks)
- Mention if something is above/below THEIR typical
- Reference THEIR best/worst patterns when relevant
- Note if current values match THEIR trend


"""
            return prompt_section
            
        except Exception as e:
            print(f"⚠️ Step 8.4: Data profile formatting failed: {e}")
            return ""
    
    def _format_profile_for_prompt(self, profile: dict) -> str:
        """
        🆕 PHASE 7.1: Format user profile for system prompt injection.
        Big Tech Pattern: Netflix "Because you watched..." style personalization.
        
        Args:
            profile: User profile dict from _get_user_profile()
            
        Returns:
            str: Formatted prompt section, or empty string if low confidence
        """
        # ========== GUARD: Low confidence profiles ==========
        confidence = profile.get('confidence', 0)
        total_interactions = profile.get('total_interactions', 0)
        
        if confidence < 0.3 or total_interactions < 5:
            return ""
        
        # ========== EXTRACT TOP PREFERENCES ==========
        try:
            # Top chart type
            chart_prefs = profile.get('chart_preferences', {})
            top_chart = max(chart_prefs.items(), key=lambda x: x[1])[0] if chart_prefs else "bar"
            chart_pct = int(chart_prefs.get(top_chart, 0) * 100)
            
            # Top limit
            limit_prefs = profile.get('limit_preferences', {})
            top_limit = max(limit_prefs.items(), key=lambda x: x[1])[0] if limit_prefs else "10"
            
            # Verbosity preference
            verbosity_prefs = profile.get('verbosity_preferences', {})
            if verbosity_prefs.get('short', 0) > verbosity_prefs.get('long', 0):
                verbosity = "concise, brief"
            else:
                verbosity = "detailed, comprehensive"
            
            # Top routes by success rate
            route_success = profile.get('route_success_rates', {})
            top_routes = sorted(route_success.items(), key=lambda x: x[1], reverse=True)[:2]
            top_routes_str = ", ".join([f"{r[0]} ({int(r[1]*100)}%)" for r in top_routes]) if top_routes else "DATABASE"
            
            # Topic interests
            topics = profile.get('topic_interests', [])[:5]
            topics_str = ", ".join(topics) if topics else "general analytics"
            
            # ========== BUILD PROMPT SECTION ==========
            prompt_section = f"""

USER PREFERENCES (learned from {total_interactions} past interactions, confidence: {confidence:.0%}):

- Preferred chart type: {top_chart} (used {chart_pct}% of time)

- Default result limit: {top_limit}

- Answer style: {verbosity}

- Best performing routes: {top_routes_str}

- Topics of interest: {topics_str}



Apply these preferences as defaults unless user explicitly requests otherwise.

"""
            return prompt_section
            
        except Exception as e:
            print(f"⚠️ Phase 7.1: Profile formatting failed: {e}")
            return ""

    def _get_few_shot_examples(self, user_question: str, route: str = None, session_id: str = None) -> str:
        """
        🆕 PHASE 7: Get few-shot examples from successful past interactions.
        Big Tech Pattern: Netflix/Spotify - "users like you also liked this"
        
        Args:
            user_question: Current user query
            route: Optional route filter (DATABASE, CODE_INTERPRETER, etc.)
            session_id: Optional session for personalization
            
        Returns:
            str: Formatted few-shot examples for LLM prompt, or empty string
        """
        # ========== GUARD: Feature not available ==========
        if not INTERACTION_RAG_AVAILABLE:
            return ""
        
        # ========== GRACEFUL EXECUTION (Big Tech: Never block) ==========
        try:
            rag_worker = get_interaction_rag()
            
            # Find similar successful interactions
            similar = rag_worker.find_similar_successes(
                question=user_question,
                top_k=3,
                min_similarity=0.5,
                route_filter=route,
                session_id=session_id
            )
            
            if not similar:
                print(f"🧠 Phase 7: No similar examples found for: {user_question[:50]}...")
                return ""
            
            # Format as few-shot examples
            examples_text = rag_worker.format_few_shot_examples(
                similar_interactions=similar,
                max_examples=2,
                max_answer_length=300
            )
            
            if examples_text:
                print(f"🧠 Phase 7: Found {len(similar)} similar examples (using top 2)")
            
            return examples_text
            
        except Exception as e:
            # Big Tech Pattern: NEVER block on few-shot failure
            print(f"⚠️ Phase 7: Few-shot lookup failed (non-critical): {e}")
            return ""
    
    def _get_user_preferences(self, user_id: str) -> dict:
        """
        🆕 PHASE 6: Get learned user preferences from universal learner.
        Big Tech Pattern: Netflix/Spotify/YouTube personalization.
        
        Args:
            user_id: User identifier (browser UUID)
            
        Returns:
            dict: User preferences with safe defaults
        """
        # ========== SAFE DEFAULTS (Big Tech: Always have fallback) ==========
        defaults = {
            # Query preferences
            'default_limit': 10,
            'default_time_filter': 'last_month',
            'preferred_order': 'DESC',
            
            # Chart preferences
            'preferred_chart_type': 'bar',
            'preferred_color_scheme': 'blue',
            'show_data_labels': True,
            'show_legend': True,
            
            # Route preferences
            'preferred_route': 'DATABASE',
            
            # Communication preferences
            'verbosity_level': 'normal',  # brief, normal, detailed
            'formality_level': 'casual',  # casual, professional
            'prefers_explanations': True,
            
            # Confidence (0 = no data, 1 = highly confident)
            'overall_confidence': 0.0
        }
        
        # ========== GUARD: No user or anonymous ==========
        if not user_id or user_id == 'anonymous':
            print("🧠 AGI: No user_id, using defaults")
            return defaults
        
        if not UNIVERSAL_LEARNER_AVAILABLE:
            print("🧠 AGI: Universal learner not available, using defaults")
            return defaults
        
        # ========== LOOKUP USER PREFERENCES ==========
        try:
            learner = UniversalLearner()
            prefs = learner.get_user_preferences(user_id)
            
            if prefs and prefs.get('overall_confidence', 0) > 0:
                # Merge: user prefs override defaults (only non-None values)
                merged = {**defaults, **{k: v for k, v in prefs.items() if v is not None}}
                
                print(f"🧠 AGI: Loaded preferences for {user_id[:8]}...")
                print(f"   📊 limit={merged.get('default_limit')}, chart={merged.get('preferred_chart_type')}")
                print(f"   🗣️ verbosity={merged.get('verbosity_level')}, explanations={merged.get('prefers_explanations')}")
                print(f"   🎯 confidence={merged.get('overall_confidence', 0):.2f}")
                
                return merged
            else:
                print(f"🧠 AGI: No learned preferences for {user_id[:8]}..., using defaults")
                return defaults
                
        except Exception as e:
            # Big Tech Pattern: NEVER block on personalization failure
            print(f"⚠️ AGI: Preference lookup failed (using defaults): {e}")
            return defaults
    
    def _apply_verbosity(self, answer: str, verbosity: str) -> str:
        """
        🆕 PHASE 6: Adjust answer length based on user's verbosity preference.
        Big Tech Pattern: Personalized response length.
        
        Args:
            answer: Original answer text
            verbosity: 'brief', 'normal', or 'detailed'
            
        Returns:
            str: Adjusted answer
        """
        if verbosity == 'brief':
            # Truncate long answers, keep first 2-3 sentences
            sentences = answer.split('. ')
            if len(sentences) > 3:
                return '. '.join(sentences[:3]) + '.'
        elif verbosity == 'detailed':
            # Could add more context, but for now just return as-is
            pass
        
        return answer
    
    def _get_learned_chart_type(self, user_id: str) -> str:
        """
        🆕 PHASE 5.2: Get learned chart type preference from user_chart_preferences table.
        Big Tech Pattern: Netflix/Spotify preference learning.
        
        Priority: User explicit request > Learned preference > Default
        
        Args:
            user_id: User identifier (browser UUID)
            
        Returns:
            str: Preferred chart type (e.g., 'bar', 'line', 'heatmap') or None
        """
        # ========== GUARD: Feature not available ==========
        if not CHART_PREFS_AVAILABLE:
            return None
        
        # ========== GUARD: No user or anonymous ==========
        if not user_id or user_id == 'anonymous':
            return None
        
        # ========== LOOKUP LEARNED PREFERENCE ==========
        try:
            worker = ChartPreferencesWorker()
            preferred = worker.get_preferred_chart_type(user_id, default=None)
            
            if preferred:
                print(f"🎨 Learned chart preference for {user_id[:8]}...: {preferred}")
            
            return preferred
            
        except Exception as e:
            # Big Tech Pattern: Never block on preference failure
            print(f"⚠️ Chart preference lookup failed (using default): {e}")
            return None
    
    def _save_to_db(self, session_id: str, user_id: str, user_question: str, final_answer: str):
        """Save conversation turn to database (helper method)"""
        if not session_id:
            return
            
        try:
            db_memory = DBConversationMemory()
            
            # Load existing messages
            messages = db_memory.load(session_id, user_id=user_id or 'demo_user')
            
            # Add new turn
            messages.append({
                'role': 'user',
                'content': user_question,
                'timestamp': datetime.now().isoformat()
            })
            messages.append({
                'role': 'assistant',
                'content': final_answer,
                'timestamp': datetime.now().isoformat()
            })
            
            # Save to database
            db_memory.save(session_id, messages, user_id=user_id or 'demo_user')
            db_memory.close()
            print(f"💾 Saved to database: session {session_id}")
            
        except Exception as e:
            print(f"⚠️ Database save failed (non-critical): {e}")

    def query(
        self, 
        user_question: str, 
        active_doc_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        try:
            # 🔧 CRITICAL FIX: Clear previous interaction state (prevent state leakage between queries)
            # Big Tech Pattern: Stateless request handling (Google/OpenAI pattern)
            self.last_code_interpreter_images = None
            self.last_code_metadata = None
            self.last_web_search_metadata = None
            self.last_template_metadata = None  # 🆕 Reset template metadata
            self.last_route_confidence = None  # 🆕 Reset route confidence
            self.last_route_execution_time_ms = None  # 🆕 Reset route execution time
            self.last_route_error_occurred = False  # 🆕 Reset route error flag
            self.last_route_alternative_exists = False  # 🆕 Reset route alternative flag
            self.last_parameters_used = None  # 🆕 PHASE 5: Reset AGI parameters
            self.last_parameters_extracted = None  # 🆕 PHASE 7.1: Reset extracted params
            self.last_route_used = None
            self.current_route = None
            
            # ========== 🆕 PHASE 6: LOAD USER PREFERENCES (Big Tech Personalization) ==========
            # Netflix/Spotify/YouTube pattern: Personalize EVERY interaction
            self._current_user_prefs = self._get_user_preferences(user_id)
            
            # Extract commonly used preferences for easy access
            _user_limit = self._current_user_prefs.get('default_limit', 10)
            _user_chart = self._current_user_prefs.get('preferred_chart_type', 'bar')
            _user_verbosity = self._current_user_prefs.get('verbosity_level', 'normal')
            _user_time_filter = self._current_user_prefs.get('default_time_filter', 'last_month')
            _user_confidence = self._current_user_prefs.get('overall_confidence', 0)
            # ========== END PHASE 6 LOAD ==========
            
            # ========== 🆕 PHASE 7.1: LOAD USER PROFILE (Big Tech AGI Personalization) ==========
            # Netflix/Spotify/YouTube pattern: "Based on your past 50 interactions..."
            # Uses pre-computed profiles from user_preference_profiles table
            # Safety: Graceful fallback to Phase 6 preferences if profile unavailable
            # ==========================================================================
            # 🐛 FIX: Use user_id for profile lookup (persists across sessions)
            # session_id changes every chat, user_id persists per browser
            self._current_user_profile = self._get_user_profile(user_id or session_id)
            self._profile_prompt_section = self._format_profile_for_prompt(self._current_user_profile)
            
            # Phase 7.1 can OVERRIDE Phase 6 preferences if profile has higher confidence
            if self._current_user_profile.get('confidence', 0) > _user_confidence:
                # Profile has more data - use profile preferences
                _profile_chart_prefs = self._current_user_profile.get('chart_preferences', {})
                if _profile_chart_prefs:
                    _top_profile_chart = max(_profile_chart_prefs.items(), key=lambda x: x[1])[0]
                    _user_chart = _top_profile_chart
                    print(f"🎯 Phase 7.1 override: Using profile chart preference: {_user_chart}")
                
                _profile_limit_prefs = self._current_user_profile.get('limit_preferences', {})
                if _profile_limit_prefs:
                    _top_profile_limit = max(_profile_limit_prefs.items(), key=lambda x: x[1])[0]
                    try:
                        _user_limit = int(_top_profile_limit)
                        print(f"🎯 Phase 7.1 override: Using profile limit preference: {_user_limit}")
                    except ValueError:
                        pass
            # ========== END PHASE 7.1 LOAD ==========
            
            # ========== 🆕 STEP 8.4: LOAD DATA PROFILE (Big Tech AGI Personalization) ==========
            # Fitbit/Spotify/Bank App pattern: "Your average is X, this is +12% above typical"
            # Uses pre-computed baselines from user_data_profiles table
            # Safety: Graceful fallback to raw numbers if profile unavailable
            # ==========================================================================
            # 🚲 CITIBIKE: Skip data profile (lubobali.com context irrelevant for citibike)
            _citibike_check = CITIBIKE_KEYWORDS
            _is_citibike = any(kw in user_question.lower() for kw in _citibike_check)
            
            if _is_citibike:
                self._current_data_profile = {}
                self._data_profile_prompt_section = ""
                print("🚲 Citibike query - skipping data profile injection")
            else:
                self._current_data_profile = self._get_data_profile(user_id or 'default')
                self._data_profile_prompt_section = self._format_data_profile_for_prompt(self._current_data_profile)
            
            if self._current_data_profile.get('confidence_score', 0) > 0.3:
                print(f"📊 Step 8.4: Data profile loaded - baseline={self._current_data_profile.get('baseline_daily_avg', 0):.1f}")
            # ========== END STEP 8.4 LOAD ==========
            
            # Check if user said "yes" after greeting offer
            if self.memory.get_turn_count() >= 1:
                # Get the last assistant message specifically
                last_context = self.memory.get_context(last_n=1)
                
                # Check if last response was the greeting that asks "Want to see what I can do?"
                if (last_context and 
                    "Want to see what I can do?" in last_context and
                    "How's it going?" in last_context):
                    
                    # Check if current message is affirmative (single word response)
                    user_input_lower = user_question.lower().strip()
                    affirmative_patterns = ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'yea', 'ya', 'yup']
                    
                    if user_input_lower in affirmative_patterns:
                        # User wants to see capabilities - trigger KB search
                        kb_result = search_knowledge_base("What can you do?")
                        if kb_result.get('success'):
                            answer = kb_result['answer']
                            self.memory.add_turn(user_question, answer)
                            self._save_to_db(session_id, user_id, user_question, answer)
                            return answer
            
            # Check if this is a conversational greeting (no SQL needed)
            greeting_keywords = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "what's up", "help"]
            is_greeting = any(user_question.lower().strip().startswith(gw) for gw in greeting_keywords)
            
            
            if is_greeting:
                # Hardcoded professional greeting (Step 1 of two-step process)
                answer = """Hi! 👋 How's it going?

Want to see what I can do?"""
                
                self.memory.add_turn(user_question, answer)
                self._save_to_db(session_id, user_id, user_question, answer)
                return answer
            
            # ============= IDENTITY QUESTIONS (LuBot Introduction) =============
            # Check if this is an identity question about Lubo or LuBot
            # Use precise pattern matching to avoid false positives
            question_lower = user_question.lower().strip()
            identity_patterns = [
                "who is lubo", "who are you", "what are you", "what is lubot",
                "who created you", "who made you", "tell me about yourself",
                "what's your name", "what is your name", "who built you"
            ]
            
            # Check if question starts with pattern OR is exactly the pattern (handles punctuation)
            is_identity_question = any(
                question_lower.startswith(pattern) or 
                question_lower.replace("?", "").replace(".", "").strip() == pattern
                for pattern in identity_patterns
            )
            
            if is_identity_question:
                answer = "I'm LuBot, an AI assistant built by Lubo. I specialize in predictive analytics, data insights, coding, document intelligence, report generation, and more. Visit [lubot.ai](https://lubot.ai) to learn more."
                self.memory.add_turn(user_question, answer)
                self._save_to_db(session_id, user_id, user_question, answer)
                self.last_route_used = "IDENTITY"
                return answer
            # ============= END IDENTITY QUESTIONS =============
            
            # ==========================================================================
            # 🏢 PHASE 9 STEP 3: Data Mode Chat Commands (Big Tech: Netflix Voice Control)
            # ==========================================================================
            # Pattern: "switch to demo", "use my data", "show my data", "use demo"
            # Netflix/Spotify: Voice commands for profile switching
            # Safety: Validates user has data before switching to 'user' mode
            # ==========================================================================
            _switch_phrases_demo = ['switch to demo', 'use demo', 'demo data', 'show demo', 'back to demo']
            _switch_phrases_user = ['switch to my data', 'use my data', 'my data', 'show my data', 'use uploaded', 'my uploaded']
            
            _q_lower_switch = user_question.lower().strip()
            _wants_demo = any(phrase in _q_lower_switch for phrase in _switch_phrases_demo)
            _wants_user = any(phrase in _q_lower_switch for phrase in _switch_phrases_user)
            
            if _wants_demo or _wants_user:
                try:
                    from sqlalchemy import create_engine, text
                    import os as _os_switch
                    
                    _db_url = _os_switch.getenv("DATABASE_URL")
                    if _db_url and user_id:
                        _engine = create_engine(_db_url)
                        
                        with _engine.connect() as _conn:
                            if _wants_user:
                                # 🛡️ SAFETY: Check user has data before switching (Netflix: can't switch to empty profile)
                                _check = _conn.execute(text("""
                                    SELECT COUNT(*) FROM user_uploaded_metrics WHERE user_id = :uid
                                """), {"uid": user_id}).scalar()
                                
                                if not _check or _check == 0:
                                    _no_data_msg = "❌ You don't have any uploaded data yet. Upload a CSV first, then say 'use my data'."
                                    self.memory.add_turn(user_question, _no_data_msg)
                                    self._save_to_db(session_id, user_id, user_question, _no_data_msg)
                                    self.last_route_used = "DATA_MODE_SWITCH"
                                    return _no_data_msg
                                
                                # Switch to user mode
                                _conn.execute(text("""
                                    UPDATE users SET data_mode = 'user', updated_at = NOW() WHERE id = :uid
                                """), {"uid": user_id})
                                _conn.commit()
                                
                                _switch_msg = "✅ Switched to **Your Data**. I'll now query your uploaded metrics."
                                print(f"🏢 Phase 9: User {user_id[:8]}... switched to USER mode")
                            else:
                                # Switch to demo mode
                                _conn.execute(text("""
                                    UPDATE users SET data_mode = 'demo', updated_at = NOW() WHERE id = :uid
                                """), {"uid": user_id})
                                _conn.commit()
                                
                                _switch_msg = "✅ Switched to **Demo Data** (lubobali.com analytics)."
                                print(f"🏢 Phase 9: User {user_id[:8]}... switched to DEMO mode")
                        
                        self.memory.add_turn(user_question, _switch_msg)
                        self._save_to_db(session_id, user_id, user_question, _switch_msg)
                        self.last_route_used = "DATA_MODE_SWITCH"
                        return _switch_msg
                    else:
                        print(f"⚠️ Phase 9: No DATABASE_URL or user_id for mode switch")
                        
                except Exception as e:
                    # Big Tech: NEVER block on mode switch failure
                    print(f"⚠️ Phase 9: Mode switch failed (non-critical): {e}")
            # ==========================================================================
            # END PHASE 9 STEP 3
            # ==========================================================================
            
            # ============= IMAGE ANALYSIS (Task 12.1.3) =============
            # Check if there are uploaded images to analyze
            try:
                from fastapi import FastAPI
                # Access FastAPI app.state to check for images
                import sys
                if 'main' in sys.modules:
                    main_module = sys.modules['main']
                    if hasattr(main_module, 'app') and hasattr(main_module.app.state, 'uploaded_images'):
                        if main_module.app.state.uploaded_images:
                            print("🖼️ Images detected - analyzing with GPT-4o Vision")
                            
                            from agent_tools import analyze_images_agent_tool
                            
                            # Get image paths
                            image_paths = [img['path'] for img in main_module.app.state.uploaded_images]
                            
                            # Analyze images with user's question (or default if empty)
                            question_to_send = user_question if user_question.strip() else None
                            analysis = analyze_images_agent_tool(image_paths, question_to_send)
                            
                            # Clear images after analysis
                            main_module.app.state.uploaded_images = []
                            
                            # Return analysis directly
                            self.memory.add_turn(user_question, analysis)
                            self._save_to_db(session_id, user_id, user_question, analysis)
                            self.last_route_used = "IMAGE_ANALYSIS"
                            return analysis
            except Exception as e:
                print(f"⚠️ Image analysis check failed: {e}")
            # ============= END IMAGE ANALYSIS =============
            
            # Check if we have uploaded documents - use semantic search
            print(f"🐛 DEBUG: uploaded_documents count: {len(self.uploaded_documents)}")
            print(f"🐛 DEBUG: vector_store stats: {self.vector_store.get_collection_stats()}")
            
            if self.uploaded_documents:
                # 🏢 BIG TECH PATTERN: Require explicit question (don't auto-analyze)
                user_question_trimmed = user_question.strip()
                is_empty_prompt = len(user_question_trimmed) == 0
                
                # If empty prompt, prompt user to ask a question (ChatGPT/Claude pattern)
                if is_empty_prompt:
                    recent_doc = self.uploaded_documents[-1]['filename']
                    helpful_msg = f"I've received your document: **{recent_doc}**\n\nWhat would you like to know about it?"
                    self.memory.add_turn(user_question, helpful_msg)
                    self._save_to_db(session_id, user_id, user_question, helpful_msg)
                    self.last_route_used = "PROMPT_REQUEST"
                    return helpful_msg
                
                # Special handling: if question mentions "document", always use docs
                doc_keywords = ['document', 'pdf', 'file', 'uploaded']
                has_doc_keyword = any(keyword in user_question.lower() for keyword in doc_keywords)
                
                print(f"🔍 Searching vector store for: {user_question}")
                # Try semantic search in vector store
                similar_chunks = self.vector_store.search(user_question, top_k=10)  # ✅ CHANGED from top_k=3 to top_k=10
                print(f"🔍 Found {len(similar_chunks)} chunks")
                if similar_chunks:
                    top_sim = similar_chunks[0].get('similarity_score', 1.0)
                    print(f"🔍 Top similarity: {top_sim:.3f}")
                
                # Prepare generic-doc detection
                generic_doc_phrases = ['this document', 'the document', 'that document', 'uploaded document', 'the file', 'this file']
                is_generic_doc_question = any(phrase in user_question.lower() for phrase in generic_doc_phrases)

                # ✅ PIN FIRST (hard filter by active_doc_id)
                pinned_chunks = []
                if active_doc_id and similar_chunks:
                    pinned_chunks = [c for c in similar_chunks if c['metadata'].get('doc_id') == active_doc_id]
                    if pinned_chunks:
                        similar_chunks = pinned_chunks
                        print(f"📌 Pinned by active_doc_id={active_doc_id}: {len(similar_chunks)} chunks")
                    else:
                        print(f"⚠️ No chunks matched active_doc_id={active_doc_id}; will try recency if needed")

                # ✅ REAL RECENCY FALLBACK
                if (not active_doc_id or (active_doc_id and not pinned_chunks)) and is_generic_doc_question and len(self.uploaded_documents) > 1:
                    most_recent = self.uploaded_documents[-1]
                    mr_doc_id = most_recent["doc_id"]
                    mr_filename = most_recent["filename"]

                    recent_hits = [c for c in similar_chunks if c["metadata"].get("doc_id") == mr_doc_id]
                    if recent_hits:
                        similar_chunks = recent_hits
                        print(f"🕐 Kept search hits for most recent: {mr_filename} ({len(similar_chunks)} chunks)")
                    else:
                        fallback = self.doc_chunks.get(mr_doc_id, [])
                        if fallback:
                            # ✅ Normalize cached chunks so each has a similarity_score
                            normalized = []
                            for c in fallback[:10]:
                                if "similarity_score" not in c:
                                    c = {**c, "similarity_score": 1.0}
                                normalized.append(c)
                            similar_chunks = normalized
                            print(f"🕐 No search hits; using cached chunks for: {mr_filename} ({len(similar_chunks)} chunks)")
                
                # Optional safety sort
                similar_chunks.sort(key=lambda x: self._parse_upload_dt(x["metadata"]), reverse=True)
                
                # --- classify analytics intent (charts, metrics, time words) ---
                ql = user_question.lower()
                has_time = self._detect_time_period(user_question) != "all_time"
                is_analytics_intent = has_time or any(w in ql for w in [
                    "chart", "graph", "visualize", "plot", "top", "page", "pages",
                    "click", "clicks", "visits", "referrer", "trend", "over time"
                ])

                top_sim = similar_chunks[0].get('similarity_score', 0.0) if similar_chunks else 0.0

                # Use docs ONLY if:
                #  - user explicitly references a document/file, OR
                #  - active_doc_id is set, OR
                #  - it's NOT an analytics-style question AND similarity >= 0.45
                should_use_docs = (
                    has_doc_keyword
                    or bool(active_doc_id)
                    or (not is_analytics_intent and top_sim >= 0.45)
                )

                print(
                    f"🔍 Should use docs: {should_use_docs} "
                    f"(doc_keyword: {has_doc_keyword}, active_doc: {bool(active_doc_id)}, "
                    f"analytics_intent: {is_analytics_intent}, similarity: {top_sim:.3f})"
                )
                
                if should_use_docs and similar_chunks:
                    # Build context from most similar chunks
                    doc_context = "\n\n".join([
                        f"[Chunk from {chunk['metadata']['source_filename']}]\n{chunk['text_content']}"
                        for chunk in similar_chunks[:3]
                    ])
                    
                    # ========== 🆕 PHASE 6: RAG VERBOSITY PERSONALIZATION (Big Tech AGI) ==========
                    # Netflix/Spotify/YouTube pattern: Personalize response length
                    # Safety: Graceful fallback to 'normal' if preference missing
                    # ============================================================================
                    rag_verbosity = ""
                    try:
                        if _user_verbosity == 'brief':
                            rag_verbosity = "Keep your answer brief (2-3 sentences maximum)."
                        elif _user_verbosity == 'detailed':
                            rag_verbosity = "Provide a detailed, comprehensive answer with context."
                        else:
                            rag_verbosity = "Provide a clear, direct answer."
                    except NameError:
                        # Big Tech Pattern: Never block on personalization failure
                        rag_verbosity = "Provide a clear, direct answer."
                    # ========== END PHASE 6 RAG ==========
                    
                    doc_prompt = f"""You are a helpful assistant. Answer this question based on the uploaded documents.

RELEVANT DOCUMENT SECTIONS:
{doc_context}

USER QUESTION: {user_question}

{rag_verbosity}

Provide your answer based on the document content above:"""
                    
                    result = self.generator.call(prompt_kwargs={"input_str": doc_prompt})
                    if result is None:
                        answer = "I found relevant documents, but I'm having trouble processing them right now."
                    else:
                        answer = result.data if hasattr(result, 'data') else str(result)
                    
                    print(f"📄 Document answer (similarity: {similar_chunks[0]['similarity_score']:.3f}): {answer[:100]}...")
                    self.last_route_used = "RAG"
                    self.memory.add_turn(user_question, answer)
                    self._save_to_db(session_id, user_id, user_question, answer)
                    return answer
            
            # ============= INTELLIGENT QUERY ROUTING (LLM-BASED) =============
            # Use LLM to decide if this query needs web search, database, or knowledge base
            # CRITICAL: This is the ONLY place we call _route_query() to avoid duplicate LLM calls
            route = self._route_query(user_question)
            
            # 🆕 PHASE 4: Start timing route execution (Big Tech: measure actual work)
            import time as _time_module
            _route_execution_start = _time_module.time()
            
            # Cache route for _select_tool() to use (avoid duplicate LLM call)
            self.current_route = route
            
            # Store route for source icon detection
            self.last_route_used = route
            
            # Handle WEB_SEARCH route (skip for Ollama + "lubo" questions - safety fallback)
            if route == "WEB_SEARCH" and not (self.model_provider == "ollama" and "lubo" in user_question.lower()):
                from agent_tools import search_web_tool
                
                try:
                    print(f"🌐 Calling search_web_tool with query: {user_question}")
                    result_df = search_web_tool(user_question, num_results=5)
                    print(f"🌐 Search returned: type={type(result_df)}, shape={result_df.shape if hasattr(result_df, 'shape') else 'N/A'}")
                    print(f"🌐 DataFrame empty? {result_df.empty if hasattr(result_df, 'empty') else 'N/A'}")
                    if not result_df.empty:
                        print(f"🌐 First result: {result_df.iloc[0].to_dict()}")
                        
                    if not result_df.empty:
                        # Build context from search results for LLM synthesis
                        search_context = ""
                        for _, row in result_df.iterrows():
                            search_context += f"Source: {row['source_domain']}\n"
                            search_context += f"Title: {row['title']}\n"
                            search_context += f"Content: {row['snippet']}\n\n"
                        
                        # Store web search metadata
                        web_sources = []
                        for _, row in result_df.iterrows():
                            web_sources.append({
                                'url': row['url'],
                                'title': row['title'],
                                'snippet': row['snippet']
                            })
                        
                        self.last_web_search_metadata = {
                            'web_search_query': user_question,
                            'web_sources_count': len(web_sources),
                            'web_sources': web_sources
                        }
                        
                        # ==========================================================================
                        # 🆕 PHASE 7: WEB_SEARCH Few-Shot Learning (Big Tech: Netflix/Spotify AGI)
                        # ==========================================================================
                        # Pattern: "Similar web searches were answered like this"
                        # Safety: Graceful fallback (returns "" if fails, never blocks)
                        # ==========================================================================
                        _web_few_shot = self._get_few_shot_examples(
                            user_question, 
                            route="WEB_SEARCH", 
                            session_id=session_id
                        )
                        _web_few_shot_section = ""
                        if _web_few_shot:
                            _web_few_shot_section = f"""

SUCCESSFUL SIMILAR WEB SEARCHES (match this answer style):

{_web_few_shot}

"""
                            print(f"🧠 Phase 7: WEB_SEARCH using {_web_few_shot.count('Example')} few-shot examples")
                        
                        # ==========================================================================
                        # END PHASE 7 WEB_SEARCH
                        # ==========================================================================
                        
                        # 🆕 PHASE 6: Adjust instructions based on verbosity preference
                        verbosity_instruction = ""
                        if _user_verbosity == 'brief':
                            verbosity_instruction = "- Be VERY concise (1-2 sentences maximum)"
                        elif _user_verbosity == 'detailed':
                            verbosity_instruction = "- Provide detailed information with context"
                        else:
                            verbosity_instruction = "- Be concise (2-3 sentences maximum)"
                        
                        explanation_instruction = ""
                        if self._current_user_prefs.get('prefers_explanations', True):
                            explanation_instruction = "- Include brief context if helpful"
                        else:
                            explanation_instruction = "- Just the facts, no explanations"
                        
                        # Synthesize a clean, conversational answer using LLM
                        synthesis_prompt = f"""You are a helpful assistant. Answer the user's question using the web search results below.
{_web_few_shot_section}
USER QUESTION: {user_question}

WEB SEARCH RESULTS:

{search_context}

INSTRUCTIONS:

{verbosity_instruction}

{explanation_instruction}

- Write naturally like you're talking to a friend

- Do NOT say "according to search results" or "based on the information"

- Do NOT mention sources in your answer

- Just answer the question directly



Your answer:"""
                        
                        result = self.generator.call(prompt_kwargs={"input_str": synthesis_prompt})
                        
                        if result is None:
                            # Fallback: if synthesis fails, show raw results
                            response = "🌐 **Web Search Results:**\n\n"
                            for _, row in result_df.iterrows():
                                response += f"**{row['rank']}. {row['title']}**\n"
                                response += f"   🌐 {row['source_domain']}\n"
                                response += f"   {row['snippet']}\n"
                                response += f"   {row['url']}\n\n"
                        else:
                            # Clean synthesized answer
                            response = result.data if hasattr(result, 'data') else str(result)
                        
                        self.memory.add_turn(user_question, response)
                        self._save_to_db(session_id, user_id, user_question, response)
                        return response
                    else:
                        print("🌐 No web results found, continuing to knowledge base check...")
                        # Fall through to knowledge base check
                        
                except Exception as e:
                    print(f"❌ Web search error: {e}")
                    print("🌐 Web search unavailable, continuing to knowledge base check...")
                    # Fall through to knowledge base check
            
            # Handle DATABASE route - skip KB check and go straight to SQL
            elif route == "DATABASE":
                print("💾 Routed to DATABASE - skipping KB check, proceeding to SQL")
                # Store for later reference
                self.last_route_used = "DATABASE"
                # Continue to SQL logic below (don't return, let it fall through)
            
            # ============= END INTELLIGENT QUERY ROUTING =============
            
            # Check if this is a knowledge base question (about agent capabilities)
            # Skip KB check if already routed to DATABASE
            is_kb_question = False
            if route != "DATABASE":
                kb_keywords = ['what can you do', 'what can you', 'how do i', 'how to', 
                              'time filter', 'chart type', 'model', 'switch', 'features',
                              'capabilities', 'help me', 'show me how']
                is_kb_question = any(keyword in user_question.lower() for keyword in kb_keywords)
                
                # Also check if route explicitly says KNOWLEDGE_BASE
                if route == "KNOWLEDGE_BASE":
                    is_kb_question = True
            
            if is_kb_question:
                print("📚 Knowledge base question detected")
                try:
                    kb_result = search_knowledge_base(user_question)
                    if kb_result.get('success') and kb_result.get('relevance', 0) > 0.3:
                        answer = kb_result['answer']
                        # Source removed - do not append to user answer
                        print(f"✅ Knowledge base answer: {answer[:100]}...")
                        self.memory.add_turn(user_question, answer)
                        self._save_to_db(session_id, user_id, user_question, answer)
                        return answer
                except Exception as e:
                    print(f"⚠️ Knowledge base search failed: {e}")
                    # Continue to normal flow if KB fails
            
            # =============================================================================
            # 🏢 BIG TECH AGI: Route-First Architecture (Netflix/Google Pattern)
            # =============================================================================
            # Principle: Router decides → Execute → Never intercept
            # 
            # Problem this solves:
            # - meta_prompt was short-circuiting data queries by returning SQL templates as text
            # - User asks "What's my revenue?" → Router says CODE_INTERPRETER
            # - But meta_prompt saw schema examples and returned them as the answer
            #
            # Solution:
            # - If router decided DATABASE or CODE_INTERPRETER → skip meta_prompt entirely
            # - Only run meta_prompt for non-data routes (WEB_SEARCH, KNOWLEDGE_BASE, CONTINUE)
            # =============================================================================
            
            if route in ["DATABASE", "CODE_INTERPRETER"]:
                # 🚀 Route-First: Skip meta_prompt, go straight to data execution
                print(f"🚀 Route-First: Skipping meta_prompt, executing {route} directly")
                needs_sql = True  # Force data path
                meta_answer = None
            else:
                # 🌐 Non-data route: Run meta_prompt for general questions
                # 🏢 Big Tech AGI: Personalized schema (Netflix/Spotify pattern)
                # If user has uploaded data, show THEIR schema, not demo data
                schema = get_database_schema(user_id=user_id)
                meta_prompt = f"""You are an analytics assistant. Answer this question using only the context provided. Do NOT generate SQL queries for general questions about the system or creator.

CONTEXT:
{schema}

USER QUESTION: {user_question}

If this is a question about the system, creator, or general information, provide a helpful answer directly without SQL.
If this is an analytics question about clicks, visits, pages, etc., say "NEEDS_SQL"

Your response (short and natural):"""
                
                meta_result = self.generator.call(prompt_kwargs={"input_str": meta_prompt})
                meta_answer = meta_result.data if hasattr(meta_result, 'data') else str(meta_result)
                
                # If LLM didn't say needs SQL (any variation), use that answer directly
                needs_sql = any(keyword in meta_answer.upper() for keyword in ["NEEDS", "SQL", "NEED SQL"]) if meta_answer else False
                if meta_answer and not needs_sql:
                    print(f"📝 Meta answer (no SQL): {meta_answer}")
                    self.memory.add_turn(user_question, meta_answer)
                    self._save_to_db(session_id, user_id, user_question, meta_answer)
                    return meta_answer
            
            # ReAct loop - multiple steps until FINISH
            max_steps = 5
            current_step = 0
            observations = []
            final_answer = None
            
            # Detect time period from question (once, before the loop)
            time_filter = self._detect_time_period(user_question)
            print(f"⏰ Time period detected: {time_filter}")
            
            while current_step < max_steps:
                current_step += 1
                print(f"🔄 Step {current_step}/{max_steps}")
                
                # Select which tool to use
                tool_choice = self._select_tool(user_question, observations)
                print(f"🔧 Tool selected: {tool_choice}")
                
                # Handle different tool choices
                
                # ========== PREDICTION TOOLS (Task 17) ==========
                # ========== REPORT GENERATION TOOLS (Task 17.3) ==========
                if tool_choice == "GENERATE_FORECAST_REPORT":
                    print("📄 Generating forecast report")
                    
                    # Detect time range
                    if 'next week' in user_question.lower():
                        time_range = 'next_week'
                    elif 'next quarter' in user_question.lower():
                        time_range = 'next_quarter'
                    else:
                        time_range = 'next_month'
                    
                    # Check for uploaded file
                    uploaded_file_path = None
                    if self.uploaded_documents:
                        for doc in reversed(self.uploaded_documents[-3:]):
                            if doc['filename'].lower().endswith(('.csv', '.xlsx', '.xls')):
                                uploaded_file_path = f"/tmp/{doc['filename']}"
                                break
                    
                    # Run forecast
                    if uploaded_file_path:
                        forecast_result = forecast_traffic(time_range, uploaded_file_path=uploaded_file_path)
                    else:
                        forecast_result = forecast_traffic(time_range)
                    
                    # Generate report
                    report_format = 'csv' if 'csv' in user_question.lower() else 'excel'
                    
                    if report_format == 'excel':
                        report_result = generate_forecast_excel_report(forecast_result)
                    else:
                        report_result = generate_forecast_csv_report(forecast_result)
                    
                    if report_result.get('success'):
                        download_url = f"http://localhost:8001{report_result['download_url']}"
                        final_answer = f"✅ Forecast report generated successfully!\n\n"
                        final_answer += f"📊 **Summary:** {forecast_result.get('summary', 'N/A')}\n\n"
                        final_answer += f"📥 **Download Report:** {download_url}\n\n"
                        final_answer += f"Report includes: Forecast data, confidence intervals, and insights."
                    else:
                        final_answer = f"❌ Failed to generate report: {report_result.get('error', 'Unknown error')}"
                    
                    observations.append(f"Step {current_step}: Generated forecast report")
                    self.last_route_used = "REPORT_GENERATION"
                    break
                
                elif tool_choice == "GENERATE_ANOMALY_REPORT":
                    print("📄 Generating anomaly report")
                    
                    # Detect time period
                    if 'last week' in user_question.lower():
                        time_period = 'last_week'
                    elif 'last quarter' in user_question.lower():
                        time_period = 'last_quarter'
                    else:
                        time_period = 'last_month'
                    
                    sensitivity = 'high' if 'all' in user_question.lower() else 'medium'
                    
                    # Check for uploaded file
                    uploaded_file_path = None
                    if self.uploaded_documents:
                        for doc in reversed(self.uploaded_documents[-3:]):
                            if doc['filename'].lower().endswith(('.csv', '.xlsx', '.xls')):
                                uploaded_file_path = f"/tmp/{doc['filename']}"
                                break
                    
                    # Run anomaly detection
                    if uploaded_file_path:
                        anomaly_result = detect_traffic_anomalies(time_period, sensitivity, uploaded_file_path=uploaded_file_path)
                    else:
                        anomaly_result = detect_traffic_anomalies(time_period, sensitivity)
                    
                    # Generate report
                    report_format = 'csv' if 'csv' in user_question.lower() else 'excel'
                    
                    if report_format == 'excel':
                        report_result = generate_anomaly_excel_report(anomaly_result)
                    else:
                        report_result = generate_anomaly_csv_report(anomaly_result)
                    
                    if report_result.get('success'):
                        download_url = f"http://localhost:8001{report_result['download_url']}"
                        final_answer = f"✅ Anomaly report generated successfully!\n\n"
                        final_answer += f"📊 **Summary:** {anomaly_result.get('summary', 'N/A')}\n\n"
                        final_answer += f"📥 **Download Report:** {download_url}\n\n"
                        final_answer += f"Report includes: {anomaly_result.get('total_anomalies', 0)} anomalies with severity levels and recommendations."
                    else:
                        final_answer = f"❌ Failed to generate report: {report_result.get('error', 'Unknown error')}"
                    
                    observations.append(f"Step {current_step}: Generated anomaly report")
                    self.last_route_used = "REPORT_GENERATION"
                    break
                
                # ========== END REPORT GENERATION TOOLS ==========
                
                elif tool_choice == "GENERATE_PDF_REPORT":
                    print("📄 Generating PDF report")
                    
                    # Detect time period - support both past and future periods
                    if 'next week' in user_question.lower():
                        time_period = 'next_week'
                    elif 'next quarter' in user_question.lower():
                        time_period = 'next_quarter'
                    elif 'next month' in user_question.lower():
                        time_period = 'next_month'
                    elif 'last week' in user_question.lower():
                        time_period = 'last_week'
                    elif 'last month' in user_question.lower():
                        time_period = 'last_month'
                    elif 'last quarter' in user_question.lower():
                        time_period = 'last_quarter'
                    elif 'last year' in user_question.lower():
                        time_period = 'last_year'
                    else:
                        time_period = 'last_month'
                    
                    # Call PDF generation tool
                    result = generate_pdf_report(
                        title="Strategic Traffic Analysis Report",
                        time_period=time_period,
                        include_scenarios=True,
                        include_recommendations=True
                    )
                    
                    # Unwrap FunctionOutput
                    if hasattr(result, 'output'):
                        result = result.output
                    
                    if result.get('error'):
                        final_answer = f"❌ Failed to generate PDF: {result['error']}"
                    else:
                        final_answer = f"✅ PDF report generated successfully!\n\n"
                        final_answer += f"📄 **{result['filename']}** ({result['file_size_kb']} KB, {result['pages']} pages)\n\n"
                        final_answer += f"📥 **Download:** {result['download_link']}\n\n"
                        final_answer += f"{result['summary']}"
                    
                    observations.append(f"Step {current_step}: Generated PDF report")
                    self.last_route_used = "DOCUMENT_GENERATION"
                    break
                
                elif tool_choice == "GENERATE_EXCEL_EXPORT":
                    print("📊 Generating Excel export")
                    
                    # Detect time period
                    if 'last week' in user_question.lower():
                        time_period = 'last_week'
                    elif 'last quarter' in user_question.lower():
                        time_period = 'last_quarter'
                    elif 'last year' in user_question.lower():
                        time_period = 'last_year'
                    else:
                        time_period = 'last_month'
                    
                    # Call Excel export tool
                    result = generate_excel_export(
                        time_period=time_period,
                        include_predictions=True
                    )
                    
                    # Unwrap FunctionOutput
                    if hasattr(result, 'output'):
                        result = result.output
                    
                    if result.get('error'):
                        final_answer = f"❌ Failed to generate Excel: {result['error']}"
                    else:
                        final_answer = f"✅ Excel spreadsheet generated successfully!\n\n"
                        final_answer += f"📊 **{result['filename']}** ({result['file_size_kb']} KB, {result['sheets']} sheets)\n\n"
                        final_answer += f"📥 **Download:** {result['download_link']}\n\n"
                        final_answer += f"{result['summary']}"
                    
                    observations.append(f"Step {current_step}: Generated Excel export")
                    self.last_route_used = "DOCUMENT_GENERATION"
                    break
                
                elif tool_choice == "GENERATE_DOCX_REPORT":
                    print("📝 Generating Word document")
                    
                    # Detect time period
                    if 'last week' in user_question.lower():
                        time_period = 'last_week'
                    elif 'last quarter' in user_question.lower():
                        time_period = 'last_quarter'
                    else:
                        time_period = 'last_month'
                    
                    # Call Word document tool
                    result = generate_docx_report(
                        title="Traffic Analysis Report",
                        time_period=time_period,
                        include_charts=True
                    )
                    
                    # Unwrap FunctionOutput
                    if hasattr(result, 'output'):
                        result = result.output
                    
                    if result.get('error'):
                        final_answer = f"❌ Failed to generate Word document: {result['error']}"
                    else:
                        final_answer = f"✅ Word document generated successfully!\n\n"
                        final_answer += f"📝 **{result['filename']}** ({result['file_size_kb']} KB, ~{result['pages']} pages)\n\n"
                        final_answer += f"📥 **Download:** {result['download_link']}\n\n"
                        final_answer += f"{result['summary']}"
                    
                    observations.append(f"Step {current_step}: Generated Word document")
                    self.last_route_used = "DOCUMENT_GENERATION"
                    break
                
                if tool_choice == "PREDICT_TRAFFIC":
                    print("🔮 Executing forecast_traffic tool")
                    
                    # Detect time range from query
                    if 'next week' in user_question.lower():
                        time_range = 'next_week'
                    elif 'next quarter' in user_question.lower():
                        time_range = 'next_quarter'
                    else:
                        time_range = 'next_month'  # default
                    
                    # Only use uploaded files if user explicitly references them
                    file_keywords = ['file', 'csv', 'excel', 'xlsx', 'xls', 'uploaded', 'this data', 'this file', 'the file', 'the csv', 'the excel', 'from file', 'from csv']
                    user_wants_file = any(keyword in user_question.lower() for keyword in file_keywords)
                    
                    uploaded_file_path = None
                    if user_wants_file and self.uploaded_documents:
                        # Check last 3 uploaded documents for CSV/Excel
                        for doc in reversed(self.uploaded_documents[-3:]):
                            if doc['filename'].lower().endswith(('.csv', '.xlsx', '.xls')):
                                uploaded_file_path = f"/tmp/{doc['filename']}"
                                print(f"📊 Found uploaded data file: {doc['filename']}")
                                break
                    
                    # Call forecast with or without file
                    if uploaded_file_path:
                        result = forecast_traffic(time_range, uploaded_file_path=uploaded_file_path)
                        observations.append(f"Step {current_step}: Forecasted traffic from uploaded file for {time_range}")
                    else:
                        result = forecast_traffic(time_range)
                        observations.append(f"Step {current_step}: Forecasted traffic from database for {time_range}")
                    
                    # Format answer
                    if result.get('error'):
                        final_answer = result.get('summary', result.get('error'))
                    else:
                        final_answer = result.get('summary', str(result))
                        # Add data source info if from file
                        if 'data_source' in result and 'uploaded' in result['data_source']:
                            final_answer = f"📊 Analysis from uploaded file:\n\n{final_answer}"
                    
                    self.last_route_used = "PREDICTION"
                    break
                
                elif tool_choice == "DETECT_ANOMALIES":
                    print("🔍 Executing detect_traffic_anomalies tool")
                    
                    # Detect time period and sensitivity
                    if 'last week' in user_question.lower():
                        time_period = 'last_week'
                    elif 'last quarter' in user_question.lower():
                        time_period = 'last_quarter'
                    else:
                        time_period = 'last_month'  # default
                    
                    sensitivity = 'high' if 'all' in user_question.lower() else 'medium'
                    
                    # Only use uploaded files if user explicitly references them
                    file_keywords = ['file', 'csv', 'excel', 'xlsx', 'xls', 'uploaded', 'this data', 'this file', 'the file', 'the csv', 'the excel', 'from file', 'from csv']
                    user_wants_file = any(keyword in user_question.lower() for keyword in file_keywords)
                    
                    uploaded_file_path = None
                    if user_wants_file and self.uploaded_documents:
                        # Check last 3 uploaded documents for CSV/Excel
                        for doc in reversed(self.uploaded_documents[-3:]):
                            if doc['filename'].lower().endswith(('.csv', '.xlsx', '.xls')):
                                uploaded_file_path = f"/tmp/{doc['filename']}"
                                print(f"📊 Found uploaded data file: {doc['filename']}")
                                break
                    
                    # Call anomaly detection with or without file
                    if uploaded_file_path:
                        result = detect_traffic_anomalies(time_period, sensitivity, uploaded_file_path=uploaded_file_path)
                        observations.append(f"Step {current_step}: Detected anomalies from uploaded file")
                    else:
                        result = detect_traffic_anomalies(time_period, sensitivity)
                        observations.append(f"Step {current_step}: Detected anomalies from database in {time_period}")
                    
                    # Format anomalies nicely
                    if result.get('anomalies'):
                        anomaly_list = []
                        for a in result['anomalies'][:5]:  # Show top 5
                            anomaly_list.append(f"- **{a['date']}**: {a['type']} ({a['severity']}) - {a['actual_value']} (expected {a['expected_value']})")
                        
                        summary_prefix = "📊 Analysis from uploaded file:\n\n" if uploaded_file_path else ""
                        final_answer = f"{summary_prefix}{result['summary']}\n\n**Anomalies Detected:**\n" + "\n".join(anomaly_list) + f"\n\n**💡 Recommendation:** {result['recommendation']}"
                    else:
                        summary_prefix = "📊 Analysis from uploaded file:\n\n" if uploaded_file_path else ""
                        final_answer = f"{summary_prefix}{result['summary']}"
                    
                    self.last_route_used = "PREDICTION"
                    break
                
                elif tool_choice == "ANALYZE_TRENDS":
                    print("📈 Executing analyze_page_trends tool")
                    # Detect time period
                    if 'last week' in user_question.lower():
                        time_period = 'last_week'
                    elif 'last quarter' in user_question.lower():
                        time_period = 'last_quarter'
                    else:
                        time_period = 'last_month'  # default
                    
                    result = analyze_page_trends(time_period, top_n=10)
                    observations.append(f"Step {current_step}: Analyzed page trends for {time_period}")
                    
                    # Format trends nicely
                    answer_parts = [result['summary']]
                    
                    if result.get('trending_up'):
                        answer_parts.append("\n**📈 Trending Up:**")
                        for page in result['trending_up'][:5]:
                            growth_pct = page['growth_rate'] * 100
                            answer_parts.append(f"- {page['page_name']}: +{growth_pct:.1f}% ({int(page['previous_clicks'])} → {int(page['current_clicks'])} clicks)")
                    
                    if result.get('trending_down'):
                        answer_parts.append("\n**📉 Trending Down:**")
                        for page in result['trending_down'][:5]:
                            decline_pct = abs(page['growth_rate'] * 100)
                            answer_parts.append(f"- {page['page_name']}: -{decline_pct:.1f}% ({int(page['previous_clicks'])} → {int(page['current_clicks'])} clicks)")
                    
                    if result.get('recommendations'):
                        answer_parts.append("\n**💡 Recommendations:**")
                        for rec in result['recommendations']:
                            answer_parts.append(f"- {rec}")
                    
                    final_answer = "\n".join(answer_parts)
                    self.last_route_used = "PREDICTION"
                    break
                
                # ========== END PREDICTION TOOLS ==========
                
                # ========== CODE INTERPRETER TOOL ==========
                elif tool_choice == "RUN_PYTHON_CODE":
                    print("🐍 Executing code interpreter")
                    
                    # 🆕 TEMPLATE SYSTEM - Try templates first, fallback to LLM
                    from tools.code_interpreter_templates import match_template, extract_parameters, get_template
                    
                    # 🆕 PHASE 6: Prepare user preferences for code interpreter
                    # 🆕 PHASE 5.2: Check learned chart preference FIRST (Big Tech: specific > general)
                    _learned_chart = self._get_learned_chart_type(user_id)
                    if _learned_chart:
                        print(f"🎨 Phase 5.2: CODE_INTERPRETER using learned chart: {_learned_chart}")
                    
                    code_user_prefs = None
                    if _user_confidence > 0.1:
                        code_user_prefs = {
                            'default_limit': _user_limit,
                            'preferred_chart_type': _learned_chart or _user_chart,  # Phase 5.2 > Phase 6
                            'preferred_color_scheme': self._current_user_prefs.get('preferred_color_scheme', 'blue'),
                            'show_data_labels': self._current_user_prefs.get('show_data_labels', True),
                            'show_legend': self._current_user_prefs.get('show_legend', True)
                        }
                        print(f"🎯 Code interpreter using learned prefs: limit={_user_limit}, chart={_learned_chart or _user_chart}")
                    
                    # Step 1: Try to match a template
                    # 🆕 PHASE 6: Pass user preferences for personalization
                    template_match = match_template(user_question, user_prefs=code_user_prefs)
                    python_code = None
                    
                    if template_match:
                        # ✅ Template found - use it!
                        print(f"✅ Using template: {template_match['name']} (confidence: {template_match['confidence']:.2f})")
                        
                        try:
                            # ================================================================
                            # 🏢 BIG TECH AGI: Detect user data context (Netflix/Spotify)
                            # ================================================================
                            # Check if user has uploaded data → use their schema
                            # No uploaded data → use demo schema (lubobali.com)
                            # ================================================================
                            _is_user_data = False
                            _user_data_context = None
                            
                            if user_id:
                                try:
                                    # Check if user has uploaded data
                                    _is_user_data = should_use_user_data(user_id)
                                    
                                    if _is_user_data:
                                        # Get user's data metadata from database
                                        _user_data_context = self._get_user_data_context(user_id)
                                        if _user_data_context:
                                            print(f"🏢 AGI: User data detected - metric='{_user_data_context.get('metric_column')}', dims={_user_data_context.get('dimensions')}")
                                        else:
                                            print(f"🏢 AGI: User data flag set but no metadata found, using demo schema")
                                            _is_user_data = False
                                    else:
                                        print(f"🏢 AGI: No user data, using demo schema (lubobali.com)")
                                except Exception as e:
                                    # Big Tech: NEVER block on context detection failure
                                    print(f"⚠️ AGI: User data detection failed (using demo): {e}")
                                    _is_user_data = False
                                    _user_data_context = None
                            # ================================================================
                            # END USER DATA DETECTION
                            # ================================================================
                            
                            # Extract parameters using LLM
                            # 🆕 PHASE 6: Pass user preferences as defaults
                            # 🏢 BIG TECH AGI: Pass user data context for schema routing
                            params = extract_parameters(
                                user_question, 
                                template_match['name'], 
                                self.model_provider,
                                user_prefs=code_user_prefs,
                                is_user_data=_is_user_data,
                                user_data_context=_user_data_context,
                                user_id=user_id
                            )
                            
                            # Get template code with parameters filled
                            python_code = get_template(template_match['name'], params)
                            
                            print(f"🐛 DEBUG Generated code (lines 85-100):\n{python_code.split(chr(10))[85:100]}")
                            print(f"✅ Template code generated ({len(python_code)} chars)")
                            
                            # 🆕 TASK 18.1 STEP 1.5: Store template metadata for interaction logging
                            self.last_template_metadata = {
                                'sql_template_used': template_match['name'],
                                'sql_template_confidence': template_match['confidence']
                            }
                            
                        except Exception as e:
                            print(f"⚠️ Template failed: {e}, falling back to LLM code generation")
                            template_match = None  # Force fallback to LLM
                    
                    # Step 2: If no template or template failed, use LLM code generation
                    if not template_match or python_code is None:
                        print("🤖 No template found - using LLM code generation")
                        
                        # 🆕 PHASE 7: Get few-shot examples for code generation (Big Tech AGI)
                        _code_few_shot = self._get_few_shot_examples(user_question, route="CODE_INTERPRETER", session_id=session_id)
                        _code_few_shot_section = f"\nSUCCESSFUL SIMILAR ANALYSES (use these as reference):\n{_code_few_shot}\n" if _code_few_shot else ""
                        
                        # Build Python code using LLM (existing code)
                        code_prompt = f"""You are a data scientist. Write Python code to answer this question using the database.
{_code_few_shot_section}
USER QUESTION: {user_question}

DATABASE CONNECTION (CRITICAL - Use this exact pattern):
```python
import psycopg2
import pandas as pd
from scipy import stats

# DATABASE_URL is pre-injected as a constant (no 'os' module needed)
# Just use: conn = psycopg2.connect(DATABASE_URL)
# Example:
conn = psycopg2.connect(DATABASE_URL)
df = pd.read_sql("SELECT * FROM table", conn)
conn.close()
```

AVAILABLE TABLES:
- click_logs: Individual click events
  Columns: timestamp, page_name, ip_hash, user_agent, session_id, referrer, time_on_page, tag
  
- daily_click_summary: Pre-aggregated daily statistics  
  Columns: date, total_clicks, top_pages, avg_time_on_page, device_split, top_referrers, repeat_visits, project_name, tag, created_at, id

CRITICAL: To calculate unique visitors, use click_logs table:
  SELECT DATE(timestamp) as date, COUNT(DISTINCT ip_hash) as unique_visitors
  FROM click_logs
  GROUP BY DATE(timestamp)

CRITICAL SECURITY RULES:
- DO NOT import: os, sys, subprocess, socket, requests, urllib
- DO import: pandas, numpy, scipy, matplotlib, seaborn, psycopg2
- DO NOT use: os.getenv(), os.environ, open(), eval(), exec()
- DO use: pandas.read_sql() for database queries

INSTRUCTIONS:
1. Import only allowed libraries (pandas, numpy, scipy.stats, matplotlib, seaborn, psycopg2)
2. Use pd.read_sql() to query database directly (connection handled internally)
3. Perform statistical analysis (correlation, regression, etc.)
4. Print clear, interpretable results
5. Create visualizations if appropriate (matplotlib/seaborn)
6. Do NOT include markdown code blocks

EXAMPLE PATTERN:
```python
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import psycopg2

# Connect to database (DATABASE_URL is pre-injected as constant, no 'os' needed)
conn = psycopg2.connect(DATABASE_URL)

# Query database
# Get total_clicks from daily_click_summary
query1 = "SELECT date, total_clicks FROM daily_click_summary"
df1 = pd.read_sql(query1, conn)

# Get unique_visitors from click_logs (no column in daily_click_summary)
query2 = "SELECT DATE(timestamp) as date, COUNT(DISTINCT ip_hash) as unique_visitors FROM click_logs GROUP BY DATE(timestamp)"
df2 = pd.read_sql(query2, conn)

# Merge data
df = pd.merge(df1, df2, on='date', how='inner')

# Perform analysis
correlation, p_value = pearsonr(df['total_clicks'], df['unique_visitors'])
print(f"Correlation: {{correlation:.3f}}")
print(f"P-value: {{p_value:.4f}}")

# Close connection
conn.close()
```

Generate ONLY Python code (no markdown, no explanations):"""
                        
                        code_result = self.generator.call(prompt_kwargs={"input_str": code_prompt})
                        
                        if code_result is None:
                            final_answer = "Error: Could not generate Python code"
                            break
                        
                        python_code = code_result.data if hasattr(code_result, 'data') else str(code_result)
                        
                        # Clean up code (remove markdown if present)
                        python_code = python_code.strip()
                        if python_code.startswith("```python"):
                            python_code = python_code.replace("```python", "").replace("```", "").strip()
                        elif python_code.startswith("```"):
                            python_code = python_code.replace("```", "").strip()
                        
                        print(f"🤖 LLM generated code ({len(python_code)} chars)")
                    
                    # Step 3: Execute code (same for both template and LLM)
                    exec_result = run_python_code(python_code, f"Statistical analysis: {user_question[:50]}")
                    
                    observations.append(f"Step {current_step}: Executed Python code for statistical analysis")
                    
                    # Format response
                    if exec_result['success']:
                        final_answer = f"**Analysis Results:**\n\n{exec_result['output']}"
                        
                        # Store images in agent instance for main.py to access
                        if exec_result.get('images'):
                            self.last_code_interpreter_images = exec_result['images']
                            final_answer += f"\n\n📊 Generated {len(exec_result['images'])} visualization(s)"
                            print(f"🖼️ Stored {len(exec_result['images'])} images in agent instance")
                        else:
                            self.last_code_interpreter_images = None
                        
                        # Store code execution metadata
                        self.last_code_metadata = {
                            'code_executed': python_code,
                            'code_output': exec_result.get('output', ''),
                            'code_error': None,
                            'execution_time_ms': int(exec_result.get('execution_time', 0) * 1000)
                        }
                        
                        final_answer += f"\n\n⏱️ Executed in {exec_result['execution_time']}s"
                    else:
                        final_answer = f"❌ Code execution failed:\n{exec_result['error']}"
                        
                        # Store error metadata
                        self.last_code_metadata = {
                            'code_executed': python_code,
                            'code_output': exec_result.get('output', ''),
                            'code_error': exec_result.get('error', 'Unknown error'),
                            'execution_time_ms': int(exec_result.get('execution_time', 0) * 1000)
                        }
                        
                        self.last_code_interpreter_images = None
                    
                    self.last_route_used = "CODE_INTERPRETER"
                    break
                # ========== END CODE INTERPRETER TOOL ==========
                
                if tool_choice == "FINISH":
                    # If we have observations with SQL results, use the last answer
                    if observations and "SQL executed" in str(observations):
                        # 🆕 PHASE 6: Apply verbosity preference to final answer
                        if _user_verbosity and _user_confidence > 0.3:
                            final_answer = self._apply_verbosity(final_answer, _user_verbosity)
                        
                        # Return the answer from the last SQL execution
                        self.memory.add_turn(user_question, final_answer)
                        self._save_to_db(session_id, user_id, user_question, final_answer)
                        return final_answer
                    else:
                        # No data available, generatea a conversational response
                        finish_prompt = f"""You are a friendly analytics assistant. The user asked: "{user_question}"
                        
Provide a helpful, conversational response without using any tools or data."""
                        
                        result = self.generator.call(prompt_kwargs={"input_str": finish_prompt})
                        answer = result.data if hasattr(result, 'data') else str(result) if result else "I'm here to help with your analytics!"
                        
                        observations.append(f"Step {current_step}: FINISH - answering without tools")
                        self.memory.add_turn(user_question, answer)
                        self._save_to_db(session_id, user_id, user_question, answer)
                        return answer
                
                elif tool_choice == "CHART":
                    print("📊 Chart tool selected")
                    
                    # Check if we have SQL data from previous step
                    if 'sql_data' not in locals() or sql_data is None:
                        observations.append(f"Step {current_step}: CHART - Need SQL data first, running SQL query")
                        # Fall through to SQL execution below
                        tool_choice = "SQL"  # Force SQL path to run correction logic
                    else:
                        # We have SQL data, generate chart
                        chart_request = self._detect_chart_request(user_question)
                        
                        if chart_request['wants_chart'] and sql_data.get('data'):
                            try:
                                # Convert SQL data to DataFrame format for chart tool
                                import pandas as pd
                                df = pd.DataFrame(sql_data['data'], columns=sql_data['columns'])
                                
                                # Detect chart configuration intelligently
                                chart_config = generate_chart_from_data(df, user_question, model_provider=self.model_provider)
                                
                                # 🆕 PHASE 5.2: Apply LEARNED chart preferences FIRST (from user_chart_preferences table)
                                # Big Tech Pattern: Netflix/Spotify - database-backed preference learning
                                learned_chart = self._get_learned_chart_type(user_id)
                                if learned_chart and not chart_request.get('chart_type'):
                                    chart_config['chart_type'] = learned_chart
                                    print(f"🎨 Phase 5.2: Using learned chart type: {learned_chart}")
                                # Fallback to Phase 6 universal learner preferences
                                elif _user_confidence > 0.3:
                                    if not chart_request.get('chart_type') and _user_chart:
                                        chart_config['chart_type'] = _user_chart
                                        print(f"🎯 Phase 6: Using universal learner chart preference: {_user_chart}")
                                
                                # Apply color scheme preference (Phase 6)
                                if _user_confidence > 0.3:
                                    user_colors = self._current_user_prefs.get('preferred_color_scheme')
                                    if user_colors:
                                        chart_config['color_scheme'] = user_colors
                                        print(f"🎯 Using learned color preference: {user_colors}")
                                
                                # Generate chart with detected config
                                chart_result = generate_chart_tool(
                                    data=df,
                                    chart_type=chart_config.get('chart_type', chart_request['chart_type']),
                                    title=user_question,
                                    config=chart_config
                                )
                                
                                if isinstance(chart_result, dict) and "chart_url" in chart_result:
                                    print(f"📊 Chart generated: {chart_result['chart_url']}")
                                    observations.append(f"Step {current_step}: Chart generated successfully")
                                    
                                    # Return markdown image embed instead of link
                                    if 'final_answer' in locals() and final_answer:
                                        final_answer = f"{final_answer}\n\n![{chart_result['title']}]({chart_result['chart_url']})"
                                    else:
                                        final_answer = f"![{chart_result['title']}]({chart_result['chart_url']})"
                                else:
                                    print(f"⚠️ Chart generation failed: {chart_result.get('error', 'Unknown error')}")
                                    observations.append(f"Step {current_step}: Chart generation failed")
                                break
                                
                            except Exception as e:
                                print(f"❌ Chart generation failed: {e}")
                                observations.append(f"Step {current_step}: Chart generation failed")
                                # Continue with text-only answer
                
                # If tool_choice == "SQL" or anything else, continue with existing SQL logic below
                
                # Get conversation context from memory
                context = self.memory.get_context(last_n=3)
                context_section = f"\nCONVERSATION HISTORY:\n{context}\n" if context else ""
                
                # Call tools manually and let LLM process
                # 🏢 Big Tech AGI: Personalized schema (Netflix/Spotify pattern)
                # If user has uploaded data, show THEIR schema, not demo data
                schema = get_database_schema(user_id=user_id)
                
                # 🚲 CITIBIKE: Skip template matching (templates are for click_logs, not citibike)
                _citibike_keywords = CITIBIKE_KEYWORDS
                _is_citibike_query = any(kw in user_question.lower() for kw in _citibike_keywords)
                
                if _is_citibike_query:
                    print("🚲 Citibike query - trying citibike templates FIRST")
                    template_result = None  # Default (triggers LLM fallback if no match)
                    
                    # 🚲 CITIBIKE TEMPLATES: Pre-validated SQL (Big Tech: Templates > LLM)
                    if CITIBIKE_TEMPLATES_AVAILABLE:
                        try:
                            citibike_result = get_citibike_sql_from_template(user_question)
                            if citibike_result:
                                # Mimic sql_templates format - existing code handles the rest
                                template_result = {
                                    'sql': citibike_result['sql'],
                                    'template_name': citibike_result['template_name'],
                                    'confidence': citibike_result['confidence'],
                                    '_meta': {'is_citibike': True}
                                }
                                print(f"🚲 Citibike template matched: {citibike_result['template_name']}")
                            else:
                                print("🚲 No citibike template matched - LLM fallback")
                        except Exception as e:
                            print(f"⚠️ Citibike template error (non-critical): {e}")
                    else:
                        print("🚲 Citibike templates not loaded - LLM fallback")
                else:
                    # 🎯 Try SQL template first (Big Tech pattern - fast path)
                    # 🆕 PHASE 6: Pass user preferences for personalization (limit, time_filter)
                    # 🏢 Big Tech AGI: Multi-tenant SQL templates (Netflix/Spotify pattern)
                    # Templates now check user data FIRST before falling back to demo data
                    template_result = get_sql_from_template(
                        user_question, 
                        self.model_provider,
                        user_prefs={
                            'default_limit': _user_limit,
                            'default_time_filter': _user_time_filter,
                            'preferred_order': self._current_user_prefs.get('preferred_order', 'DESC')
                        } if _user_confidence > 0.1 else None,  # Only apply if we have learned data
                        user_id=user_id  # 🏢 Multi-tenant: Query user's data, not demo data
                    )
                
                # 🆕 STEP 1.5: Extract SQL + metadata from template result
                sql_query = None
                if template_result:
                    sql_query = template_result['sql']
                    # 🆕 PHASE 5 AGI: Capture parameters used for learning
                    if template_result.get('_meta'):
                        self.last_parameters_used = template_result['_meta'].get('parameters_used', [])
                        print(f"🧠 AGI: Captured {len(self.last_parameters_used)} parameters for learning")
                    # Store template metadata for interaction logging (same as CODE_INTERPRETER)
                    self.last_template_metadata = {
                        'sql_template_used': template_result['template_name'],
                        'sql_template_confidence': template_result['confidence']
                    }
                    print(f"📋 DATABASE template metadata captured: {template_result['template_name']} ({template_result['confidence']:.2f})")
                
                # If no template match, use LLM fallback
                if not sql_query:
                    # 🆕 PHASE 7: Get few-shot examples for SQL generation (Big Tech AGI)
                    _sql_few_shot = self._get_few_shot_examples(user_question, route="DATABASE", session_id=session_id)
                    _sql_few_shot_section = f"\nSUCCESSFUL SIMILAR QUERIES (use these as reference):\n{_sql_few_shot}\n" if _sql_few_shot else ""
                    
                    # 🆕 Phase 7.1: Inject user profile into SQL prompt
                    _profile_section = self._profile_prompt_section if hasattr(self, '_profile_prompt_section') else ""
                    
                    # Generate SQL query using LLM
                    sql_prompt = f"""You are an analytics assistant. Generate a SQL query to answer this question.

DATABASE SCHEMA:
{schema}
{context_section}{_sql_few_shot_section}{_profile_section}
USER QUESTION: {user_question}

IMPORTANT INSTRUCTIONS:
- CRITICAL: ALWAYS add a space before WHERE keyword. WRONG: "FROM click_logsWHERE" CORRECT: "FROM click_logs WHERE"
- CRITICAL: Questions like "show me clicks on X" or "clicks from X" mean COUNT clicks, use: SELECT COUNT(*) FROM click_logs WHERE page_name = 'X'
- For "how many" or "total clicks" questions, use: SELECT COUNT(*) FROM click_logs
- Do NOT use GROUP BY for simple count questions
- Do NOT select page_name for total count questions
- For device type queries, ALWAYS use CASE statement (no 'device' column exists): SELECT CASE WHEN user_agent LIKE '%Mobile%' OR user_agent LIKE '%Android%' OR user_agent LIKE '%iPhone%' THEN 'Mobile' WHEN user_agent LIKE '%Tablet%' OR user_agent LIKE '%iPad%' THEN 'Tablet' ELSE 'Desktop' END as device, COUNT(*) as count FROM click_logs GROUP BY device ORDER BY count DESC
- For "top pages" or "which pages" questions, use: SELECT page_name, COUNT(*) FROM click_logs GROUP BY page_name ORDER BY COUNT(*) DESC
- CRITICAL: When filtering by page_name, use ONLY the actual page name (e.g., 'portfolio', 'home', 'about'). Do NOT include words like 'page' or 'clicks' in the page_name value. Example: WHERE page_name = 'portfolio' NOT WHERE page_name = 'portfolio page'
- CRITICAL MULTI-PAGE LINE CHART DETECTION (Check FIRST before other rules):
  * Keywords that REQUIRE multi-series: "top N pages", "2 pages", "3 pages", "most visited pages", "compare", "each page", "multiple pages", "all pages"
  * If query has ANY of these + "over time" or "line" or "trend" → YOU MUST generate 3-column SQL
  * REQUIRED FORMAT: SELECT DATE(timestamp) as date, page_name, COUNT(*) as count FROM click_logs WHERE page_name IN (SELECT page_name FROM click_logs GROUP BY page_name ORDER BY COUNT(*) DESC LIMIT N) GROUP BY date, page_name ORDER BY date, page_name
  * CRITICAL: You MUST include page_name in SELECT and GROUP BY for multi-page charts
  * Example query "top 2 pages over time" → SELECT DATE(timestamp) as date, page_name, COUNT(*) as count FROM click_logs WHERE page_name IN (SELECT page_name FROM click_logs GROUP BY page_name ORDER BY COUNT(*) DESC LIMIT 2) GROUP BY date, page_name ORDER BY date, page_name
- For SINGLE-PAGE line charts: SELECT DATE(timestamp) as date, COUNT(*) as count FROM click_logs WHERE page_name = 'specific_page' GROUP BY date ORDER BY date
- For GENERAL time trends (no page specified): SELECT DATE(timestamp) as date, COUNT(*) as count FROM click_logs GROUP BY date ORDER BY date
- WRONG: Never use UNION ALL for multi-series - this causes GROUP BY errors
- Do NOT add WHERE clauses for time filtering - the system handles time filters automatically

Generate ONLY a SQL query WITHOUT time filters (no explanations, no markdown):"""
                    
                    sql_result = self.generator.call(prompt_kwargs={"input_str": sql_prompt})
                    
                    # Better error handling for None results
                    if sql_result is None:
                        return "Error: No response from LLM for SQL generation"
                    
                    # Debug: Print the raw result
                    print(f"🔍 DEBUG: sql_result object: {sql_result}")
                    print(f"🔍 DEBUG: sql_result type: {type(sql_result)}")
                    
                    sql_query = sql_result.data if hasattr(sql_result, 'data') else str(sql_result)
                    
                    print(f"🔍 DEBUG: sql_query value: {sql_query}")
                    print(f"🔍 DEBUG: sql_query type: {type(sql_query)}")
                    
                    # =============================================================================
                    # 🧠 AGI PHASE 5: Parameter Extraction (Big Tech Pattern)
                    # =============================================================================
                    # REMOVED: Inline parameter tracking that caused time_filter variable collision
                    # 
                    # NEW ARCHITECTURE (Big Tech: Netflix/Spotify/Google):
                    # - Request path: Execute fast, log sql_query to interaction_log
                    # - Batch job: sql_parameter_worker.py extracts parameters ASYNC
                    # - Benefits: No variable collision, zero request overhead, complete data
                    #
                    # Parameter extraction now handled by:
                    # 1. Template matches: sql_templates.py returns _meta.parameters_used
                    # 2. LLM fallback: sql_parameter_worker.py batch job (runs at 5 AM)
                    # =============================================================================
                    
                    # Check if sql_query is None or empty
                    if sql_query is None or sql_query == "":
                        print(f"⚠️ DEBUG: sql_query is None or empty")
                        return "Error: LLM returned empty SQL query"
                    
                    # End of LLM generation
                # Template or LLM - continue with SQL query
                
                # Clean up the SQL query (remove markdown formatting)
                sql_query = sql_query.strip()
                if sql_query.startswith("```sql"):
                    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
                elif sql_query.startswith("```"):
                    sql_query = sql_query.replace("```", "").strip()
                
                # ====== SQL SPACING FIX ======
                import re
                # Robust fix: Remove any junk text between table name and SQL keywords
                # Handles: click_logsWHERE, click_logsFilterWhere, click_logsSomeJunkWHERE
                # Does NOT touch SQL functions like CURRENT_DATE
                sql_query = re.sub(r'(click_logs|daily_click_summary)\w*(WHERE|FROM)', r'\1 \2', sql_query, flags=re.IGNORECASE)
                sql_query = re.sub(r'\s+', ' ', sql_query).strip()  # clean multiple spaces
                print(f"🔧 FIXED SQL spacing: {sql_query}")
                # ====== END FIX ======
                
                # Defensive programming: sanitize only dangerous patterns, preserve safe WHERE filters
                dangerous_patterns = [
                    r'\bINSERT\b', r'\bUPDATE\b', r'\bDELETE\b', r'\bDROP\b',
                    r'\bALTER\b', r'\bCREATE\b', r'\bEXEC\b', r'\bEXECUTE\b', r'--', r'/\*', r'\bxp_cmdshell\b'
                ]
                for pattern in dangerous_patterns:
                    if re.search(pattern, sql_query, re.IGNORECASE):
                        print(f"🚫 Blocked dangerous SQL pattern: {pattern}")
                        sql_query = "SELECT COUNT(*) as count FROM click_logs LIMIT 1"
                        break
                if re.search(r'\bWHERE\b', sql_query, re.IGNORECASE):
                    print("✅ Preserving WHERE clause for filtering")
                sql_query = sql_query.strip()
                print(f"🛡️ Sanitized SQL: {sql_query}")
                
                print(f"📊 Generated SQL: {sql_query}")
                
                # 🎯 SMART PAGE NAME CORRECTION (before executing SQL)
                # Check if this is a page-specific query (multi-page or single-page)
                if 'page_name in' in sql_query.lower() or 'page_name =' in sql_query.lower():
                    import re
                    # Extract requested page names from SQL query (handles both IN and = patterns)
                    match = re.search(r"(?:IN\s*\(\s*'([^']+)'(?:,\s*'([^']+)')?\s*\)|=\s*'([^']+)')", sql_query, re.IGNORECASE)
                    if match:
                        requested_pages = []
                        if match.group(1):  # IN clause - first page
                            requested_pages.append(match.group(1))
                        if match.group(2):  # IN clause - second page
                            requested_pages.append(match.group(2))
                        if match.group(3):  # = clause - single page
                            requested_pages.append(match.group(3))
                        
                        print(f"🔍 Checking page names: {requested_pages}")
                        
                        # Fuzzy match each requested page
                        corrected_pages = []
                        needs_clarification = False
                        clarification_options = []
                        
                        for req_page in requested_pages:
                            match_result = self._fuzzy_match_pages(req_page)
                            
                            if match_result['exact_match']:
                                # Exact match found - use it
                                corrected_pages.append(match_result['exact_match'])
                                print(f"✅ Exact match: '{req_page}' → '{match_result['exact_match']}'")
                            elif match_result['fuzzy_matches'] and len(match_result['fuzzy_matches']) == 1:
                                # Single fuzzy match - auto-correct
                                best_match = match_result['fuzzy_matches'][0][0]
                                corrected_pages.append(best_match)
                                print(f"🔧 Auto-corrected: '{req_page}' → '{best_match}'")
                            elif match_result['fuzzy_matches'] and len(match_result['fuzzy_matches']) > 1:
                                # Multiple matches - need clarification
                                needs_clarification = True
                                clarification_options.extend(match_result['fuzzy_matches'])
                                print(f"⚠️ Multiple matches for '{req_page}': {match_result['fuzzy_matches']}")
                            else:
                                # No match - need clarification
                                needs_clarification = True
                                clarification_options.extend(match_result['fuzzy_matches'][:3])
                                print(f"❌ No match for '{req_page}'")
                        
                        # Handle clarification or auto-correction
                        if needs_clarification:
                            # Ask user to clarify
                            options_text = '\n'.join([f"• {page} ({clicks} clicks)" for page, clicks in clarification_options[:5]])
                            clarification_msg = f"""I couldn't find exact matches for those page names.

Did you mean one of these pages?
{options_text}

Try asking: \"compare [page1] and [page2] with different colors\" """
                            self.memory.add_turn(user_question, clarification_msg)
                            self._save_to_db(session_id, user_id, user_question, clarification_msg)
                            return clarification_msg
                        elif corrected_pages and len(corrected_pages) == len(requested_pages):
                            # All pages auto-corrected - update SQL query
                            corrected_sql = sql_query
                            for i, (old, new) in enumerate(zip(requested_pages, corrected_pages)):
                                corrected_sql = corrected_sql.replace(f"'{old}'", f"'{new}'", 1)
                            sql_query = corrected_sql
                            print(f"✅ Corrected SQL: {sql_query}")
                
                # Execute the SQL query
                # 🏢 Big Tech AGI: Multi-tenant SQL execution (Netflix/Spotify pattern)
                # Security: Auto-injects user_id filter for user tables (prevents data leakage)
                sql_data = run_sql_query(sql_query, time_filter, user_id=user_id)
                
                # 🧠 AGI: Store SQL for main.py to capture
                self.last_sql_query = sql_query
                
                # ========== 🧠 PHASE 7.1 STEP 6: Real-Time Parameter Extraction ==========
                # Big Tech Pattern: Netflix/Spotify instant learning (don't wait for batch)
                # Safety: Never blocks, graceful failure, batch job catches misses
                # ==========================================================================
                try:
                    from workers.sql_parameter_worker import extract_sql_parameters
                    
                    _extracted = extract_sql_parameters(sql_query)
                    if _extracted:
                        self.last_parameters_extracted = _extracted
                        print(f"🧠 AGI Real-Time: Extracted {len(_extracted)} parameters: {list(_extracted.keys())}")
                    else:
                        self.last_parameters_extracted = None
                        print(f"🧠 AGI Real-Time: No parameters extracted")
                except Exception as e:
                    # Big Tech: NEVER block on extraction failure
                    self.last_parameters_extracted = None
                    print(f"⚠️ AGI Real-Time: Extraction skipped (non-critical): {e}")
                # ========== END PHASE 7.1 STEP 6 ==========
                
                print(f"📊 SQL Result: {sql_data}")
                
                # Check for errors
                if sql_data.get('error'):
                    return f"❌ Error: {sql_data['error']}"
                
                # Process result - use Python formatting for simple count queries
                # Check if this is a simple COUNT(*) query with single number result
                if sql_data['row_count'] == 1 and len(sql_data['data'][0]) == 1:
                    # Extract the total count
                    total_count = sql_data['data'][0][0]
                    
                    # Use Python string formatting for clean, consistent answers
                    answer = None
                    
                    if "last week" in user_question.lower():
                        answer = f"There were {total_count} clicks last week."
                    elif "last month" in user_question.lower():
                        answer = f"There were {total_count} clicks last month."
                    elif "last year" in user_question.lower():
                        answer = f"There were {total_count} clicks last year."
                    elif "yesterday" in user_question.lower():
                        answer = f"There were {total_count} clicks yesterday."
                    elif "today" in user_question.lower():
                        answer = f"There were {total_count} clicks today."
                    elif "last 2 months" in user_question.lower():
                        answer = f"There were {total_count} clicks in the last 2 months."
                    elif "last 3 months" in user_question.lower():
                        answer = f"There were {total_count} clicks in the last 3 months."
                    elif "last 6 months" in user_question.lower():
                        answer = f"There were {total_count} clicks in the last 6 months."
                    
                    if answer:
                        # 🆕 STEP 8.4: Contextualize simple counts (Big Tech AGI)
                        if hasattr(self, '_current_data_profile') and self._current_data_profile.get('confidence_score', 0) > 0.3:
                            try:
                                context_insight = self._contextualize_result(
                                    float(total_count), 
                                    "daily_clicks", 
                                    self._current_data_profile
                                )
                                # Append context to answer
                                answer = f"{answer}\n\n📊 **Your Context:** {context_insight}"
                            except Exception as e:
                                print(f"⚠️ Step 8.4: Contextualization failed for simple count (non-critical): {e}")
                        
                        # Python formatted the answer - save and return immediately
                        print(f"✅ Python-formatted answer: {answer}")
                        observations.append(f"Step {current_step}: SQL executed successfully with results")
                        final_answer = answer
                        break
                
                # Check if user wants a chart after getting SQL data
                chart_request = self._detect_chart_request(user_question)
                
                # 🆕 AUTO-DETECT CHART DATA (like ChatGPT Advanced Data Analysis)
                # Auto-generate charts for both time-series AND categorical data
                auto_chart = False
                is_time_series = False
                is_categorical = False
                is_ranking = False  # 🧠 AGI: Detect ranking queries (top N, most, best)
                
                # ========== 🧠 AGI: RANKING DETECTION (Big Tech: Intent > Data Shape) ==========
                # Netflix/Spotify pattern: User INTENT overrides data structure
                # "top 4 pages" = ranking = bar chart (even if data has dates)
                # "pages over time" = trend = line chart
                # ==========================================================================
                _q_lower = user_question.lower()
                _ranking_keywords = ['top', 'rank', 'most', 'least', 'best', 'worst', 'highest', 'lowest', 'popular']
                _trend_keywords = ['over time', 'trend', 'timeline', 'daily', 'weekly', 'monthly', 'history']
                
                is_ranking = any(kw in _q_lower for kw in _ranking_keywords)
                is_trend_query = any(kw in _q_lower for kw in _trend_keywords)
                
                # 🧠 AGI RULE: Ranking WITHOUT trend keywords = bar chart (ignore date columns)
                if is_ranking and not is_trend_query:
                    print(f"🧠 AGI: Ranking query detected - will use bar chart (ignoring date column)")
                # ========== END AGI RANKING DETECTION ==========
                
                if sql_data.get('data') and len(sql_data['columns']) >= 2:
                    first_col = sql_data['columns'][0].lower()
                    second_col = sql_data['columns'][1].lower()
                    
                    # Check if first column is a date (time-series chart)
                    # 🧠 AGI: Only treat as time-series if NOT a ranking query
                    is_time_series = first_col in ['date', 'timestamp', 'time', 'day', 'month', 'year'] and not (is_ranking and not is_trend_query)
                    
                    # Check if first column is categorical (bar chart)
                    # Categorical columns: device, page_name, referrer, tag, session_id, etc.
                    is_categorical = first_col in ['device', 'page_name', 'referrer', 'tag', 'hour', 'source']
                    
                    # Check if second column is numeric (count, clicks, visits, etc.)
                    is_numeric = second_col in ['count', 'clicks', 'visits', 'total', 'sum', 'avg', 'unique_visitors']
                    
                    # Check if we have enough data points
                    has_enough_data = sql_data['row_count'] >= 2  # Lowered from 5 to 2 for categorical data
                    
                    if is_time_series and has_enough_data:
                        auto_chart = True
                        print(f"📊 Auto-detected time-series data: {sql_data['row_count']} rows with date column")
                    elif is_categorical and is_numeric and has_enough_data:
                        auto_chart = True
                        print(f"📊 Auto-detected categorical data: {sql_data['row_count']} rows with {first_col} column")
                
                # Generate chart if user requested OR if auto-detected
                if (chart_request['wants_chart'] or auto_chart) and sql_data.get('data'):
                    try:
                        import pandas as pd
                        df = pd.DataFrame(sql_data['data'], columns=sql_data['columns'])
                        
                        # Detect chart configuration intelligently
                        chart_config = generate_chart_from_data(df, user_question, model_provider=self.model_provider)
                        
                        # 🆕 PHASE 5.2: Apply LEARNED chart preferences FIRST (from user_chart_preferences table)
                        # Big Tech Pattern: Netflix/Spotify - database-backed preference learning
                        learned_chart = self._get_learned_chart_type(user_id)
                        if learned_chart and not chart_request.get('chart_type'):
                            chart_config['chart_type'] = learned_chart
                            print(f"🎨 Phase 5.2: Using learned chart type: {learned_chart}")
                        # Fallback to Phase 6 universal learner preferences
                        elif _user_confidence > 0.3:
                            if not chart_request.get('chart_type') and _user_chart:
                                chart_config['chart_type'] = _user_chart
                                print(f"🎯 Phase 6: Using universal learner chart preference: {_user_chart}")
                        
                        # Apply color scheme preference (Phase 6)
                        if _user_confidence > 0.3:
                            user_colors = self._current_user_prefs.get('preferred_color_scheme')
                            if user_colors:
                                chart_config['color_scheme'] = user_colors
                                print(f"🎯 Using learned color preference: {user_colors}")
                        
                        # Generate chart with detected config
                        # ========== 🧠 AGI: Smart Chart Type Selection (Big Tech Pattern) ==========
                        # Priority order (Netflix/Spotify: User intent > Learned prefs > Data shape):
                        # 1. User explicit request ("bar chart", "line chart") - handled by chart_request
                        # 2. Ranking queries ("top N") - always bar chart
                        # 3. Trend queries ("over time") - always line chart  
                        # 4. Learned preference (Phase 5.2/7.1)
                        # 5. Data shape detection (categorical=bar, time-series=line)
                        # ==========================================================================
                        
                        # Start with data-shape default
                        default_chart_type = 'line'  # Default for time-series
                        
                        # 🧠 AGI Override 1: Categorical data → bar
                        if auto_chart and is_categorical:
                            default_chart_type = 'bar'
                        
                        # 🧠 AGI Override 2: Ranking queries → ALWAYS bar (highest priority)
                        if is_ranking and not is_trend_query:
                            default_chart_type = 'bar'
                            print(f"🧠 AGI: Ranking query → forcing bar chart")
                        
                        # 🧠 AGI Override 3: Trend queries → ALWAYS line
                        if is_trend_query and not is_ranking:
                            default_chart_type = 'line'
                            print(f"🧠 AGI: Trend query → forcing line chart")
                        # ========== END AGI CHART TYPE SELECTION ==========
                        
                        chart_result = generate_chart_tool(
                            data=df,
                            chart_type=chart_config.get('chart_type', chart_request.get('chart_type', default_chart_type)),
                            title=user_question,
                            config=chart_config
                        )
                        
                        if isinstance(chart_result, dict) and "chart_url" in chart_result:
                            print(f"📊 Chart generated after SQL: {chart_result['chart_url']}")
                            summary = answer if 'answer' in locals() else ''
                            final_answer = f"{summary}\n\n![{chart_result['title']}]({chart_result['chart_url']})"
                        else:
                            summary = answer if 'answer' in locals() else ''
                            final_answer = summary
                        break
                    except Exception as e:
                        print(f"❌ Chart generation failed: {e}")
                        # Continue with text-only answer
                
                # If not a simple count, continue with LLM answer generation for complex queries
                if not final_answer:
                    # 🆕 PHASE 7: Get few-shot examples for answer formatting (Big Tech AGI)
                    _answer_few_shot = self._get_few_shot_examples(user_question, route=route, session_id=session_id)
                    _answer_few_shot_section = f"\nSUCCESSFUL SIMILAR ANSWERS (match this style):\n{_answer_few_shot}\n" if _answer_few_shot else ""
                    
                    # 🆕 STEP 8.4: Include data profile in answer prompt
                    _data_profile_section = self._data_profile_prompt_section if hasattr(self, '_data_profile_prompt_section') else ""
                    
                    answer_prompt = f"""Based on the query results below, answer the user's question.
{_answer_few_shot_section}{_data_profile_section}
USER QUESTION: {user_question}

QUERY RESULTS:
{sql_data}

CRITICAL: When showing numbers, contextualize them against the user's baselines if available.
Example: Instead of "14.8 clicks", say "14.8 clicks (+12% above your typical 13.2)"

Provide a clear, personalized answer:"""
                    
                    answer_result = self.generator.call(prompt_kwargs={"input_str": answer_prompt})
                    
                    if answer_result is None:
                        return "Error: No response from LLM for answer generation"
                    
                    answer = answer_result.data if hasattr(answer_result, 'data') else str(answer_result)
                    
                    if answer is None:
                        return "Error: LLM returned empty answer"
                    
                    observations.append(f"Step {current_step}: SQL executed successfully with results")
                    final_answer = answer
            
            # If we reach here, max steps reached without FINISH
            if final_answer:
                # 🆕 PHASE 6: Apply verbosity preference to final answer
                if _user_verbosity and _user_confidence > 0.3:
                    final_answer = self._apply_verbosity(final_answer, _user_verbosity)
                
                # Save to old memory (backward compatibility)
                self.memory.add_turn(user_question, final_answer)
                
                # 🆕 NEW: Save to database if session_id provided (Big Tech pattern)
                self._save_to_db(session_id, user_id, user_question, final_answer)
                
                # 🆕 PHASE 4: Capture route execution time before return (Big Tech: measure before exit)
                if '_route_execution_start' in locals():
                    self.last_route_execution_time_ms = int((_time_module.time() - _route_execution_start) * 1000)
                    print(f"⏱️ Route execution time: {self.last_route_execution_time_ms}ms")
                
                # Clear cached route after query completes
                self.current_route = None
                return final_answer
            else:
                # Clear cached route even on error
                self.current_route = None
                return "Error: Maximum steps reached without completion"
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"❌ Agent error: {error_msg}")
            # 🆕 PHASE 4: Track that an error occurred
            self.last_route_error_occurred = True
            print(f"🚨 Route error occurred: True")
            # Clear cached route on exception
            self.current_route = None
            return error_msg
    
    def reset_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        print("🧹 Memory cleared")
    
    def switch_model(self, provider: str):
        self.model_client, self.model_kwargs = self._get_model_config(provider)
        self.generator = Generator(
            model_client=self.model_client,
            model_kwargs=self.model_kwargs,
            use_cache=False
        )
        self.model_provider = provider
        print(f"✅ Switched to {provider.upper()}")
    
    def get_current_model(self) -> str:
        return self.model_provider
    
    def add_document(self, filename: str, content: str):
        """Add an uploaded document to the agent's knowledge with semantic search."""
        global _shared_uploaded_documents  # ✅ ADD THIS LINE
        
        # Generate unique doc_id
        doc_id = str(uuid.uuid4())
        
        # Store original document in SHARED list with doc_id and upload_date
        upload_date = datetime.now().isoformat()
        _shared_uploaded_documents.append({
            'doc_id': doc_id,
            'filename': filename,
            'content': content,
            'upload_date': upload_date
        })
        print(f"📄 Document added: {filename} (doc_id: {doc_id}) ({len(content)} chars)")
        
        # Create chunks for vector store with doc_id metadata
        metadata = {
            'doc_id': doc_id,  # ✅ ADD doc_id
            'source_filename': filename,  # ✅ unified key
            'filename': filename,  # backward compatible
            'file_path': f'/uploads/{filename}',
            'file_extension': os.path.splitext(filename)[1],
            'upload_date': upload_date
        }
        
        chunks = self.text_chunker.chunk_text(content, metadata)
        
        # Add chunks to vector store
        success = self.vector_store.add_chunks(chunks)
        
        if success:
            print(f"✅ Added {len(chunks)} chunks to vector store")
        else:
            print(f"⚠️ Failed to add chunks to vector store")
        
        # ✅ Store chunks and metadata for caching
        self.doc_chunks[doc_id] = chunks
        self.doc_meta[doc_id] = {"filename": filename, "upload_date": upload_date}
    
_agent_instance = None
_shared_vector_store = None
_shared_text_chunker = None
_shared_uploaded_documents = []  # ✅ ADD THIS LINE

def get_agent(model_provider: str = "openai"):
    global _agent_instance, _shared_vector_store, _shared_text_chunker, _shared_uploaded_documents  # ✅ ADD _shared_uploaded_documents
    
    # Initialize shared components once
    if _shared_vector_store is None:
        _shared_vector_store = SimpleFAISSVectorStore()
        _shared_text_chunker = TextChunker(chunk_size=200, overlap_size=30)
        print("✅ Shared vector store initialized")
    
    # Check if we need to switch models
    if _agent_instance is None or _agent_instance.get_current_model() != model_provider:
        print(f"🔄 Switching agent to {model_provider}")
        _agent_instance = AnalyticsReActAgent(model_provider)
        
        # Inject shared vector store and chunker
        _agent_instance.vector_store = _shared_vector_store
        _agent_instance.text_chunker = _shared_text_chunker
        _agent_instance.uploaded_documents = _shared_uploaded_documents  # ✅ ADD THIS LINE
        print(f"✅ Shared vector store injected ({_shared_vector_store.get_collection_stats()['total_chunks']} chunks)")
    
    return _agent_instance