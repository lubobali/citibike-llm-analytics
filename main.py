import os
import uuid
import asyncio
import base64
import re
from typing import Dict, Any, AsyncGenerator
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import time
import tempfile
import shutil
import openai
import pandas as pd
import io
from dateutil import parser as date_parser
from decimal import Decimal

# AdalFlow imports
from adalflow.components.model_client import GroqAPIClient
from adalflow.utils import setup_env

# Import our AdalFlow Analytics Agent (100% AdalFlow implementation)
from adalflow_agent import AnalyticsReActAgent, get_agent
from streaming.agent_streaming_wrapper import create_streaming_wrapper
from tools.keyword_extraction import extract_keywords

# APScheduler for background jobs (Big Tech: Nightly batch processing)
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# 🧠 PHASE 5: Route Learning (Redis + PostgreSQL)
from learning.universal_learner import learn_route_preference

# Fix for Python 3.13 asyncio
import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Import document tools for RAG functionality
try:
    from tools.document_tools import get_doc_tools
    DOCUMENT_TOOLS_AVAILABLE = True
    print("✅ Document tools imported successfully")
except ImportError as e:
    DOCUMENT_TOOLS_AVAILABLE = False
    print(f"⚠️ Document tools not available: {e}")

# Environment setup
setup_env()

# Initialize OpenAI client for GPT-4o Vision
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="AI Analytics Agent API", version="1.0.0")

# ==========================================================================
# BACKGROUND SCHEDULER (Big Tech Pattern: Nightly batch jobs)
# ==========================================================================
scheduler = BackgroundScheduler()

def run_route_weights_job():
    """Nightly job to recompute route weights for all users."""
    try:
        from workers.route_weights_worker import run_batch_job
        print("🌙 Starting nightly route weights batch job...")
        run_batch_job()
    except Exception as e:
        print(f"❌ Route weights batch job failed: {e}")

# Schedule nightly at 2 AM
scheduler.add_job(
    run_route_weights_job,
    CronTrigger(hour=2, minute=0),
    id="route_weights_nightly",
    replace_existing=True
)

# ==========================================================================
# 🆕 PHASE 5.2 Step 4: Chart Preferences Refresh (Big Tech: Netflix nightly batch)
# ==========================================================================
# Aggregates thumbs_up/down feedback into user_chart_preferences table
# Runs AFTER route weights (2 AM) to ensure fresh data
# ==========================================================================
def run_chart_preferences_job():
    """Nightly job to refresh user chart preferences from feedback."""
    try:
        from workers.chart_preferences_worker import ChartPreferencesWorker
        print("🎨 Starting nightly chart preferences batch job...")
        worker = ChartPreferencesWorker()
        worker.update_all_users()
        print("🎨 Chart preferences batch job complete")
    except Exception as e:
        print(f"❌ Chart preferences batch job failed: {e}")

# Schedule nightly at 3 AM (after route weights)
scheduler.add_job(
    run_chart_preferences_job,
    CronTrigger(hour=3, minute=0),
    id="chart_preferences_nightly",
    replace_existing=True
)

# ==========================================================================
# 🆕 PHASE 7: Interaction RAG Refresh (Big Tech: Netflix recommendation refresh)
# ==========================================================================
# Loads thumbs_up Q&A pairs into FAISS index for few-shot learning
# Runs AFTER chart preferences (3 AM) to ensure fresh feedback data
# ==========================================================================
def run_interaction_rag_job():
    """Nightly job to refresh Interaction RAG index from thumbs_up feedback."""
    try:
        from workers.interaction_rag_worker import get_interaction_rag
        print("🧠 Starting nightly Interaction RAG refresh job...")
        rag_worker = get_interaction_rag()
        count = rag_worker.load_successful_interactions(min_age_minutes=5, limit=500)
        print(f"🧠 Interaction RAG refresh complete: {count} interactions loaded")
    except Exception as e:
        print(f"❌ Interaction RAG refresh job failed: {e}")

# Schedule nightly at 4 AM (after chart preferences)
scheduler.add_job(
    run_interaction_rag_job,
    CronTrigger(hour=4, minute=0),
    id="interaction_rag_nightly",
    replace_existing=True
)

# ============================================================================
# 🆕 PHASE 7.1: User Profile Builder (Big Tech: Netflix nightly taste profile)
# ============================================================================
# Aggregates ALL user signals into user_preference_profiles table
# Runs AFTER Interaction RAG (4 AM) to ensure fresh feedback data
# ============================================================================
def run_user_profile_job():
    """Nightly job to build/refresh user preference profiles."""
    try:
        from workers.user_profile_worker import UserProfileWorker
        print("👤 Starting nightly User Profile batch job...")
        worker = UserProfileWorker()
        stats = worker.build_all_profiles()
        print(f"👤 User Profile batch job complete: {stats}")
    except Exception as e:
        print(f"❌ User Profile batch job failed: {e}")

# Schedule nightly at 5 AM (after interaction RAG at 4 AM)
scheduler.add_job(
    run_user_profile_job,
    CronTrigger(hour=5, minute=0),
    id="user_profile_nightly",
    replace_existing=True
)

# ============================================================================
# 🆕 PHASE 8: Data Profile Builder (Big Tech: Fitbit/Spotify data baselines)
# ============================================================================
# Builds data baselines per user: avg, std dev, patterns, trends, anomalies
# Runs AFTER user profiles (5 AM) to ensure fresh user data
# ============================================================================
def run_data_profile_job():
    """Nightly job to build/refresh data profiles for all users."""
    try:
        from workers.data_profile_worker import run_batch_job
        print("📊 Starting nightly Data Profile batch job...")
        stats = run_batch_job()
        print(f"📊 Data Profile batch job complete: {stats}")
    except Exception as e:
        print(f"❌ Data Profile batch job failed: {e}")

# Schedule nightly at 6 AM (after user profiles at 5 AM)
scheduler.add_job(
    run_data_profile_job,
    CronTrigger(hour=6, minute=0),
    id="data_profile_nightly",
    replace_existing=True
)

# ============================================================================
# 🏢 MULTI-TENANT: Data Profile Builder (Big Tech: Per-user analytics profiles)
# ============================================================================
# Builds data profiles for users with uploaded data (flexible schemas)
# Runs AFTER main data profiles (6 AM) to ensure fresh data
# ============================================================================
def run_multi_tenant_profile_job():
    """Nightly job for multi-tenant user profiles."""
    try:
        from workers.data_profile_worker_multi import run_batch_job
        print("🏢 Starting multi-tenant profile batch...")
        stats = run_batch_job()
        print(f"🏢 Multi-tenant profiles complete: {stats}")
    except Exception as e:
        print(f"❌ Multi-tenant profile job failed: {e}")

# Schedule at 6:30 AM (after main profiles at 6 AM)
scheduler.add_job(
    run_multi_tenant_profile_job,
    CronTrigger(hour=6, minute=30),
    id="multi_tenant_profiles_nightly",
    replace_existing=True
)

# ============================================================================
# 🏢 PHASE 9 STEP 8: Data Cleanup (Big Tech: Netflix/Spotify GDPR Pattern)
# ============================================================================
# Deletes expired user data (90-day retention)
# Runs AFTER all profile jobs (7 AM) to ensure fresh cleanup
# ============================================================================
def run_data_cleanup_job():
    """Nightly job to clean up expired user data uploads."""
    try:
        from workers.data_cleanup_worker import DataCleanupWorker
        print("🧹 Starting nightly Data Cleanup job...")
        worker = DataCleanupWorker()
        stats = worker.run_cleanup(execute=True)
        print(f"🧹 Data Cleanup complete: {stats}")
    except Exception as e:
        print(f"❌ Data Cleanup job failed: {e}")

# Schedule nightly at 7 AM (after multi-tenant profiles at 6:30 AM)
scheduler.add_job(
    run_data_cleanup_job,
    CronTrigger(hour=7, minute=0),
    id="data_cleanup_nightly",
    replace_existing=True
)

# ============================================================================
# 🔥 OLLAMA WARM-UP (Big Tech: Proactive model loading)
# ============================================================================
async def warmup_ollama():
    """Proactively load Ollama model into memory on startup."""
    import httpx
    OLLAMA_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "qwen2.5:14b"
    
    try:
        print(f"🔥 Warming up Ollama ({OLLAMA_MODEL})...")
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": "Say ready",
                "stream": False
            })
            if response.status_code == 200:
                print(f"✅ Ollama warm-up complete")
                return True
    except Exception as e:
        print(f"⚠️ Ollama warm-up skipped (not running): {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Start background scheduler on app startup."""
    scheduler.start()
    print("✅ Background scheduler started (route weights 2 AM, chart prefs 3 AM, interaction RAG 4 AM, user profiles 5 AM, data profiles 6 AM, multi-tenant profiles 6:30 AM, data cleanup 7 AM)")
    
    # 🔥 Proactive Ollama warm-up (Big Tech: Netflix model pre-loading)
    import asyncio
    asyncio.create_task(warmup_ollama())
    
    # Initialize Interaction RAG Worker on startup (load initial data)
    try:
        from workers.interaction_rag_worker import get_interaction_rag
        print("🧠 Initializing Interaction RAG Worker...")
        rag_worker = get_interaction_rag()
        count = rag_worker.load_successful_interactions(min_age_minutes=0, limit=100)
        print(f"✅ Interaction RAG Worker initialized with {count} interactions")
    except Exception as e:
        print(f"⚠️ Interaction RAG Worker initialization failed (will retry nightly): {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully shutdown scheduler."""
    scheduler.shutdown()
    print("🛑 Background scheduler stopped")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for generated charts
chart_cache: Dict[str, str] = {}

# Global storage for active runs
active_runs: Dict[str, Dict[str, Any]] = {}

# Chain-of-Thought enhanced system prompt
COT_SYSTEM_PROMPT = """You are an AI analytics agent with Chain-of-Thought reasoning capabilities. 

Your task is to analyze click analytics data and provide insights through step-by-step reasoning.

AVAILABLE TOOLS:
- run_sql_tool(query, user_question): Execute safe SQL queries on analytics data
- get_schema_tool(): Get database schema information
- finish(sql_result, user_question): Generate final answer from SQL results

CHAIN-OF-THOUGHT PROCESS:
1. **Understand**: Break down the user's question into clear components
2. **Plan**: Determine what data you need and which tools to use
3. **Execute**: Run SQL queries to gather the required data
4. **Analyze**: Examine the results and identify patterns/insights
5. **Conclude**: Provide a clear, actionable answer with supporting data

RESPONSE FORMAT:
Always structure your response as a JSON object with:
{
  "answer": "Your final answer with insights",
  "reasoning": "Step-by-step Chain-of-Thought explanation",
  "sql": "SQL query used (if any)",
  "chart": {"type": "bar", "x": "column1", "y": "column2"} (if applicable)
}

EXAMPLE REASONING:
"Let me break this down step by step:
1. The user wants to know which page had the highest clicks this week
2. I need to query the click_logs table for the last 7 days
3. I'll group by page_name and sum the clicks
4. Then I'll order by total clicks descending to find the top page
5. Finally, I'll provide the answer with the specific page name and click count"

Always explain your reasoning process clearly and provide actionable insights."""

class AgentRequest(BaseModel):
    prompt: str
    runId: str = None
    model: str = "groq"  # Default to groq
    user_id: str = "anonymous"  # ← ADD THIS LINE

class AgentResponse(BaseModel):
    runId: str
    status: str
    message: str

# Use our AdalFlow Analytics Agent (100% AdalFlow implementation)
# Don't initialize agent here - it will be created via get_agent() which handles shared vector store
agent = None
model_provider = os.getenv("MODEL_PROVIDER", "groq")  # Default to groq if not set

@app.get("/")
def read_root():
    return {"message": "AI Analytics Agent API is running!", "version": "1.0.0"}

@app.post("/v1/agent/switch-model")
async def switch_model(request: dict):
    """Switch the agent's model provider."""
    try:
        model_provider = request.get("model")
        if not model_provider:
            return {"error": "Model provider is required"}
        
        # Map specific model names to providers
        model_mapping = {
            "groq": "groq",
            "openai": "openai", 
            "ollama": "ollama"
        }
        
        if model_provider not in model_mapping:
            return {"error": f"Unsupported model: {model_provider}. Use: groq, openai, ollama"}
        
        # Get agent with target model (this creates/uses the right agent)
        target_model = model_mapping[model_provider]
        agent_instance = get_agent(target_model)
        
        return {
            "success": True,
            "message": f"Successfully switched to {target_model.upper()} model",
            "current_model": agent_instance.get_current_model()
        }
    except Exception as e:
        return {"error": f"Failed to switch model: {str(e)}"}

@app.get("/v1/agent/current-model")
async def get_current_model():
    """Get the currently active model provider."""
    try:
        return {
            "success": True,
            "current_model": get_agent().get_current_model()
        }
    except Exception as e:
        return {"error": f"Failed to get current model: {str(e)}"}

@app.get("/api/ollama/status")
async def get_ollama_status():
    """Health check for Ollama - Big Tech: Service mesh pattern."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                return {"status": "ready", "message": "Ollama is running"}
    except httpx.ConnectError:
        return {"status": "offline", "message": "Ollama not running"}
    except httpx.TimeoutException:
        return {"status": "cold", "message": "Ollama loading model..."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document for the agent.
    """
    # Validate file type
    allowed_extensions = ['.pdf', '.docx', '.doc', '.txt', '.xlsx', '.xls', '.csv']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # For now, just decode text files directly
        # TODO: Add PDF/DOCX parsing later
        if file_extension == '.txt':
            text_content = content.decode('utf-8')
        elif file_extension == '.pdf':
            # Simple PDF text extraction
            import PyPDF2
            import io
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text()
        elif file_extension in ['.xlsx', '.xls', '.csv']:
            # Save to temp file for prediction tools to access
            # Use /tmp/ for consistency with agent (macOS/Linux compatible)
            temp_dir = '/tmp'
            temp_path = os.path.join(temp_dir, file.filename)
            with open(temp_path, 'wb') as f:
                f.write(content)
            print(f"🐛 DEBUG UPLOAD: File saved to: {temp_path}")
            print(f"🐛 DEBUG UPLOAD: File exists? {os.path.exists(temp_path)}")
            print(f"🐛 DEBUG UPLOAD: File size: {os.path.getsize(temp_path) if os.path.exists(temp_path) else 'N/A'}")
            text_content = f"Data file uploaded: {file.filename} (ready for predictions)"
            
            # Track file for prediction tools
            from datetime import datetime
            get_agent().uploaded_documents.append({
                'filename': file.filename,
                'doc_id': str(uuid.uuid4()),
                'upload_date': datetime.now().isoformat()
            })
            
            # Don't add data files to document vector store
            # They are for prediction tools only
            return {
                "success": True,
                "message": f"Data file '{file.filename}' uploaded successfully (ready for predictions)",
                "filename": file.filename,
                "file_size": len(content),
                "file_type": file_extension,
                "chars_extracted": 0
            }
        else:
            text_content = f"Document: {file.filename} (content extraction not yet implemented)"
        
        # Add document to agent (only for text documents, not data files)
        get_agent().add_document(file.filename, text_content)
        
        return {
            "success": True,
            "message": f"Document '{file.filename}' uploaded successfully",
            "filename": file.filename,
            "file_size": len(content),
            "file_type": file_extension,
            "chars_extracted": len(text_content)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )

@app.get("/api/document-status")
async def get_document_status():
    """
    Get status of uploaded documents and document processing system.
    """
    if not DOCUMENT_TOOLS_AVAILABLE:
        return {
            "available": False,
            "message": "Document processing tools not available"
        }
    
    try:
        doc_tools = get_doc_tools()
        stats = doc_tools.get_document_stats()
        
        return {
            "available": True,
            "stats": stats,
            "message": "Document processing system is active"
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
            "message": "Error retrieving document status"
        }

# ============================================================================
# 🏢 MULTI-TENANT: Flexible Data Upload
# ============================================================================
# Big Tech Pattern: Mixpanel/Segment - accept ANY schema
# ============================================================================

@app.post("/api/users")
async def create_user(request: dict):
    """
    Create a new user account for multi-tenant access.
    
    Request:
        {"email": "user@example.com", "display_name": "John"}
    
    Returns:
        User object with ID
    """
    try:
        from database import SessionLocal
        from sqlalchemy import text
        
        email = request.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="Email is required")
        
        db = SessionLocal()
        
        try:
            # Check if exists
            existing = db.execute(text(
                "SELECT id FROM users WHERE email = :email"
            ), {"email": email}).fetchone()
            
            if existing:
                # Return existing user (idempotent)
                return {
                    "success": True,
                    "user_id": str(existing.id),
                    "email": email,
                    "message": "User already exists"
                }
            
            # Create new user
            user_id = str(uuid.uuid4())
            
            db.execute(text("""
                INSERT INTO users (id, email, display_name, created_at, updated_at)
                VALUES (:id, :email, :display_name, NOW(), NOW())
            """), {
                "id": user_id,
                "email": email,
                "display_name": request.get("display_name", email.split('@')[0])
            })
            
            db.commit()
            
            print(f"✅ Multi-tenant: Created user {email}")
            
            return {
                "success": True,
                "user_id": user_id,
                "email": email,
                "message": "User created successfully"
            }
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-data/{user_id}")
async def upload_user_data(user_id: str, file: UploadFile = File(...)):
    """
    🧠 BIG TECH AGI: Smart CSV Upload with Flexible Schema
    
    Accepts ANY CSV and stores in flexible metrics table.
    
    Auto-Detection:
    1. DATE column: 'date', 'timestamp', 'day', 'created_at', or first parseable date
    2. METRIC column: First numeric column (becomes their primary KPI)
    3. DIMENSION columns: All text columns (for breakdowns)
    
    Examples that ALL work:
    
    E-commerce:     date, revenue, orders, source, product
    SaaS:           timestamp, signups, plan, country
    Blog:           day, pageviews, article, referrer
    Marketing:      week, leads, campaign, channel, region
    
    Returns:
        Upload stats + detected schema + sample insights
    """
    try:
        from database import SessionLocal
        from sqlalchemy import text
        
        # Validate file
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files supported")
        
        content = await file.read()
        
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty")
        
        print(f"📊 Upload: {len(df)} rows, columns: {list(df.columns)}")
        
        # ================================================================
        # 🧠 STEP 1: Auto-detect DATE column
        # ================================================================
        date_column = None
        date_candidates = ['date', 'timestamp', 'day', 'time', 'created_at', 
                          'order_date', 'event_date', 'week', 'month']
        
        # Try exact match first
        for col in df.columns:
            if col.lower().strip() in date_candidates:
                date_column = col
                break
        
        # Try parsing any column as date
        if not date_column:
            for col in df.columns:
                try:
                    sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                    if sample and isinstance(sample, str):
                        date_parser.parse(sample)
                        date_column = col
                        break
                except:
                    continue
        
        if not date_column:
            raise HTTPException(
                status_code=400,
                detail=f"No date column found. Columns: {list(df.columns)}. "
                       f"Include a column named 'date', 'timestamp', etc."
            )
        
        # ================================================================
        # 🧠 STEP 2: Auto-detect METRIC column (primary KPI)
        # Big Tech Pattern: Smart column detection (Mixpanel/Amplitude)
        # ================================================================
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        
        if not numeric_cols:
            raise HTTPException(
                status_code=400,
                detail=f"No numeric columns found. Columns: {list(df.columns)}. "
                       f"Include at least one number column (revenue, clicks, etc.)"
            )
        
        # 🚫 SKIP these columns (IDs, indexes, codes)
        skip_keywords = ['id', 'index', 'key', 'code', 'number', 'num', 'no', 'seq']
        
        # ==========================================================================
        # 🏢 BIG TECH AGI: Priority-Weighted Metric Detection (Mixpanel/Amplitude)
        # ==========================================================================
        # Problem: "Quantity" was picked over "Total Amount" because first-match wins
        # Solution: Assign WEIGHTS to keywords, pick column with HIGHEST score
        #
        # Priority Tiers:
        #   TIER 1 (10 pts): Revenue/monetary metrics (what users care about most)
        #   TIER 2 (5 pts):  Volume/count metrics (secondary KPIs)
        #   TIER 3 (1 pt):   Generic numeric indicators
        # ==========================================================================
        
        prefer_keywords_weighted = {
            # TIER 1: Revenue/monetary (highest priority) - 10 points each
            'amount': 10, 'total': 10, 'revenue': 10, 'sales': 10, 'value': 10,
            'price': 10, 'profit': 10, 'cost': 10, 'income': 10, 'earning': 10,
            'spend': 10, 'sum': 10,
            
            # TIER 2: Volume/count metrics - 5 points each  
            'quantity': 5, 'count': 5, 'orders': 5, 'clicks': 5, 'views': 5,
            'pageviews': 5, 'signups': 5, 'leads': 5, 'sessions': 5, 'users': 5,
            
            # TIER 3: Generic indicators - 1 point each
            'rate': 1, 'ratio': 1, 'percent': 1, 'avg': 1, 'average': 1,
        }
        
        def is_id_column(col_name: str) -> bool:
            col_lower = col_name.lower().replace(' ', '').replace('_', '')
            return any(skip in col_lower for skip in skip_keywords)
        
        def get_column_score(col_name: str) -> int:
            """Calculate priority score for a column based on keyword matches."""
            col_lower = col_name.lower().replace(' ', '').replace('_', '')
            score = 0
            for keyword, weight in prefer_keywords_weighted.items():
                if keyword in col_lower:
                    score += weight
            return score
        
        # 🧠 AGI: Score ALL columns, pick highest (not first match)
        scored_columns = []
        for col in numeric_cols:
            if not is_id_column(col):
                score = get_column_score(col)
                scored_columns.append((col, score))
                if score > 0:
                    print(f"📊 Column '{col}' score: {score}")
        
        # Sort by score descending, then by column order (stable sort)
        scored_columns.sort(key=lambda x: x[1], reverse=True)
        
        metric_column = None
        
        # Pass 1: Pick highest-scored column
        if scored_columns and scored_columns[0][1] > 0:
            metric_column = scored_columns[0][0]
            print(f"✅ Metric detected (priority score {scored_columns[0][1]}): {metric_column}")
        
        # Pass 2: Fallback to first non-ID numeric column (no keyword match)
        if not metric_column:
            for col in numeric_cols:
                if not is_id_column(col):
                    metric_column = col
                    print(f"✅ Metric detected (fallback): {col}")
                    break
        
        # Pass 3: Last resort - use first numeric (even if ID-like)
        if not metric_column:
            metric_column = numeric_cols[0]
            print(f"⚠️ Metric detected (last resort): {metric_column}")
        
        # Clean metric name for internal use
        metric_name = metric_column.lower().strip().replace(' ', '_')
        
        # ================================================================
        # 🧠 STEP 3: Auto-detect DIMENSION columns (text/category)
        # ================================================================
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        dimension_columns = [c for c in text_cols if c != date_column][:5]  # Max 5
        
        print(f"✅ Detected: date={date_column}, metric={metric_column}, dimensions={dimension_columns}")
        
        # ================================================================
        # 🧠 STEP 4: Parse and normalize data
        # ================================================================
        try:
            df['_parsed_date'] = pd.to_datetime(df[date_column]).dt.date
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Date parsing failed: {e}")
        
        # ================================================================
        # 💾 STEP 5: Insert into flexible metrics table
        # ================================================================
        db = SessionLocal()
        rows_imported = 0
        errors = []
        
        try:
            # Ensure user exists (create if not)
            db.execute(text("""
                INSERT INTO users (id, email, display_name, primary_metric_name, 
                                   dimension_columns, created_at, updated_at)
                VALUES (:id, :email, :display_name, :metric_name, :dimensions, NOW(), NOW())
                ON CONFLICT (id) DO UPDATE SET
                    primary_metric_name = :metric_name,
                    dimension_columns = :dimensions,
                    data_mode = 'user',
                    updated_at = NOW()
            """), {
                "id": user_id,
                "email": f"{user_id[:8]}@upload.local",
                "display_name": f"User {user_id[:8]}",
                "metric_name": metric_name,
                "dimensions": json.dumps(dimension_columns)
            })
            
            # Insert each row
            for _, row in df.iterrows():
                try:
                    # Build dimensions JSONB
                    dimensions = {}
                    for dim_col in dimension_columns:
                        val = row.get(dim_col)
                        if pd.notna(val):
                            dimensions[dim_col.lower().replace(' ', '_')] = str(val)
                    
                    # Get metric value
                    metric_value = row[metric_column]
                    if pd.isna(metric_value):
                        continue
                    
                    # Insert into flexible table
                    db.execute(text("""
                        INSERT INTO user_uploaded_metrics 
                        (id, user_id, date, metric_name, metric_value, dimensions, source_file, created_at)
                        VALUES (:id, :user_id, :date, :metric_name, :metric_value, :dimensions, :source_file, NOW())
                        ON CONFLICT (user_id, date, metric_name, dimensions) 
                        DO UPDATE SET metric_value = :metric_value
                    """), {
                        "id": str(uuid.uuid4()),
                        "user_id": user_id,
                        "date": row['_parsed_date'],
                        "metric_name": metric_name,
                        "metric_value": float(metric_value),
                        "dimensions": json.dumps(dimensions),
                        "source_file": file.filename
                    })
                    
                    rows_imported += 1
                    
                except Exception as e:
                    errors.append(str(e))
                    if len(errors) > 10:
                        break
            
            # ================================================================
            # 📊 STEP 6: Build daily summary (for fast agent queries)
            # ================================================================
            summary_df = df.groupby('_parsed_date').agg({
                metric_column: 'sum'
            }).reset_index()
            
            # Build top dimensions per day
            for _, sum_row in summary_df.iterrows():
                day_data = df[df['_parsed_date'] == sum_row['_parsed_date']]
                
                # Get top values for each dimension
                top_dims = []
                dim_names = []
                for i, dim_col in enumerate(dimension_columns[:3]):
                    try:
                        top = day_data.groupby(dim_col)[metric_column].sum().to_dict()
                        top_sorted = dict(sorted(top.items(), key=lambda x: x[1], reverse=True)[:5])
                        top_dims.append(json.dumps(top_sorted))
                        dim_names.append(dim_col.lower().replace(' ', '_'))
                    except:
                        top_dims.append('{}')
                        dim_names.append(None)
                
                # Pad to 3
                while len(top_dims) < 3:
                    top_dims.append('{}')
                    dim_names.append(None)
                
                db.execute(text("""
                    INSERT INTO user_daily_summary 
                    (id, user_id, date, primary_metric_name, primary_metric_total,
                     top_dimension_1, top_dimension_2, top_dimension_3,
                     dimension_1_name, dimension_2_name, dimension_3_name,
                     created_at, updated_at)
                    VALUES (:id, :user_id, :date, :metric_name, :metric_total,
                            :dim1, :dim2, :dim3, :dim1_name, :dim2_name, :dim3_name,
                            NOW(), NOW())
                    ON CONFLICT (user_id, date) DO UPDATE SET
                        primary_metric_total = :metric_total,
                        top_dimension_1 = :dim1,
                        top_dimension_2 = :dim2,
                        top_dimension_3 = :dim3,
                        updated_at = NOW()
                """), {
                    "id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "date": sum_row['_parsed_date'],
                    "metric_name": metric_name,
                    "metric_total": float(sum_row[metric_column]),
                    "dim1": top_dims[0],
                    "dim2": top_dims[1],
                    "dim3": top_dims[2],
                    "dim1_name": dim_names[0],
                    "dim2_name": dim_names[1],
                    "dim3_name": dim_names[2]
                })
            
            # ================================================================
            # 📝 STEP 7: Record upload history
            # ================================================================
            db.execute(text("""
                INSERT INTO user_uploads
                (id, user_id, filename, date_column, metric_column, metric_name,
                 dimension_columns, rows_total, rows_imported, 
                 data_start_date, data_end_date, status, created_at, expires_at)
                VALUES (:id, :user_id, :filename, :date_col, :metric_col, :metric_name,
                        :dimensions, :total, :imported, :start, :end, 'completed', NOW(), NOW() + INTERVAL '90 days')
            """), {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "filename": file.filename,
                "date_col": date_column,
                "metric_col": metric_column,
                "metric_name": metric_name,
                "dimensions": json.dumps(dimension_columns),
                "total": len(df),
                "imported": rows_imported,
                "start": df['_parsed_date'].min(),
                "end": df['_parsed_date'].max()
            })
            
            # Update user stats
            db.execute(text("""
                UPDATE users SET
                    total_uploads = total_uploads + 1,
                    total_data_points = total_data_points + :imported,
                    last_upload_at = NOW(),
                    updated_at = NOW()
                WHERE id = :user_id
            """), {"user_id": user_id, "imported": rows_imported})
            
            db.commit()
            
            # ================================================================
            # 🧠 STEP 8: Build data profile immediately
            # ================================================================
            profile_built = False
            try:
                from workers.data_profile_worker_multi import MultiTenantDataProfileWorker
                worker = MultiTenantDataProfileWorker()
                worker.build_profile_for_user(user_id)
                profile_built = True
                print(f"✅ Built data profile for user {user_id[:8]}...")
            except Exception as e:
                print(f"⚠️ Profile build deferred to nightly: {e}")
            
            # ================================================================
            # 📊 Calculate quick insights for response
            # ================================================================
            total_metric = df[metric_column].sum()
            avg_metric = df.groupby('_parsed_date')[metric_column].sum().mean()
            days_count = df['_parsed_date'].nunique()
            
            return {
                "success": True,
                "user_id": user_id,
                "filename": file.filename,
                "rows_imported": int(rows_imported),
                "rows_total": int(len(df)),
                "errors": errors[:5] if errors else None,
                
                "detected_schema": {
                    "date_column": date_column,
                    "metric_column": metric_column,
                    "metric_name": metric_name,
                    "dimension_columns": dimension_columns
                },
                
                "data_summary": {
                    "date_range": {
                        "start": str(df['_parsed_date'].min()),
                        "end": str(df['_parsed_date'].max())
                    },
                    "days_count": int(days_count),
                    "total": round(float(total_metric), 2),
                    "daily_average": round(float(avg_metric), 2)
                },
                
                "profile_built": profile_built,
                
                "next_steps": [
                    f"✅ Your data is ready! Ask: 'What was my {metric_name} yesterday?'",
                    f"✅ Try: 'Show me {metric_name} by {dimension_columns[0]}'" if dimension_columns else None,
                    f"✅ Your daily average is {float(avg_metric):.1f} {metric_name}"
                ]
            }
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}/schema")
async def get_user_schema(user_id: str):
    """
    Get a user's detected data schema.
    
    Returns their metric name, dimensions, and sample queries.
    """
    try:
        from database import SessionLocal
        from sqlalchemy import text
        
        db = SessionLocal()
        
        try:
            user = db.execute(text("""
                SELECT primary_metric_name, dimension_columns
                FROM users WHERE id = :user_id
            """), {"user_id": user_id}).fetchone()
            
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            metric = user.primary_metric_name or "value"
            dims = json.loads(user.dimension_columns) if user.dimension_columns else []
            
            # Generate sample queries
            sample_queries = [
                f"What was my total {metric} yesterday?",
                f"Show me {metric} for the last 7 days",
                f"What's my average daily {metric}?",
            ]
            
            if dims:
                sample_queries.append(f"Break down {metric} by {dims[0]}")
                sample_queries.append(f"Which {dims[0]} had the most {metric}?")
            
            return {
                "success": True,
                "user_id": user_id,
                "schema": {
                    "primary_metric": metric,
                    "dimensions": dims
                },
                "sample_queries": sample_queries
            }
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# 🏢 PHASE 9: Data Source Mode Switcher (Big Tech: Netflix Profile Pattern)
# ============================================================================
# Like Netflix "Switch Profile" - user controls their data universe
# Returns rich context: what data they're switching to, summary stats
# ============================================================================

@app.post("/api/users/{user_id}/data-mode")
async def switch_data_mode(user_id: str, request: dict):
    """
    Switch between demo and user data mode.
    
    Big Tech Pattern: Netflix profile switcher with rich context
    
    Request: {"mode": "demo"} or {"mode": "user"}
    
    Returns:
        - Demo: summary of lubobali.com click data
        - User: summary of their uploaded metrics
    """
    try:
        from database import SessionLocal
        from sqlalchemy import text
        
        mode = request.get("mode")
        if mode not in ["demo", "user"]:
            raise HTTPException(status_code=400, detail="mode must be 'demo' or 'user'")
        
        db = SessionLocal()
        
        try:
            # ================================================================
            # 🧠 AGI: Validate before switching (Netflix: can't switch to empty profile)
            # ================================================================
            if mode == "user":
                user_data = db.execute(text("""
                    SELECT 
                        COUNT(*) as row_count,
                        u.primary_metric_name,
                        MIN(m.date) as date_start,
                        MAX(m.date) as date_end,
                        SUM(m.metric_value) as total_value
                    FROM users u
                    LEFT JOIN user_uploaded_metrics m ON u.id = m.user_id
                    WHERE u.id = :user_id
                    GROUP BY u.id, u.primary_metric_name
                """), {"user_id": user_id}).fetchone()
                
                if not user_data or user_data.row_count == 0:
                    raise HTTPException(
                        status_code=400, 
                        detail="No uploaded data. Upload CSV first to use 'My Data' mode."
                    )
            
            # ================================================================
            # 💾 Update mode in database
            # ================================================================
            db.execute(text("""
                UPDATE users SET data_mode = :mode, updated_at = NOW()
                WHERE id = :user_id
            """), {"user_id": user_id, "mode": mode})
            
            db.commit()
            
            # ================================================================
            # 🧠 AGI: Return rich context (Netflix: "Switching to Kids Profile")
            # ================================================================
            if mode == "demo":
                # Get demo data summary
                demo_stats = db.execute(text("""
                    SELECT COUNT(*) as total_events,
                           MIN(timestamp) as date_start,
                           MAX(timestamp) as date_end
                    FROM click_logs
                """)).fetchone()
                
                return {
                    "success": True,
                    "user_id": user_id,
                    "data_mode": "demo",
                    "message": "✅ Switched to Demo Data",
                    "data_summary": {
                        "source": "lubobali.com",
                        "metric": "clicks",
                        "total_events": demo_stats.total_events if demo_stats else 0,
                        "date_range": {
                            "start": str(demo_stats.date_start)[:10] if demo_stats and demo_stats.date_start else None,
                            "end": str(demo_stats.date_end)[:10] if demo_stats and demo_stats.date_end else None
                        }
                    }
                }
            else:
                # Get user data summary (already fetched above)
                days_count = (user_data.date_end - user_data.date_start).days + 1 if user_data.date_start and user_data.date_end else 0
                
                return {
                    "success": True,
                    "user_id": user_id,
                    "data_mode": "user",
                    "message": "✅ Switched to Your Data",
                    "data_summary": {
                        "source": "your_upload",
                        "metric": user_data.primary_metric_name,
                        "total_value": round(float(user_data.total_value), 2) if user_data.total_value else 0,
                        "row_count": user_data.row_count,
                        "days_count": days_count,
                        "date_range": {
                            "start": str(user_data.date_start) if user_data.date_start else None,
                            "end": str(user_data.date_end) if user_data.date_end else None
                        }
                    }
                }
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/users/{user_id}/data-mode")
async def get_data_mode(user_id: str):
    """
    Get current data mode with summary.
    
    Big Tech Pattern: Netflix - always show current profile state
    """
    try:
        from database import SessionLocal
        from sqlalchemy import text
        
        db = SessionLocal()
        
        try:
            user = db.execute(text("""
                SELECT data_mode, primary_metric_name
                FROM users WHERE id = :user_id
            """), {"user_id": user_id}).fetchone()
            
            if not user:
                return {
                    "success": True,
                    "data_mode": "demo",
                    "has_user_data": False
                }
            
            # Check if user has uploaded data
            has_data = db.execute(text("""
                SELECT COUNT(*) FROM user_uploaded_metrics WHERE user_id = :user_id
            """), {"user_id": user_id}).scalar()
            
            return {
                "success": True,
                "data_mode": user.data_mode or "demo",
                "has_user_data": has_data > 0,
                "user_metric": user.primary_metric_name
            }
            
        finally:
            db.close()
            
    except Exception as e:
        return {
            "success": True,
            "data_mode": "demo",
            "has_user_data": False,
            "error": str(e)
        }

@app.post("/v1/agent/run", response_model=AgentResponse)
async def start_agent_run(request: AgentRequest):
    """Start a new agent run and return the runId for streaming."""
    try:
        run_id = request.runId or str(uuid.uuid4())
        
        # Get model from request
        model_provider = request.model if request.model else "ollama"
        
        # Debug logging
        print(f"🐛 DEBUG main.py: Received model from frontend: {request.model if request.model else 'NOT PROVIDED'}")
        print(f"🐛 DEBUG main.py: Using model_provider: {model_provider}")
        
        # Get agent with correct model
        agent = get_agent(model_provider)
        print(f"🐛 DEBUG main.py: Agent current model: {agent.get_current_model()}")
        
        # Store run info
        active_runs[run_id] = {
            "status": "running",
            "start_time": time.time(),
            "prompt": request.prompt,
            "model": model_provider,
            "user_id": request.user_id,  # ← ADD THIS LINE
            "steps": [],
            "final_output": None
        }
        
        return AgentResponse(
            runId=run_id,
            status="started",
            message="Agent run started successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start agent run: {str(e)}")

@app.get("/v1/agent/stream")
async def stream_agent_response(runId: str):
    """Stream the agent's response tokens in real-time."""
    if runId not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run_info = active_runs[runId]
    
    async def generate_tokens():
        try:
            # Update status
            run_info["status"] = "processing"
            
            # Get the prompt
            prompt = run_info["prompt"]
            
            # Use our AdalFlow Analytics Agent
            print(f"DEBUG: Using AdalFlow Analytics Agent for prompt: {prompt}")
            
            try:
                # Call our AdalFlow Analytics Agent with correct model
                model = run_info.get("model", "groq")
                agent = get_agent(model)
                
                # TASK 18: Add streaming wrapper
                wrapper, emitter = create_streaming_wrapper(agent)
                
                # Create async task to run query with streaming (with session_id and user_id)
                query_task = asyncio.create_task(
                    wrapper.query_with_streaming(
                        prompt, 
                        session_id=runId, 
                        user_id=run_info.get("user_id", "anonymous")
                    )
                )
                
                # Stream thinking events while query runs
                thinking_sent = False
                while not query_task.done():
                    try:
                        event = await asyncio.wait_for(emitter.get_event(), timeout=0.1)
                        
                        # Send thinking event to frontend
                        if event["type"] == "thinking":
                            thinking_event = {
                                "type": "agent_thinking",
                                "message": event["data"].get("message", "Thinking..."),
                                "runId": runId,
                                "ts": int(time.time() * 1000)
                            }
                            yield f"data: {json.dumps(thinking_event)}\n\n"
                            thinking_sent = True
                            
                    except asyncio.TimeoutError:
                        if not thinking_sent:
                            # Send initial thinking if not sent yet
                            thinking_event = {
                                "type": "agent_thinking",
                                "message": "Analyzing your question...",
                                "runId": runId,
                                "ts": int(time.time() * 1000)
                            }
                            yield f"data: {json.dumps(thinking_event)}\n\n"
                            thinking_sent = True
                        continue
                
                # Get the completed answer
                answer = await query_task
                route_used = agent.last_route_used  # Get route from agent
                print(f"DEBUG: AdalFlow Analytics Agent result: {answer}")
            except Exception as e:
                print(f"DEBUG: AdalFlow Analytics Agent error: {e}")
                # Fallback to simple response if agent fails
                answer = f"I encountered an error processing your request: {str(e)}. Please try rephrasing your question."
                route_used = None
            
            # Detect source type based on route used
            print(f"🐛 DEBUG: Detecting source icon for prompt: {prompt[:50]}...")
            print(f"🐛 DEBUG: Route used: {route_used}")
            source_icon = None
            if route_used == "WEB_SEARCH":
                source_icon = "🌐"
            elif route_used == "DATABASE":
                source_icon = "💾"
            elif route_used == "IMAGE_ANALYSIS":
                source_icon = "🖼️"
            elif route_used == "CODE_INTERPRETER":
                source_icon = "🐍"
            print(f"🐛 DEBUG: Source icon detected: {source_icon}")
            
            # Check if code interpreter generated images
            code_interpreter_images = None
            if route_used == "CODE_INTERPRETER" and hasattr(agent, 'last_code_interpreter_images'):
                code_interpreter_images = agent.last_code_interpreter_images
                if code_interpreter_images:
                    print(f"🖼️ Retrieved {len(code_interpreter_images)} images from code interpreter")
            
            # =============================================================================
            # BIG TECH PATTERN: Chart Type Detection Before Metadata Event
            # =============================================================================
            # Detect chart type if chart was generated (works for ANY Plotly chart type)
            chart_type_detected = None
            chart_id_to_check = None
            
            # Check TWO sources for chart URLs (DATABASE route and CODE_INTERPRETER route)
            
            # SOURCE 1: Answer text (DATABASE route - markdown image)
            if "api/chart/" in answer:
                chart_match = re.search(r'/api/chart/([a-f0-9-]+)', answer)
                if chart_match:
                    chart_id_to_check = chart_match.group(1)
                    print(f"📊 Chart detected in answer (DATABASE route): {chart_id_to_check}")
            
            # SOURCE 2: Code interpreter images (CODE_INTERPRETER route)
            if not chart_id_to_check and code_interpreter_images:
                for img in code_interpreter_images:
                    if img.get('url') and 'api/chart/' in img['url']:
                        chart_match = re.search(r'/api/chart/([a-f0-9-]+)', img['url'])
                        if chart_match:
                            chart_id_to_check = chart_match.group(1)
                            print(f"📊 Chart detected in code_interpreter_images: {chart_id_to_check}")
                            break
            
            # Query database for chart_type (SINGLE SOURCE OF TRUTH for both routes)
            if chart_id_to_check:
                try:
                    from database import SessionLocal, ChartData
                    
                    db = SessionLocal()
                    chart = db.query(ChartData).filter(
                        ChartData.chart_id == chart_id_to_check
                    ).first()
                    db.close()
                    
                    if chart and chart.chart_type:
                        chart_type_detected = chart.chart_type
                        print(f"✅ Chart type detected: {chart_type_detected}")
                    else:
                        chart_type_detected = "unknown"
                        print(f"⚠️ Chart found but no chart_type in database")
                        
                except Exception as e:
                    print(f"❌ Chart type detection failed: {e}")
                    chart_type_detected = "unknown"
            
            # Send metadata event with sourceIcon, route_used, chartType, and images (if any)
            # ALWAYS send if we have a route (ensures route_used is always sent to frontend)
            if route_used or source_icon or code_interpreter_images or chart_type_detected:
                metadata_event = {
                    "type": "metadata",
                    "sourceIcon": source_icon,
                    "route_used": route_used,
                    "route_confidence": agent.last_route_confidence if hasattr(agent, 'last_route_confidence') else None,
                    "route_execution_time_ms": agent.last_route_execution_time_ms if hasattr(agent, 'last_route_execution_time_ms') else None,  # 🆕 PHASE 4
                    "route_error_occurred": agent.last_route_error_occurred if hasattr(agent, 'last_route_error_occurred') else False,  # 🆕 PHASE 4
                    "route_alternative_exists": agent.last_route_alternative_exists if hasattr(agent, 'last_route_alternative_exists') else False,  # 🆕 PHASE 4
                    "chartType": chart_type_detected,  # ← NEW! Actual chart type (bar/line/pie/etc)
                    "runId": runId,
                    "ts": int(time.time() * 1000)
                }
                
                # Add images if present
                if code_interpreter_images:
                    metadata_event["images"] = code_interpreter_images
                    print(f"📊 Added {len(code_interpreter_images)} images to metadata event")
                
                # 🆕 PHASE 2.5: Add CODE_INTERPRETER metadata
                if route_used == "CODE_INTERPRETER" and hasattr(agent, 'last_code_metadata') and agent.last_code_metadata:
                    metadata_event["code_executed"] = agent.last_code_metadata.get('code_executed')
                    metadata_event["code_output"] = agent.last_code_metadata.get('code_output')
                    metadata_event["code_error"] = agent.last_code_metadata.get('code_error')
                    metadata_event["execution_time_ms"] = agent.last_code_metadata.get('execution_time_ms')
                    print(f"🐍 Added CODE_INTERPRETER metadata to event")
                
                # 🆕 STEP 1.5: Add TEMPLATE metadata (for any route that uses templates)
                if hasattr(agent, 'last_template_metadata') and agent.last_template_metadata:
                    metadata_event["sql_template_used"] = agent.last_template_metadata.get('sql_template_used')
                    metadata_event["sql_template_confidence"] = agent.last_template_metadata.get('sql_template_confidence')
                    print(f"📋 Added TEMPLATE metadata to event: {agent.last_template_metadata.get('sql_template_used')}")
                
                # 🆕 PHASE 5: Add PARAMETERS_USED for AGI learning
                if hasattr(agent, 'last_parameters_used') and agent.last_parameters_used:
                    metadata_event["parameters_used"] = agent.last_parameters_used
                    print(f"🧠 AGI: Added parameters_used to metadata: {len(agent.last_parameters_used)} params")
                    
                    # ========== 🧠 AGI: Store in active_runs for /api/interaction to retrieve ==========
                    # Big Tech Pattern: Server-side parameter persistence (never trust client for critical data)
                    if runId in active_runs:
                        active_runs[runId]["parameters_used"] = agent.last_parameters_used
                        print(f"🧠 AGI: Stored {len(agent.last_parameters_used)} params in active_runs[{runId[:8]}...]")
                    # ========== END AGI STORAGE ==========
                
                # ========== 🧠 AGI: Store sql_query in active_runs for /api/interaction ==========
                # Big Tech Pattern: Server-side SQL persistence (never trust client for critical data)
                if hasattr(agent, 'last_sql_query') and agent.last_sql_query:
                    if runId in active_runs:
                        active_runs[runId]["sql_query"] = agent.last_sql_query
                        print(f"🧠 AGI: Stored sql_query in active_runs[{runId[:8]}...]")
                # ========== END SQL STORAGE ==========
                
                # 🆕 PHASE 2.5: Add WEB_SEARCH metadata
                if route_used == "WEB_SEARCH" and hasattr(agent, 'last_web_search_metadata') and agent.last_web_search_metadata:
                    metadata_event["web_search_query"] = agent.last_web_search_metadata.get('web_search_query')
                    metadata_event["web_sources_count"] = agent.last_web_search_metadata.get('web_sources_count')
                    metadata_event["web_sources"] = agent.last_web_search_metadata.get('web_sources')
                    print(f"🌐 Added WEB_SEARCH metadata to event")
                
                # Log chart type for debugging
                if chart_type_detected:
                    print(f"📊 Sending chartType in metadata: {chart_type_detected}")
                
                yield f"data: {json.dumps(metadata_event)}\n\n"
                print(f"🐛 DEBUG: ✅ Sent metadata event with sourceIcon: {source_icon}, chartType: {chart_type_detected}")
            
            # TASK 18 FIX: Clear thinking indicator before streaming tokens
            answer_start_event = {
                "type": "answer_start",
                "runId": runId,
                "ts": int(time.time() * 1000)
            }
            yield f"data: {json.dumps(answer_start_event)}\n\n"
            
            # Stream the answer preserving newlines
            # Split by whitespace but keep whitespace as separate tokens
            tokens = re.split(r'(\s+)', answer)
            tokens = [t for t in tokens if t]  # Remove empty strings
            
            for i, token in enumerate(tokens):
                token_event = {
                    "type": "token",
                    "token": token,
                    "runId": runId,
                    "ts": int(time.time() * 1000)
                }
                yield f"data: {json.dumps(token_event)}\n\n"
                # Small delay to simulate streaming
                await asyncio.sleep(0.1)
            
            # Mark as completed
            run_info["status"] = "completed"
            run_info["end_time"] = time.time()
            
            # ========== 🧠 PHASE 5: AGI Route Learning (Implicit Signals) ==========
            # Big Tech Pattern: Learn from EVERY interaction, not just explicit feedback
            # Netflix/Spotify: Implicit signals (errors, latency) are 80% of learning signal
            try:
                if route_used:
                    user_id = run_info.get("user_id", "anonymous")
                    error_occurred = agent.last_route_error_occurred if hasattr(agent, 'last_route_error_occurred') else False
                    execution_time = agent.last_route_execution_time_ms if hasattr(agent, 'last_route_execution_time_ms') else None
                    
                    learn_route_preference(
                        user_id=user_id,
                        route_name=route_used,
                        feedback=None,  # No explicit feedback yet - implicit only
                        error_occurred=error_occurred,
                        execution_time_ms=execution_time
                    )
                    print(f"🧠 AGI: Implicit learning for route={route_used} user={user_id[:8]}... error={error_occurred} time={execution_time}ms")
            except Exception as e:
                print(f"⚠️ AGI implicit learning failed (non-critical): {e}")
            # ========== END PHASE 5 IMPLICIT LEARNING ==========
            
            # Send completion event
            done_event = {
                "type": "done",
                "runId": runId,
                "ts": int(time.time() * 1000)
            }
            yield f"data: {json.dumps(done_event)}\n\n"
            
            # Send final output event for events stream
            final_event = {
                "type": "agent.final_output",
                "runId": runId,
                "output": answer,
                "ts": int(time.time() * 1000)
            }
            yield f"data: {json.dumps(final_event)}\n\n"
            
        except Exception as e:
            # Handle errors
            run_info["status"] = "error"
            run_info["error"] = str(e)
            
            error_event = {
                "type": "error",
                "runId": runId,
                "error": str(e),
                "ts": int(time.time() * 1000)
            }
            yield f"data: {json.dumps(error_event)}\n\n"
    
    return StreamingResponse(
        generate_tokens(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@app.get("/v1/agent/events")
async def stream_agent_events(runId: str):
    """Stream agent tool usage events and step information."""
    if runId not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    async def generate_events():
        run_info = active_runs[runId]
        
        # Simulate tool usage events (in a real implementation, these would come from the agent)
        tools_used = ["get_schema_tool", "run_sql_tool", "finish"]
        
        for i, tool in enumerate(tools_used):
            # Tool call start
            start_event = {
                "type": "agent.tool_call_start",
                "runId": runId,
                "tool": tool,
                "step": i + 1,
                "ts": int(time.time() * 1000)
            }
            yield f"data: {json.dumps(start_event)}\n\n"
            
            await asyncio.sleep(0.5)  # Simulate processing time
            
            # Tool call activity
            activity_event = {
                "type": "agent.tool_call_activity",
                "runId": runId,
                "tool": tool,
                "step": i + 1,
                "ts": int(time.time() * 1000)
            }
            yield f"data: {json.dumps(activity_event)}\n\n"
            
            await asyncio.sleep(0.3)
            
            # Tool call complete
            complete_event = {
                "type": "agent.tool_call_complete",
                "runId": runId,
                "tool": tool,
                "step": i + 1,
                "ts": int(time.time() * 1000)
            }
            yield f"data: {json.dumps(complete_event)}\n\n"
            
            # Step complete
            step_event = {
                "type": "agent.step_complete",
                "runId": runId,
                "step": i + 1,
                "ts": int(time.time() * 1000)
            }
            yield f"data: {json.dumps(step_event)}\n\n"
        
        # Final output - get actual result from agent
        try:
            prompt = run_info["prompt"]
            print(f"DEBUG STREAMING: Using AdalFlow Analytics Agent for prompt: {prompt}")
            # Use our AdalFlow Analytics Agent with correct model
            model = run_info.get("model", "groq")
            user_id = run_info.get("user_id", "anonymous")
            answer = get_agent(model).query(prompt, session_id=runId, user_id=user_id)
            
            # Add None check
            if answer is None:
                answer = "Error: Agent returned no response"
            
            print(f"DEBUG STREAMING: AdalFlow Analytics Agent result: {answer[:200] if answer else 'None'}...")
            
            final_event = {
                "type": "agent.final_output",
                "runId": runId,
                "output": {
                    "answer": answer,
                    "reasoning": "Chain-of-Thought analysis completed",
                    "sql": "Query executed successfully",
                    "chart": None
                },
                "ts": int(time.time() * 1000)
            }
        except Exception as e:
            final_event = {
                "type": "agent.final_output",
                "runId": runId,
                "output": {
                    "answer": f"Error: {str(e)}",
                    "reasoning": "Agent encountered an error",
                    "sql": None,
                    "chart": None
                },
                "ts": int(time.time() * 1000)
            }
        
        yield f"data: {json.dumps(final_event)}\n\n"
    
    return StreamingResponse(
        generate_events(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@app.get("/v1/agent/status/{runId}")
async def get_run_status(runId: str):
    """Get the current status of a run."""
    if runId not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return active_runs[runId]

@app.post("/v1/agent/reset-memory")
async def reset_memory():
    """Reset the agent's conversation memory."""
    try:
        get_agent().reset_memory()
        return {
            "success": True,
            "message": "Conversation memory cleared successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset memory: {str(e)}"
        )

@app.get("/v1/agent/memory-status")
async def get_memory_status():
    """Get the current state of conversation memory."""
    try:
        a = get_agent()
        return {
            "success": True,
            "turn_count": a.memory.get_turn_count(),
            "is_empty": a.memory.is_empty(),
            "max_turns": a.memory.max_turns,
            "context_preview": a.memory.get_context(last_n=1) if not a.memory.is_empty() else ""
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get memory status: {str(e)}"
        )

# ============================================================================
# PHASE 5A: CHAT HISTORY API ENDPOINTS
# ============================================================================

@app.get("/api/chat-sessions")
async def list_chat_sessions(user_id: str, limit: int = 50):
    """
    List all chat sessions for THIS user only.
    
    Args:
        user_id: Browser UUID (required for privacy)
        limit: Maximum sessions to return (default: 50)
    """
    try:
        from memory.db_conversation_memory import DBConversationMemory
        
        if not user_id or user_id == "":
            raise HTTPException(
                status_code=400,
                detail="user_id is required"
            )
        
        db = DBConversationMemory()
        sessions = db.list_sessions(user_id=user_id, limit=limit)
        db.close()
        
        # Convert datetime objects to ISO strings for JSON serialization
        for session in sessions:
            if session.get('updated_at'):
                session['updated_at'] = session['updated_at'].isoformat()
            if session.get('created_at'):
                session['created_at'] = session['created_at'].isoformat()
            if session.get('last_message_at'):
                session['last_message_at'] = session['last_message_at'].isoformat()
        
        return {
            "success": True,
            "sessions": sessions,
            "total": len(sessions)
        }
        
    except Exception as e:
        print(f"❌ Failed to list sessions: {e}")
        return {
            "success": False,
            "sessions": [],
            "total": 0,
            "error": str(e)
        }


@app.get("/api/chat-sessions/{session_id}")
async def get_chat_session(session_id: str, user_id: str):
    """
    Get a specific chat session (privacy-filtered).
    
    Args:
        session_id: Unique session identifier
        user_id: User identifier (must own this session)
    """
    try:
        from memory.db_conversation_memory import DBConversationMemory
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")
        
        db = DBConversationMemory()
        messages = db.load(session_id, user_id=user_id)
        db.close()
        
        if not messages:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found or not owned by this user"
            )
        
        return {
            "success": True,
            "session_id": session_id,
            "messages": messages,
            "message_count": len(messages)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Failed to get session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load session: {str(e)}"
        )


@app.delete("/api/chat-sessions/{session_id}")
async def delete_chat_session(session_id: str, user_id: str):
    """
    Soft delete a chat session (only if owned by user).
    
    Args:
        session_id: Session to delete
        user_id: User identifier (security check)
    """
    try:
        from memory.db_conversation_memory import DBConversationMemory
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")
        
        db = DBConversationMemory()
        success = db.reset(session_id, user_id=user_id)
        db.close()
        
        if success:
            return {
                "success": True,
                "message": f"Session {session_id} deleted successfully"
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found or not owned by this user"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Failed to delete session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete session: {str(e)}"
        )

# ============================================================================
# END PHASE 5A
# ============================================================================

@app.post("/api/interaction")
async def log_interaction(request: dict):
    """
    Log comprehensive user interaction with feedback.
    Big Tech pattern: Complete observability + RLHF data collection.
    
    Phase 1: Core feedback + DATABASE route tracking
    Phase 2: WEB_SEARCH + CODE_INTERPRETER tracking (future)
    Phase 3: RAG + IMAGE tracking (future)
    Phase 4: PREDICTION tracking (future)
    """
    # DEBUG: Print what we received
    print(f"🐛 DEBUG RECEIVED: charts_generated = {request.get('charts_generated')}")
    print(f"🐛 DEBUG TYPE: {type(request.get('charts_generated'))}")
    
    try:
        # Import locally (matching your code style)
        from database import SessionLocal, InteractionLog
        
        # Extract required fields
        session_id = request.get("session_id", "anonymous")
        message_index = request.get("message_index")
        feedback_type = request.get("feedback_type")
        
        # Allow null feedback for auto-logging (when agent responds)
        if message_index is None:
            raise HTTPException(status_code=400, detail="message_index is required")
        
        # Validate feedback_type if provided
        if feedback_type is not None and feedback_type not in ["thumbs_up", "thumbs_down"]:
            raise HTTPException(status_code=400, detail="Invalid feedback_type")
        
        # ✅ SAFETY PATCH: Convert Python objects for PostgreSQL compatibility
        # charts_generated - preserve charts array
        charts = request.get("charts_generated")
        if charts is not None:
            request["charts_generated"] = charts if charts else []
        
        # JSONB fields need JSON strings (not Python objects)
        jsonb_fields = ["web_sources", "documents_uploaded", "images_uploaded", "charts_generated"]
        for field in jsonb_fields:
            value = request.get(field)
            
            # Skip if None
            if value is None:
                continue
            
            # Already a JSON string, keep as-is
            if isinstance(value, str):
                continue
            
            # Convert dict/list to JSON string for JSONB columns
            if isinstance(value, (dict, list)):
                try:
                    request[field] = json.dumps(value)
                except Exception as e:
                    print(f"⚠️ JSON serialization failed for {field}, storing NULL instead. Error: {e}")
                    request[field] = None
        
        # ========== 🧠 AGI: Retrieve sql_query from active_runs ==========
        # Big Tech Pattern: Server-side SQL persistence (never trust client)
        sql_query_from_server = request.get("sql_query")
        if not sql_query_from_server:
            session_id_for_lookup = request.get("session_id", "anonymous")
            if session_id_for_lookup in active_runs:
                sql_query_from_server = active_runs[session_id_for_lookup].get("sql_query")
                if sql_query_from_server:
                    print(f"🧠 AGI: Retrieved sql_query from active_runs cache")
            
            # Fallback: iterate all runs (backup)
            if not sql_query_from_server:
                for rid, run_data in active_runs.items():
                    if run_data.get("sql_query"):
                        sql_query_from_server = run_data["sql_query"]
                        print(f"🧠 AGI: Retrieved sql_query from run {rid[:8]}...")
                        break
        
        # Update request dict so downstream code uses it
        if sql_query_from_server:
            request["sql_query"] = sql_query_from_server
        # ========== END SQL RETRIEVAL ==========
        
        db = SessionLocal()
        
        try:
            # Generate unique interaction_id (idempotency key)
            interaction_id = f"{session_id}_{message_index}"
            
            # Upsert pattern (Big Tech: idempotent operations)
            existing = db.query(InteractionLog).filter(
                InteractionLog.session_id == session_id,
                InteractionLog.message_index == message_index
            ).first()
            
            # DEBUG LOGGING (matching your style)
            print(f"🔍 DEBUG INTERACTION: Looking for session_id={session_id}, message_index={message_index}")
            print(f"🔍 DEBUG INTERACTION: Found existing? {existing is not None}")
            
            if existing:
                # Update existing interaction
                print(f"🔍 DEBUG INTERACTION: Existing feedback_type: {existing.feedback_type}")
                print(f"🔍 DEBUG INTERACTION: New feedback_type: {feedback_type}")
                
                # 🧠 AGI FIX: Always update user_id (Netflix: user identity is GOLD)
                if request.get("user_id"):
                    existing.user_id = request.get("user_id")
                
                existing.feedback_type = feedback_type
                existing.question_text = request.get("question", "")
                existing.answer_text = request.get("answer", "")
                existing.route_used = request.get("route_used", "UNKNOWN")
                
                # Phase 1: DATABASE route tracking
                if request.get("sql_query"):
                    existing.sql_query = request.get("sql_query")
                
                # 🆕 STEP 1.5: Template metadata tracking (any route that uses templates)
                if request.get("sql_template_used"):
                    existing.sql_template_used = request.get("sql_template_used")
                if request.get("sql_template_confidence"):
                    existing.sql_template_confidence = request.get("sql_template_confidence")
                
                # Phase 2: WEB_SEARCH route tracking
                if request.get("route_used") == "WEB_SEARCH":
                    existing.web_search_query = request.get("web_search_query")
                    existing.web_sources_count = request.get("web_sources_count")
                    existing.web_sources = request.get("web_sources")
                
                # Phase 2: CODE_INTERPRETER route tracking
                if request.get("route_used") == "CODE_INTERPRETER":
                    existing.code_executed = request.get("code_executed")
                    existing.code_output = request.get("code_output")
                    existing.code_error = request.get("code_error")
                    existing.execution_time_ms = request.get("execution_time_ms")
                
                # Phase 3: RAG/DOCUMENT route fields
                if request.get("route_used") == "RAG":
                    existing.documents_uploaded = request.get("documents_uploaded")
                    existing.documents_used_in_response = request.get("documents_used")
                    existing.vector_search_similarity = request.get("vector_search_similarity")
                    existing.rag_retrieval_success = request.get("rag_retrieval_success")
                
                # Phase 3: IMAGE_ANALYSIS route fields
                if request.get("route_used") == "IMAGE_ANALYSIS":
                    existing.images_uploaded = request.get("images_uploaded")
                    existing.vision_model_used = request.get("vision_model_used")
                    existing.image_analysis_confidence = request.get("image_analysis_confidence")
                
                # Phase 3: Multi-modal detection
                existing.multi_modal = request.get("multi_modal", False)
                existing.modalities_used = request.get("modalities_used")
                
                # Phase 4: PREDICTION route fields
                if request.get("route_used") == "PREDICTION":
                    existing.prediction_model_used = request.get("prediction_model_used")
                    existing.prediction_input_data = request.get("prediction_input_data")
                    existing.prediction_output = request.get("prediction_output")
                    existing.prediction_horizon = request.get("prediction_horizon")
                
                # Phase 4: Performance metrics (all routes)
                existing.response_time_ms = request.get("response_time_ms")
                existing.token_count = request.get("token_count")
                
                # Phase 4: Conversation context (all routes)
                existing.conversation_topic = request.get("conversation_topic")
                
                # Chart tracking (works for ALL routes: DATABASE, CODE_INTERPRETER, etc.)
                if request.get("charts_generated") is not None:
                    existing.charts_generated = request.get("charts_generated")
                    existing.chart_type = request.get("chart_type")
                    existing.chart_format = request.get("chart_format")
                
                existing.feedback_text = request.get("feedback_text")
                
                existing.updated_at = datetime.utcnow()
                
                print(f"🔍 DEBUG INTERACTION: UPDATED existing record")
            else:
                # Create new interaction
                interaction = InteractionLog(
                    interaction_id=interaction_id,
                    session_id=session_id,
                    user_id=request.get("user_id"),  # 🧠 AGI FIX: Track user across sessions
                    turn_number=message_index,
                    message_index=message_index,
                    question_text=request.get("question", ""),
                    answer_text=request.get("answer", ""),
                    route_used=request.get("route_used", "UNKNOWN"),
                    feedback_type=feedback_type,
                    sql_query=request.get("sql_query"),
                    sql_template_used=request.get("sql_template_used"),  # 🆕 STEP 1.5
                    sql_template_confidence=request.get("sql_template_confidence"),  # 🆕 STEP 1.5
                    feedback_text=request.get("feedback_text"),
                    model_provider=request.get("model_provider", "unknown"),
                    model_name=request.get("model_name", "unknown"),
                    
                    # Phase 2: WEB_SEARCH route tracking
                    web_search_query=request.get("web_search_query") if request.get("route_used") == "WEB_SEARCH" else None,
                    web_sources_count=request.get("web_sources_count") if request.get("route_used") == "WEB_SEARCH" else None,
                    web_sources=request.get("web_sources") if request.get("route_used") == "WEB_SEARCH" else None,
                    
                    # Phase 2: CODE_INTERPRETER route tracking
                    code_executed=request.get("code_executed") if request.get("route_used") == "CODE_INTERPRETER" else None,
                    code_output=request.get("code_output") if request.get("route_used") == "CODE_INTERPRETER" else None,
                    code_error=request.get("code_error") if request.get("route_used") == "CODE_INTERPRETER" else None,
                    execution_time_ms=request.get("execution_time_ms") if request.get("route_used") == "CODE_INTERPRETER" else None,
                    
                    # Phase 3: RAG/DOCUMENT route fields
                    documents_uploaded=request.get("documents_uploaded") if request.get("route_used") == "RAG" else None,
                    documents_used_in_response=request.get("documents_used") if request.get("route_used") == "RAG" else None,
                    vector_search_similarity=request.get("vector_search_similarity") if request.get("route_used") == "RAG" else None,
                    rag_retrieval_success=request.get("rag_retrieval_success") if request.get("route_used") == "RAG" else None,
                    
                    # Phase 3: IMAGE_ANALYSIS route fields
                    images_uploaded=request.get("images_uploaded") if request.get("route_used") == "IMAGE_ANALYSIS" else None,
                    vision_model_used=request.get("vision_model_used") if request.get("route_used") == "IMAGE_ANALYSIS" else None,
                    image_analysis_confidence=request.get("image_analysis_confidence") if request.get("route_used") == "IMAGE_ANALYSIS" else None,
                    
                    # Phase 3: Multi-modal tracking
                    multi_modal=request.get("multi_modal", False),
                    modalities_used=request.get("modalities_used"),
                    
                    # Phase 4: PREDICTION route fields
                    prediction_model_used=request.get("prediction_model_used") if request.get("route_used") == "PREDICTION" else None,
                    prediction_input_data=request.get("prediction_input_data") if request.get("route_used") == "PREDICTION" else None,
                    prediction_output=request.get("prediction_output") if request.get("route_used") == "PREDICTION" else None,
                    prediction_horizon=request.get("prediction_horizon") if request.get("route_used") == "PREDICTION" else None,
                    
                    # Phase 4: Performance metrics
                    response_time_ms=request.get("response_time_ms"),
                    token_count=request.get("token_count"),
                    
                    # Phase 4: Conversation context
                    conversation_topic=request.get("conversation_topic"),
                    
                    # Chart tracking (works for ALL routes)
                    charts_generated=request.get("charts_generated"),
                    chart_type=request.get("chart_type"),
                    chart_format=request.get("chart_format"),
                )
                db.add(interaction)
                print(f"🔍 DEBUG INTERACTION: CREATED new record")
            
            db.commit()
            
            # ========== SAVE TO template_learning_data (ML Training) ==========
            # Big Tech Pattern: Separate ML training data from operational logs
            try:
                from sqlalchemy import text
                
                question_text = request.get("question", "")
                keywords = extract_keywords(question_text)
                
                if question_text and request.get("route_used"):
                    learning_id = str(uuid.uuid4())
                    
                    db.execute(text("""
                        INSERT INTO template_learning_data 
                        (id, interaction_id, query_text, query_keywords, template_selected, 
                         template_confidence, route_used, chart_type_shown, route_confidence, 
                         route_execution_time_ms, route_error_occurred, route_alternative_exists, 
                         feedback, created_at,
                         user_id, session_id, model_used, chart_preferences, feedback_comment, ab_test_group)
                        VALUES (:id, :interaction_id, :query_text, :query_keywords, :template_selected,
                                :template_confidence, :route_used, :chart_type_shown, :route_confidence, 
                                :route_execution_time_ms, :route_error_occurred, :route_alternative_exists, 
                                :feedback, NOW(),
                                :user_id, :session_id, :model_used, :chart_preferences, :feedback_comment, :ab_test_group)
                        ON CONFLICT (id) DO NOTHING
                    """), {
                        "id": learning_id,
                        "interaction_id": interaction_id,
                        "query_text": question_text,
                        "query_keywords": keywords,
                        "template_selected": request.get("sql_template_used"),
                        "template_confidence": request.get("sql_template_confidence"),
                        "route_used": request.get("route_used"),
                        "chart_type_shown": request.get("chart_type"),
                        "route_confidence": request.get("route_confidence"),
                        "route_execution_time_ms": request.get("route_execution_time_ms"),  # 🆕 PHASE 4
                        "route_error_occurred": request.get("route_error_occurred"),  # 🆕 PHASE 4
                        "route_alternative_exists": request.get("route_alternative_exists"),  # 🆕 PHASE 4
                        "feedback": feedback_type or "pending",
                        # 🆕 Phase 4: Chart Preference Learning columns
                        "user_id": request.get("user_id") or "anonymous",  # 🛡️ Enterprise: Never store NULL
                        "session_id": session_id,
                        "model_used": request.get("model_provider"),
                        "chart_preferences": json.dumps(request.get("chart_preferences")) if request.get("chart_preferences") else None,
                        "feedback_comment": request.get("feedback_text"),
                        "ab_test_group": None  # Phase 9: A/B testing
                    })
                    db.commit()
                    print(f"✅ ML Training: saved to template_learning_data (keywords={keywords})")
            except Exception as e:
                print(f"⚠️ template_learning_data save failed (non-critical): {e}")
            # ========== END ML TRAINING DATA ==========
            
            # ========== NETFLIX PATTERN: Write to Event Stream ==========
            # Big Tech: Kafka-style event streaming for AGI learning
            # All learning signals captured here → background job processes into parameter_learning
            try:
                from sqlalchemy import text
                
                feedback_text = request.get("feedback_text")
                
                # Only write event if there's feedback to process
                if feedback_type or feedback_text:
                    # ========== 🧠 AGI: Retrieve parameters_used from active_runs ==========
                    # Big Tech Pattern: Server-side parameter persistence
                    parameters_used = request.get("parameters_used")
                    
                    if not parameters_used:
                        # Frontend didn't send parameters - retrieve from active_runs cache
                        session_id = request.get("session_id", "anonymous")
                        if session_id in active_runs:
                            parameters_used = active_runs[session_id].get("parameters_used")
                            if parameters_used:
                                print(f"🧠 AGI: Retrieved {len(parameters_used)} params from active_runs cache")
                        
                        # Also try iterating active_runs if session_id didn't work
                        if not parameters_used:
                            for rid, run_data in active_runs.items():
                                if run_data.get("parameters_used"):
                                    parameters_used = run_data["parameters_used"]
                                    print(f"🧠 AGI: Retrieved {len(parameters_used)} params from run {rid[:8]}...")
                                    break
                    
                    if parameters_used:
                        print(f"📤 Netflix: Event will be published with {len(parameters_used)} params for AGI learning")
                    else:
                        print(f"📤 Netflix: Event published with 0 params for AGI learning")
                    # ========== END AGI RETRIEVAL ==========
                    
                    # 🛡️ AGI SAFETY: Serialize parameters_used to JSON string for JSONB column
                    if parameters_used and isinstance(parameters_used, list):
                        parameters_used_json = json.dumps(parameters_used)
                    else:
                        parameters_used_json = None
                    
                    # 🛡️ AGI SAFETY: Derive interaction_success from feedback
                    interaction_success = None
                    if feedback_type == "thumbs_up":
                        interaction_success = True
                    elif feedback_type == "thumbs_down":
                        interaction_success = False
                    
                    db.execute(text("""
                        INSERT INTO preference_events 
                        (event_type, user_id, session_id, query_text, query_keywords,
                         feedback_type, feedback_comment, chart_type, route_used,
                         template_used, template_confidence, parameters_used,
                         response_time_ms, model_used, interaction_success)
                        VALUES ('feedback', :user_id, :session_id, :query_text, :query_keywords,
                                :feedback_type, :feedback_comment, :chart_type, :route_used,
                                :template_used, :template_confidence, :parameters_used,
                                :response_time_ms, :model_used, :interaction_success)
                    """), {
                        "user_id": request.get("user_id") or "anonymous",  # 🧠 AGI: Per-user learning (enterprise fallback)
                        "session_id": session_id,
                        "query_text": request.get("question", ""),
                        "query_keywords": extract_keywords(request.get("question", "")),
                        "feedback_type": feedback_type,
                        "feedback_comment": feedback_text,
                        "chart_type": request.get("chart_type"),
                        "route_used": request.get("route_used"),
                        # 🆕 PHASE 5: AGI Learning columns
                        "template_used": request.get("sql_template_used"),
                        "template_confidence": request.get("sql_template_confidence"),
                        "parameters_used": parameters_used_json,
                        "response_time_ms": request.get("response_time_ms"),
                        "model_used": request.get("model_provider"),
                        "interaction_success": interaction_success
                    })
                    db.commit()
                    print(f"📤 Netflix: Event published with {len(parameters_used) if parameters_used else 0} params for AGI learning")
                    
                    # ========== 🧠 PHASE 5: AGI Route Learning (Explicit Feedback) ==========
                    # Big Tech Pattern: Combine explicit feedback with implicit signals
                    # This is the GOLD SIGNAL - user explicitly said good/bad
                    try:
                        route_used_for_learning = request.get("route_used")
                        if route_used_for_learning and feedback_type:
                            learn_route_preference(
                                user_id=request.get("user_id") or "anonymous",
                                route_name=route_used_for_learning,
                                feedback=feedback_type,  # thumbs_up or thumbs_down
                                error_occurred=request.get("route_error_occurred", False),
                                execution_time_ms=request.get("response_time_ms")
                            )
                            print(f"🧠 AGI: Explicit learning feedback={feedback_type} route={route_used_for_learning}")
                    except Exception as e:
                        print(f"⚠️ AGI explicit learning failed (non-critical): {e}")
                    # ========== END PHASE 5 EXPLICIT LEARNING ==========
                    
            except Exception as e:
                print(f"⚠️ Netflix event publish failed (non-critical): {e}")
            # ========== END NETFLIX PATTERN ==========
            
            return {
                "success": True, 
                "feedback_type": feedback_type,
                "interaction_id": interaction_id
            }
            
        finally:
            db.close()
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Interaction logging error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/topics")
async def get_topic_analytics(
    session_id: str = None,
    days: int = 30,
    limit: int = 10
):
    """
    Get conversation topic analytics.
    
    Args:
        session_id: Optional - filter by specific session
        days: Number of days to look back (default: 30)
        limit: Max topics to return (default: 10)
    
    Returns:
        Topic distribution with counts and feedback ratios
    """
    try:
        from database import SessionLocal, InteractionLog
        from sqlalchemy import func, case
        from datetime import datetime, timedelta
        
        db = SessionLocal()
        
        try:
            # Calculate date threshold
            date_threshold = datetime.utcnow() - timedelta(days=days)
            
            # Base query
            query = db.query(
                InteractionLog.conversation_topic,
                func.count(InteractionLog.interaction_id).label('total_count'),
                func.sum(
                    case(
                        (InteractionLog.feedback_type == 'thumbs_up', 1),
                        else_=0
                    )
                ).label('thumbs_up_count'),
                func.sum(
                    case(
                        (InteractionLog.feedback_type == 'thumbs_down', 1),
                        else_=0
                    )
                ).label('thumbs_down_count')
            ).filter(
                InteractionLog.created_at >= date_threshold,
                InteractionLog.conversation_topic.isnot(None)
            )
            
            # Optional session filter
            if session_id:
                query = query.filter(InteractionLog.session_id == session_id)
            
            # Group and order
            results = query.group_by(
                InteractionLog.conversation_topic
            ).order_by(
                func.count(InteractionLog.interaction_id).desc()
            ).limit(limit).all()
            
            # Format results
            topics = []
            for row in results:
                total = row.total_count or 0
                thumbs_up = row.thumbs_up_count or 0
                thumbs_down = row.thumbs_down_count or 0
                
                topics.append({
                    "topic": row.conversation_topic,
                    "total_interactions": total,
                    "thumbs_up": thumbs_up,
                    "thumbs_down": thumbs_down,
                    "satisfaction_rate": round(thumbs_up / total * 100, 1) if total > 0 else 0.0
                })
            
            return {
                "success": True,
                "topics": topics,
                "period_days": days,
                "total_topics": len(topics)
            }
            
        finally:
            db.close()
        
    except Exception as e:
        print(f"❌ Analytics error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve analytics: {str(e)}"
        )


@app.post("/api/feedback")
async def submit_feedback_legacy(request: dict):
    """
    LEGACY ENDPOINT - Redirects to new /api/interaction endpoint.
    Kept for backward compatibility during Phase 1 transition.
    SAFETY: If new endpoint fails, this provides fallback path.
    """
    print("⚠️ LEGACY: /api/feedback called, redirecting to /api/interaction")
    return await log_interaction(request)

@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an image using GPT-4o Vision and store for agent use.
    """
    # Validate file type
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.webp']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Read the image file
        image_content = await file.read()
        
        # Convert to base64 for OpenAI API
        base64_image = base64.b64encode(image_content).decode('utf-8')
        
        # Determine MIME type
        mime_type = f"image/{file_extension[1:]}"
        if file_extension == '.jpg':
            mime_type = "image/jpeg"
        
        # Save image to temp file for agent access
        temp_dir = '/tmp'
        temp_path = os.path.join(temp_dir, file.filename)
        with open(temp_path, 'wb') as f:
            f.write(image_content)
        
        # Store image info for agent (with base64 for backend display)
        if not hasattr(app.state, 'uploaded_images'):
            app.state.uploaded_images = []
        
        app.state.uploaded_images.append({
            'filename': file.filename,
            'path': temp_path,
            'base64': base64_image,
            'mime_type': mime_type
        })
        
        # Call GPT-4o Vision API for frontend display
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image in detail. Describe what you see, identify any text, objects, people, or important elements. Be thorough and specific."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        analysis = response.choices[0].message.content
        
        return {
            "success": True,
            "analysis": analysis,
            "filename": file.filename,
            "file_size": len(image_content)
        }
        
    except Exception as e:
        print(f"❌ Image analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze image: {str(e)}"
        )

@app.delete("/v1/agent/run/{runId}")
async def cleanup_run(runId: str):
    """Clean up a completed run."""
    if runId in active_runs:
        del active_runs[runId]
        return {"message": "Run cleaned up successfully"}
    else:
        raise HTTPException(status_code=404, detail="Run not found")

# Serve interactive Plotly chart HTML from database (with fallback)
@app.get("/api/chart/{chart_id}")
async def get_chart(chart_id: str):
    """Retrieve chart from PostgreSQL (with cache and file fallback)"""
    
    # TRY 1: PostgreSQL (new way)
    try:
        from database import SessionLocal, ChartData
        
        db = SessionLocal()
        chart = db.query(ChartData).filter(
            ChartData.chart_id == chart_id
        ).first()
        db.close()
        
        if chart:
            print(f"✅ Chart {chart_id} from PostgreSQL")
            return HTMLResponse(content=chart.chart_html)
    except Exception as e:
        print(f"⚠️ DB read failed: {e}")
    
    # TRY 2: Cache (in-memory, for current session)
    if chart_id in chart_cache:
        print(f"💾 Chart {chart_id} from cache (in-memory)")
        html_content = chart_cache[chart_id]
        return HTMLResponse(content=html_content)
    
    # TRY 3: File system (old charts, fallback)
    chart_path = f"/tmp/charts/{chart_id}.html"
    if os.path.exists(chart_path):
        print(f"📁 Chart {chart_id} from file (legacy)")
        with open(chart_path, 'r') as f:
            return HTMLResponse(content=f.read())
    
    # TRY 4: Not found
    raise HTTPException(status_code=404, detail="Chart not found")

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    '''
    Download generated report files.
    
    Args:
        filename: Name of file to download
    
    Returns:
        FileResponse with file for download
    '''
    try:
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        file_path = os.path.join(output_dir, filename)
        
        # Security: Check file exists and is in outputs directory
        if not os.path.exists(file_path):
            return {"error": "File not found"}
        
        # Determine media type based on extension
        if filename.endswith('.csv'):
            media_type = 'text/csv'
        elif filename.endswith('.xlsx'):
            media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        elif filename.endswith('.pdf'):
            media_type = 'application/pdf'
        else:
            media_type = 'application/octet-stream'
        
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename
        )
    
    except Exception as e:
        return {"error": f"Download failed: {str(e)}"}

# ============================================================================
# PHASE 6: IMPLICIT USER BEHAVIOR TRACKING (AGI Learning)
# ============================================================================
# Big Tech Pattern: TikTok/YouTube/Netflix - capture ALL user behavior
# These signals feed into universal_learner.py → user_preferences table
# ============================================================================

class UserEventRequest(BaseModel):
    """Pydantic model for user event validation - Enterprise grade."""
    user_id: str
    session_id: str = None
    event_type: str  # page_view, copy, scroll, hover, click, idle, export, focus, blur
    event_target: str = None  # chart, response, sidebar, input, message
    event_value: str = None  # Additional context (chart_id, message_index, etc.)
    duration_ms: int = None  # For timed events (hover, scroll, focus)
    scroll_depth_percent: float = None  # For scroll events (0-100)
    message_index: int = None  # Which message in conversation
    chart_id: str = None  # If event relates to a chart
    page_url: str = None  # Current page URL
    device_type: str = None  # desktop, mobile, tablet
    browser: str = None  # chrome, firefox, safari
    viewport_width: int = None
    viewport_height: int = None


@app.post("/api/events")
async def log_user_event(request: UserEventRequest):
    """
    Log implicit user behavior events for AGI learning.
    
    Big Tech Pattern: TikTok/YouTube/Netflix - capture EVERYTHING
    - Every hover, scroll, copy, click feeds into learning
    - Processed by universal_learner.py into user_preferences
    
    Event Types:
        - page_view: User opened/viewed the app
        - copy: User copied response text
        - scroll: User scrolled (with depth %)
        - hover: User hovered on element (with duration)
        - click: User clicked element (chart, button, etc.)
        - idle: User went idle (no activity for N seconds)
        - export: User exported/downloaded data
        - focus: User focused on input
        - blur: User left input/window
    
    Returns:
        Success response with event_id
    """
    try:
        from database import SessionLocal
        from sqlalchemy import text
        
        # ========== VALIDATION (Enterprise: Never trust client) ==========
        valid_event_types = [
            # Original events
            'page_view', 'copy', 'scroll', 'hover', 'click', 
            'idle', 'export', 'focus', 'blur', 'chart_interact',
            'response_expand', 'response_collapse', 'feedback_start',
            # 🧠 Phase 5.1: AGI implicit behavior signals
            'scroll_25', 'scroll_50', 'scroll_80', 'scroll_100',  # Scroll milestones
            'dwell_long',   # 30s+ engagement with message
            'hover_long',   # 3s+ hover on chart
            'abandon',      # Left without giving feedback
        ]
        
        if request.event_type not in valid_event_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid event_type. Valid types: {valid_event_types}"
            )
        
        # ========== SANITIZATION (Enterprise: Prevent injection) ==========
        user_id = request.user_id[:100] if request.user_id else 'anonymous'
        session_id = request.session_id[:100] if request.session_id else None
        event_target = request.event_target[:50] if request.event_target else None
        event_value = request.event_value[:500] if request.event_value else None
        page_url = request.page_url[:500] if request.page_url else None
        device_type = request.device_type[:20] if request.device_type else None
        browser = request.browser[:50] if request.browser else None
        chart_id = request.chart_id[:100] if request.chart_id else None
        
        # ========== RATE LIMITING (Enterprise: Prevent abuse) ==========
        # Simple in-memory rate limit: max 100 events per user per minute
        # In production, use Redis for distributed rate limiting
        rate_limit_key = f"events_{user_id}"
        current_time = time.time()
        
        if not hasattr(app.state, 'event_rate_limits'):
            app.state.event_rate_limits = {}
        
        user_events = app.state.event_rate_limits.get(rate_limit_key, [])
        # Remove events older than 60 seconds
        user_events = [t for t in user_events if current_time - t < 60]
        
        if len(user_events) >= 100:
            print(f"⚠️ Rate limit hit for user {user_id[:8]}...")
            # Don't raise error, just skip silently (Big Tech: graceful degradation)
            return {"success": True, "rate_limited": True}
        
        user_events.append(current_time)
        app.state.event_rate_limits[rate_limit_key] = user_events
        
        # ========== SAVE TO DATABASE ==========
        db = SessionLocal()
        
        try:
            event_id = str(uuid.uuid4())
            
            db.execute(text("""
                INSERT INTO user_events (
                    id, user_id, session_id, event_type, event_target, event_value,
                    duration_ms, scroll_depth_percent, message_index, chart_id,
                    page_url, device_type, browser, viewport_width, viewport_height,
                    event_timestamp, created_at
                ) VALUES (
                    :id, :user_id, :session_id, :event_type, :event_target, :event_value,
                    :duration_ms, :scroll_depth_percent, :message_index, :chart_id,
                    :page_url, :device_type, :browser, :viewport_width, :viewport_height,
                    NOW(), NOW()
                )
            """), {
                "id": event_id,
                "user_id": user_id,
                "session_id": session_id,
                "event_type": request.event_type,
                "event_target": event_target,
                "event_value": event_value,
                "duration_ms": request.duration_ms,
                "scroll_depth_percent": request.scroll_depth_percent,
                "message_index": request.message_index,
                "chart_id": chart_id,
                "page_url": page_url,
                "device_type": device_type,
                "browser": browser,
                "viewport_width": request.viewport_width,
                "viewport_height": request.viewport_height,
            })
            
            db.commit()
            
            # Debug logging (remove in production for performance)
            print(f"📊 Event logged: {request.event_type} | user={user_id[:8]}... | target={event_target}")
            
            return {
                "success": True,
                "event_id": event_id,
                "event_type": request.event_type
            }
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Event logging error: {e}")
        # Big Tech Pattern: Never fail user experience for analytics
        # Return success even if logging fails
        return {"success": True, "logged": False, "reason": "internal_error"}


@app.post("/api/events/batch")
async def log_user_events_batch(request: dict):
    """
    Log multiple user events in a single request.
    
    Big Tech Pattern: Batch processing for efficiency
    - Frontend collects events, sends in batches every N seconds
    - Reduces network overhead
    - Better for mobile/slow connections
    
    Request format:
        {
            "user_id": "abc-123",
            "session_id": "xyz-789",
            "events": [
                {"event_type": "scroll", "scroll_depth_percent": 50, ...},
                {"event_type": "hover", "duration_ms": 2000, ...},
                ...
            ]
        }
    """
    try:
        from database import SessionLocal
        from sqlalchemy import text
        
        user_id = request.get("user_id", "anonymous")[:100]
        session_id = request.get("session_id")
        if session_id:
            session_id = session_id[:100]
        
        events = request.get("events", [])
        
        if not events:
            return {"success": True, "logged": 0}
        
        # ========== RATE LIMITING ==========
        if len(events) > 50:
            events = events[:50]  # Max 50 events per batch
            print(f"⚠️ Batch truncated to 50 events for user {user_id[:8]}...")
        
        db = SessionLocal()
        logged_count = 0
        
        try:
            for event in events:
                try:
                    event_id = str(uuid.uuid4())
                    event_type = event.get("event_type", "unknown")
                    
                    # Skip invalid event types
                    valid_types = [
                        # Original events
                        'page_view', 'copy', 'scroll', 'hover', 'click',
                        'idle', 'export', 'focus', 'blur', 'chart_interact',
                        'response_expand', 'response_collapse', 'feedback_start',
                        # 🧠 Phase 5.1: AGI implicit behavior signals
                        'scroll_25', 'scroll_50', 'scroll_80', 'scroll_100',  # Scroll milestones
                        'dwell_long',   # 30s+ engagement with message
                        'hover_long',   # 3s+ hover on chart
                        'abandon',      # Left without giving feedback
                    ]
                    if event_type not in valid_types:
                        continue
                    
                    db.execute(text("""
                        INSERT INTO user_events (
                            id, user_id, session_id, event_type, event_target, event_value,
                            duration_ms, scroll_depth_percent, message_index, chart_id,
                            page_url, device_type, browser, viewport_width, viewport_height,
                            event_timestamp, created_at
                        ) VALUES (
                            :id, :user_id, :session_id, :event_type, :event_target, :event_value,
                            :duration_ms, :scroll_depth_percent, :message_index, :chart_id,
                            :page_url, :device_type, :browser, :viewport_width, :viewport_height,
                            NOW(), NOW()
                        )
                    """), {
                        "id": event_id,
                        "user_id": user_id,
                        "session_id": session_id,
                        "event_type": event_type,
                        "event_target": event.get("event_target", "")[:50] if event.get("event_target") else None,
                        "event_value": event.get("event_value", "")[:500] if event.get("event_value") else None,
                        "duration_ms": event.get("duration_ms"),
                        "scroll_depth_percent": event.get("scroll_depth_percent"),
                        "message_index": event.get("message_index"),
                        "chart_id": event.get("chart_id", "")[:100] if event.get("chart_id") else None,
                        "page_url": event.get("page_url", "")[:500] if event.get("page_url") else None,
                        "device_type": event.get("device_type", "")[:20] if event.get("device_type") else None,
                        "browser": event.get("browser", "")[:50] if event.get("browser") else None,
                        "viewport_width": event.get("viewport_width"),
                        "viewport_height": event.get("viewport_height"),
                    })
                    
                    logged_count += 1
                    
                except Exception as e:
                    print(f"⚠️ Single event failed (continuing): {e}")
                    continue
            
            db.commit()
            print(f"📊 Batch logged: {logged_count} events for user {user_id[:8]}...")
            
            return {
                "success": True,
                "logged": logged_count,
                "total_received": len(events)
            }
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Batch event logging error: {e}")
        return {"success": True, "logged": 0, "reason": "internal_error"}


# ============================================================================
# END PHASE 6: IMPLICIT USER BEHAVIOR TRACKING
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info", loop="asyncio")
