"""
Database models using SQLAlchemy ORM
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

# Convert SQLAlchemy format if needed
if DATABASE_URL.startswith('postgresql+psycopg2://'):
    DATABASE_URL = DATABASE_URL.replace('postgresql+psycopg2://', 'postgresql://')

# Create engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Create base class for models
Base = declarative_base()

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Existing models like ChatSession, ChatMessage, etc.


class ChartData(Base):
    """Store Plotly charts in PostgreSQL"""
    __tablename__ = "chart_data"
    
    id = Column(Integer, primary_key=True)
    chart_id = Column(String(255), unique=True, nullable=False, index=True)
    session_id = Column(String(255))
    chart_html = Column(Text, nullable=False)
    chart_type = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


class MessageFeedback(Base):
    """Store user feedback (thumbs up/down) - Big Tech pattern: RLHF data collection"""
    __tablename__ = "message_feedback"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    feedback_id = Column(String(255), unique=True, nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    message_index = Column(Integer, nullable=False)
    question_text = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=False)
    feedback_type = Column(String(20), nullable=False)  # thumbs_up | thumbs_down
    route_used = Column(String(50))  # DATABASE | WEB_SEARCH | CODE_INTERPRETER
    sql_query = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class InteractionLog(Base):
    """
    Comprehensive interaction tracking - Big Tech pattern: Complete observability
    
    Tracks ALL user interactions across ALL agent routes:
    - DATABASE: SQL queries
    - WEB_SEARCH: Brave Search
    - CODE_INTERPRETER: Python execution
    - RAG: Document upload + retrieval
    - IMAGE_ANALYSIS: GPT-4 Vision
    - PREDICTION: Prophet forecasting
    - IDENTITY: Agent self-description
    
    Design: Single wide table with NULLs (Google Analytics pattern)
    Scale: Optimized for 10K-100K interactions/month
    """
    __tablename__ = "interaction_log"
    
    # ==================== CORE IDENTITY ====================
    # Every interaction has these fields
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    interaction_id = Column(String(255), unique=True, nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    user_id = Column(String(255), index=True, nullable=True)  # 🧠 AGI: Track user across sessions
    turn_number = Column(Integer, nullable=False)
    message_index = Column(Integer, nullable=False)  # For UI feedback buttons
    
    # ==================== QUESTION & ANSWER ====================
    # Universal fields for all routes
    
    question_text = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=False)
    route_used = Column(String(50), nullable=False, index=True)
    
    # ==================== DATABASE ROUTE ====================
    # Phase 1 implementation
    
    sql_query = Column(Text)
    sql_template_used = Column(String(100))
    sql_template_confidence = Column(Float)
    data_returned = Column(Boolean)
    
    # ==================== WEB_SEARCH ROUTE ====================
    # Phase 2 implementation (columns ready, not tracking yet)
    
    web_search_query = Column(Text)
    web_sources_count = Column(Integer)
    web_sources = Column(JSONB)  # [{"url", "title", "snippet"}]
    
    # ==================== CODE_INTERPRETER ROUTE ====================
    # Phase 2 implementation (columns ready, not tracking yet)
    
    code_executed = Column(Text)
    code_language = Column(String(50), default="python")
    code_output = Column(Text)
    code_error = Column(Text)
    execution_time_ms = Column(Integer)
    charts_generated = Column(JSONB)  # Array of chart objects: [{"chart_id", "filename", "format", "size"}]
    chart_type = Column(String(50))  # Type of chart (line, bar, pie, etc.)
    chart_format = Column(String(50))  # Format (html, png, etc.)
    
    # ==================== RAG/DOCUMENT ROUTE ====================
    # Phase 3 implementation (columns ready, not tracking yet)
    
    documents_uploaded = Column(JSONB)  # [{"doc_id", "filename", "file_type", "size"}]
    documents_used_in_response = Column(JSONB)  # [{"doc_id", "chunks", "relevance"}]
    vector_search_similarity = Column(Float)
    rag_retrieval_success = Column(Boolean)
    
    # ==================== IMAGE_ANALYSIS ROUTE ====================
    # Phase 3 implementation (columns ready, not tracking yet)
    
    images_uploaded = Column(JSONB)  # [{"image_id", "filename", "type", "size"}]
    vision_model_used = Column(String(50))  # gpt-4o, gpt-4-vision-preview
    image_analysis_confidence = Column(Float)
    
    # ==================== PREDICTION ROUTE ====================
    # Phase 4 implementation (columns ready, not tracking yet)
    
    prediction_model_used = Column(String(50))  # prophet, scikit-learn
    prediction_input_data = Column(JSONB)  # CSV/Excel info
    prediction_output = Column(JSONB)  # Forecast values
    prediction_accuracy_score = Column(Float)
    prediction_horizon = Column(String(50))  # 7_days, 30_days, 90_days
    
    # ==================== FEEDBACK & LEARNING ====================
    # User feedback for RLHF-style improvement
    
    feedback_type = Column(String(20))  # thumbs_up | thumbs_down | null
    feedback_text = Column(Text, nullable=True)  # Optional user feedback text
    feedback_comment = Column(Text)
    
    # ==================== METADATA & PERFORMANCE ====================
    # System metrics and context
    
    model_provider = Column(String(50))  # openai | ollama | groq
    model_name = Column(String(100))  # gpt-4o | qwen2.5:14b
    response_time_ms = Column(Integer)
    token_count = Column(Integer)
    error_occurred = Column(Boolean, default=False)
    error_message = Column(Text)
    
    # ==================== CONTEXT & CLASSIFICATION ====================
    # Semantic understanding
    
    conversation_topic = Column(String(100))  # sales | traffic | product | general
    user_intent = Column(String(50))  # query | analysis | report | prediction
    multi_modal = Column(Boolean, default=False)  # Text + images/files
    modalities_used = Column(JSONB)  # ["text", "chart", "image", "document"]
    
    # ==================== TIMESTAMPS ====================
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

