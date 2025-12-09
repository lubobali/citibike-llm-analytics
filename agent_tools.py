# agent_tools.py

from __future__ import annotations

import re
import json
import requests
from typing import Any, Iterable

import pandas as pd

from db import run_sql
from schema_utils import load_schema_card


# ===== Safety / Config =====

# =============================================================================
# 🔐 SECURITY: Allowed Tables Allowlist (Big Tech: Zero Trust Pattern)
# =============================================================================
# Only tables in this list can be queried. Schema-qualified names required
# for non-public schemas (e.g., citibike.fact_rides, not just fact_rides).
# =============================================================================
ALLOWED_TABLES = [
    # LuBot analytics (public schema)
    "daily_click_summary",
    "click_logs",
    # Citibike star schema (TheCommons XR Homework - Task 4)
    "citibike.fact_rides",
    "citibike.dim_date",
    "citibike.dim_time_of_day",
    "citibike.dim_bike_type",
    "citibike.dim_member_type",
]

DANGEROUS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|MERGE)\b",
    re.IGNORECASE,
)


# ===== Utilities =====

def _is_select(query: str) -> bool:
    return bool(re.match(r"^\s*SELECT\b", query or "", flags=re.IGNORECASE))


def _contains_allowed_tables(query: str, allowed: Iterable[str]) -> bool:
    ql = (query or "").lower()
    return any(tbl.lower() in ql for tbl in allowed)


def _rewrite_mysql_date_sub_to_pg(query: str) -> str:
    """
    Convert MySQL DATE_SUB(CURRENT_DATE, INTERVAL N DAY) to PostgreSQL syntax:
    CURRENT_DATE - INTERVAL 'N days'
    """
    pattern = r"DATE_SUB\s*\(\s*CURRENT_DATE\s*,\s*INTERVAL\s*(\d+)\s*DAY\s*\)"
    return re.sub(pattern, r"CURRENT_DATE - INTERVAL '\1 days'", query, flags=re.IGNORECASE)


# ----- Dimension aliasing: map human words -> real fields we can return -----
DIMENSION_ALIASES = {
    "project": "page",     # treat "project" as real page/project URL
    "projects": "page",
    "page": "page",
    "pages": "page",
}


def _normalize_common_intents(user_question: str, query: str) -> str:
    """
    Map common natural-language questions to proven, safe SQL on your schema.
    Uses daily_click_summary (JSONB columns for device/referrer) and click_logs for page-level.
    If no pattern matches, returns a harmless default SELECT.
    """
    # ====== PERMANENT SQL SPACING FIX ======
    import re
    if query:
        query = re.sub(r'(\w)(WHERE|FROM|JOIN|ON|AND|OR)', r'\1 \2', query, flags=re.IGNORECASE)
        query = re.sub(r'\s+', ' ', query)
        print(f"🔧 SQL spacing fixed: {query}")
    # ====== END FIX ======
    
    uq = (user_question or "").strip().lower()

    # ---------- Time window helpers ----------
    last_7 = (
        "last 7 days" in uq
        or "past 7 days" in uq
        or "last week" in uq
        or "past week" in uq
        or ("week" in uq and "last" in uq)
    )
    last_30 = ("last 30 days" in uq) or ("past 30 days" in uq) or ("last month" in uq) or ("past month" in uq)
    yesterday = "yesterday" in uq
    today = ("today" in uq) or ("current day" in uq)

    def window_between(days: int) -> str:
        return f"date BETWEEN CURRENT_DATE - INTERVAL '{days} days' AND CURRENT_DATE"

    def ts_window_between(days: int) -> str:
        return f"timestamp >= CURRENT_DATE - INTERVAL '{days} days'"

    # ---------- Detect "by page" intents (use click_logs.page_name) ----------
    asks_by_page = (
        (" by page" in uq) or ("top pages" in uq) or ("pages" in uq) or
        (("page" in uq) and any(w in uq for w in ["top", "most", "visited", "popular"]))
    )
    wants_timeseries = any(k in uq for k in ("line", "over time", "by date", "daily", "trend"))

    if asks_by_page:
        # We aggregate directly from click_logs so we return real page names
        if yesterday:
            if wants_timeseries:
                # daily breakdown for a single day (still grouped by date for consistency)
                return (
                    "SELECT DATE_TRUNC('day', timestamp)::date AS date, page_name AS page, "
                    "COUNT(*)::int AS total_clicks "
                    "FROM click_logs "
                    "WHERE timestamp::date = CURRENT_DATE - INTERVAL '1 day' "
                    "GROUP BY date, page "
                    "ORDER BY date, page;"
                )
            # Totals by page for yesterday
            return (
                "SELECT page_name AS page, COUNT(*)::int AS total_clicks "
                "FROM click_logs "
                "WHERE timestamp::date = CURRENT_DATE - INTERVAL '1 day' "
                "GROUP BY page "
                "ORDER BY total_clicks DESC;"
            )

        if last_30:
            if wants_timeseries:
                return (
                    "SELECT DATE_TRUNC('day', timestamp)::date AS date, page_name AS page, "
                    "COUNT(*)::int AS total_clicks "
                    "FROM click_logs "
                    f"WHERE {ts_window_between(30)} "
                    "GROUP BY date, page "
                    "ORDER BY date, page;"
                )
            return (
                "SELECT page_name AS page, COUNT(*)::int AS total_clicks "
                "FROM click_logs "
                f"WHERE {ts_window_between(30)} "
                "GROUP BY page "
                "ORDER BY total_clicks DESC;"
            )

        # Default window = last 7 days
        if wants_timeseries:
            return (
                "SELECT DATE_TRUNC('day', timestamp)::date AS date, page_name AS page, "
                "COUNT(*)::int AS total_clicks "
                "FROM click_logs "
                f"WHERE {ts_window_between(7)} "
                "GROUP BY date, page "
                "ORDER BY date, page;"
            )
        return (
            "SELECT page_name AS page, COUNT(*)::int AS total_clicks "
            "FROM click_logs "
            f"WHERE {ts_window_between(7)} "
            "GROUP BY page "
            "ORDER BY total_clicks DESC;"
        )

    # ---------- Averages ----------
    if ("average time on page" in uq) or ("avg time on page" in uq):
        if yesterday:
            return (
                "SELECT AVG(avg_time_on_page) AS avg_time_on_page "
                "FROM daily_click_summary "
                "WHERE date = CURRENT_DATE - INTERVAL '1 day';"
            )
        if last_30:
            return (
                "SELECT AVG(avg_time_on_page) AS avg_time_on_page "
                "FROM daily_click_summary "
                f"WHERE {window_between(30)};"
            )
        # default 7d
        return (
            "SELECT AVG(avg_time_on_page) AS avg_time_on_page "
            "FROM daily_click_summary "
            f"WHERE {window_between(7)};"
        )

    # ---------- Highest total clicks (top page) — use click_logs ----------
    if (("highest total clicks" in uq) or
        ("which page had the most clicks" in uq) or
        ("most clicks" in uq and "page" in uq) or
        ("most visited" in uq and "page" in uq) or
        ("most popular" in uq and "page" in uq)):
        if yesterday:
            return (
                "SELECT page_name AS page, COUNT(*)::int AS total_clicks "
                "FROM click_logs "
                "WHERE timestamp::date = CURRENT_DATE - INTERVAL '1 day' "
                "GROUP BY page "
                "ORDER BY total_clicks DESC "
                "LIMIT 1;"
            )
        if last_30:
            return (
                "SELECT page_name AS page, COUNT(*)::int AS total_clicks "
                "FROM click_logs "
                f"WHERE {ts_window_between(30)} "
                "GROUP BY page "
                "ORDER BY total_clicks DESC "
                "LIMIT 1;"
            )
        # default 7d
        return (
            "SELECT page_name AS page, COUNT(*)::int AS total_clicks "
            "FROM click_logs "
            f"WHERE {ts_window_between(7)} "
            "GROUP BY page "
            "ORDER BY total_clicks DESC "
            "LIMIT 1;"
        )

    # ---------- Total clicks (site-wide) ----------
    if ("how many total clicks" in uq) or (("total clicks" in uq) and ("site" in uq or "overall" in uq or "all" in uq)):
        if yesterday:
            return (
                "SELECT SUM(total_clicks) AS total_clicks "
                "FROM daily_click_summary "
                "WHERE date = CURRENT_DATE - INTERVAL '1 day';"
            )
        if last_30:
            return (
                "SELECT SUM(total_clicks) AS total_clicks "
                "FROM daily_click_summary "
                f"WHERE {window_between(30)};"
            )
        # default 7d
        return (
            "SELECT SUM(total_clicks) AS total_clicks "
            "FROM daily_click_summary "
            f"WHERE {window_between(7)};"
        )

    # ---------- Referrers (traffic sources) ----------
    if ("referrer" in uq and yesterday) or ("most visitors" in uq and yesterday):
        return (
            "SELECT key AS referrer, (value)::int AS visits "
            "FROM daily_click_summary, jsonb_each(top_referrers::jsonb) "
            "WHERE date = CURRENT_DATE - INTERVAL '1 day' "
            "ORDER BY visits DESC "
            "LIMIT 1;"
        )

    if (
        ("most traffic" in uq)
        or ("getting the most traffic" in uq)
        or ("where am i getting" in uq and "traffic" in uq)
        or ("where i am getting" in uq and "traffic" in uq)
        or ("where do i get the most traffic" in uq)
    ):
        if yesterday:
            return (
                "SELECT key AS referrer, (value)::int AS visits "
                "FROM daily_click_summary, jsonb_each(top_referrers::jsonb) "
                "WHERE date = CURRENT_DATE - INTERVAL '1 day' "
                "ORDER BY visits DESC "
                "LIMIT 1;"
            )
        if last_30:
            return (
                "SELECT key AS referrer, SUM((value)::int) AS visits "
                "FROM daily_click_summary, jsonb_each(top_referrers::jsonb) "
                f"WHERE {window_between(30)} "
                "GROUP BY referrer "
                "ORDER BY visits DESC "
                "LIMIT 1;"
            )
        # default 7d
        return (
            "SELECT key AS referrer, SUM((value)::int) AS visits "
            "FROM daily_click_summary, jsonb_each(top_referrers::jsonb) "
            f"WHERE {window_between(7)} "
            "GROUP BY referrer "
            "ORDER BY visits DESC "
            "LIMIT 1;"
        )

    # ---------- Device split ----------
    if ("device type" in uq or "device types" in uq or "by device" in uq
        or "device" in uq or "mobile" in uq or "desktop" in uq):
        if last_30:
            return (
                "SELECT key AS device, SUM((value)::int) AS visits "
                "FROM daily_click_summary, jsonb_each(device_split::jsonb) "
                f"WHERE {window_between(30)} "
                "GROUP BY device "
                "ORDER BY visits DESC;"
            )
        if yesterday:
            return (
                "SELECT key AS device, (value)::int AS visits "
                "FROM daily_click_summary, jsonb_each(device_split::jsonb) "
                "WHERE date = CURRENT_DATE - INTERVAL '1 day' "
                "ORDER BY visits DESC;"
            )
        if last_7:
            return (
                "SELECT key AS device, SUM((value)::int) AS visits "
                "FROM daily_click_summary, jsonb_each(device_split::jsonb) "
                f"WHERE {window_between(7)} "
                "GROUP BY device "
                "ORDER BY visits DESC;"
            )
        # default snapshot (3 days)
        return (
            "SELECT key AS device, SUM((value)::int) AS visits "
            "FROM daily_click_summary, jsonb_each(device_split::jsonb) "
            f"WHERE {window_between(3)} "
            "GROUP BY device "
            "ORDER BY visits DESC;"
        )

    # ---------- Repeat visits ----------
    if "repeat visits" in uq:
        if last_30:
            return (
                "SELECT SUM(repeat_visits) AS repeat_visits "
                "FROM daily_click_summary "
                f"WHERE {window_between(30)};"
            )
        if yesterday:
            return (
                "SELECT SUM(repeat_visits) AS repeat_visits "
                "FROM daily_click_summary "
                "WHERE date = CURRENT_DATE - INTERVAL '1 day';"
            )
        # default 14d
        return (
            "SELECT SUM(repeat_visits) AS repeat_visits "
            "FROM daily_click_summary "
            f"WHERE {window_between(14)};"
        )

    # ---------- Generic "yesterday" ----------
    if "yesterday" in uq and ("summary" in uq) or ("yesterday" in uq and ("stats" in uq or "clicks" in uq)):
        return (
            "SELECT date, project_name, total_clicks, avg_time_on_page, top_referrers "
            "FROM daily_click_summary "
            "WHERE date = CURRENT_DATE - INTERVAL '1 day' "
            "ORDER BY total_clicks DESC "
            "LIMIT 50;"
        )

    # ---------- Otherwise: honor a provided SELECT (after DATE_SUB rewrite) ----------
    q2 = _rewrite_mysql_date_sub_to_pg(query or "")
    if _is_select(q2):
        return q2

    # ---------- Safe default ----------
    return (
        "SELECT date, project_name, total_clicks, avg_time_on_page "
        "FROM daily_click_summary "
        "ORDER BY date DESC "
        "LIMIT 50;"
    )


def _coerce_to_df(rows: Any) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def _nullish_metric(df: pd.DataFrame) -> bool:
    """True if df has no meaningful metric (empty or 1x1 NULL or metric column all NaN/0)."""
    if df is None or getattr(df, "empty", True):
        return True
    if df.shape == (1, 1):
        v = df.iat[0, 0]
        try:
            return v is None or (pd.isna(v)) or (isinstance(v, str) and not v.strip())
        except Exception:
            return False
    cols_lower = [c.lower() for c in df.columns]
    for cand in ("visits", "total_clicks", "clicks"):
        if cand in cols_lower:
            col = df.columns[cols_lower.index(cand)]
            try:
                s = pd.to_numeric(df[col], errors="coerce")
                return s.notna().sum() == 0 or (s.fillna(0).sum() == 0)
            except Exception:
                return False
    return False


# ===== Main: SQL tool (fetch data) =====

def run_sql_tool(query: str, user_question: str = "", **kwargs: Any) -> pd.DataFrame:
    """
    Securely run a read-only SQL query and return a pandas DataFrame.
    - Normalizes common intents from `user_question`
    - Rewrites MySQL DATE_SUB to PostgreSQL syntax
    - Enforces SELECT-only and table allowlist
    - Fallbacks:
        • If "yesterday" referrer/traffic query returns empty/NULL ⇒ retry last 7 days
        • If device split via daily_click_summary is empty ⇒ try click_logs aggregation
    - Adds df.attrs['window_note'] for clearer, human output
    """
    if not query and not user_question:
        raise ValueError("Missing query and user_question — nothing to run.")

    uq = (user_question or "").lower()
    # requested window (for messaging)
    window_note = None
    if "yesterday" in uq:
        window_note = "yesterday"
    elif ("last 7 days" in uq) or ("past 7 days" in uq) or ("past week" in uq) or ("this week" in uq):
        window_note = "last 7 days"

    # First attempt: honor both user_question + query
    # If a specific query is provided, use it directly instead of normalizing
    if query and query.strip():
        normalized_query = query
    else:
        normalized_query = _normalize_common_intents(user_question, query or "")

    # Safety checks
    if DANGEROUS.search(normalized_query or ""):
        raise ValueError("Unsafe SQL detected. Only read-only queries are allowed.")
    if not _is_select(normalized_query):
        raise ValueError("Only SELECT queries are permitted.")

    # If disallowed tables, try re-mapping from natural language alone (ignore bad query)
    if not _contains_allowed_tables(normalized_query, ALLOWED_TABLES):
        if user_question:
            normalized_query = _normalize_common_intents(user_question, "")
        if not _contains_allowed_tables(normalized_query, ALLOWED_TABLES):
            normalized_query = (
                "SELECT date, project_name, total_clicks, avg_time_on_page "
                "FROM daily_click_summary "
                "ORDER BY date DESC "
                "LIMIT 50;"
            )

    # Execute primary query
    rows = run_sql(normalized_query, question=user_question)
    df = _coerce_to_df(rows)

    # ---- Smart fallback: "yesterday" + (referrer/traffic) empty OR NULL metric -> try 7 days ----
    wants_referrer = ("traffic" in uq) or ("referrer" in uq) or ("most visitors" in uq)
    mentions_yesterday = "yesterday" in uq
    if mentions_yesterday and wants_referrer and _nullish_metric(df):
        alt_question = uq.replace("yesterday", "last 7 days")
        alt_query = _normalize_common_intents(alt_question, "")
        if _contains_allowed_tables(alt_query, ALLOWED_TABLES) and _is_select(alt_query):
            rows2 = run_sql(alt_query, question=user_question)
            df2 = _coerce_to_df(rows2)
            if not _nullish_metric(df2):
                df2.attrs["window_note"] = "last 7 days (fallback from yesterday)"
                return df2

    # ---- Device split fallback ----
    wants_device = ("device" in uq or "mobile" in uq or "desktop" in uq or "device type" in uq or "device types" in uq or "by device" in uq)
    if wants_device and _nullish_metric(df):
        try:
            if "last 30" in uq:
                alt = (
                    "SELECT device, COUNT(*)::int AS visits "
                    "FROM click_logs "
                    "WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days' "
                    "GROUP BY device "
                    "ORDER BY visits DESC;"
                )
                win = "last 30 days"
            elif "yesterday" in uq:
                alt = (
                    "SELECT device, COUNT(*)::int AS visits "
                    "FROM click_logs "
                    "WHERE timestamp >= CURRENT_DATE - INTERVAL '1 day' "
                    "GROUP BY device "
                    "ORDER BY visits DESC;"
                )
                win = "yesterday"
            else:
                alt = (
                    "SELECT device, COUNT(*)::int AS visits "
                    "FROM click_logs "
                    "WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days' "
                    "GROUP BY device "
                    "ORDER BY visits DESC;"
                )
                win = "last 7 days"
            rows3 = run_sql(alt, question=user_question)
            df3 = _coerce_to_df(rows3)
            if not _nullish_metric(df3):
                df3.attrs["window_note"] = win
                return df3
        except Exception:
            pass

    # Tag the intended window so the finisher can say it in plain English
    if isinstance(df, pd.DataFrame) and window_note and not df.attrs.get("window_note"):
        df.attrs["window_note"] = window_note

    return df


# ===== Human-friendly finisher (build text from data) =====

def _summarize(df: pd.DataFrame) -> str:
    """Turn a result DataFrame into a short, human sentence (no tables)."""
    if df is None or getattr(df, "empty", True):
        return "No data available."

    note = ""
    wn = getattr(df, "attrs", {}).get("window_note")
    if wn:
        note = f" ({wn})"

    # Single scalar
    if df.shape == (1, 1):
        col = str(df.columns[0])
        value = df.iat[0, 0]
        if col.lower() in ("avg_time_on_page", "average_time_on_page", "avg_time"):
            try:
                val = float(value)
                return f"Average time on page{note}: {val:.2f} seconds."
            except Exception:
                return f"Average time on page{note}: {value}"
        if col.lower() in ("total_clicks", "clicks", "visits"):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return f"{col}{note}: No data"
            try:
                val = int(float(value))
            except Exception:
                val = value
            label = "Total clicks" if "click" in col.lower() else col
            return f"{label}{note}: {val}"
        return f"{col}{note}: {value}"

    cols = list(df.columns)
    cols_lower = [c.lower() for c in cols]

    # Device split
    if "device" in cols_lower and "visits" in cols_lower and len(df) >= 1:
        dcol = cols[cols_lower.index("device")]
        vcol = cols[cols_lower.index("visits")]
        try:
            s = pd.to_numeric(df[vcol], errors="coerce").fillna(0)
            total = float(s.sum())
            if total > 0:
                ordered = (
                    df.assign(__vis=pd.to_numeric(df[vcol], errors="coerce").fillna(0))
                    .sort_values("__vis", ascending=False)
                    .drop(columns="__vis")
                )
                parts = []
                for _, r in ordered.head(4).iterrows():
                    pct = (float(r[vcol]) / total) * 100.0
                    parts.append(f"{r[dcol]}: {pct:.1f}%")
                return f"Device split{note}: " + ", ".join(parts)
        except Exception:
            pass

    # Top pages
    if ("page" in cols_lower and "total_clicks" in cols_lower) or ("page" in cols_lower and "clicks" in cols_lower):
        dcol = cols[cols_lower.index("page")]
        vname = "total_clicks" if "total_clicks" in cols_lower else "clicks"
        vcol = cols[cols_lower.index(vname)]
        ordered = (
            df.assign(__v=pd.to_numeric(df[vcol], errors="coerce").fillna(0))
            .sort_values("__v", ascending=False)
            .drop(columns="__v")
        )
        parts = []
        for _, r in ordered.head(5).iterrows():
            try:
                parts.append(f"{r[dcol]} ({int(float(r[vcol]))})")
            except Exception:
                parts.append(f"{r[dcol]} ({r[vcol]})")
        return f"Top pages{note}: " + ", ".join(parts)

    # Referrers
    if "referrer" in cols_lower and ("visits" in cols_lower or "total_clicks" in cols_lower):
        dcol = cols[cols_lower.index("referrer")]
        vname = "visits" if "visits" in cols_lower else "total_clicks"
        vcol = cols[cols_lower.index(vname)]
        ordered = (
            df.assign(__v=pd.to_numeric(df[vcol], errors="coerce").fillna(0))
            .sort_values("__v", ascending=False)
            .drop(columns="__v")
        )
        parts = []
        for _, r in ordered.head(5).iterrows():
            try:
                parts.append(f"{r[dcol]} ({int(float(r[vcol]))})")
            except Exception:
                parts.append(f"{r[dcol]} ({r[vcol]})")
        return f"Top referrers{note}: " + ", ".join(parts)

    # Time series totals
    if "date" in cols_lower and "total_clicks" in cols_lower:
        vcol = cols[cols_lower.index("total_clicks")]
        try:
            total = int(pd.to_numeric(df[vcol], errors="coerce").fillna(0).sum())
            return f"Total clicks{note}: {total}"
        except Exception:
            pass

    # Generic key/value fallback
    text_col = None
    num_col = None
    for c in cols:
        if text_col is None and df[c].dtype == object:
            text_col = c
        if num_col is None:
            try:
                pd.to_numeric(df[c], errors="raise")
                num_col = c
            except Exception:
                pass
        if text_col and num_col:
            break
    if text_col and num_col:
        ordered = (
            df.assign(__v=pd.to_numeric(df[num_col], errors="coerce").fillna(0))
            .sort_values("__v", ascending=False)
            .drop(columns="__v")
        )
        parts = []
        for _, r in ordered.head(5).iterrows():
            try:
                parts.append(f"{r[text_col]} ({int(float(r[num_col]))})")
            except Exception:
                parts.append(f"{r[text_col]} ({r[num_col]})")
        return f"Top results{note}: " + ", ".join(parts)

    row = df.iloc[0]
    comp = []
    for c in df.columns[:5]:
        comp.append(f"{c}={row[c]}")
    return f"Results{note}: " + ", ".join(comp)


def get_schema_tool(**kwargs: Any) -> str:
    """
    Get the database schema information for the agent to understand available tables and columns.
    """
    schema_card = load_schema_card()
    if not schema_card:
        return "Schema information not available."

    schema_text = "Database Schema:\n"
    if "tables" in schema_card:
        for table_info in schema_card["tables"]:
            table_name = table_info["name"]
            schema_text += f"\n{table_name}:\n"
            if "columns" in table_info:
                for col_info in table_info["columns"]:
                    col_name = col_info["name"]
                    col_type = col_info["type"]
                    col_desc = col_info.get("description", "")
                    schema_text += f"  - {col_name} ({col_type}): {col_desc}\n"
            
            if "example_queries" in table_info:
                schema_text += "  Example queries:\n"
                for query in table_info["example_queries"]:
                    schema_text += f"    - {query}\n"

    return schema_text


def finish(sql_result: pd.DataFrame, user_question: str = "") -> str:
    """
    Build the human-friendly answer from the DataFrame produced by run_sql_tool.
    """
    return _summarize(sql_result)


def _create_chart_title(user_question: str, chart_type: str) -> str:
    """Generate a clean, professional chart title from user question."""
    import re
    
    # Remove command words and chart type mentions
    title = user_question.lower()
    
    # Remove all command phrases (expanded list)
    title = re.sub(r'\b(make|show|give|create|generate|visualize|display|plot|draw)\s+(me\s+)?(a\s+)?', '', title, flags=re.IGNORECASE)
    
    # Remove chart type mentions
    title = re.sub(r'\b(line|bar|pie)\s*(chart|graph)?\s*', '', title, flags=re.IGNORECASE)
    
    # Remove "in a" phrases
    title = re.sub(r'\s+in\s+a\s+', ' ', title, flags=re.IGNORECASE)
    
    # Remove question marks and extra spaces
    title = title.replace('?', '').strip()
    title = re.sub(r'\s+', ' ', title)
    
    # Capitalize first letter of each major word
    title = ' '.join(word.capitalize() if len(word) > 2 else word for word in title.split())
    
    # Normalize formatting: "of X From Y" → "X — Y" (cosmetic only)
    if isinstance(title, str) and title:
        # remove a leading "of " (case-insensitive)
        title = re.sub(r'^\s*of\s+', '', title, flags=re.IGNORECASE)
        # replace " from " with an em dash (case-insensitive)
        title = re.sub(r'\s+from\s+', ' — ', title, flags=re.IGNORECASE)
    
    # If title is empty or too short, use generic
    if len(title) < 3:
        title = f"{chart_type.capitalize()} Chart"
    
    return title


def generate_chart_tool(data: pd.DataFrame, chart_type: str = "bar", title: str = "Analytics Chart") -> dict:
    """
    Generate charts using QuickChart API that can be embedded inline.
    
    Args:
        data: DataFrame with the data to chart
        chart_type: Type of chart ('line', 'bar', 'pie')
        title: Chart title
    
    Returns:
        Dict with chart URL and metadata
    """
    try:
        # Convert DataFrame to chart format
        if len(data.columns) >= 2:
            labels = [str(x) for x in data.iloc[:, 0].tolist()]
            values = data.iloc[:, 1].tolist()
        else:
            return {"error": "Need at least 2 columns for chart generation"}
        
        # Create chart configuration with different handling for pie charts
        if chart_type == 'pie':
            chart_config = {
                "type": "pie",
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "data": values,
                        "backgroundColor": [
                            "rgba(255, 99, 132, 0.8)",   # Red
                            "rgba(54, 162, 235, 0.8)",   # Blue
                            "rgba(255, 206, 86, 0.8)",   # Yellow
                            "rgba(75, 192, 192, 0.8)",   # Green
                            "rgba(153, 102, 255, 0.8)",  # Purple
                            "rgba(255, 159, 64, 0.8)",   # Orange
                            "rgba(199, 199, 199, 0.8)",  # Grey
                            "rgba(83, 102, 255, 0.8)",   # Indigo
                        ]
                    }]
                },
                "options": {
                    "title": {
                        "display": True,
                        "text": _create_chart_title(title, chart_type)
                    },
                    "legend": {
                        "display": True,
                        "position": "right"
                    }
                }
            }
        else:
            # Line and bar charts keep existing config
            chart_config = {
                "type": chart_type,
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "data": values,
                        "backgroundColor": "rgba(54, 162, 235, 0.8)"
                    }]
                },
                "options": {
                    "title": {
                        "display": True,
                        "text": _create_chart_title(title, chart_type)
                    },
                    "legend": {
                        "display": False
                    }
                }
            }
        
        # Generate chart URL
        import urllib.parse
        chart_config_json = json.dumps(chart_config, separators=(',', ':'))
        chart_url = f"https://quickchart.io/chart?c={urllib.parse.quote(chart_config_json)}"
        
        return {
            "chart_url": chart_url,
            "chart_type": chart_type,
            "title": _create_chart_title(title, chart_type),
            "data_summary": f"{len(labels)} data points"
        }
        
    except Exception as e:
        return {"error": f"Error generating chart: {str(e)}"}


# ====== New: Intelligent chart configuration detection (ADD ONLY) ======
def detect_chart_config(user_question: str, data: pd.DataFrame, model_provider: str = "openai") -> dict:
    """
    Use LLM to intelligently detect chart type and generate Plotly code.
    Falls back to simple config if LLM fails.
    """
    try:
        from adalflow.core import Generator
        from adalflow.components.model_client import OpenAIClient, GroqAPIClient
        
        # Choose model client
        if model_provider == "groq":
            model_client = GroqAPIClient()
            model_kwargs = {"model": "llama-3.1-8b-instant"}
        else:
            model_client = OpenAIClient()
            model_kwargs = {"model": "gpt-4o-mini"}
        
        # Prepare data info for LLM
        col1_name = data.columns[0]
        col2_name = data.columns[1]
        sample_values = data.head(5).to_dict('records')
        
        # Create ENHANCED prompt with strict structure requirements
        prompt = f"""You are a Plotly expert. Generate Python code for the PERFECT chart.

USER REQUEST: {user_question}

DATA STRUCTURE (CRITICAL):
- DataFrame name: df
- Column 1: '{col1_name}' (categories/labels) - type: {data[col1_name].dtype}
- Column 2: '{col2_name}' (values/counts) - type: {data[col2_name].dtype}
- Sample data: {sample_values}
- Total rows: {len(data)}

AVAILABLE CHART TYPES:
1. bar - Compare categories (default choice)
2. line - Show trends over time
3. scatter - Show individual data points
4. pie - Show proportions (max 10 categories)
5. area - Filled line chart
6. box - Distribution statistics (needs raw values column)
7. violin - Distribution shape (needs raw values column)
8. histogram - Frequency distribution (needs raw values column)
9. heatmap - 2D correlation (needs matrix data)
10. funnel - Conversion steps
11. waterfall - Cumulative changes
12. sunburst - Hierarchical data (needs hierarchy columns)
13. treemap - Nested rectangles (needs hierarchy columns)

CODE REQUIREMENTS:
1. Use EXACT column names: df['{col1_name}'] and df['{col2_name}']
2. Create variable 'fig' using plotly.graph_objects (go) or plotly.express (px)
3. For bar/line/scatter/area: Use x=df['{col1_name}'], y=df['{col2_name}']
4. For pie: Use labels=df['{col1_name}'], values=df['{col2_name}']
5. For HORIZONTAL bar charts: REVERSE the data order so biggest bar is on TOP:
   df_reversed = df.iloc[::-1]
   Then use df_reversed instead of df in the chart
6. For PIE CHARTS with >8 rows: Auto-limit to top 7 + Other using this pattern:
   top_df = df.nlargest(7, '{col2_name}')
   other_sum = df.nsmallest(len(df)-7, '{col2_name}')['{col2_name}'].sum()
   other_row = pd.DataFrame([{{'{col1_name}': 'Other', '{col2_name}': other_sum}}])
   df = pd.concat([top_df, other_row], ignore_index=True)
7. Add title, axis labels, and hover tooltips
8. Use professional colors and styling
9. NO markdown code blocks - just plain Python code

RESPONSE FORMAT (JSON only):
{{
  "chart_type": "exact_type",
  "plotly_code": "import plotly.graph_objects as go\\nfig = go.Figure(...)\\nfig.update_layout(...)",
  "reasoning": "why this chart type"
}}

EXAMPLES:

REQUEST: "create a pie chart of clicks by referrer"
RESPONSE:
{{
  "chart_type": "pie",
  "plotly_code": "import plotly.graph_objects as go\\nfig = go.Figure(data=[go.Pie(labels=df['{col1_name}'], values=df['{col2_name}'], hole=0.3)])\\nfig.update_layout(showlegend=True)",
  "reasoning": "Pie chart shows proportion of clicks from each referrer"
}}

REQUEST: "show me a bar chart of top pages"
RESPONSE:
{{
  "chart_type": "bar",
  "plotly_code": "import plotly.graph_objects as go\\nfig = go.Figure(data=[go.Bar(x=df['{col1_name}'], y=df['{col2_name}'], marker_color='lightblue')])\\nfig.update_layout(xaxis_title='{col1_name}', yaxis_title='{col2_name}')",
  "reasoning": "Bar chart best compares values across categories"
}}

REQUEST: "show me a horizontal bar chart of top referrers"
RESPONSE:
{{
  "chart_type": "bar",
  "plotly_code": "import plotly.graph_objects as go\\ndf_reversed = df.iloc[::-1]\\nfig = go.Figure(data=[go.Bar(x=df_reversed['{col2_name}'], y=df_reversed['{col1_name}'], orientation='h', marker_color='lightblue')])\\nfig.update_layout(xaxis_title='{col2_name}', yaxis_title='{col1_name}')",
  "reasoning": "Horizontal bar chart with reversed data so biggest bar is on top"
}}

REQUEST: "create a line chart of clicks over time"
RESPONSE:
{{
  "chart_type": "line",
  "plotly_code": "import plotly.graph_objects as go\\nfig = go.Figure(data=[go.Scatter(x=df['{col1_name}'], y=df['{col2_name}'], mode='lines+markers', line=dict(width=2))])\\nfig.update_layout(xaxis_title='{col1_name}', yaxis_title='{col2_name}')",
  "reasoning": "Line chart shows trend over time"
}}

Now generate code for: {user_question}"""
        
        # Generate chart config with code
        generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            use_cache=False
        )
        
        result = generator(prompt_kwargs={"input_str": prompt})
        
        # Parse JSON response
        import json
        import re
        
        response_text = result.data if hasattr(result, 'data') else str(result)
        
        # Extract JSON (handle markdown code blocks)
        json_match = re.search(r'\{[^{{}}]*"chart_type"[^{{}}]*"plotly_code"[^{{}}]*\}', response_text, re.DOTALL)
        
        if json_match:
            config = json.loads(json_match.group())
            
            # Validate that code uses correct column names
            plotly_code = config.get('plotly_code', '')
            if col1_name not in plotly_code or col2_name not in plotly_code:
                print(f"⚠️ LLM code doesn't use correct column names, using fallback")
                raise ValueError("Invalid column usage in generated code")
            
            print(f"🎨 Dynamic chart: {config.get('chart_type')} - {config.get('reasoning', '')[:50]}...")
            return config
        else:
            raise ValueError("No valid JSON in LLM response")
            
    except Exception as e:
        print(f"⚠️ Dynamic chart generation failed: {e}, using fallback")
        
        # CRITICAL: Check data structure FIRST (before keywords)
        # If we have 3 columns with page_name → multi-series line chart
        columns_lower = [col.lower() for col in data.columns]
        if len(data.columns) == 3 and 'page_name' in columns_lower:
            print(f"📊 Fallback: Detected multi-series data (3 columns with page_name) → forcing line chart")
            return {"chart_type": "line", "mode": "lines+markers", "plotly_code": None}
        
        # ENHANCED FALLBACK with more chart types (keywords)
        user_q_lower = user_question.lower()
        
        # Map keywords to chart types
        if 'scatter' in user_q_lower or 'scatter plot' in user_q_lower:
            return {"chart_type": "scatter", "mode": "markers", "plotly_code": None}
        elif 'area' in user_q_lower:
            return {"chart_type": "area", "mode": "none", "plotly_code": None}
        elif 'funnel' in user_q_lower:
            return {"chart_type": "bar", "mode": "none", "plotly_code": None}  # Fallback to bar
        elif 'waterfall' in user_q_lower:
            return {"chart_type": "bar", "mode": "none", "plotly_code": None}  # Fallback to bar
        elif 'sunburst' in user_q_lower:
            return {"chart_type": "pie", "mode": "none", "plotly_code": None}  # Fallback to pie
        elif 'treemap' in user_q_lower:
            return {"chart_type": "pie", "mode": "none", "plotly_code": None}  # Fallback to pie
        elif 'heatmap' in user_q_lower or 'heat map' in user_q_lower:
            return {"chart_type": "bar", "mode": "none", "plotly_code": None}  # Fallback to bar (heatmap needs matrix)
        elif 'box' in user_q_lower or 'box plot' in user_q_lower:
            return {"chart_type": "bar", "mode": "none", "plotly_code": None}  # Fallback to bar (box needs raw data)
        elif 'violin' in user_q_lower:
            return {"chart_type": "bar", "mode": "none", "plotly_code": None}  # Fallback to bar
        elif 'histogram' in user_q_lower:
            return {"chart_type": "bar", "mode": "none", "plotly_code": None}  # Histogram needs raw data
        elif 'stacked' in user_q_lower or 'stack' in user_q_lower:
            return {"chart_type": "bar", "barmode": "stack", "plotly_code": None}
        elif 'line' in user_q_lower:
            return {"chart_type": "line", "mode": "lines+markers", "plotly_code": None}
        elif 'pie' in user_q_lower:
            return {"chart_type": "pie", "mode": "none", "plotly_code": None}
        else:
            return {"chart_type": "bar", "mode": "none", "plotly_code": None}


# ====== WEB SEARCH TOOLS (Task 12.4) ======

def search_web_tool(query: str, num_results: int = 5) -> pd.DataFrame:
    """
    Search the web using Brave Search API and return results as DataFrame.
    
    Args:
        query: Search query
        num_results: Number of results (default: 5, max: 10)
    
    Returns:
        DataFrame with columns: rank, title, snippet, url, source_domain
    """
    from tools.web_search_tools import search_web
    
    result = search_web(query, num_results)
    
    if result.get('error') or not result.get('results'):
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=['rank', 'title', 'snippet', 'url', 'source_domain'])
    
    # Convert results to DataFrame
    return pd.DataFrame(result['results'])


def get_web_search_capabilities_tool() -> str:
    """
    Get information about web search capabilities.
    
    Returns:
        String describing web search capabilities
    """
    from tools.web_search_tools import get_search_capabilities
    return get_search_capabilities()


# ====== IMAGE ANALYSIS TOOL (Task 12.1.3) ======

def analyze_images_agent_tool(image_paths: list, user_question: str = "") -> str:
    """
    Agent-facing wrapper for image analysis.
    Automatically called when images are uploaded.
    
    Args:
        image_paths: List of image file paths
        user_question: User's question about the images
        
    Returns:
        Analysis text from GPT-4o Vision
    """
    from tools.image_tools import analyze_images_tool
    
    result = analyze_images_tool(image_paths, user_question if user_question else None)
    return result
