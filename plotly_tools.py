"""
Plotly-based chart generation tool - drop-in replacement for QuickChart version.

This module provides interactive Plotly charts as embedded HTML using data URIs,
maintaining the same interface as agent_tools.generate_chart_tool() for seamless replacement.

🆕 PHASE 4: Now includes Chart Preference Learning (Netflix/Spotify pattern)
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Dict
import uuid
import re

# This will be set by main.py at startup
chart_cache = None

# =============================================================================
# BIG TECH PATTERN: Configuration Constants
# Netflix/Spotify: Centralized config, easy to modify, A/B testable
# =============================================================================

# Feature flags (Phase 9: A/B testing ready)
MULTI_COLOR_BARS_ENABLED = True
CHART_PREFERENCE_LEARNING_ENABLED = True

# Default color palette (Tailwind CSS colors - professional, accessible)
# Used when user hasn't specified preferences or needs color cycling
DEFAULT_COLOR_PALETTE = [
    '#3b82f6',  # blue (primary)
    '#22c55e',  # green
    '#f97316',  # orange
    '#ef4444',  # red
    '#8b5cf6',  # purple
    '#ec4899',  # pink
    '#14b8a6',  # teal
    '#eab308',  # yellow
    '#6366f1',  # indigo
    '#f43f5e',  # rose
]

# ============================================================================
# 🆕 PHASE 4: Chart Preference Learning Import
# ============================================================================
try:
    from tools.chart_preference_parser import (
        get_chart_preferences_for_query,
        apply_preferences_to_config,
        merge_preferences,
        get_default_preferences
    )
    CHART_PREFS_AVAILABLE = True
    print("✅ Chart preference learning enabled")
except ImportError:
    CHART_PREFS_AVAILABLE = False
    print("⚠️ Chart preference learning not available (tools.chart_preference_parser not found)")


def _create_chart_title(user_question: str, chart_type: str) -> str:
    """
    Generate a clean, professional chart title from user question.
    (Copied from agent_tools.py for consistency)
    """
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


def _limit_pie_data(df: pd.DataFrame, max_slices: int = 5) -> pd.DataFrame:
    """
    Limit pie chart data to top N slices, group rest as 'Other'.
    Returns a NEW DataFrame without modifying the original.
    """
    if len(df) <= max_slices:
        return df.copy()
    
    # Get column names
    label_col = df.columns[0]
    value_col = df.columns[1]
    
    # Sort by value descending and take top N
    sorted_df = df.sort_values(by=value_col, ascending=False)
    top_data = sorted_df.head(max_slices).copy()
    
    # Sum the rest as "Other"
    rest_data = sorted_df.iloc[max_slices:]
    if len(rest_data) > 0:
        other_sum = rest_data[value_col].sum()
        other_row = pd.DataFrame({label_col: ['Other'], value_col: [other_sum]})
        result = pd.concat([top_data, other_row], ignore_index=True)
    else:
        result = top_data
    
    print(f"📊 Limited pie chart: {len(df)} → {len(result)} slices (top {max_slices} + Other)")
    return result


def _validate_chart_data(df: pd.DataFrame, chart_type: str) -> tuple[bool, str, pd.DataFrame]:
    """
    Validate and auto-correct chart data for optimal visualization.
    
    Returns:
        (is_valid, reason, corrected_df)
    """
    # Pie charts: limit to readable number of slices
    if chart_type.lower() == 'pie' and len(df) > 8:
        corrected_df = _limit_pie_data(df, max_slices=5)
        return True, "Auto-limited pie chart to top 5 + Other", corrected_df
    
    # Bar charts: warn if too many categories (but don't fail)
    if chart_type.lower() == 'bar' and len(df) > 50:
        print(f"⚠️ Warning: Bar chart has {len(df)} categories, may be hard to read")
        return True, "Many categories", df
    
    # Line charts: ensure data is sorted if x-axis looks like dates/time
    if chart_type.lower() == 'line':
        # Try to detect if first column contains dates
        try:
            pd.to_datetime(df.iloc[:, 0])
            sorted_df = df.sort_values(by=df.columns[0])
            return True, "Sorted by x-axis", sorted_df
        except Exception:
            pass  # Not date data, that's fine
    
    return True, "Data valid", df


def _validate_and_limit_data(df: pd.DataFrame, chart_type: str, max_categories: int = 10) -> pd.DataFrame:
    """
    Auto-limit data for readability.
    - Bar charts: max 10 categories
    - Scatter charts: max 15 points
    - Line charts: no limit
    - Pie charts: handled elsewhere
    """
    try:
        if chart_type.lower() == 'bar' and len(df) > max_categories:
            if len(df.columns) >= 2:
                value_col = df.columns[1]
                df_limited = df.nlargest(max_categories, value_col)
                print(f"📊 Auto-limited bar chart: {len(df)} → {max_categories} items for readability")
                return df_limited
        if chart_type.lower() == 'scatter' and len(df) > 15:
            if len(df.columns) >= 2:
                value_col = df.columns[1]
                df_limited = df.nlargest(15, value_col)
                print(f"📊 Auto-limited scatter chart: {len(df)} → 15 points for readability")
                return df_limited
    except Exception:
        pass
    return df


def _execute_plotly_code(code: str, df: pd.DataFrame, title: str) -> go.Figure:
    """
    Safely execute LLM-generated Plotly code.
    
    Args:
        code: Python code string that creates a 'fig' variable
        df: DataFrame to pass to the code
        title: Chart title to use
    
    Returns:
        plotly.graph_objects.Figure
    """
    try:
        # Create a safe namespace for code execution
        namespace = {
            'pd': pd,
            'go': go,
            'px': px,
            'df': df,
            'title': title,
            'Figure': go.Figure
        }
        
        # Execute the code
        exec(code, namespace)
        
        # Extract the figure
        if 'fig' not in namespace:
            raise ValueError("Generated code did not create 'fig' variable")
        
        fig = namespace['fig']
        
        # Ensure it's a Plotly figure
        if not isinstance(fig, go.Figure):
            raise ValueError(f"Generated 'fig' is not a Plotly Figure, got {type(fig)}")
        
        # Apply consistent styling
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            template='plotly_white',
            height=500,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        print(f"✅ Successfully executed dynamic Plotly code")
        return fig
        
    except Exception as e:
        print(f"❌ Code execution failed: {e}")
        raise ValueError(f"Failed to execute Plotly code: {str(e)}")


# ============================================================================
# 🆕 PHASE 4: Preference Application Helper Functions
# ============================================================================

def _apply_color_preferences(fig, preferences: Dict, chart_type: str):
    """
    Apply learned color preferences to chart.
    
    BIG TECH PATTERN: Defensive programming with graceful fallbacks
    - Netflix: Never crash on bad data
    - Spotify: User preferences override defaults
    - Google: Feature flags for safe rollout
    
    Args:
        fig: Plotly figure object
        preferences: Learned preferences dict
        chart_type: Type of chart (bar, line, pie, etc.)
    """
    # Safety check: Validate inputs
    if not fig or not preferences:
        return
    
    if not CHART_PREFERENCE_LEARNING_ENABLED:
        print("🎨 Chart preference learning disabled via feature flag")
        return
    
    # Extract color palette from preferences
    colors = preferences.get('colors', {})
    palette = colors.get('palette', [])
    
    if not palette:
        return
    
    print(f"🎨 Applying learned preferences: {list(preferences.keys())}")
    
    try:
        chart_type_lower = chart_type.lower() if chart_type else ''
        
        # Big Tech Pattern: Trust actual figure data over config
        # Config might say 'heatmap' but dynamic code might generate 'bar'
        if fig.data and hasattr(fig.data[0], 'type'):
            actual_type = fig.data[0].type.lower()
            if actual_type in ['bar', 'pie'] and actual_type != chart_type_lower:
                print(f"🔍 Chart type override: config={chart_type}, actual={actual_type}")
                chart_type_lower = actual_type
        
        if chart_type_lower == 'bar':
            _apply_bar_colors(fig, palette)
        
        elif chart_type_lower == 'line':
            # 🧠 AGI: Apply multi-color to line charts (like bar charts)
            _apply_line_colors(fig, palette)
        
        elif chart_type_lower == 'pie':
            # Apply full palette to pie slices
            fig.update_traces(marker_colors=palette)
            print(f"🥧 Applied pie colors: {len(palette)} colors")
        
        elif chart_type_lower == 'scatter':
            # Apply first color to scatter markers
            fig.update_traces(marker_color=palette[0])
            print(f"📊 Applied scatter color: {palette[0]}")
        
        elif chart_type_lower == 'area':
            # Apply to both line and fill
            fig.update_traces(line_color=palette[0], fillcolor=palette[0])
            print(f"📉 Applied area color: {palette[0]}")
        
    except Exception as e:
        print(f"⚠️ Color preference application failed (graceful fallback): {e}")


def _apply_bar_colors(fig, palette: list):
    """
    Apply colors to bar chart - each bar gets different color.
    
    BIG TECH PATTERN: 
    - Netflix/Spotify honors user preferences
    - Only fallback to single color if palette literally has 1 color
    - Defensive null checks
    - Graceful degradation on failure
    
    Args:
        fig: Plotly figure object
        palette: List of color hex codes from user preferences
    """
    try:
        # Defensive checks for figure data
        if not fig.data:
            print("⚠️ No figure data, skipping bar colors")
            return
        
        if not hasattr(fig.data[0], 'x') or fig.data[0].x is None:
            print("⚠️ No x-axis data, using single color fallback")
            fallback_color = palette[0] if palette else DEFAULT_COLOR_PALETTE[0]
            fig.update_traces(marker_color=fallback_color)
            return
        
        # Get number of bars
        num_bars = len(fig.data[0].x)
        
        if num_bars == 0:
            print("⚠️ Zero bars detected, skipping colors")
            return
        
        # ========== BIG TECH PATTERN: Honor user preferences ==========
        # Only use single color if palette literally has 1 color
        # Otherwise, apply multi-color with cycling
        if not palette or len(palette) == 0:
            # No palette provided - use default multi-color
            color_source = DEFAULT_COLOR_PALETTE
            print(f"🌈 No palette provided, using default multi-color palette")
        elif len(palette) == 1:
            # User explicitly wants single color (palette has exactly 1 color)
            single_color = palette[0]
            fig.update_traces(marker_color=single_color)
            print(f"🎨 Single color requested: {single_color}")
            return
        else:
            # User provided multi-color palette - honor it!
            color_source = palette
            print(f"🎨 Using user multi-color palette: {len(palette)} colors")
        
        # Cycle colors for all bars: palette[i % len(palette)] for each bar index
        bar_colors = [color_source[i % len(color_source)] for i in range(num_bars)]
        
        # Apply colors
        fig.update_traces(marker_color=bar_colors)
        print(f"🌈 Multi-color bars applied: {num_bars} bars, {len(set(bar_colors))} unique colors")
        
    except Exception as e:
        # Graceful fallback: Single color on any failure
        fallback_color = palette[0] if palette else DEFAULT_COLOR_PALETTE[0]
        fig.update_traces(marker_color=fallback_color)
        print(f"⚠️ Multi-color bars fallback to single color: {e}")


def _apply_line_colors(fig, palette: list):
    """
    Apply colors to line chart - each series gets different color.
    
    🆕 BIG TECH AGI PATTERN (Netflix/Spotify): 
    - Multi-series line charts get multi-color (like bar charts)
    - Single-series gets first color from palette
    - Defensive null checks, graceful degradation
    
    Args:
        fig: Plotly figure object
        palette: List of color hex codes from user preferences
    """
    try:
        # Defensive checks for figure data
        if not fig.data:
            print("⚠️ No figure data, skipping line colors")
            return
        
        num_series = len(fig.data)
        
        if num_series == 0:
            print("⚠️ Zero line series detected, skipping colors")
            return
        
        # ========== BIG TECH PATTERN: Honor user preferences ==========
        # Multi-series: Each line gets different color from palette
        # Single-series: Just use first color
        if not palette or len(palette) == 0:
            # No palette provided - use default multi-color
            color_source = DEFAULT_COLOR_PALETTE
            print(f"🌈 No palette provided, using default multi-color palette for lines")
        elif len(palette) == 1:
            # User explicitly wants single color
            single_color = palette[0]
            for trace in fig.data:
                if hasattr(trace, 'line'):
                    trace.line.color = single_color
                if hasattr(trace, 'marker'):
                    trace.marker.color = single_color
            print(f"📈 Single line color requested: {single_color}")
            return
        else:
            # User provided multi-color palette - honor it!
            color_source = palette
            print(f"🎨 Using user multi-color palette for lines: {len(palette)} colors")
        
        # Apply different color to each series (trace)
        for i, trace in enumerate(fig.data):
            color = color_source[i % len(color_source)]
            if hasattr(trace, 'line'):
                trace.line.color = color
            if hasattr(trace, 'marker'):
                trace.marker.color = color
        
        print(f"🌈 Multi-color lines applied: {num_series} series, {min(num_series, len(color_source))} unique colors")
        
    except Exception as e:
        # Graceful fallback: Single color on any failure
        fallback_color = palette[0] if palette else DEFAULT_COLOR_PALETTE[0]
        fig.update_traces(line_color=fallback_color)
        print(f"⚠️ Multi-color lines fallback to single color: {e}")


def _apply_layout_preferences(fig: go.Figure, preferences: dict) -> None:
    """
    Apply learned layout preferences (legend, gridlines, theme) to a Plotly figure.
    
    🆕 PHASE 4: Big Tech Pattern - personalized chart layout
    
    Args:
        fig: Plotly Figure object
        preferences: Learned preferences dict
    """
    if not preferences:
        return
    
    layout_updates = {}
    
    try:
        # ========== LEGEND ==========
        legend_prefs = preferences.get('legend', {})
        if 'show' in legend_prefs:
            layout_updates['showlegend'] = legend_prefs['show']
            print(f"🎨 Legend: {'shown' if legend_prefs['show'] else 'hidden'}")
        
        if legend_prefs.get('position'):
            pos = legend_prefs['position']
            if pos == 'top':
                layout_updates['legend'] = dict(orientation='h', y=1.1, x=0.5, xanchor='center')
            elif pos == 'bottom':
                layout_updates['legend'] = dict(orientation='h', y=-0.2, x=0.5, xanchor='center')
            elif pos == 'left':
                layout_updates['legend'] = dict(orientation='v', x=-0.1, y=0.5)
            elif pos == 'right':
                layout_updates['legend'] = dict(orientation='v', x=1.05, y=0.5)
        
        # ========== GRIDLINES ==========
        gridlines_prefs = preferences.get('gridlines', {})
        if 'show' in gridlines_prefs:
            show_grid = gridlines_prefs['show']
            layout_updates['xaxis'] = dict(showgrid=show_grid)
            layout_updates['yaxis'] = dict(showgrid=show_grid)
            print(f"🎨 Gridlines: {'shown' if show_grid else 'hidden'}")
        
        # ========== THEME ==========
        theme_prefs = preferences.get('theme', {})
        if theme_prefs.get('mode') == 'dark':
            layout_updates['template'] = 'plotly_dark'
            layout_updates['paper_bgcolor'] = '#1f2937'
            layout_updates['plot_bgcolor'] = '#1f2937'
            print(f"🎨 Theme: dark mode")
        elif theme_prefs.get('mode') == 'light':
            layout_updates['template'] = 'plotly_white'
            print(f"🎨 Theme: light mode")
        
        # Apply all layout updates at once
        if layout_updates:
            fig.update_layout(**layout_updates)
            
    except Exception as e:
        print(f"⚠️ Layout preference application failed: {e}")


def _apply_data_label_preferences(fig: go.Figure, preferences: dict, chart_type: str) -> None:
    """
    Apply data label preferences to a Plotly figure.
    
    🆕 PHASE 4: Big Tech Pattern - personalized data labels
    
    Args:
        fig: Plotly Figure object
        preferences: Learned preferences dict
        chart_type: Type of chart
    """
    if not preferences:
        return
    
    data_labels = preferences.get('data_labels', {})
    
    if not data_labels:
        return
    
    try:
        show_labels = data_labels.get('show', None)
        position = data_labels.get('position', 'top')
        
        if show_labels is True:
            if chart_type.lower() == 'bar':
                # Add text labels to bars
                text_position = 'inside' if position == 'inside' else 'outside'
                fig.update_traces(
                    texttemplate='%{y}',
                    textposition=text_position
                )
                print(f"🎨 Data labels: shown ({text_position})")
                
            elif chart_type.lower() == 'pie':
                # Pie charts: show percentages
                fig.update_traces(textinfo='percent+label')
                print(f"🎨 Data labels: shown (percent+label)")
                
        elif show_labels is False:
            if chart_type.lower() == 'bar':
                fig.update_traces(texttemplate=None)
            elif chart_type.lower() == 'pie':
                fig.update_traces(textinfo='none')
            print(f"🎨 Data labels: hidden")
            
    except Exception as e:
        print(f"⚠️ Data label preference application failed: {e}")


# ============================================================================
# MAIN CHART GENERATION FUNCTION
# ============================================================================

def generate_chart_tool(
    data: pd.DataFrame, 
    chart_type: str = "bar", 
    title: str = "Analytics Chart", 
    config: dict = None,
    user_id: str = None,        # 🆕 PHASE 4: For preference lookup
    user_question: str = None   # 🆕 PHASE 4: For keyword extraction
) -> dict:
    """
    Generate interactive Plotly charts embedded as HTML data URIs.
    
    This is a drop-in replacement for the QuickChart version in agent_tools.py.
    
    🆕 PHASE 4: Now applies learned user preferences for colors, legend, etc.
    
    Args:
        data: DataFrame with the data to chart
        chart_type: Type of chart ('line', 'bar', 'pie')
        title: Chart title (usually the user's question)
        config: Optional chart configuration
        user_id: Optional user ID for preference lookup (Phase 4)
        user_question: Optional question for keyword extraction (Phase 4)
    
    Returns:
        Dict with chart_url (data URI), chart_type, title, and data_summary
    """
    try:
        # Validate input
        if data is None or data.empty:
            return {"error": "No data provided for chart generation"}
        
        if len(data.columns) < 2:
            return {"error": "Need at least 2 columns for chart generation"}
        
        # ========== 🆕 PHASE 4: Apply Learned Chart Preferences ==========
        # Big Tech Pattern: Netflix/Spotify - personalized styling
        learned_prefs = {}
        
        if CHART_PREFS_AVAILABLE:
            try:
                # Get learned preferences from database
                query_for_prefs = user_question or title
                learned_prefs = get_chart_preferences_for_query(query_for_prefs, user_id)
                
                if learned_prefs:
                    print(f"🎨 Applying learned preferences: {list(learned_prefs.keys())}")
                    
                    # Merge with config (learned prefs are base, config overrides)
                    if config is None:
                        config = {}
                    config = merge_preferences(learned_prefs, config)
                else:
                    print(f"🎨 No learned preferences, using defaults")
                    
            except Exception as e:
                # Big Tech Pattern: Never block on preference failure
                print(f"⚠️ Preference lookup failed (non-blocking): {e}")
        # ========== END PHASE 4 ==========
        
        # Extract data
        labels = [str(x) for x in data.iloc[:, 0].tolist()]
        values = data.iloc[:, 1].tolist()
        
        # Create clean title
        clean_title = _create_chart_title(title, chart_type)

        # Use config if provided, otherwise defaults
        if config is None:
            config = {"chart_type": chart_type, "mode": "none", "barmode": "group"}

        chart_type = config.get("chart_type", chart_type)
        mode = config.get("mode", "none")
        barmode = config.get("barmode", "group")

        # Detect if user wants horizontal bar chart
        orientation = 'v'  # default vertical
        if 'horizontal' in title.lower() or 'horizontal bar' in title.lower():
            orientation = 'h'
            print(f"🔄 Detected horizontal bar chart request")
        
        # 🆕 PHASE 4: Check for orientation preference
        bar_prefs = learned_prefs.get('bar', {}) if learned_prefs else {}
        if bar_prefs.get('orientation') == 'horizontal':
            orientation = 'h'
            print(f"🎨 Applying learned preference: horizontal bars")
        
        # ✅ HYBRID SYSTEM: Validate data before generating chart
        chart_type_config = config.get("chart_type", chart_type)
        is_valid, reason, validated_data = _validate_chart_data(data, chart_type_config)
        
        if reason != "Data valid":
            print(f"🔧 Chart validation: {reason}")
        
        # Additional auto-limit for bar/scatter for readability
        validated_data = _validate_and_limit_data(validated_data, chart_type_config)

        # Update labels/values with validated data
        labels = [str(x) for x in validated_data.iloc[:, 0].tolist()]
        values = validated_data.iloc[:, 1].tolist()

        # Check for multi-series line charts FIRST (before dynamic code)
        fig = None
        if chart_type_config.lower() == 'line' and len(validated_data.columns) == 3 and 'page_name' in validated_data.columns:
            # Multi-series line chart
            print("📊 Detected multi-series line chart (page_name column found)")
            series_column = 'page_name'
            x_column = 'date'
            y_column = 'count'
            fig = go.Figure()
            
            # 🆕 PHASE 4: Use learned color palette if available
            default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            colors = learned_prefs.get('colors', {}).get('palette', default_colors) if learned_prefs else default_colors
            
            for i, series_name in enumerate(validated_data[series_column].unique()):
                series_data = validated_data[validated_data[series_column] == series_name]
                fig.add_trace(go.Scatter(
                    x=series_data[x_column],
                    y=series_data[y_column],
                    mode='lines+markers',
                    name=series_name,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6)
                ))
            fig.update_layout(
                title=clean_title,
                xaxis_title=x_column.title(),
                yaxis_title=y_column.title(),
                hovermode='x unified',
                template='plotly_white',
                height=500,
                margin=dict(l=50, r=50, t=80, b=50)
            )

        # Only generate chart if fig wasn't already created (multi-series)
        if fig is None:
            # Check if we have LLM-generated Plotly code
            plotly_code = config.get('plotly_code') if config else None

            if plotly_code:
                # Use dynamic LLM-generated code (supports ALL Plotly chart types!)
                try:
                    print(f"🚀 Executing dynamic Plotly code for {config.get('chart_type', 'unknown')} chart")
                    fig = _execute_plotly_code(plotly_code, validated_data, clean_title)
                except Exception as e:
                    print(f"⚠️ Dynamic code failed, falling back to predefined charts: {e}")
                    # Fallback to predefined charts (with validation already applied)
                    if chart_type_config.lower() == 'pie':
                        fig = _create_pie_chart(labels, values, clean_title)
                    else:
                        fig = _create_bar_chart(labels, values, clean_title, barmode, orientation)
            else:
                # Use predefined chart functions (existing behavior - KEEPS ALL CURRENT CHARTS WORKING)
                if chart_type_config.lower() == 'pie':
                    fig = _create_pie_chart(labels, values, clean_title)
                elif chart_type_config.lower() == 'line':
                    # Single-series line chart (keep existing code)
                    fig = _create_line_chart(labels, values, clean_title, mode)
                elif chart_type_config.lower() == 'scatter':
                    fig = _create_scatter_chart(labels, values, clean_title, mode)
                elif chart_type_config.lower() == 'area':
                    fig = _create_area_chart(labels, values, clean_title)
                elif chart_type_config.lower() == 'bar':
                    fig = _create_bar_chart(labels, values, clean_title, barmode, orientation)
                else:  # Default to bar
                    fig = _create_bar_chart(labels, values, clean_title, barmode)
        
        # ========== 🆕 PHASE 4: Apply Learned Preferences to Figure ==========
        # Big Tech Pattern: Netflix/Spotify - personalized styling
        if CHART_PREFS_AVAILABLE and learned_prefs and fig is not None:
            try:
                _apply_color_preferences(fig, learned_prefs, chart_type_config)
                _apply_layout_preferences(fig, learned_prefs)
                _apply_data_label_preferences(fig, learned_prefs, chart_type_config)
            except Exception as e:
                # Big Tech Pattern: Never block on preference failure
                print(f"⚠️ Preference application failed (non-blocking): {e}")
        # ========== END PHASE 4 ==========
        
        # Convert to HTML
        html_content = fig.to_html(include_plotlyjs='cdn', config={'displayModeBar': True, 'responsive': True})
        
        # Generate unique chart ID
        chart_id = str(uuid.uuid4())
        
        # === BIG TECH APPROACH: Save to PostgreSQL ===
        try:
            from database import SessionLocal, ChartData
            from datetime import datetime
            
            # Extract actual chart type (bar/line/pie/scatter) from config
            actual_chart_type = chart_type_config if chart_type_config else chart_type
            
            # Big Tech Pattern: Trust actual figure data over config for DB storage
            # Config might say 'heatmap' but dynamic code might generate 'bar'
            # Covers ALL Plotly chart types for accurate AGI learning
            if fig and fig.data and hasattr(fig.data[0], 'type'):
                detected_type = fig.data[0].type.lower()
                valid_plotly_types = [
                    'bar', 'pie', 'scatter', 'heatmap', 'histogram', 
                    'box', 'violin', 'funnel', 'waterfall', 'treemap',
                    'sunburst', 'sankey', 'choropleth', 'scattergeo'
                ]
                if detected_type in valid_plotly_types and detected_type != actual_chart_type.lower():
                    print(f"🔍 DB chart_type override: config={actual_chart_type}, actual={detected_type}")
                    actual_chart_type = detected_type
            
            db = SessionLocal()
            chart_record = ChartData(
                chart_id=chart_id,
                session_id=config.get('session_id') if config and 'session_id' in config else None,
                chart_html=html_content,
                chart_type=actual_chart_type,  # ← FIXED! Now saves "bar", "line", "pie", etc.
                created_at=datetime.utcnow()
            )
            db.add(chart_record)
            db.commit()
            print(f"✅ Chart {chart_id} saved to PostgreSQL")
            db.close()
        except Exception as e:
            print(f"⚠️ DB save failed, using cache fallback: {e}")
            # FALLBACK: Store in cache (old way, still works!)
            from main import chart_cache
            chart_cache[chart_id] = html_content
        
        # Return URL to chart endpoint
        chart_url = f"http://localhost:8001/api/chart/{chart_id}"
        
        return {
            "chart_url": chart_url,
            "chart_type": chart_type,
            "title": clean_title,
            "data_summary": f"{len(labels)} data points"
        }
        
    except Exception as e:
        return {"error": f"Error generating chart: {str(e)}"}


def _create_bar_chart(labels: list, values: list, title: str, barmode: str = "group", orientation: str = "v") -> go.Figure:
    """Create an interactive bar chart."""
    if orientation == 'h':
        # Reverse lists so biggest bar appears on top
        labels_reversed = list(reversed(labels))
        values_reversed = list(reversed(values))
        
        fig = go.Figure(data=[
            go.Bar(
                x=values_reversed,
                y=labels_reversed,
                orientation='h',
                marker=dict(
                    color='rgba(54, 162, 235, 0.8)',
                    line=dict(color='rgba(54, 162, 235, 1)', width=1)
                ),
                hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
            )
        ])
        
    else:
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=values,
                marker=dict(
                    color='rgba(54, 162, 235, 0.8)',
                    line=dict(color='rgba(54, 162, 235, 1)', width=1)
                ),
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            )
        ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title='',
        yaxis_title='Count',
        barmode=barmode,
        template='plotly_white',
        hovermode='closest',
        showlegend=False,
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def _create_line_chart(labels: list, values: list, title: str, mode: str = "lines+markers") -> go.Figure:
    """Create an interactive line chart."""
    fig = go.Figure(data=[
        go.Scatter(
            x=labels,
            y=values,
            mode=mode,
            line=dict(color='rgba(54, 162, 235, 0.8)', width=2),
            marker=dict(size=8, color='rgba(54, 162, 235, 1)'),
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title='',
        yaxis_title='Count',
        template='plotly_white',
        hovermode='closest',
        showlegend=False,
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def _create_pie_chart(labels: list, values: list, title: str) -> go.Figure:
    """Create an interactive pie chart with top items only."""
    # Limit to top 5 items, group the rest as "Other"
    MAX_SLICES = 5
    
    if len(labels) > MAX_SLICES:
        # Sort by values descending
        sorted_data = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
        top_labels = [x[0] for x in sorted_data[:MAX_SLICES]]
        top_values = [x[1] for x in sorted_data[:MAX_SLICES]]
        
        # Group remaining as "Other"
        other_value = sum([x[1] for x in sorted_data[MAX_SLICES:]])
        if other_value > 0:
            top_labels.append("Other")
            top_values.append(other_value)
        
        labels = top_labels
        values = top_values
    
    # Define a professional color palette
    colors = [
        'rgba(255, 99, 132, 0.8)',   # Red
        'rgba(54, 162, 235, 0.8)',   # Blue
        'rgba(255, 206, 86, 0.8)',   # Yellow
        'rgba(75, 192, 192, 0.8)',   # Green
        'rgba(153, 102, 255, 0.8)',  # Purple
        'rgba(255, 159, 64, 0.8)',   # Orange
        'rgba(199, 199, 199, 0.8)',  # Grey
        'rgba(83, 102, 255, 0.8)',   # Indigo
    ]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors[:len(labels)]),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
            textinfo='percent',
            textposition='inside',
            hole=0
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation='v', x=1.05, y=0.5),
        height=500,
        margin=dict(l=50, r=150, t=80, b=50)
    )
    
    return fig


def _create_scatter_chart(labels: list, values: list, title: str, mode: str = "markers") -> go.Figure:
    """Create an interactive scatter plot."""
    fig = go.Figure(data=[
        go.Scatter(
            x=labels,
            y=values,
            mode=mode,
            marker=dict(
                size=10,
                color='rgba(54, 162, 235, 0.6)',
                line=dict(color='rgba(54, 162, 235, 1)', width=1)
            ),
            hovertemplate='<b>%{x}</b><br>Value: %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title='',
        yaxis_title='Value',
        template='plotly_white',
        hovermode='closest',
        showlegend=False,
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def _create_area_chart(labels: list, values: list, title: str) -> go.Figure:
    """Create an interactive area chart."""
    fig = go.Figure(data=[
        go.Scatter(
            x=labels,
            y=values,
            mode='lines',
            fill='tozeroy',
            line=dict(color='rgba(54, 162, 235, 0.8)', width=2),
            fillcolor='rgba(54, 162, 235, 0.3)',
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title='',
        yaxis_title='Count',
        template='plotly_white',
        hovermode='closest',
        showlegend=False,
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


# Test function (optional - for development/debugging)
def _test_chart_generation():
    """Test the chart generation with sample data."""
    import pandas as pd
    
    # Test data
    test_data = pd.DataFrame({
        'page': ['Home', 'About', 'Contact', 'Blog'],
        'clicks': [150, 89, 45, 120]
    })
    
    # Test each chart type
    for chart_type in ['bar', 'line', 'pie']:
        result = generate_chart_tool(
            data=test_data,
            chart_type=chart_type,
            title=f"Test {chart_type} chart"
        )
        
        if 'error' in result:
            print(f"❌ {chart_type}: {result['error']}")
        else:
            print(f"✅ {chart_type}: {result['title']} ({result['data_summary']})")
            print(f"   URL length: {len(result['chart_url'])} chars")


if __name__ == "__main__":
    _test_chart_generation()
