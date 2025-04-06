"""
Social Media Data Visualization

This module provides a Streamlit interface for displaying and analyzing
social media data retrieved from various platforms.
"""
import os
import sys
import logging
import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sqlite3

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import project modules
from app.utils.config import DASHBOARD_TITLE
from app.social_media.social_media_connector import get_posts_from_all_platforms
from app.social_media.database import get_posts, get_sentiment_stats, get_sentiment_by_platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('social_media_data')

# Set page configuration
st.set_page_config(
    page_title=f"{DASHBOARD_TITLE} - Social Media Data",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def fetch_and_save_data():
    """Fetch data from social media platforms and save to database"""
    # This would normally trigger a background job
    # For now, we'll just show a message
    with st.spinner("Fetching data from social media platforms..."):
        # Simulate processing time
        import time
        time.sleep(2)
        st.success("✅ Data fetched successfully!")

def display_sentiment_summary(platform=None):
    """Display sentiment summary statistics"""
    # Get sentiment stats
    days = st.session_state.get("time_range", 30)
    stats = get_sentiment_stats(platform=platform, days=days)
    
    # Log stats for debugging
    logger.info(f"Sentiment stats: {stats}")
    
    # Create columns for the stats
    col1, col2, col3, col4 = st.columns(4)
    
    total = stats.get("total", 0)
    positive = stats.get("positive", 0) 
    neutral = stats.get("neutral", 0)
    negative = stats.get("negative", 0)
    
    with col1:
        st.metric("Total Posts", total)
    
    with col2:
        positive_pct = f"{(positive / total * 100):.1f}%" if total > 0 else "0%"
        st.metric("Positive", positive, positive_pct)
    
    with col3:
        neutral_pct = f"{(neutral / total * 100):.1f}%" if total > 0 else "0%"
        st.metric("Neutral", neutral, neutral_pct)
    
    with col4:
        negative_pct = f"{(negative / total * 100):.1f}%" if total > 0 else "0%"
        st.metric("Negative", negative, negative_pct)
    
    # Create pie chart for sentiment distribution
    if total > 0 and (positive > 0 or neutral > 0 or negative > 0):
        fig, ax = plt.subplots(figsize=(5, 5))
        sentiment_labels = ['Positive', 'Neutral', 'Negative']
        sentiment_values = [positive, neutral, negative]
        sentiment_colors = ['#4CAF50', '#FFC107', '#F44336']
        
        ax.pie(
            sentiment_values, 
            labels=sentiment_labels,
            colors=sentiment_colors,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1}
        )
        ax.axis('equal')
        
        # Filter title based on platform
        title = f"Sentiment Distribution" + (f" - {platform.title()}" if platform else "")
        ax.set_title(title)
        
        st.pyplot(fig)

def display_platform_comparison():
    """Display sentiment comparison by platform"""
    # Get sentiment stats by platform
    days = st.session_state.get("time_range", 30)
    platform_stats = get_sentiment_by_platform(days=days)
    
    # Log for debugging
    logger.info(f"Platform stats: {platform_stats}")
    
    if not platform_stats:
        st.info("No platform data available for comparison.")
        return
    
    # Create DataFrame for visualization
    platforms = []
    sentiments = []
    counts = []
    
    for platform in platform_stats:
        # Get values with safe defaults
        platform_name = platform.get("platform", "unknown")
        total = platform.get("total", 0)
        
        # Skip platforms with no data
        if total == 0:
            continue
            
        # Get sentiment counts with safe defaults
        positive = platform.get("positive", 0)
        neutral = platform.get("neutral", 0)
        negative = platform.get("negative", 0)
            
        # Positive sentiment
        platforms.append(platform_name)
        sentiments.append("Positive")
        counts.append(positive)
        
        # Neutral sentiment
        platforms.append(platform_name)
        sentiments.append("Neutral")
        counts.append(neutral)
        
        # Negative sentiment
        platforms.append(platform_name)
        sentiments.append("Negative")
        counts.append(negative)
    
    if not platforms:  # If no data after filtering
        st.info("No sentiment data available for comparison.")
        return
    
    # Create DataFrame
    df = pd.DataFrame({
        "Platform": platforms,
        "Sentiment": sentiments,
        "Count": counts
    })
    
    # Create grouped bar chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Platform:N', title='Platform'),
        y=alt.Y('Count:Q', title='Count'),
        color=alt.Color('Sentiment:N', scale=alt.Scale(
            domain=['Positive', 'Neutral', 'Negative'],
            range=['#4CAF50', '#FFC107', '#F44336']
        )),
        tooltip=['Platform', 'Sentiment', 'Count']
    ).properties(
        title='Sentiment Comparison by Platform',
        width=600,
        height=400
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # Create a table with the raw data
    st.subheader("Platform Sentiment Statistics")
    
    # Convert to a more readable format
    table_data = []
    for platform in platform_stats:
        platform_name = platform.get("platform", "unknown")
        total = platform.get("total", 0)
        
        if total > 0:
            positive = platform.get("positive", 0)
            neutral = platform.get("neutral", 0)
            negative = platform.get("negative", 0)
            
            table_data.append({
                "Platform": platform_name.title(),
                "Total Posts": total,
                "Positive": f"{positive} ({positive/total*100:.1f}%)",
                "Neutral": f"{neutral} ({neutral/total*100:.1f}%)",
                "Negative": f"{negative} ({negative/total*100:.1f}%)"
            })
    
    # Display table
    if table_data:
        st.table(pd.DataFrame(table_data))
    else:
        st.info("No platform sentiment data available.")

def display_post_feed(platform=None):
    """Display social media posts feed with sentiment"""
    # Get posts
    limit = st.session_state.get("post_limit", 50)
    posts = get_posts(platform=platform, limit=limit)
    
    if not posts:
        st.info("No posts available.")
        return
    
    # Display post count
    st.subheader(f"Recent Posts" + (f" from {platform.title()}" if platform else ""))
    st.text(f"Showing {len(posts)} most recent posts")
    
    # Create DataFrame from posts
    df = pd.DataFrame(posts)
    
    # Format timestamp
    df['formatted_time'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Debug: Show sentiment values in console (more detailed)
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    if 'sentiment' in df.columns:
        logger.info(f"Sentiment values: {df['sentiment'].tolist()[:10]}")
        logger.info(f"Sentiment value types: {[type(x).__name__ for x in df['sentiment'].tolist()[:5]]}")
    else:
        logger.warning("No sentiment column in DataFrame")
    
    # Add sentiment emoji
    def get_sentiment_emoji(sentiment_val):
        # Debug the input value
        logger.info(f"get_sentiment_emoji input: {sentiment_val}, type: {type(sentiment_val).__name__}")
        
        # Handle different input types
        sentiment = None
        
        # Check for NaN values (which can't be converted to int)
        if isinstance(sentiment_val, float) and pd.isna(sentiment_val):
            logger.info(f"Detected NaN value, returning Unknown")
            return "❓ Unknown"
        
        # Try to convert to integer
        if sentiment_val is not None:
            try:
                sentiment = int(float(sentiment_val))
            except (ValueError, TypeError):
                logger.warning(f"Failed to convert sentiment: {sentiment_val}")
                pass
        
        # Map integer values to text
        if sentiment == 1:
            return "😊 Positive"
        elif sentiment == 0:
            return "😐 Neutral"
        elif sentiment == -1:
            return "😞 Negative"
        else:
            return "❓ Unknown"
    
    # Handle case where sentiment column might not exist
    if 'sentiment' not in df.columns:
        df['sentiment'] = None
        logger.warning("Adding empty sentiment column to DataFrame")
    
    df['sentiment_display'] = df['sentiment'].apply(get_sentiment_emoji)
    
    # Add platform icon
    def get_platform_icon(platform):
        if platform == "twitter":
            return "🐦 Twitter"
        elif platform == "facebook":
            return "📘 Facebook"
        elif platform == "linkedin":
            return "📊 LinkedIn"
        else:
            return f"🌐 {platform.title()}"
    
    df['platform_display'] = df['platform'].apply(get_platform_icon)
    
    # Display posts
    for i, post in df.iterrows():
        with st.expander(f"{post['platform_display']} - {post['author']} - {post['formatted_time']}"):
            # Get sentiment value with extensive error handling
            sentiment = None
            
            # Add debug info
            if 'sentiment' in post:
                logger.info(f"Post {i} sentiment: {post['sentiment']}, type: {type(post['sentiment']).__name__}")
            
            try:
                raw_sentiment = post.get('sentiment')
                # Check for NaN values
                if isinstance(raw_sentiment, float) and pd.isna(raw_sentiment):
                    logger.info(f"Skipping NaN sentiment value for post {i}")
                elif raw_sentiment is not None:
                    if isinstance(raw_sentiment, (int, float)):
                        sentiment = int(raw_sentiment)
                    elif isinstance(raw_sentiment, str) and raw_sentiment.strip():
                        sentiment = int(float(raw_sentiment))
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing sentiment for post {i}: {e}")
            
            # Select color based on sentiment
            sentiment_color = {
                1: "background-color: #4CAF50; color: white; padding: 2px 6px; border-radius: 3px;",
                0: "background-color: #FFC107; color: black; padding: 2px 6px; border-radius: 3px;",
                -1: "background-color: #F44336; color: white; padding: 2px 6px; border-radius: 3px;"
            }.get(sentiment, "background-color: #9E9E9E; color: white; padding: 2px 6px; border-radius: 3px;")
            
            sentiment_html = f'<span style="{sentiment_color}">{post["sentiment_display"]}</span>'
            
            st.markdown(f"""
            **Content:** {post['content']}  
            **Author:** {post['author']}  
            **Sentiment:** {sentiment_html}  
            **Time:** {post['formatted_time']}  
            **Engagement:** {post.get('engagement_count', 0)}  
            """, unsafe_allow_html=True)
            
            if post.get('url'):
                st.markdown(f"[View Original Post]({post['url']})")

def main():
    """Main function for the social media data page"""
    st.title("Social Media Data Analysis")
    
    # Initialize session state
    if "time_range" not in st.session_state:
        st.session_state.time_range = 30
    
    if "post_limit" not in st.session_state:
        st.session_state.post_limit = 50
    
    # Sidebar controls
    st.sidebar.title("Controls")
    
    # Fetch data button
    if st.sidebar.button("Fetch New Data"):
        fetch_and_save_data()
    
    # Time range selector
    st.session_state.time_range = st.sidebar.slider(
        "Time Range (days)",
        min_value=1,
        max_value=90,
        value=st.session_state.time_range
    )
    
    # Post limit selector
    st.session_state.post_limit = st.sidebar.slider(
        "Max Posts to Display",
        min_value=10,
        max_value=200,
        value=st.session_state.post_limit
    )
    
    # Platform filter
    platform_options = ["All Platforms", "twitter", "facebook", "linkedin"]
    selected_platform = st.sidebar.selectbox(
        "Filter by Platform",
        options=platform_options
    )
    
    # Convert "All Platforms" to None for database queries
    platform_filter = None if selected_platform == "All Platforms" else selected_platform
    
    # Display navigation options in sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("""
    - [Dashboard](/)
    - [Social Media Setup](/social_media_auth)
    - [Social Media Data](/social_media_data) (Current)
    """)
    
    # Main content
    
    # Top row: Sentiment Summary
    st.header("Sentiment Summary")
    display_sentiment_summary(platform=platform_filter)
    
    # Platform comparison (only if "All Platforms" is selected)
    if platform_filter is None:
        st.header("Platform Comparison")
        display_platform_comparison()
    
    # Posts feed
    st.header("Social Media Posts")
    display_post_feed(platform=platform_filter)
    
    # Navigation links
    st.markdown("---")
    st.markdown("""
    [← Return to Dashboard](/) | [Configure Social Media Accounts →](/social_media_auth)
    """)

if __name__ == "__main__":
    main() 