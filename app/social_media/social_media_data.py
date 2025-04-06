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
    
    # Create columns for the stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Posts", stats["total"])
    
    with col2:
        positive_pct = f"{(stats['positive'] / stats['total'] * 100):.1f}%" if stats["total"] > 0 else "0%"
        st.metric("Positive", stats["positive"], positive_pct)
    
    with col3:
        neutral_pct = f"{(stats['neutral'] / stats['total'] * 100):.1f}%" if stats["total"] > 0 else "0%"
        st.metric("Neutral", stats["neutral"], neutral_pct)
    
    with col4:
        negative_pct = f"{(stats['negative'] / stats['total'] * 100):.1f}%" if stats["total"] > 0 else "0%"
        st.metric("Negative", stats["negative"], negative_pct)
    
    # Create pie chart for sentiment distribution
    if stats["total"] > 0:
        fig, ax = plt.subplots(figsize=(5, 5))
        sentiment_labels = ['Positive', 'Neutral', 'Negative']
        sentiment_values = [stats['positive'], stats['neutral'], stats['negative']]
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
    
    if not platform_stats:
        st.info("No platform data available for comparison.")
        return
    
    # Create DataFrame for visualization
    platforms = []
    sentiments = []
    counts = []
    
    for platform in platform_stats:
        # Skip platforms with no data
        if platform["total"] == 0:
            continue
            
        # Positive sentiment
        platforms.append(platform["platform"])
        sentiments.append("Positive")
        counts.append(platform["positive"])
        
        # Neutral sentiment
        platforms.append(platform["platform"])
        sentiments.append("Neutral")
        counts.append(platform["neutral"])
        
        # Negative sentiment
        platforms.append(platform["platform"])
        sentiments.append("Negative")
        counts.append(platform["negative"])
    
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
        total = platform["total"]
        if total > 0:
            table_data.append({
                "Platform": platform["platform"].title(),
                "Total Posts": total,
                "Positive": f"{platform['positive']} ({platform['positive']/total*100:.1f}%)",
                "Neutral": f"{platform['neutral']} ({platform['neutral']/total*100:.1f}%)",
                "Negative": f"{platform['negative']} ({platform['negative']/total*100:.1f}%)"
            })
    
    # Display table
    st.table(pd.DataFrame(table_data))

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
    
    # Add sentiment emoji
    def get_sentiment_emoji(sentiment):
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
            # Post content with sentiment badge
            sentiment = post.get('sentiment')
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