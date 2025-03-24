import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calendar

# Set page config
st.set_page_config(
    page_title="Bike Activity Dashboard",
    page_icon="🚲",
    layout="wide"
)

# Load the data
@st.cache_data
def load_data():
    with open('bike_activities_with_routes.json', 'r') as f:
        data = json.load(f)
    return data

def process_data(data):
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Parse dates
    df['start_datetime'] = pd.to_datetime(df['start_time'])
    df['end_datetime'] = pd.to_datetime(df['end_time'])
    df['duration_minutes'] = df['duration_millis'] / 60000  # Convert ms to minutes
    
    # Extract date components
    df['year'] = df['start_datetime'].dt.year
    df['month'] = df['start_datetime'].dt.month
    df['month_name'] = df['start_datetime'].dt.strftime('%b')
    df['day'] = df['start_datetime'].dt.day
    df['hour'] = df['start_datetime'].dt.hour
    df['day_of_week'] = df['start_datetime'].dt.dayofweek
    df['day_name'] = df['start_datetime'].dt.strftime('%a')
    
    # Create year-month field for aggregation
    df['year_month'] = df['start_datetime'].dt.strftime('%Y-%m')
    
    # Check if route data exists
    df['has_route'] = df['route'].apply(lambda x: len(x) > 0 if x else False)
    
    return df

def create_monthly_chart(df):
    # Create monthly aggregation
    monthly_counts = df.groupby('year_month').size().reset_index(name='count')
    monthly_counts['year_month_dt'] = pd.to_datetime(monthly_counts['year_month'] + '-01')
    monthly_counts = monthly_counts.sort_values('year_month_dt')
    
    # Create monthly chart
    fig = px.bar(
        monthly_counts, 
        x='year_month', 
        y='count',
        labels={'count': 'Number of Activities', 'year_month': 'Month'},
        title='Monthly Biking Activities'
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_hourly_chart(df):
    # Create hourly aggregation
    hourly_counts = df.groupby('hour').size().reset_index(name='count')
    
    # Create hourly chart
    fig = px.bar(
        hourly_counts, 
        x='hour', 
        y='count',
        labels={'count': 'Number of Activities', 'hour': 'Hour of Day'},
        title='Time of Day Analysis'
    )
    
    # Add custom x-axis labels for time periods
    time_labels = {
        0: '12 AM', 3: '3 AM', 6: '6 AM', 9: '9 AM', 
        12: '12 PM', 15: '3 PM', 18: '6 PM', 21: '9 PM'
    }
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(time_labels.keys()),
            ticktext=list(time_labels.values())
        )
    )
    return fig

def create_map_visualization(df):
    # Only consider activities with routes
    df_with_routes = df[df['has_route']]
    
    if len(df_with_routes) == 0:
        return None
    
    # Create the figure
    fig = go.Figure()
    
    # Add routes to the map
    for idx, row in df_with_routes.iterrows():
        route = row['route']
        if not route:
            continue
        
        # Extract lat/lon from the route
        lats = [point['latitude'] for point in route]
        lons = [point['longitude'] for point in route]
        
        # Add a line for each route
        fig.add_trace(go.Scattermapbox(
            mode='lines',
            lon=lons,
            lat=lats,
            name=f"Route {row['start_datetime'].strftime('%Y-%m-%d %H:%M')}",
            line=dict(width=2)
        ))
    
    # Set map center (average of all coordinates)
    all_lats = [point['latitude'] for route in df_with_routes['route'] for point in route if route]
    all_lons = [point['longitude'] for route in df_with_routes['route'] for point in route if route]
    
    if all_lats and all_lons:
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
    else:
        # Default center (fallback)
        center_lat, center_lon = 0, 0
    
    # Update the layout for the map
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600
    )
    
    return fig

def create_day_of_week_chart(df):
    # Create day of week aggregation
    day_counts = df.groupby(['day_of_week', 'day_name']).size().reset_index(name='count')
    day_counts = day_counts.sort_values('day_of_week')
    
    # Create day of week chart
    fig = px.bar(
        day_counts, 
        x='day_name', 
        y='count',
        labels={'count': 'Number of Activities', 'day_name': 'Day of Week'},
        title='Activities by Day of Week',
        category_orders={'day_name': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']}
    )
    return fig

def main():
    st.title("🚲 Bike Activity Dashboard")
    
    st.write("""
    This dashboard visualizes your biking activities extracted from Google Takeout data.
    Explore patterns in your biking habits, view routes on the map, and analyze when you're most active.
    """)
    
    try:
        data = load_data()
        df = process_data(data)
        
        # Display overview statistics
        st.header("Overview Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Activities", len(df))
        
        with col2:
            st.metric("Activities with Routes", df['has_route'].sum())
        
        with col3:
            start_date = df['start_datetime'].min().strftime('%Y-%m-%d')
            end_date = df['start_datetime'].max().strftime('%Y-%m-%d')
            st.metric("Date Range", f"{start_date} to {end_date}")
        
        with col4:
            total_minutes = df['duration_minutes'].sum()
            st.metric("Total Biking Time", f"{total_minutes:.0f} minutes")
        
        # Monthly activity chart
        st.header("Monthly Activity")
        monthly_fig = create_monthly_chart(df)
        st.plotly_chart(monthly_fig, use_container_width=True)
        
        # Create two columns for the next charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Time of day analysis
            st.header("Time of Day Analysis")
            hourly_fig = create_hourly_chart(df)
            st.plotly_chart(hourly_fig, use_container_width=True)
        
        with col2:
            # Day of week analysis
            st.header("Activities by Day of Week")
            day_fig = create_day_of_week_chart(df)
            st.plotly_chart(day_fig, use_container_width=True)
        
        # Map visualization
        st.header("Bike Routes Map")
        
        # Add a selector for activities with routes
        if df['has_route'].sum() > 0:
            map_fig = create_map_visualization(df)
            st.plotly_chart(map_fig, use_container_width=True)
            
            # Additional information about the map
            st.info(f"Displaying {df['has_route'].sum()} activities with GPS route data. Some activities may not have associated route information.")
        else:
            st.warning("No GPS route data available in the dataset.")
        
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        st.info("Make sure 'bike_activities_with_routes.json' is in the same directory as this script.")

if __name__ == "__main__":
    main()

