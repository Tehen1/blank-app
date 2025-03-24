#!/usr/bin/env python3
"""
Cycling Activity Analysis Dashboard

This is the main Streamlit application that provides a dashboard for analyzing cycling activities.
It uses the data_parser, data_processor, and visualization modules to parse, process, and visualize 
cycling activity data from TCX files.
"""

import os
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
from pathlib import Path

# Import our custom modules
from data_parser import parse_tcx, get_activity_files
from data_processor import process_cycling_data, build_activities_dataframe
from visualization import (
    display_activity_map, 
    display_elevation_profile,
    display_speed_chart, 
    display_summary_cards,
    export_activity_data,
    display_monthly_trend,
    compare_activities
)

# Set page configuration
st.set_page_config(
    page_title="Cycling Activity Analysis",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data loading to improve performance
@st.cache_data(ttl=3600)
def load_activity_data(data_dir):
    """
    Load and cache cycling activity data.
    
    Args:
        data_dir: Directory containing TCX files
        
    Returns:
        Tuple of (activities_df, flattened_df)
    """
    return process_cycling_data(data_dir)

def format_activity_name(row):
    """
    Format activity name for display in the selector.
    
    Args:
        row: DataFrame row containing activity data
        
    Returns:
        Formatted activity name string
    """
    date_str = row['date'].strftime('%Y-%m-%d')
    distance = f"{row['distance']:.1f}" if 'distance' in row and row['distance'] else "?"
    elevation = f"{row['elevation_gain']:.0f}" if 'elevation_gain' in row and row['elevation_gain'] else "?"
    
    return f"{date_str}: {distance} km, {elevation}m elevation"

def filter_activities_by_date(df, start_date, end_date):
    """
    Filter activities dataframe by date range.
    
    Args:
        df: Activities DataFrame
        start_date: Start date for filtering
        end_date: End date for filtering
        
    Returns:
        Filtered DataFrame
    """
    if df is None or df.empty or 'date' not in df.columns:
        return df
        
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
        
    # Filter by date range
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    return df[mask].reset_index(drop=True)

def main():
    """Main application function."""
    
    # Header
    st.title("🚴 Cycling Activity Analysis")
    
    # Sidebar
    st.sidebar.header("Controls")
    
    # Set up data directory path
    default_data_dir = "./Takeout 2/Takeout/Fit/Activités"
    data_dir = st.sidebar.text_input("Data Directory", value=default_data_dir)
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        st.sidebar.warning(f"Directory not found: {data_dir}")
        st.warning("Please specify a valid data directory in the sidebar.")
        return
    
    # Load data
    with st.spinner("Loading cycling activity data..."):
        try:
            activities_df, flattened_df = load_activity_data(data_dir)
            
            if activities_df.empty:
                st.warning("No cycling activities found in the specified directory.")
                return
                
            st.sidebar.success(f"Loaded {len(activities_df)} cycling activities")
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return
    
    # Date range filter
    if 'date' in activities_df.columns:
        # Get min and max dates from data
        min_date = activities_df['date'].min()
        max_date = activities_df['date'].max()
        
        # Default to last 3 months
        default_start = max_date - timedelta(days=90)
        
        # Date picker widgets
        start_date = st.sidebar.date_input(
            "Start Date", 
            value=default_start,
            min_value=min_date,
            max_value=max_date
        )
        
        end_date = st.sidebar.date_input(
            "End Date", 
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
        
        # Ensure end_date is not before start_date
        if end_date < start_date:
            st.sidebar.error("End date cannot be before start date")
            end_date = start_date
        
        # Filter activities by date
        filtered_df = filter_activities_by_date(
            activities_df, 
            pd.Timestamp(start_date), 
            pd.Timestamp(end_date)
        )
        
        st.sidebar.info(f"Showing {len(filtered_df)} activities")
    else:
        filtered_df = activities_df
    
    # Activity selector
    activity_options = {}
    for _, row in filtered_df.iterrows():
        activity_options[format_activity_name(row)] = row.name
    
    selected_activity_name = st.sidebar.selectbox(
        "Select Activity",
        options=list(activity_options.keys())
    )
    
    if selected_activity_name:
        selected_idx = activity_options[selected_activity_name]
        selected_activity = filtered_df.loc[selected_idx]
    else:
        # Default to the most recent activity
        selected_activity = filtered_df.iloc[0] if not filtered_df.empty else None
    
    # Comparison mode
    enable_comparison = st.sidebar.checkbox("Enable Activity Comparison", value=False)
    
    if enable_comparison:
        # Multi-select for activities to compare
        comparison_options = [format_activity_name(row) for _, row in filtered_df.iterrows()]
        comparison_selected = st.sidebar.multiselect(
            "Compare Activities",
            options=comparison_options,
            default=[selected_activity_name] if selected_activity_name in comparison_options else []
        )
        
        # Get indices of selected activities
        comparison_indices = [activity_options[name] for name in comparison_selected if name in activity_options]
    
    # Export option
    st.sidebar.subheader("Export Data")
    if st.sidebar.button("Export to CSV"):
        export_activity_data(filtered_df, "cycling_activities.csv")
    
    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(["Summary", "Maps", "Trends"])
    
    with tab1:
        st.header("Activity Summary")
        
        if selected_activity is not None:
            # Prepare summary data
            summary_data = {
                'date': selected_activity['date'],
                'distance': selected_activity['distance'] / 1000,  # Convert to km
                'duration': selected_activity['duration'],
                'elevation_gain': selected_activity['elevation_gain'],
                'avg_speed': selected_activity.get('average_speed', 0) * 3.6  # Convert m/s to km/h
            }
            
            # Add max speed if available
            if 'max_speed' in selected_activity:
                summary_data['max_speed'] = selected_activity['max_speed'] * 3.6  # Convert m/s to km/h
            
            # Display summary cards
            display_summary_cards(summary_data)
            
            # Show comparison if enabled
            if enable_comparison and len(comparison_indices) >= 2:
                st.subheader("Activity Comparison")
                comparison_data = filtered_df.loc[comparison_indices]
                compare_activities(comparison_data, comparison_data['date'].tolist())
        else:
            st.info("Select an activity to view summary")
    
    with tab2:
        st.header("Activity Maps")
        
        if selected_activity is not None and 'coordinates' in selected_activity:
            # Display activity map
            coordinates = selected_activity['coordinates']
            speed_data = selected_activity.get('speed_data', None)
            display_activity_map(coordinates, speed_data)
            
            # Display elevation profile if data available
            if 'elevation_data' in selected_activity and len(selected_activity['elevation_data']) > 0:
                # Create distance data (cumulative)
                if 'distance' in selected_activity and selected_activity['distance'] > 0:
                    # Create evenly spaced distance points
                    distance_data = [
                        i * (selected_activity['distance'] / 1000) / len(selected_activity['elevation_data'])
                        for i in range(len(selected_activity['elevation_data']))
                    ]
                    display_elevation_profile(distance_data, selected_activity['elevation_data'])
            
            # Display speed chart if data available
            if 'speed_data' in selected_activity and len(selected_activity['speed_data']) > 0:
                time_data = [item.get('timestamp') for item in selected_activity['speed_data']]
                speed_values = [item.get('speed') * 3.6 for item in selected_activity['speed_data']]  # Convert to km/h
                
                # Create distance data if possible
                distance_data = None
                if 'distance' in selected_activity and selected_activity['distance'] > 0:
                    # Create cumulative distance based on speed and time intervals
                    distance_data = []
                    total = 0
                    for i, item in enumerate(selected_activity['speed_data']):
                        if i > 0 and time_data[i] and time_data[i-1]:
                            # Calculate time difference in seconds
                            time_diff = (time_data[i] - time_data[i-1]).total_seconds()
                            # Calculate distance increment in km
                            increment = (item.get('speed', 0) * time_diff) / 1000
                            total += increment
                        distance_data.append(total)
                
                display_speed_chart(time_data, speed_values, distance_data)
        else:
            st.info("Select an activity with GPS data to view maps")
    
    with tab3:
        st.header("Cycling Trends")
        
        # Display monthly distance trend
        if not filtered_df.empty:
            display_monthly_trend(filtered_df)
            
            # Summary statistics
            st.subheader("Statistics")
            
            # Create a copy with unit conversions
            stats_df = filtered_df.copy()
            if 'distance' in stats_df.columns:
                stats_df['distance_km'] = stats_df['distance'] / 1000  # Convert to km
            
            if 'duration' in stats_df.columns:
                stats_df['duration_hours'] = stats_df['duration'] / 3600  # Convert to hours
            
            # Calculate summary statistics
            total_distance = stats_df['distance_km'].sum() if 'distance_km' in stats_df.columns else 0
            total_activities = len(stats_df)
            total_time = stats_df['duration_hours'].sum() if 'duration_hours' in stats_df.columns else 0
            avg_distance = stats_df['distance_km'].mean() if 'distance_km' in stats_df.columns else 0
            
            # Display statistics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Activities", f"{total_activities}")
            with col2:
                st.metric("Total Distance", f"{total_distance:.1f} km")
            with col3:
                st.metric("Total Time", f"{total_time:.1f} hours")
            with col4:
                st.metric("Avg Distance", f"{avg_distance:.1f} km")
            
            # Show activity table
            st.subheader("Activity List")
            
            # Create a display dataframe with formatted columns
            display_df = filtered_df[['date', 'distance', 'duration', 'elevation_gain']].copy()
            
            # Format columns
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df['distance'] = display_df['distance'].apply(lambda x: f"{x/1000:.1f} km")
            display_df['duration'] = display_df['duration'].apply(
                lambda x: f"{int(x//3600)}:{int((x%3600)//60):02d}:{int(x%60):02d}"
            )
            display_df['elevation_gain'] = display_df['elevation_gain'].apply(lambda x: f"{x:.0f} m")
            
            # Rename columns for display
            display_df.columns = ['Date', 'Distance', 'Duration', 'Elevation Gain']
            
            # Show the table
            st.dataframe(display_df, use_container_width=True)
            
            # Export option
            if st.button("Export Activity List"):
                export_activity_data(display_df, "cycling_activities_list.csv")
        else:
            st.info("No data available for trend analysis")

if __name__ == "__main__":
    main()

"""
Cycling Activity Analysis Dashboard

A Streamlit application to analyze and visualize cycling activity data from TCX files.
Features:
- Activity summary metrics
- Interactive maps with route visualization
- Elevation profiles and speed charts
- Monthly trends and activity comparison
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
from streamlit_folium import folium_static
import time

# Import custom modules
from data_parser import parse_tcx, get_activity_files
from data_processor import process_cycling_data, build_activities_dataframe
from visualization import (
    display_activity_map, 
    display_elevation_profile, 
    display_speed_chart, 
    display_summary_cards,
    export_activity_data,
    display_monthly_trend,
    compare_activities
)

# Set page configuration
st.set_page_config(
    page_title="Cycling Activity Analysis",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data loading to improve performance
@st.cache_data(ttl=3600)
def load_data(data_dir):
    """
    Load and cache cycling activity data
    
    Parameters:
    -----------
    data_dir : str
        Path to directory containing TCX files
    
    Returns:
    --------
    tuple
        (activities_df, flattened_df) containing all processed cycling data
    """
    try:
        start_time = time.time()
        st.info(f"Loading data from {data_dir}...")
        
        # Process cycling data
        activities_df, flattened_df = process_cycling_data(data_dir)
        
        # Calculate processing time
        elapsed_time = time.time() - start_time
        st.success(f"Loaded {len(activities_df)} activities in {elapsed_time:.2f} seconds")
        
        return activities_df, flattened_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def filter_activities(df, date_range, activity_selection=None):
    """
    Filter activities based on date range and activity selection
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing activity data
    date_range : tuple
        (start_date, end_date) tuple for filtering
    activity_selection : list, optional
        List of specific activity dates to include
    
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    # Filter by date range if provided
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        df_filtered = df[mask]
    else:
        df_filtered = df.copy()
    
    # Filter by specific activities if provided
    if activity_selection and len(activity_selection) > 0:
        df_filtered = df_filtered[df_filtered['date'].isin(activity_selection)]
    
    return df_filtered

def format_activity_options(df):
    """
    Format activity dates for the selection dropdown
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing activity data with 'date' and 'distance' columns
    
    Returns:
    --------
    list
        List of formatted date strings with distance information
    """
    if df.empty or 'date' not in df.columns or 'distance' not in df.columns:
        return []
    
    options = []
    for _, row in df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        distance = row['distance']
        options.append(f"{date_str} ({distance:.1f} km)")
    
    return options

def display_summary_tab(df):
    """
    Display the Summary tab content
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing activity data
    """
    st.header("Activity Summary")
    
    if df.empty:
        st.warning("No activities found in the selected date range.")
        return
    
    # Display overall statistics
    st.subheader("Overall Statistics")
    
    # Calculate summary statistics
    total_distance = df['distance'].sum()
    total_activities = len(df)
    total_duration = df['duration'].sum()
    avg_speed = df['average_speed'].mean() if 'average_speed' in df.columns else 0
    
    # Create summary columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Activities", f"{total_activities}")
    with col2:
        st.metric("Total Distance", f"{total_distance:.1f} km")
    with col3:
        # Format duration from seconds to hours:minutes
        hours, remainder = divmod(total_duration, 3600)
        minutes, _ = divmod(remainder, 60)
        st.metric("Total Duration", f"{int(hours)}h {int(minutes)}m")
    with col4:
        st.metric("Avg Speed", f"{avg_speed:.1f} km/h")
    
    # Show recent activities table
    st.subheader("Recent Activities")
    recent_activities = df.sort_values('date', ascending=False).head(10)
    
    # Format the table
    table_data = recent_activities[['date', 'distance', 'duration', 'elevation_gain', 'average_speed']].copy()
    
    # Convert duration to readable format
    if 'duration' in table_data.columns:
        table_data['duration'] = table_data['duration'].apply(
            lambda x: f"{int(x//3600)}h {int((x%3600)//60)}m" if pd.notnull(x) else "N/A"
        )
    
    # Format columns
    formatted_table = table_data.copy()
    if 'distance' in formatted_table.columns:
        formatted_table['distance'] = formatted_table['distance'].apply(lambda x: f"{x:.1f} km")
    if 'elevation_gain' in formatted_table.columns:
        formatted_table['elevation_gain'] = formatted_table['elevation_gain'].apply(lambda x: f"{x:.0f} m")
    if 'average_speed' in formatted_table.columns:
        formatted_table['average_speed'] = formatted_table['average_speed'].apply(lambda x: f"{x:.1f} km/h")
    
    # Format date as string if it's datetime
    if 'date' in formatted_table.columns and pd.api.types.is_datetime64_any_dtype(formatted_table['date']):
        formatted_table['date'] = formatted_table['date'].dt.strftime('%Y-%m-%d')
    
    # Rename columns
    formatted_table.columns = ['Date', 'Distance', 'Duration', 'Elevation Gain', 'Avg Speed']
    
    # Display table
    st.dataframe(formatted_table, use_container_width=True)
    
    # Add export button
    export_activity_data(table_data, "cycling_activities.csv")
    
    # Display monthly trend chart
    st.subheader("Monthly Distance Trend")
    display_monthly_trend(df)

def display_maps_tab(df, selected_activity=None):
    """
    Display the Maps tab content
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing activity data
    selected_activity : datetime, optional
        Specific activity date to display
    """
    st.header("Activity Maps")
    
    if df.empty:
        st.warning("No activities found in the selected date range.")
        return
    
    # If no specific activity is selected, use the most recent one
    if selected_activity is None:
        selected_activity = df['date'].max()
    
    # Filter to selected activity
    activity_data = df[df['date'] == selected_activity].iloc[0] if not df[df['date'] == selected_activity].empty else None
    
    if activity_data is None:
        st.warning("Selected activity not found.")
        return
    
    # Display activity summary cards
    activity_summary = {
        'date': activity_data['date'],
        'distance': activity_data['distance'],
        'duration': activity_data['duration'],
        'elevation_gain': activity_data['elevation_gain'],
        'avg_speed': activity_data['average_speed'] if 'average_speed' in activity_data else 0,
        'max_speed': activity_data['max_speed'] if 'max_speed' in activity_data else 0
    }
    
    display_summary_cards(activity_summary)
    
    # Create columns for map and elevation profile
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Display activity map
        coordinates = activity_data['coordinates']
        speed_data = activity_data['speed_data'] if len(activity_data['speed_data']) == len(coordinates) else None
        display_activity_map(coordinates, speed_data)
    
    with col2:
        # Display elevation profile
        if 'elevation_data' in activity_data and len(activity_data['elevation_data']) > 0:
            # Create distance data array based on array length
            if 'distance' in activity_data and activity_data['distance'] > 0:
                total_distance = activity_data['distance']
                num_points = len(activity_data['elevation_data'])
                distance_data = [i * total_distance / (num_points - 1) for i in range(num_points)]
                
                display_elevation_profile(distance_data, activity_data['elevation_data'])
            else:
                st.warning("Distance data not available for elevation profile.")
        else:
            st.warning("Elevation data not available for this activity.")
    
    # Display speed chart
    if 'speed_data' in activity_data and len(activity_data['speed_data']) > 0:
        # Create time array based on duration
        if 'duration' in activity_data and activity_data['duration'] > 0:
            duration = activity_data['duration']
            num_points = len(activity_data['speed_data'])
            time_data = [i * duration / (num_points - 1) for i in range(num_points)]
            
            # Convert to timestamps for better display
            start_time = activity_data['date']
            if isinstance(start_time, datetime):
                time_data = [start_time + timedelta(seconds=t) for t in time_data]
            
            display_speed_chart(time_data, activity_data['speed_data'])
        else:
            st.warning("Duration data not available for speed chart.")
    else:
        st.warning("Speed data not available for this activity.")

def display_trends_tab(df):
    """
    Display the Trends tab content
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing activity data
    """
    st.header("Activity Trends and Comparison")
    
    if df.empty:
        st.warning("No activities found in the selected date range.")
        return
    
    # Monthly trend chart
    st.subheader("Monthly Distance and Activities")
    
    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Extract month and year
    df['month_year'] = df['date'].dt.strftime('%Y-%m')
    
    # Group by month and calculate metrics
    monthly_metrics = df.groupby('month_year').agg({
        'distance': 'sum',
        'date': 'count',
        'duration': 'sum',
        'elevation_gain': 'sum'
    }).reset_index()
    
    monthly_metrics.columns = ['Month', 'Distance (km)', 'Activities', 'Duration (s)', 'Elevation (m)']
    
    # Convert duration to hours
    monthly_metrics['Duration (h)'] = monthly_metrics['Duration (s)'] / 3600
    
    # Create the chart
    fig = px.bar(
        monthly_metrics,
        x='Month',
        y=['Distance (km)', 'Activities'],
        barmode='group',
        labels={'value': 'Value', 'Month': 'Month', 'variable': 'Metric'},
        title='Monthly Cycling Metrics'
    )
    
    # Customize the layout
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Value',
        xaxis={'categoryorder':'category ascending'},
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Activity comparison
    st.subheader("Activity Comparison")
    st.write("Select activities to compare:")
    
    # Format activity options for selection
    activity_options = df['date'].dt.strftime('%Y-%m-%d').tolist()
    
    # Select activities to compare
    selected_activities = st.multiselect(
        "Choose activities to compare:",
        options=activity_options,
        default=activity_options[:2] if len(activity_options) >= 2 else activity_options
    )
    
    # Convert selected activities back to datetime
    selected_dates = [datetime.strptime(date, '%Y-%m-%d') for date in selected_activities]
    
    # Display comparison if activities are selected
    if len(selected_dates) >= 2:
        compare_activities(df, selected_dates)
    else:
        st.info("Select at least 2 activities for comparison.")

def main():
    """
    Main function to run the Streamlit app
    """
    # Set app title and header
    st.title("🚴 Cycling Activity Analysis Dashboard")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Data directory input
    data_dir = st.sidebar.text_input(
        "Data Directory",
        value="./Takeout 2/Takeout/Fit/Activités",
        help="Path to directory containing TCX files"
    )
    
    # Load data
    activities_df, flattened_df = load_data(data_dir)
    
    if activities_df.empty:
        st.warning(f"No cycling activities found in {data_dir}. Please check the directory path.")
        return
    
    # Date range filter
    min_date = activities_df['date'].min().date()
    max_date = activities_df['date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Activity selection
    st.sidebar.subheader("Activity Selection")
    
    # Format activity options (date + distance)
    formatted_options = {}
    for _, row in activities_df.iterrows():
        date_str = row['date'].str

import streamlit as st
import pandas as pd
import datetime
import os
from data_parser import parse_tcx
from data_processor import load_activities, process_activities
from visualization import (
    display_summary_stats, 
    show_activity_map,
    plot_elevation_profile,
    plot_speed_chart,
    display_monthly_mileage,
    export_activity_data
)

# Set page configuration
st.set_page_config(
    page_title="Cycling Activity Analysis",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def get_cached_activities():
    """Cache the activities data for better performance"""
    data_path = "./Takeout 2/Takeout/Fit/Activités"
    tcx_files = load_activities(data_path, activity_type="Biking")
    return process_activities(tcx_files)

def main():
    """Main function to run the Streamlit app"""
    st.title("🚴 Cycling Activity Analysis")
    
    # Load data
    try:
        df = get_cached_activities()
        if df.empty:
            st.warning("No cycling activities found. Please check the data directory.")
            return
    except Exception as e:
        st.error(f"Error loading activities: {str(e)}")
        return
    
    # Sidebar Controls
    st.sidebar.header("Filters")
    
    # Date range selector
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Convert date inputs to datetime for filtering
    start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
    end_datetime = datetime.datetime.combine(end_date, datetime.time.max)
    
    # Filter data based on date range
    filtered_df = df[(df['date'] >= start_datetime) & (df['date'] <= end_datetime)]
    
    # Activity selector
    if not filtered_df.empty:
        activity_options = filtered_df['date'].dt.strftime('%Y-%m-%d %H:%M - %A').tolist()
        activity_indices = list(range(len(activity_options)))
        activity_map = dict(zip(activity_options, activity_indices))
        
        selected_activity_names = st.sidebar.multiselect(
            "Choose Activities",
            options=activity_options,
            default=[activity_options[0]] if activity_options else []
        )
        
        selected_indices = [activity_map[name] for name in selected_activity_names]
        selected_activities = filtered_df.iloc[selected_indices] if selected_indices else filtered_df
    else:
        st.warning("No activities found in the selected date range.")
        selected_activities = filtered_df
    
    # Sidebar download section
    st.sidebar.header("Export Data")
    if st.sidebar.button("Export to CSV"):
        csv_data = export_activity_data(selected_activities)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"cycling_activities_{start_date}_{end_date}.csv",
            mime="text/csv"
        )
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Summary", "Maps", "Trends"])
    
    with tab1:
        st.header("Activity Summary")
        
        if not filtered_df.empty:
            display_summary_stats(filtered_df)
            
            # Show details of selected activities
            if not selected_activities.empty:
                st.subheader("Selected Activities Details")
                st.dataframe(
                    selected_activities[['date', 'duration', 'distance', 'elevation_gain']].reset_index(drop=True),
                    use_container_width=True
                )
            
        else:
            st.info("No activities found in the selected date range")
    
    with tab2:
        st.header("Activity Maps")
        
        if not selected_activities.empty:
            selected_activity_for_map = None
            
            if len(selected_activities) == 1:
                selected_activity_for_map = selected_activities.iloc[0]
            else:
                map_activity_options = selected_activities['date'].dt.strftime('%Y-%m-%d %H:%M - %A').tolist()
                if map_activity_options:
                    selected_map_activity = st.selectbox(
                        "Select activity to view on map",
                        options=map_activity_options,
                        index=0
                    )
                    selected_idx = map_activity_options.index(selected_map_activity)
                    selected_activity_for_map = selected_activities.iloc[selected_idx]
            
            if selected_activity_for_map is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    show_activity_map(selected_activity_for_map)
                
                with col2:
                    st.subheader("Elevation Profile")
                    if 'elevation_data' in selected_activity_for_map and selected_activity_for_map['elevation_data']:
                        plot_elevation_profile(selected_activity_for_map)
                    else:
                        st.info("No elevation data available for this activity")
                
                st.subheader("Speed Chart")
                if 'speed_data' in selected_activity_for_map and selected_activity_for_map['speed_data']:
                    plot_speed_chart(selected_activity_for_map)
                else:
                    st.info("No speed data available for this activity")
        else:
            st.info("Please select at least one activity to view on map")
    
    with tab3:
        st.header("Activity Trends")
        
        if not filtered_df.empty:
            # Monthly mileage trends
            st.subheader("Monthly Mileage")
            display_monthly_mileage(filtered_df)
            
            # Activity comparison
            st.subheader("Activity Comparison")
            if len(selected_activities) > 1:
                # Create comparison chart
                comparison_df = selected_activities[['date', 'distance', 'duration', 'elevation_gain']].copy()
                comparison_df['date'] = comparison_df['date'].dt.strftime('%Y-%m-%d')
                comparison_df.set_index('date', inplace=True)
                
                metric_to_compare = st.selectbox(
                    "Select metric to compare",
                    options=["distance", "duration", "elevation_gain"],
                    index=0
                )
                
                st.bar_chart(comparison_df[metric_to_compare])
            else:
                st.info("Select multiple activities to compare them")
        else:
            st.info("No activities found in the selected date range")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Import custom modules
from data_parser import parse_tcx
from data_processor import load_activities, build_activity_dataframe
from visualization import (
    display_activity_map, 
    display_elevation_profile, 
    display_speed_chart, 
    display_summary_cards,
    export_activity_data,
    display_monthly_trend,
    compare_activities
)

# Page configuration
st.set_page_config(
    page_title="Cycling Activity Analysis",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the data loading function to improve performance
@st.cache_data(ttl=3600, show_spinner=True)
def load_cached_data(data_path):
    """
    Load and cache activity data for better performance.
    """
    activities = load_activities(data_path)
    if not activities:
        return None
        
    return build_activity_dataframe(activities)

def main():
    # App title
    st.title("🚴 Cycling Activity Analysis Dashboard")
    
    # Set default data path
    DEFAULT_DATA_PATH = "./Takeout 2/Takeout/Fit/Activités"
    
    # Sidebar
    st.sidebar.header("Data Controls")
    
    # Data path input
    data_path = st.sidebar.text_input(
        "Data Directory Path", 
        value=DEFAULT_DATA_PATH,
        help="Path to directory containing TCX files"
    )
    
    # Load data
    if os.path.exists(data_path):
        with st.spinner("Loading activity data..."):
            df = load_cached_data(data_path)
        
        if df is None or df.empty:
            st.error(f"No cycling activities found in {data_path}")
            return
            
        st.sidebar.success(f"Found {len(df)} cycling activities")
        
        # Date range filter
        min_date = df['date'].min().date() if not df.empty else datetime.now().date()
        max_date = df['date'].max().date() if not df.empty else datetime.now().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Handle case when only one date is selected
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range
        
        # Filter by date range
        filtered_df = df[
            (df['date'].dt.date >= start_date) & 
            (df['date'].dt.date <= end_date)
        ]
        
        # Activity selector
        activity_dates = filtered_df['date'].dt.strftime('%Y-%m-%d').tolist()
        activity_distances = filtered_df['distance'].tolist()
        
        # Create more informative options by including distance
        activity_options = [
            f"{date} ({dist:.1f} km)" 
            for date, dist in zip(activity_dates, activity_distances)
        ]
        
        selected_activities = st.sidebar.multiselect(
            "Select Activities to View",
            options=activity_options,
            default=activity_options[:1] if activity_options else [],
            help="Choose one or more activities to analyze"
        )
        
        # Extract dates from the selected options
        selected_dates = [
            datetime.strptime(option.split(" ")[0], '%Y-%m-%d').date() 
            for option in selected_activities
        ]
        
        # Filter the dataframe based on selected activities
        selected_df = filtered_df[filtered_df['date'].dt.date.isin(selected_dates)]
        
        # Add comparison mode checkbox
        comparison_mode = st.sidebar.checkbox(
            "Comparison Mode", 
            value=False,
            help="Compare multiple selected activities"
        )
        
        # Add export option
        st.sidebar.subheader("Export Data")
        export_activity_data(filtered_df)
        
        # Add About section in sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("About")
        st.sidebar.info(
            "This dashboard analyzes cycling activities from TCX files. "
            "Select date ranges and specific activities to view detailed analysis."
        )
        
        # Set up tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Summary", "🗺️ Maps", "📈 Trends"])
        
        with tab1:
            if comparison_mode and len(selected_dates) >= 2:
                # Show comparison of selected activities
                compare_activities(filtered_df, selected_dates)
            elif not selected_df.empty:
                # Show individual activity details
                st.subheader("Activity Summary")
                
                # For summary tab, use the first selected activity
                activity_data = selected_df.iloc[0].to_dict()
                
                # Display summary cards
                display_summary_cards(activity_data)
                
                # Display charts
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'speed_data' in activity_data and 'time_data' in activity_data:
                        display_speed_chart(
                            activity_data['time_data'], 
                            activity_data['speed_data'],
                            activity_data.get('distance_data')
                        )
                
                with col2:
                    if 'distance_data' in activity_data and 'elevation_data' in activity_data:
                        display_elevation_profile(
                            activity_data['distance_data'], 
                            activity_data['elevation_data']
                        )
            else:
                st.info("Please select an activity from the sidebar to view details.")
                
                # Show overall statistics
                st.subheader("Overall Statistics")
                
                # Create metrics for entire dataset
                total_distance = filtered_df['distance'].sum()
                avg_speed = filtered_df['avg_speed'].mean()
                total_elevation = filtered_df['elevation_gain'].sum()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Activities", f"{len(filtered_df)}")
                col2.metric("Total Distance", f"{total_distance:.1f} km")
                col3.metric("Total Elevation", f"{total_elevation:.0f} m")
                col4.metric("Avg Speed", f"{avg_speed:.1f} km/h")
                
                # Display monthly trend
                st.subheader("Monthly Distance Trend")
                display_monthly_trend(filtered_df)
        
        with tab2:
            if not selected_df.empty:
                # For map tab, show the first selected activity
                activity_data = selected_df.iloc[0].to_dict()
                
                if 'coordinates' in activity_data and activity_data['coordinates']:
                    display_activity_map(
                        activity_data['coordinates'], 
                        activity_data.get('speed_data')
                    )
                else:
                    st.warning("No GPS data available for the selected activity.")
            else:
                st.warning("No activities selected. Please select at least one activity from the sidebar to view it on the map.")
