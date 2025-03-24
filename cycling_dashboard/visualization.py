import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import branca.colormap as cm
from datetime import datetime
import io

def create_activity_map(coordinates, speed_data=None, start_point=None, end_point=None):
    """
    Create a Folium map with a polyline representing the cycling route.
    
    Parameters:
    -----------
    coordinates : list
        List of [lat, lon] coordinates
    speed_data : list, optional
        List of speed values corresponding to coordinates for coloring
    start_point : list, optional
        [lat, lon] of the starting point
    end_point : list, optional
        [lat, lon] of the ending point
        
    Returns:
    --------
    folium.Map
        Map with the cycling route plotted
    """
    if not coordinates or len(coordinates) < 2:
        st.warning("Insufficient coordinate data to display map")
        return None
    
    # Calculate the center of the map
    center_lat = sum(lat for lat, _ in coordinates) / len(coordinates)
    center_lon = sum(lon for _, lon in coordinates) / len(coordinates)
    
    # Create a map centered at the midpoint of the route
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # If speed data is available, use it for coloring the polyline
    if speed_data and len(speed_data) == len(coordinates):
        # Create a colormap
        min_speed = min(speed_data)
        max_speed = max(speed_data)
        colormap = cm.LinearColormap(
            colors=['blue', 'green', 'yellow', 'orange', 'red'],
            vmin=min_speed,
            vmax=max_speed
        )
        
        # Create segments of the route colored by speed
        for i in range(len(coordinates) - 1):
            folium.PolyLine(
                [coordinates[i], coordinates[i+1]],
                color=colormap(speed_data[i]),
                weight=5,
                opacity=0.7
            ).add_to(m)
        
        # Add the colormap to the map
        colormap.caption = 'Speed (km/h)'
        m.add_child(colormap)
    else:
        # Create a simple polyline with a single color
        folium.PolyLine(
            coordinates,
            color='blue',
            weight=5,
            opacity=0.7
        ).add_to(m)
    
    # Add markers for start and end points
    if start_point:
        folium.Marker(
            location=start_point,
            popup='Start',
            icon=folium.Icon(color='green', icon='play', prefix='fa')
        ).add_to(m)
    else:
        # Use the first coordinate as the start point
        folium.Marker(
            location=coordinates[0],
            popup='Start',
            icon=folium.Icon(color='green', icon='play', prefix='fa')
        ).add_to(m)
    
    if end_point:
        folium.Marker(
            location=end_point,
            popup='End',
            icon=folium.Icon(color='red', icon='stop', prefix='fa')
        ).add_to(m)
    else:
        # Use the last coordinate as the end point
        folium.Marker(
            location=coordinates[-1],
            popup='End',
            icon=folium.Icon(color='red', icon='stop', prefix='fa')
        ).add_to(m)
    
    # Fit the map to the bounds of the route
    m.fit_bounds([coordinates[0], coordinates[-1]])
    
    return m

def display_activity_map(coordinates, speed_data=None):
    """
    Display a Folium map with a cycling route in a Streamlit app.
    
    Parameters:
    -----------
    coordinates : list
        List of [lat, lon] coordinates
    speed_data : list, optional
        List of speed values corresponding to coordinates for coloring
    """
    map_obj = create_activity_map(coordinates, speed_data)
    if map_obj:
        st.subheader("Activity Map")
        folium_static(map_obj)
    else:
        st.warning("Unable to display map due to insufficient coordinate data")

def create_elevation_profile(distance_data, elevation_data):
    """
    Create an elevation profile chart using Plotly.
    
    Parameters:
    -----------
    distance_data : list
        List of cumulative distance values (in km)
    elevation_data : list
        List of elevation values (in meters)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with the elevation profile
    """
    if not distance_data or not elevation_data or len(distance_data) != len(elevation_data):
        st.warning("Insufficient or mismatched data for elevation profile")
        return None
    
    # Create a line chart
    fig = px.line(
        x=distance_data, 
        y=elevation_data,
        labels={'x': 'Distance (km)', 'y': 'Elevation (m)'},
        title='Elevation Profile'
    )
    
    # Customize the layout
    fig.update_layout(
        xaxis_title='Distance (km)',
        yaxis_title='Elevation (m)',
        hovermode='x unified',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    # Add shading below the line
    fig.add_trace(
        go.Scatter(
            x=distance_data,
            y=elevation_data,
            fill='tozeroy',
            fillcolor='rgba(0, 100, 80, 0.2)',
            line=dict(color='rgba(0, 100, 80, 0.8)', width=2),
            name='Elevation'
        )
    )
    
    # Calculate elevation gain and add annotation
    elevation_diffs = [max(0, elevation_data[i+1] - elevation_data[i]) 
                       for i in range(len(elevation_data)-1)]
    elevation_gain = sum(elevation_diffs)
    
    # Add annotation for elevation gain
    fig.add_annotation(
        x=distance_data[-1] * 0.9,
        y=max(elevation_data) * 0.9,
        text=f"Elevation Gain: {elevation_gain:.1f}m",
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgba(0, 100, 80, 1)",
        borderwidth=1,
        borderpad=4
    )
    
    return fig

def display_elevation_profile(distance_data, elevation_data):
    """
    Display an elevation profile chart in a Streamlit app.
    
    Parameters:
    -----------
    distance_data : list
        List of cumulative distance values (in km)
    elevation_data : list
        List of elevation values (in meters)
    """
    fig = create_elevation_profile(distance_data, elevation_data)
    if fig:
        st.subheader("Elevation Profile")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Unable to display elevation profile due to insufficient data")

def create_speed_chart(time_data, speed_data, distance_data=None):
    """
    Create an interactive speed chart using Plotly.
    
    Parameters:
    -----------
    time_data : list
        List of datetime objects or timestamps
    speed_data : list
        List of speed values (in km/h)
    distance_data : list, optional
        List of cumulative distance values (in km)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with the speed chart
    """
    if not time_data or not speed_data or len(time_data) != len(speed_data):
        st.warning("Insufficient or mismatched data for speed chart")
        return None
    
    # Create a dataframe for better handling in Plotly
    data = {'Time': time_data, 'Speed': speed_data}
    if distance_data and len(distance_data) == len(time_data):
        data['Distance'] = distance_data
    
    df = pd.DataFrame(data)
    
    # Format time data if it's datetime objects
    if isinstance(df['Time'].iloc[0], datetime):
        df['Time'] = df['Time'].dt.strftime('%H:%M:%S')
    
    # Create the line chart
    if 'Distance' in df.columns:
        fig = px.line(
            df, 
            x='Time', 
            y='Speed',
            hover_data=['Distance'],
            labels={'Speed': 'Speed (km/h)', 'Time': 'Time', 'Distance': 'Distance (km)'},
            title='Speed Over Time'
        )
    else:
        fig = px.line(
            df, 
            x='Time', 
            y='Speed',
            labels={'Speed': 'Speed (km/h)', 'Time': 'Time'},
            title='Speed Over Time'
        )
    
    # Customize the layout
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Speed (km/h)',
        hovermode='x unified',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    # Add a horizontal line for average speed
    avg_speed = sum(speed_data) / len(speed_data)
    fig.add_shape(
        type="line",
        x0=df['Time'].iloc[0],
        y0=avg_speed,
        x1=df['Time'].iloc[-1],
        y1=avg_speed,
        line=dict(
            color="red",
            width=1,
            dash="dash",
        )
    )
    
    # Add annotation for average speed
    fig.add_annotation(
        x=df['Time'].iloc[int(len(df['Time'])/2)],
        y=avg_speed,
        text=f"Avg: {avg_speed:.1f} km/h",
        showarrow=False,
        yshift=10,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgba(255, 0, 0, 0.8)",
        borderwidth=1,
        borderpad=4
    )
    
    return fig

def display_speed_chart(time_data, speed_data, distance_data=None):
    """
    Display a speed chart in a Streamlit app.
    
    Parameters:
    -----------
    time_data : list
        List of datetime objects or timestamps
    speed_data : list
        List of speed values (in km/h)
    distance_data : list, optional
        List of cumulative distance values (in km)
    """
    fig = create_speed_chart(time_data, speed_data, distance_data)
    if fig:
        st.subheader("Speed Analysis")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Unable to display speed chart due to insufficient data")

def display_summary_cards(activity_data):
    """
    Display summary metrics using Streamlit's metric component.
    
    Parameters:
    -----------
    activity_data : dict
        Dictionary containing activity metrics such as:
        - distance: Total distance in km
        - duration: Duration in seconds or as a timedelta
        - elevation_gain: Total elevation gain in meters
        - avg_speed: Average speed in km/h
        - max_speed: Maximum speed in km/h
        - date: Date of the activity
    """
    if not activity_data:
        st.warning("No activity data available to display summary")
        return
    
    # Create a row with multiple columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Format duration (convert seconds to hours:minutes:seconds if needed)
    if 'duration' in activity_data:
        duration = activity_data['duration']
        if isinstance(duration, (int, float)):
            hours, remainder = divmod(duration, 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_formatted = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        else:
            # Assume it's already formatted or a timedelta
            duration_formatted = str(duration)
    else:
        duration_formatted = "N/A"
    
    # Display metrics
    with col1:
        if 'distance' in activity_data:
            st.metric("Distance", f"{activity_data['distance']:.2f} km")
        else:
            st.metric("Distance", "N/A")
    
    with col2:
        st.metric("Duration", duration_formatted)
    
    with col3:
        if 'elevation_gain' in activity_data:
            st.metric("Elevation Gain", f"{activity_data['elevation_gain']:.0f} m")
        else:
            st.metric("Elevation Gain", "N/A")
    
    with col4:
        if 'avg_speed' in activity_data:
            st.metric("Avg Speed", f"{activity_data['avg_speed']:.1f} km/h")
        else:
            st.metric("Avg Speed", "N/A")
    
    # Create a second row for additional metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'max_speed' in activity_data:
            st.metric("Max Speed", f"{activity_data['max_speed']:.1f} km/h")
        else:
            st.metric("Max Speed", "N/A")
    
    with col2:
        if 'date' in activity_data:
            date = activity_data['date']
            if isinstance(date, datetime):
                date_formatted = date.strftime('%Y-%m-%d')
            else:
                date_formatted = str(date)
            st.metric("Date", date_formatted)
        else:
            st.metric("Date", "N/A")
    
    with col3:
        if 'calories' in activity_data:
            st.metric("Calories", f"{activity_data['calories']:.0f} kcal")
        elif 'energy' in activity_data:
            st.metric("Energy", f"{activity_data['energy']:.0f} kcal")
        else:
            st.metric("Calories", "N/A")
    
    with col4:
        if 'avg_heart_rate' in activity_data:
            st.metric("Avg Heart Rate", f"{activity_data['avg_heart_rate']:.0f} bpm")
        else:
            st.metric("Avg Heart Rate", "N/A")

def export_activity_data(df, filename="cycling_activity_data.csv"):
    """
    Create a download button for exporting activity data as CSV.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing activity data
    filename : str, optional
        Name of the CSV file to download (default: "cycling_activity_data.csv")
    """
    if df is None or df.empty:
        st.warning("No data available to export")
        return
    
    # Convert DataFrame to CSV
    csv = df.to_csv(index=False)
    
    # Create a download button
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
        help="Click to download the data as a CSV file"
    )

def display_monthly_trend(df):
    """
    Create and display a monthly mileage trend chart.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing activity data with 'date' and 'distance' columns
    """
    if df is None or df.empty or 'date' not in df.columns or 'distance' not in df.columns:
        st.warning("Insufficient data to display monthly trends")
        return
    
    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Extract month and year
    df['month_year'] = df['date'].dt.strftime('%Y-%m')
    
    # Group by month and sum the distance
    monthly_data = df.groupby('month_year')['distance'].sum().reset_index()
    
    # Create the bar chart
    fig = px.bar(
        monthly_data,
        x='month_year',
        y='distance',
        labels={'month_year': 'Month', 'distance': 'Distance (km)'},
        title='Monthly Cycling Distance'
    )
    
    # Add a trend line
    fig.add_trace(
        go.Scatter(
            x=monthly_data['month_year'],
            y=monthly_data['distance'],
            mode='lines',
            name='Trend',
            line=dict(color='red', width=2, dash='dot')
        )
    )
    
    # Customize the layout
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Distance (km)',
        xaxis={'categoryorder':'category ascending'},
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def compare_activities(df, activity_ids):
    """
    Create and display a comparison chart for multiple activities.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing activity data
    activity_ids : list
        List of activity IDs to compare
    """
    if df is None or df.empty or len(activity_ids) < 2:
        st.warning("Need at least 2 activities to compare")
        return
    
    # Filter activities to compare
    comparison_df = df[df['date'].isin(activity_ids)]
    
    if len(comparison_df) < 2:
        st.warning("Could not find enough activities to compare")
        return
    
    # Create comparison metrics
    st.subheader("Activity Comparison")
    
    # Show summary table
    comparison_table = comparison_df[['date', 'distance', 'duration', 'elevation_gain', 'avg_speed']].copy()
    if 'duration' in comparison_table.columns and pd.api.types.is_numeric_dtype(comparison_table['duration']):
        # Format duration (convert seconds to hours:minutes:seconds)
        comparison_table['duration'] = comparison_table['duration'].apply(lambda x: 
            f"{int(x//3600)}h {int((x%3600)//60)}m {int(x%60)}s" if pd.notnull(x) else "N/A")
    
    # Format date if it's datetime
    if pd.api.types.is_datetime64_any_dtype(comparison_table['date']):
        comparison_table['date'] = comparison_table['date'].dt.strftime('%Y-%m-%d')
    
    st.dataframe(comparison_table, use_container_width=True)
    
    # Create comparison chart for distance and speed
    fig = go.Figure()
    
    # Add distance bars
    fig.add_trace(go.Bar(
        x=comparison_table['date'],
        y=comparison_df['distance'],
        name='Distance (km)',
        marker_color='blue'
    ))
    
    # Add speed line on secondary y-axis
    fig.add_trace(go.Scatter(
        x=comparison_table['date'],
        y=comparison_df['avg_speed'],
        name='Avg Speed (km/h)',
        marker_color='red',
        mode='lines+markers',
        yaxis='y2'
    ))
    
    # Set up the layout with two y-axes
    fig.update_layout(
        title='Distance vs Speed Comparison',
        yaxis=dict(
            title='Distance (km)',
            side='left'
        ),
        yaxis2=dict(
            title='Speed (km/h)',
            side='right',
            overlaying='y',
            rangemode='tozero'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
