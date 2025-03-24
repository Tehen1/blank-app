import os
import json
import glob
import datetime
from datetime import timedelta
from collections import defaultdict
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Configuration
ACTIVITY_TYPES = ["BIKING", "WALKING", "RUNNING"]
COLOR_MAP = {
    "BIKING": "#1f77b4",  # Blue
    "WALKING": "#2ca02c",  # Green
    "RUNNING": "#d62728",  # Red
}


def find_activity_files(base_dir='.'):
    """
    Find all activity files (BIKING, WALKING, RUNNING) in the Takeout directory.

    Args:
        base_dir: Base directory to start searching from

    Returns:
        Dict mapping activity types to lists of file paths
    """
    activity_files = {activity_type: [] for activity_type in ACTIVITY_TYPES}

    # Look for activity files in common Takeout directory structures
    patterns = [
        f"{base_dir}/**/Takeout/**/Fit/**/Sessions/**/*_{activity_type}.json",
        f"{base_dir}/**/Takeout/**/Fit/**/*_{activity_type}.json",
        f"{base_dir}/**/Takeout/Fit/**/*_{activity_type}.json",
    ]

    for activity_type in ACTIVITY_TYPES:
        for pattern in patterns:
            found_files = glob.glob(pattern, recursive=True)
            activity_files[activity_type].extend(found_files)

        # Remove duplicates while preserving order
        activity_files[activity_type] = list(
            dict.fromkeys(activity_files[activity_type]))

        print(
            f"Found {len(activity_files[activity_type])} {activity_type} files")

    return activity_files


def find_location_files(base_dir='.'):
    """
    Find all location sample files in the Takeout directory.

    Args:
        base_dir: Base directory to start searching from

    Returns:
        List of location file paths
    """
    patterns = [
        f"{base_dir}/**/Takeout/**/derived_com.google.location.sample*.json",
        f"{base_dir}/**/Takeout/Fit/**/derived_com.google.location.sample*.json",
    ]

    location_files = []
    for pattern in patterns:
        location_files.extend(glob.glob(pattern, recursive=True))

    # Remove duplicates while preserving order
    location_files = list(dict.fromkeys(location_files))
    print(f"Found {len(location_files)} location files")
    return location_files


def extract_activity_data(activity_files):
    """
    Extract metadata from activity files.

    Args:
        activity_files: Dict mapping activity types to lists of file paths

    Returns:
        List of activity data dictionaries
    """
    all_activities = []

    for activity_type, files in activity_files.items():
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract basic information
                activity = {
                    'type': activity_type,
                    'file_path': file_path,
                    'filename': os.path.basename(file_path),
                    'start_time': data.get('startTime', ''),
                    'end_time': data.get('endTime', ''),
                    'duration_ms': data.get('duration', {}).get('millis', 0),
                }

                # Convert string dates to datetime objects
                if activity['start_time']:
                    start_dt = datetime.datetime.fromisoformat(
                        activity['start_time'].replace('Z', '+00:00'))
                    activity['start_dt'] = start_dt
                    activity['start_time_nanos'] = int(
                        start_dt.timestamp() * 1_000_000_000)

                if activity['end_time']:
                    end_dt = datetime.datetime.fromisoformat(
                        activity['end_time'].replace('Z', '+00:00'))
                    activity['end_dt'] = end_dt
                    activity['end_time_nanos'] = int(
                        end_dt.timestamp() * 1_000_000_000)

                # Extract additional metrics if available
                if 'aggregate' in data:
                    for metric in data['aggregate']:
                        if metric.get('metricName') == 'com.google.calories.expended':
                            activity['calories'] = metric.get('floatValue', 0)
                        elif metric.get('metricName') == 'com.google.step_count.delta':
                            activity['steps'] = metric.get('intValue', 0)
                        elif metric.get('metricName') == 'com.google.distance.delta':
                            activity['distance'] = metric.get('floatValue', 0)
                        elif metric.get('metricName') == 'com.google.heart_minutes.summary':
                            activity['heart_minutes'] = metric.get(
                                'floatValue', 0)

                all_activities.append(activity)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # Sort by start time
    all_activities.sort(key=lambda x: x.get('start_dt', datetime.datetime.min))
    print(f"Extracted metadata for {len(all_activities)} activities")
    return all_activities


def extract_location_data(location_files):
    """
    Extract location data from all location files.

    Args:
        location_files: List of location file paths

    Returns:
        List of location data points
    """
    all_location_data = []

    for file_path in location_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract location data points
            if 'Data Points' in data:
                data_points = data['Data Points']
                for point in data_points:
                    # Extract timestamp and coordinates
                    if 'startTimeNanos' in point and 'fitValue' in point:
                        try:
                            start_time_nanos = int(point['startTimeNanos'])
                            end_time_nanos = int(point['endTimeNanos'])

                            # Extract lat/long from fitValue array
                            if len(point['fitValue']) >= 2:
                                lat = point['fitValue'][0]['value'].get(
                                    'fpVal')
                                lng = point['fitValue'][1]['value'].get(
                                    'fpVal')

                                # Some data points have optional accuracy and altitude
                                accuracy = None
                                altitude = None
                                if len(point['fitValue']) >= 3:
                                    accuracy = point['fitValue'][2]['value'].get(
                                        'fpVal')
                                if len(point['fitValue']) >= 4:
                                    altitude = point['fitValue'][3]['value'].get(
                                        'fpVal')

                                if lat is not None and lng is not None:
                                    loc_data = {
                                        'start_time_nanos': start_time_nanos,
                                        'end_time_nanos': end_time_nanos,
                                        'latitude': lat,
                                        'longitude': lng,
                                        'accuracy': accuracy,
                                        'altitude': altitude,
                                        'timestamp': datetime.datetime.fromtimestamp(start_time_nanos / 1_000_000_000)
                                    }
                                    all_location_data.append(loc_data)
                        except (KeyError, ValueError, TypeError) as e:
                            # Skip invalid data points
                            continue
        except Exception as e:
            print(f"Error processing location file {file_path}: {e}")

    # Sort by timestamp
    all_location_data.sort(key=lambda x: x['start_time_nanos'])
    print(f"Extracted {len(all_location_data)} location data points")
    return all_location_data


def match_locations_to_activities(activities, location_data, accuracy_threshold=30):
    """
    Match location data to activities based on timestamps.

    Args:
        activities: List of activity data dictionaries
        location_data: List of location data points
        accuracy_threshold: Maximum GPS accuracy value (in meters) to include

    Returns:
        Activities list with routes added
    """
    # Create a new list to store activities with routes
    activities_with_routes = []

    # Count activities with routes
    activities_with_routes_count = 0

    for activity in activities:
        # Skip activities missing timestamp data
        if 'start_time_nanos' not in activity or 'end_time_nanos' not in activity:
            activities_with_routes.append(activity)
            continue

        # Find location points within the activity time range
        route_points = []
        for loc in location_data:
            # Check if the location timestamp falls within the activity time range
            if (activity['start_time_nanos'] <= loc['start_time_nanos'] <= activity['end_time_nanos']):
                # Filter out low-accuracy points if accuracy data is available
                if loc['accuracy'] is not None and loc['accuracy'] > accuracy_threshold:
                    continue

                route_points.append({
                    'latitude': loc['latitude'],
                    'longitude': loc['longitude'],
                    'timestamp': loc['timestamp'],
                    'accuracy': loc['accuracy'],
                    'altitude': loc['altitude']
                })

        # Add route to activity if we found matching points
        activity['route'] = route_points
        activity['has_route'] = len(route_points) > 0

        if activity['has_route']:
            activities_with_routes_count += 1

        activities_with_routes.append(activity)

    print(
        f"Matched location data to {activities_with_routes_count} activities")
    return activities_with_routes


def create_activity_dataframe(activities):
    """
    Convert activity data to a pandas DataFrame for easier analysis.

    Args:
        activities: List of activity data dictionaries

    Returns:
        Pandas DataFrame with activity data
    """
    # Extract relevant fields for the DataFrame
    data = []
    for activity in activities:
        if 'start_dt' in activity:
            row = {
                'activity_type': activity['type'],
                'start_time': activity['start_dt'],
                'end_time': activity.get('end_dt'),
                'duration_minutes': activity.get('duration_ms', 0) / 60000,
                'has_route': activity.get('has_route', False),
                'route_points': len(activity.get('route', [])),
                'calories': activity.get('calories', 0),
                'steps': activity.get('steps', 0),
                'distance': activity.get('distance', 0),
                'heart_minutes': activity.get('heart_minutes', 0),
                'hour_of_day': activity['start_dt'].hour,
                'day_of_week': activity['start_dt'].strftime('%A'),
                'month': activity['start_dt'].strftime('%Y-%m'),
                'year_month': activity['start_dt'].strftime('%Y-%m'),
                'date': activity['start_dt'].date()
            }
            data.append(row)

    df = pd.DataFrame(data)

    # Add day of week as categorical with proper order
    days_order = ['Monday', 'Tuesday', 'Wednesday',
                  'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_of_week'] = pd.Categorical(
        df['day_of_week'], categories=days_order, ordered=True)

    return df


def create_dashboard(df, activities):
    """
    Create a Streamlit dashboard to visualize activity data.

    Args:
        df: Pandas DataFrame with activity data
        activities: List of activity data dictionaries with routes
    """
    # Set page title and configuration - moved to main() to avoid duplicate configuration

    # Title and introduction
    st.title("Activity Dashboard")
    st.markdown(
        "This dashboard visualizes your fitness activities from Google Takeout data.")

    # Get the start and end dates of the data
    min_date = df['start_time'].min().date()
    max_date = df['start_time'].max().date()

    # Sidebar with filters
    st.sidebar.title("Filters")

    # Activity type filter
    activity_types = df['activity_type'].unique().tolist()
    selected_types = st.sidebar.multiselect(
        "Activity Types",
        options=activity_types,
        default=activity_types
    )

    # Date range filter
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Filter the dataframe based on selections
    filtered_df = df.copy()

    if selected_types:
        filtered_df = filtered_df[filtered_df['activity_type'].isin(
            selected_types)]

    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (
            filtered_df['date'] <= end_date)]

    # Overview statistics
    st.header("Activity Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Activities", filtered_df.shape[0])

    with col2:
        st.metric(
            "Date Range", f"{filtered_df['date'].min()} to {filtered_df['date'].max()}")

    with col3:
        activities_with_routes = filtered_df[filtered_df['has_route']].shape[0]
        st.metric("Activities with Routes", activities_with_routes)

    with col4:
        total_duration = filtered_df['duration_minutes'].sum()
        st.metric("Total Duration", f"{total_duration:.1f} minutes")

    # Activity counts by type
    st.subheader("Activity Counts by Type")
    activity_counts = filtered_df['activity_type'].value_counts().reset_index()
    activity_counts.columns = ['Activity Type', 'Count']

    fig = px.bar(
        activity_counts,
        x='Activity Type',
        y='Count',
        color='Activity Type',
        color_discrete_map=COLOR_MAP,
        text='Count'
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Monthly activity distribution
    st.header("Activity Patterns")

    # Group by month and activity type
    monthly_counts = filtered_df.groupby(
        ['year_month', 'activity_type']).size().reset_index(name='count')

    # Continue with the dashboard implementation
    # Create a monthly trend chart
    fig = px.line(
        monthly_counts,
        x='year_month',
        y='count',
        color='activity_type',
        title='Monthly Activity Trends',
        labels={'count': 'Number of Activities',
                'year_month': 'Month', 'activity_type': 'Activity Type'}
    )
    st.plotly_chart(fig, use_container_width=True)

# Remove duplicate function definitions


def process_all_data():
    """Process all activity and location data"""
    print("Finding activity files...")
    activity_files = find_activity_files(base_dir)
    total_files = sum(len(files) for files in activity_files.values())
    print(f"Found {total_files} activity files")

    print("Finding location files...")
    location_files = find_location_files()
    print(f"Found {len(location_files)} location files")

    print("Extracting activity metadata...")
    activities = extract_activity_data(activity_files)
    print(f"Extracted metadata for {len(activities)} activities")

    print("Extracting location data...")
    location_data = extract_location_data(location_files)
    print(f"Extracted {len(location_data)} location data points")

    print("Matching locations to activities...")
    activities_with_routes = match_locations_to_activities(
        activities, location_data)

    # Count activities with routes
    activities_with_routes_count = sum(
        1 for a in activities_with_routes if len(a.get("route", [])) > 0)
    print(
        f"Matched location data to {activities_with_routes_count} activities")

    return activities_with_routes

# Dashboard Visualization Functions


def create_activity_overview(activities):
    """Create overview statistics for the dashboard"""
    st.header("Activity Overview")

    # Calculate overview statistics
    total_activities = len(activities)
    activities_with_routes = sum(
        1 for a in activities if len(a.get("route", [])) > 0)
    total_duration_mins = sum(a.get("duration_ms", 0)
                              for a in activities) / (1000 * 60)

    # Get date range
    start_dates = [datetime.fromisoformat(a["start_time"].replace("Z", "+00:00"))
                   for a in activities if a.get("start_time")]
    if start_dates:
        min_date = min(start_dates).strftime("%Y-%m-%d")
        max_date = max(start_dates).strftime("%Y-%m-%d")
        date_range = f"{min_date} to {max_date}"
    else:
        date_range = "N/A"

    # Activity counts by type
    activity_types = {}
    for a in activities:
        activity_type = a.get("activity_type", "UNKNOWN")
        activity_types[activity_type] = activity_types.get(
            activity_type, 0) + 1

    # Create columns for statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Activities", total_activities)
        st.metric("Date Range", date_range)

    with col2:
        st.metric("Activities with GPS Routes", activities_with_routes)
        st.metric("Total Duration (hours)", f"{total_duration_mins/60:.1f}")

    with col3:
        for activity_type, count in activity_types.items():
            st.metric(f"{activity_type} Activities", count)


def create_monthly_chart(activities):
    """Create a monthly activity chart"""
    st.header("Monthly Activity Distribution")

    # Prepare data
    monthly_data = defaultdict(lambda: defaultdict(int))

    for activity in activities:
        if not activity.get("start_time"):
            continue

        try:
            start_time = datetime.fromisoformat(
                activity["start_time"].replace("Z", "+00:00"))
            month_key = start_time.strftime("%Y-%m")
            activity_type = activity.get("activity_type", "UNKNOWN")
            monthly_data[month_key][activity_type] += 1
        except Exception as e:
            print(f"Error processing activity for monthly chart: {e}")

    # Convert to DataFrame
    df_data = []
    for month, type_counts in monthly_data.items():
        for activity_type, count in type_counts.items():
            df_data.append({
                "Month": month,
                "Activity Type": activity_type,
                "Count": count
            })

    df = pd.DataFrame(df_data)

    if not df.empty:
        # Sort by month
        df["Month"] = pd.to_datetime(df["Month"])
        df = df.sort_values("Month")
        df["Month"] = df["Month"].dt.strftime("%Y-%m")

        # Create the chart
        fig = px.bar(
            df,
            x="Month",
            y="Count",
            color="Activity Type",
            title="Activities per Month",
            labels={"Count": "Number of Activities", "Month": "Month"},
            barmode="group"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for monthly chart")


def create_time_of_day_chart(activities):
    """Create a time of day analysis chart"""
    st.header("Time of Day Analysis")

    # Prepare data
    time_data = defaultdict(lambda: defaultdict(int))

    for activity in activities:
        if not activity.get("start_time"):
            continue

        try:
            start_time = datetime.fromisoformat(
                activity["start_time"].replace("Z", "+00:00"))
            hour = start_time.hour

            # Define time of day
            if 5 <= hour < 12:
                time_of_day = "Morning (5-12)"
            elif 12 <= hour < 17:
                time_of_day = "Afternoon (12-17)"
            elif 17 <= hour < 22:
                time_of_day = "Evening (17-22)"
            else:
                time_of_day = "Night (22-5)"

            activity_type = activity.get("activity_type", "UNKNOWN")
            time_data[time_of_day][activity_type] += 1
        except Exception as e:
            print(f"Error processing activity for time of day chart: {e}")

    # Convert to DataFrame
    df_data = []
    for time_of_day, type_counts in time_data.items():
        for activity_type, count in type_counts.items():
            df_data.append({
                "Time of Day": time_of_day,
                "Activity Type": activity_type,
                "Count": count
            })

    df = pd.DataFrame(df_data)

    if not df.empty:
        # Sort by time of day
        time_order = ["Morning (5-12)", "Afternoon (12-17)",
                      "Evening (17-22)", "Night (22-5)"]
        df["Time of Day"] = pd.Categorical(
            df["Time of Day"], categories=time_order, ordered=True)
        df = df.sort_values("Time of Day")

        # Create the chart
        fig = px.bar(
            df,
            x="Time of Day",
            y="Count",
            color="Activity Type",
            title="Activities by Time of Day",
            labels={"Count": "Number of Activities",
                    "Time of Day": "Time of Day"},
            barmode="group"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for time of day chart")


def create_day_of_week_chart(activities):
    """Create a day of week analysis chart"""
    st.header("Activities by Day of Week")

    # Prepare data
    day_data = defaultdict(lambda: defaultdict(int))

    for activity in activities:
        if not activity.get("start_time"):
            continue

        try:
            start_time = datetime.fromisoformat(
                activity["start_time"].replace("Z", "+00:00"))
            day_of_week = start_time.strftime("%A")

            activity_type = activity.get("activity_type", "UNKNOWN")
            day_data[day_of_week][activity_type] += 1
        except Exception as e:
            print(f"Error processing activity for day of week chart: {e}")

    # Convert to DataFrame
    df_data = []
    for day_of_week, type_counts in day_data.items():
        for activity_type, count in type_counts.items():
            df_data.append({
                "Day of Week": day_of_week,
                "Activity Type": activity_type,
                "Count": count
            })

    df = pd.DataFrame(df_data)

    if not df.empty:
        # Sort by day of week
        day_order = ["Monday", "Tuesday", "Wednesday",
                     "Thursday", "Friday", "Saturday", "Sunday"]
        df["Day of Week"] = pd.Categorical(
            df["Day of Week"], categories=day_order, ordered=True)
        df = df.sort_values("Day of Week")

        # Create the chart
        fig = px.bar(
            df,
            x="Day of Week",
            y="Count",
            color="Activity Type",
            title="Activities by Day of Week",
            labels={"Count": "Number of Activities",
                    "Day of Week": "Day of Week"},
            barmode="group"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for day of week chart")


def create_duration_comparison(activities):
    """Create a duration comparison chart for different activity types"""
    st.header("Activity Duration Comparison")

    # Prepare data
    duration_data = []

    for activity in activities:
        if not activity.get("duration_ms"):
            continue

        try:
            duration_mins = activity.get("duration_ms", 0) / (1000 * 60)
            activity_type = activity.get("activity_type", "UNKNOWN")

            duration_data.append({
                "Activity Type": activity_type,
                "Duration (minutes)": duration_mins
            })
        except Exception as e:
            print(f"Error processing activity for duration comparison: {e}")

    # Create DataFrame for duration analysis
    df = pd.DataFrame(duration_data)
    if not df.empty:
        # Create the chart
        fig = px.box(
            df,
            x="Activity Type",
            y="Duration (minutes)",
            color="Activity Type",
            title="Activity Duration Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for duration comparison chart")

    return df


# Configuration constants
ACTIVITY_COLORS = {
    "BIKING": "#1f77b4",  # blue
    "WALKING": "#2ca02c",  # green
    "RUNNING": "#d62728",  # red
}


def find_activity_files(root_dir, activity_types=None):
    """Find all activity files in the Takeout directory"""
    if activity_types is None:
        activity_types = ["BIKING", "WALKING", "RUNNING"]

    all_files = []
    for activity_type in activity_types:
        pattern = os.path.join(root_dir, "**", f"*_{activity_type}.json")
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)

    print(f"Found {len(all_files)} activity files")
    return all_files


def find_location_files(root_dir):
    """Find all location sample files in the Takeout directory"""
    pattern = os.path.join(
        root_dir, "**", "derived_com.google.location.sample*.json")
    files = glob.glob(pattern, recursive=True)
    print(f"Found {len(files)} location files")
    return files


def extract_activity_data(activity_files):
    """Extract metadata from all activity files"""
    activities = []

    for file_path in activity_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                # Determine activity type from filename
                activity_type = file_path.split('_')[-1].split('.')[0]

                # Extract basic metadata
                start_time = data.get('startTime')
                end_time = data.get('endTime')
                duration_millis = 0
                if isinstance(data.get('duration'), dict):
                    duration_millis = data.get('duration', {}).get('millis', 0)

                # Initialize aggregate data
                aggregate = {}
                # Extract aggregate metrics - handle different possible formats
                if isinstance(data.get('aggregate'), list) and len(data.get('aggregate', [])) > 0:
                    aggregate = data.get('aggregate')[0]
                elif isinstance(data.get('aggregate'), dict):
                    aggregate = data.get('aggregate')

                activity = {
                    'activity_type': activity_type,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_minutes': duration_millis / 60000 if duration_millis else 0,
                    'file_path': file_path,
                    'has_route': False,
                    'route': [],
                }
                # Add type-specific metrics - safely extract data with proper error handling
                try:
                    if activity_type == "BIKING":
                        # Safely extract values using helper functions
                        if isinstance(aggregate, dict):
                            activity['calories'] = _safe_get_value(
                                aggregate, 'calories')
                            activity['heart_minutes'] = _safe_get_value(
                                aggregate, 'heartMinutes')
                    elif activity_type == "WALKING":
                        if isinstance(aggregate, dict):
                            activity['steps'] = _safe_get_value(
                                aggregate, 'steps')
                            activity['calories'] = _safe_get_value(
                                aggregate, 'calories')
                            activity['heart_minutes'] = _safe_get_value(
                                aggregate, 'heartMinutes')
                            activity['distance_meters'] = _safe_get_value(
                                aggregate, 'distance', 'meters')
                    elif activity_type == "RUNNING":
                        if isinstance(aggregate, dict):
                            activity['steps'] = _safe_get_value(
                                aggregate, 'steps')
                            activity['calories'] = _safe_get_value(
                                aggregate, 'calories')
                            activity['heart_minutes'] = _safe_get_value(
                                aggregate, 'heartMinutes')
                            activity['distance_meters'] = _safe_get_value(
                                aggregate, 'distance', 'meters')
                except Exception as e:
                    print(
                        f"Error extracting metrics for {activity_type} activity: {e}")
                    activity['distance_meters'] = aggregate.get(
                        'distance', {}).get('meters', 0)

                activities.append(activity)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Extracted metadata from {len(activities)} activities")
    return activities


def _safe_get_value(data_dict, key, nested_key=None):
    """Safely extract a value from a dictionary, handling various structures"""
    if not isinstance(data_dict, dict):
        return 0

    # Handle case where the value is directly accessible
    if key in data_dict:
        if isinstance(data_dict[key], (int, float)):
            return data_dict[key]
        elif isinstance(data_dict[key], dict) and nested_key and nested_key in data_dict[key]:
            return data_dict[key][nested_key]
        elif isinstance(data_dict[key], dict) and 'value' in data_dict[key]:
            return data_dict[key]['value']

    # Handle case where the data might be in a list of metrics
    if isinstance(data_dict, dict) and 'metrics' in data_dict and isinstance(data_dict['metrics'], list):
        for metric in data_dict['metrics']:
            if isinstance(metric, dict) and metric.get('metricName') == key:
                if 'floatValue' in metric:
                    return metric['floatValue']
                elif 'intValue' in metric:
                    return metric['intValue']

    return 0


def extract_location_data(location_files):
    """Extract location data from all location sample files"""
    location_data = []

    for file_path in location_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                # Process data points - handle different possible formats
                if "Data Points" in data:
                    for point in data["Data Points"]:
                        try:
                            # Check if fitValue exists and has at least 2 values
                            if "fitValue" in point and isinstance(point["fitValue"], list) and len(point["fitValue"]) >= 2:
                                # Safely extract latitude and longitude
                                lat_val = None
                                lon_val = None

                                # Handle different potential structures
                                if isinstance(point["fitValue"][0], dict) and "value" in point["fitValue"][0]:
                                    if isinstance(point["fitValue"][0]["value"], dict) and "fpVal" in point["fitValue"][0]["value"]:
                                        lat_val = point["fitValue"][0]["value"]["fpVal"]
                                    elif isinstance(point["fitValue"][0]["value"], (int, float)):
                                        lat_val = point["fitValue"][0]["value"]

                                if isinstance(point["fitValue"][1], dict) and "value" in point["fitValue"][1]:
                                    if isinstance(point["fitValue"][1]["value"], dict) and "fpVal" in point["fitValue"][1]["value"]:
                                        lon_val = point["fitValue"][1]["value"]["fpVal"]
                                    elif isinstance(point["fitValue"][1]["value"], (int, float)):
                                        lon_val = point["fitValue"][1]["value"]

                            if lat_val is not None and lon_val is not None:
                                timestamp_nanos = point.get("startTimeNanos")

                                if timestamp_nanos:
                                    # Convert to numeric and add to location data
                                    timestamp_nanos = int(timestamp_nanos)
                                    location_data.append({
                                        'latitude': lat_val,
                                        'longitude': lon_val,
                                        'timestamp_nanos': timestamp_nanos,
                                        'timestamp': datetime.datetime.fromtimestamp(timestamp_nanos / 1e9)
                                    })
                        except Exception as e:
                            print(
                                f"Error processing location file {file_path}: {e}")
    # Sort by timestamp
    location_data.sort(key=lambda x: x['timestamp_nanos'])
    print(f"Extracted {len(location_data)} location data points")
    return location_data


def match_activities_with_location(activities, location_data):
    """Match activities with location data based on timestamps"""
    matched_count = 0
    total_points_matched = 0

    for activity in activities:
        if not activity['start_time'] or not activity['end_time']:
            continue

        # Convert ISO timestamps to nanoseconds
        try:
            # Convert ISO timestamps to nanoseconds
            start_nanos = int(datetime.datetime.fromisoformat(
                activity['start_time'].replace('Z', '+00:00')).timestamp() * 1e9)
            end_nanos = int(datetime.datetime.fromisoformat(
                activity['end_time'].replace('Z', '+00:00')).timestamp() * 1e9)
        except Exception as e:
            print(f"Error converting timestamps: {e}")
            continue
        # Find location points within the activity timeframe
        route_points = []
        for point in location_data:
            if start_nanos <= point['timestamp_nanos'] <= end_nanos:
                route_points.append({
                    'latitude': point['latitude'],
                    'longitude': point['longitude'],
                    'timestamp_nanos': point['timestamp_nanos']
                })

        # Add route to activity if points found
        if route_points:
            activity['has_route'] = True
            activity['route'] = route_points
            matched_count += 1
            total_points_matched += len(route_points)

    print(
        f"Matched {total_points_matched} location points to {matched_count} activities")
    return activities


def preprocess_data(activities):
    """Create pandas DataFrames for easier data manipulation"""
    # Filter out activities with missing start_time
    valid_activities = [a for a in activities if a.get('start_time')]

    # Create DataFrame
    activities_df = pd.DataFrame(valid_activities)

    # Handle empty DataFrame case
    if activities_df.empty:
        # Create empty DataFrame with expected columns
        activities_df = pd.DataFrame(columns=[
            'activity_type', 'start_time', 'end_time', 'duration_minutes',
            'has_route', 'route', 'file_path', 'date', 'year', 'month', 'day',
            'hour', 'day_of_week', 'month_year', 'time_of_day', 'day_name'
        ])
        return activities_df

    # Add date-related columns
    if 'start_time' in activities_df.columns:
        # Handle potential errors in datetime conversion
        try:
            activities_df['date'] = pd.to_datetime(
                activities_df['start_time'], errors='coerce').dt.date
            activities_df['year'] = pd.to_datetime(
                activities_df['start_time'], errors='coerce').dt.year
            activities_df['month'] = pd.to_datetime(
                activities_df['start_time'], errors='coerce').dt.month
            activities_df['day'] = pd.to_datetime(
                activities_df['start_time'], errors='coerce').dt.day
            activities_df['hour'] = pd.to_datetime(
                activities_df['start_time'], errors='coerce').dt.hour
            activities_df['day_of_week'] = pd.to_datetime(
                activities_df['start_time'], errors='coerce').dt.dayofweek
            activities_df['day_of_week'] = pd.to_datetime(
                activities_df['start_time'], errors='coerce').dt.dayofweek
            activities_df['month_year'] = pd.to_datetime(
                activities_df['start_time'], errors='coerce').dt.strftime('%Y-%m')

            # Time of day category
            activities_df['time_of_day'] = pd.cut(
                activities_df['hour'],
                bins=[0, 6, 12, 18, 24],
                labels=['Night (0-6)', 'Morning (6-12)',
                        'Afternoon (12-18)', 'Evening (18-24)']
            )

            # Day of week name
            activities_df['day_name'] = pd.to_datetime(
                activities_df['start_time']).dt.day_name()
        except Exception as e:
            print(f"Error processing datetime columns: {e}")
        except Exception as e:
            print(f"Error processing datetime columns: {e}")

    return activities_df


def create_dashboard(activities_df):
    """Create Streamlit dashboard"""
    st.title("🏃‍♂️ Activity Dashboard")
    st.write("Comprehensive analysis of all your activity data from Google Takeout")

    # Show overview stats
    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Activities", f"{len(activities_df)}")

    with col2:
        # Safely handle activities_df
        if 'activity_type' in activities_df.columns:
            activity_types = activities_df['activity_type'].value_counts(
            ).to_dict()
            st.metric("Activity Types", ", ".join(
                f"{k}: {v}" for k, v in activity_types.items()))
        else:
            st.metric("Activity Types", "No data")

    with col3:
        # Safely handle date range
        if 'date' in activities_df.columns and not activities_df['date'].isna().all():
            date_range = f"{activities_df['date'].min()} to {activities_df['date'].max()}"
            st.metric("Date Range", date_range)
        else:
            st.metric("Date Range", "No data")

    with col4:
        activities_with_routes = activities_df['has_route'].sum()
        st.metric("Activities with GPS",
                  f"{activities_with_routes} ({activities_with_routes/len(activities_df):.1%})")

    # Activity breakdown
    st.header("Activity Breakdown")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            activities_df,
            names='activity_type',
            title="Activities by Type",
            color='activity_type',
            color_discrete_map=ACTIVITY_COLORS
        )
        st.plotly_chart(fig)

    with col2:
        monthly_activities = activities_df.groupby(
            ['month_year', 'activity_type']).size().reset_index(name='count')
        fig = px.bar(
            monthly_activities,
            x='month_year',
            y='count',
            color='activity_type',
            title="Monthly Activity Distribution",
            labels={'month_year': 'Month', 'count': 'Number of Activities'},
            color_discrete_map=ACTIVITY_COLORS
        )
        st.plotly_chart(fig)

    # Time analysis
    st.header("Time Analysis")

    col1, col2 = st.columns(2)

    with col1:
        time_of_day = activities_df.groupby(
            ['time_of_day', 'activity_type']).size().reset_index(name='count')
        fig = px.bar(
            time_of_day,
            x='time_of_day',
            y='count',
            color='activity_type',
            title="Activities by Time of Day",
            labels={'time_of_day': 'Time of Day',
                    'count': 'Number of Activities'},
            color_discrete_map=ACTIVITY_COLORS
        )
        st.plotly_chart(fig)

    with col2:
        day_of_week = activities_df.groupby(
            ['day_name', 'activity_type']).size().reset_index(name='count')
        # Sort by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday',
                     'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_of_week['day_name'] = pd.Categorical(
            day_of_week['day_name'], categories=day_order, ordered=True)
        day_of_week = day_of_week.sort_values('day_name')

        fig = px.bar(
            day_of_week,
            x='day_name',
            y='count',
            color='activity_type',
            title="Activities by Day of Week",
            labels={'day_name': 'Day of Week',
                    'count': 'Number of Activities'},
            color_discrete_map=ACTIVITY_COLORS
        )
        st.plotly_chart(fig)

    # Duration analysis
    st.header("Duration Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(
            activities_df,
            x='activity_type',
            y='duration_minutes',
            color='activity_type',
            title="Activity Duration Distribution",
            labels={
                'duration_minutes': 'Duration (minutes)', 'activity_type': 'Activity Type'},
            color_discrete_map=ACTIVITY_COLORS
        )
        st.plotly_chart(fig)

    with col2:
        # Calculate average duration by month and activity type
        monthly_duration = activities_df.groupby(['month_year', 'activity_type'])[
            'duration_minutes'].mean().reset_index()
        fig = px.line(
            monthly_duration,
            x='month_year',
            y='duration_minutes',
            color='activity_type',
            title="Average Duration by Month",
            labels={'month_year': 'Month',
                    'duration_minutes': 'Average Duration (minutes)'},
            color_discrete_map=ACTIVITY_COLORS
        )
        st.plotly_chart(fig)

    # Route visualization
    st.header("Route Visualization")

    # Filter to activities with route data
    activities_with_routes = activities_df[activities_df['has_route'] == True].copy(
    )

    if not activities_with_routes.empty:
        # Create a map with all routes
        fig = go.Figure()

        for idx, activity in activities_with_routes.iterrows():
            route = activity['route']
            if route and len(route) > 0:
                latitudes = [point['latitude'] for point in route]
                longitudes = [point['longitude'] for point in route]

                fig.add_trace(go.Scattermapbox(
                    lat=latitudes,
                    lon=longitudes,
                    mode='lines',
                    line=dict(width=2, color=ACTIVITY_COLORS.get(
                        activity['activity_type'], '#636EFA')),
                    name=f"{activity['activity_type']} on {activity['date']}",
                    showlegend=True
                ))

        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                zoom=10,
                center=dict(
                    lat=np.mean([point['latitude'] for activity in activities_with_routes.itertuples()
                                for point in activity.route]) if len(activities_with_routes) > 0 else 0,
                    lon=np.mean([point['longitude'] for activity in activities_with_routes.itertuples()
                                for point in activity.route]) if len(activities_with_routes) > 0 else 0
                )
            ),
            height=600,
            margin=dict(l=0, r=0, t=0, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No activities with GPS route data available to display on the map.")


def main():
    """
    Main function to load data and run the dashboard
    """
    # Set page configuration once at the beginning
    st.set_page_config(
        page_title="Activity Dashboard",
        page_icon="🏃‍♂️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("Activity Dashboard")
    st.sidebar.info(
        "Analyze and visualize your activity data from Google Takeout.")

    # Set default Takeout directory
    default_dir = "./Takeout"
    takeout_dir = st.sidebar.text_input(
        "Takeout directory path", value=default_dir)

    # Processing status placeholder
    status = st.sidebar.empty()

    # Load data button
    if st.sidebar.button("Load Data"):
        status.info("Finding activity files...")
        activity_files = find_activity_files(takeout_dir)

        status.info("Finding location files...")
        location_files = find_location_files(takeout_dir)

        status.info("Extracting activity data...")
        activities = extract_activity_data(activity_files)

        status.info("Extracting location data...")
        location_data = extract_location_data(location_files)

        status.info("Matching activities with location data...")
        activities_with_routes = match_activities_with_location(
            activities, location_data)

        status.info("Preprocessing data...")
        activities_df = preprocess_data(activities_with_routes)

        status.success("Data loaded successfully!")

        # Create the dashboard with the data
        create_dashboard(activities_df)
    else:
        # Show instructions for first-time users
        st.markdown("""
        ## Welcome to the Activity Dashboard!
        
        This dashboard will help you visualize and analyze your fitness activities 
        from Google Takeout data, including:
        
        - **BIKING** activities
        - **WALKING** activities
        - **RUNNING** activities
        
        To get started:
        1. Make sure your Takeout data is extracted
        2. Check the directory path in the sidebar
        3. Click "Load Data" to process your activities
        
        The dashboard will show you patterns in your activities, including:
        - Activity distribution by type
        - Monthly trends
        - Time of day analysis
        - Day of week patterns
        - Activity routes on a map (when GPS data is available)
        """)


def find_google_maps_history(base_dir='.'):
    """
    Find all Google Maps history files in the Takeout directory.

    Args:
        base_dir: Base directory to start searching from

    Returns:
        List of Google Maps history file paths
    """
    patterns = [
        f"{base_dir}/**/Takeout/**/Location History/**/*.json",
        f"{base_dir}/**/Takeout/Location History/**/*.json",
    ]

    history_files = []
    for pattern in patterns:
        history_files.extend(glob.glob(pattern, recursive=True))

    # Remove duplicates while preserving order
    history_files = list(dict.fromkeys(history_files))
    print(f"Found {len(history_files)} Google Maps history files")
    return history_files


if __name__ == "__main__":
    main()
