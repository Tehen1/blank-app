import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from data_parser import parse_tcx

def scan_for_cycling_files(data_dir: str) -> List[str]:
    """
    Scan the specified directory for TCX files that contain cycling activities.
    
    Args:
        data_dir: Path to directory containing TCX files
        
    Returns:
        List of file paths for cycling activities
    """
    cycling_files = []
    
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory '{data_dir}' does not exist")
        return cycling_files
        
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.tcx'):
                file_path = os.path.join(root, file)
                try:
                    # Check if this is a cycling activity
                    activity_data = parse_tcx(file_path)
                    if activity_data.get('sport') == 'Biking':
                        cycling_files.append(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    return cycling_files

def extract_coordinates(track_points: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    """
    Extract latitude and longitude coordinates from track points.
    
    Args:
        track_points: List of track point dictionaries
    
    Returns:
        List of (latitude, longitude) tuples
    """
    coordinates = []
    for point in track_points:
        if 'position' in point and 'latitude' in point['position'] and 'longitude' in point['position']:
            lat = point['position']['latitude']
            lon = point['position']['longitude']
            coordinates.append((lat, lon))
    return coordinates

def extract_speed_data(track_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract time and speed data from track points for time-series analysis.
    
    Args:
        track_points: List of track point dictionaries
    
    Returns:
        List of dictionaries with time and speed information
    """
    speed_data = []
    for point in track_points:
        time_str = point.get('time')
        speed = point.get('speed')
        
        if time_str and speed is not None:
            try:
                # Parse ISO-format timestamp
                timestamp = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                speed_data.append({
                    'timestamp': timestamp,
                    'speed': float(speed)
                })
            except (ValueError, TypeError) as e:
                print(f"Error parsing time or speed: {e}")
    
    return speed_data

def calculate_elevation_gain(track_points: List[Dict[str, Any]]) -> float:
    """
    Calculate total elevation gain from track points.
    
    Args:
        track_points: List of track point dictionaries
    
    Returns:
        Total elevation gain in meters
    """
    elevation_gain = 0.0
    prev_altitude = None
    
    for point in track_points:
        altitude = point.get('altitude')
        if altitude is not None:
            altitude = float(altitude)
            if prev_altitude is not None and altitude > prev_altitude:
                elevation_gain += (altitude - prev_altitude)
            prev_altitude = altitude
    
    return elevation_gain

def build_activities_dataframe(data_dir: str) -> pd.DataFrame:
    """
    Build a DataFrame with cycling activity data.
    
    Args:
        data_dir: Path to directory containing TCX files
    
    Returns:
        DataFrame with columns ['date', 'duration', 'distance', 'elevation_gain', 'coordinates', 'speed_data']
    """
    cycling_files = scan_for_cycling_files(data_dir)
    activities_data = []
    
    for file_path in cycling_files:
        try:
            activity = parse_tcx(file_path)
            
            # Skip if required data is missing
            if not all(k in activity for k in ['start_time', 'duration', 'distance', 'track_points']):
                print(f"Skipping {file_path}: Missing required data")
                continue
                
            # Calculate elevation gain
            elevation_gain = calculate_elevation_gain(activity['track_points'])
            
            # Extract coordinates for mapping
            coordinates = extract_coordinates(activity['track_points'])
            
            # Extract speed data for time-series analysis
            speed_data = extract_speed_data(activity['track_points'])
            
            activities_data.append({
                'file_path': file_path,
                'date': activity['start_time'],
                'duration': activity['duration'],  # in seconds
                'distance': activity['distance'],  # in meters
                'elevation_gain': elevation_gain,  # in meters
                'coordinates': coordinates,        # list of (lat, lon) tuples
                'speed_data': speed_data,          # list of dicts with timestamp and speed
                'track_points': activity['track_points']  # full track point data
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Create DataFrame and sort by date
    df = pd.DataFrame(activities_data)
    if not df.empty:
        df = df.sort_values('date', ascending=False).reset_index(drop=True)
    
    return df

def flatten_track_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten track points data for time-series analysis.
    
    Args:
        df: DataFrame with activity data including track_points column
    
    Returns:
        DataFrame with flattened track points, each row representing a single track point
    """
    flattened_data = []
    
    for _, row in df.iterrows():
        activity_id = row.name
        for point in row['track_points']:
            point_data = {
                'activity_id': activity_id,
                'activity_date': row['date'],
            }
            
            # Add track point data
            if 'time' in point:
                try:
                    point_data['timestamp'] = datetime.fromisoformat(point['time'].replace('Z', '+00:00'))
                except ValueError:
                    point_data['timestamp'] = None
            
            # Add other metrics
            for metric in ['altitude', 'distance', 'speed', 'heart_rate']:
                point_data[metric] = point.get(metric)
            
            # Add position if available
            if 'position' in point:
                point_data['latitude'] = point['position'].get('latitude')
                point_data['longitude'] = point['position'].get('longitude')
            
            flattened_data.append(point_data)
    
    # Create DataFrame from flattened data
    flat_df = pd.DataFrame(flattened_data)
    
    # Sort by activity_id and timestamp
    if not flat_df.empty and 'timestamp' in flat_df.columns:
        flat_df = flat_df.sort_values(['activity_id', 'timestamp']).reset_index(drop=True)
    
    return flat_df

def process_cycling_data(data_dir: str = "./Takeout 2/Takeout/Fit/Activités") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to process cycling activity data.
    
    Args:
        data_dir: Path to directory containing TCX files
        
    Returns:
        Tuple of (activities_df, flattened_df) where:
            - activities_df: DataFrame with one row per activity
            - flattened_df: DataFrame with flattened track points for time-series analysis
    """
    print(f"Scanning for cycling activities in: {data_dir}")
    
    # Build DataFrame with activity data
    activities_df = build_activities_dataframe(data_dir)
    print(f"Found {len(activities_df)} cycling activities")
    
    # Create flattened DataFrame for time-series analysis
    flattened_df = flatten_track_points(activities_df)
    print(f"Created flattened dataset with {len(flattened_df)} track points")
    
    return activities_df, flattened_df


if __name__ == "__main__":
    # Example usage
    activities_df, flattened_df = process_cycling_data()
    
    if not activities_df.empty:
        print("\nActivity Summary:")
        summary = activities_df[['date', 'duration', 'distance', 'elevation_gain']].copy()
        summary['duration_min'] = summary['duration'] / 60  # Convert to minutes
        summary['distance_km'] = summary['distance'] / 1000  # Convert to kilometers
        print(summary.head())
    else:
        print("No cycling activities found")

