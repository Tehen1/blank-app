#!/usr/bin/env python3

import os
import json
import glob
import datetime
from datetime import timezone
import re

def find_files(pattern):
    """Find all files matching the given pattern in the directory structure."""
    return glob.glob(pattern, recursive=True)

def iso_to_nanos(iso_timestamp):
    """Convert ISO 8601 timestamp to nanoseconds since epoch."""
    # Parse the ISO timestamp
    # Handle timezone offset in the format +HH:MM or -HH:MM
    if 'Z' in iso_timestamp:
        dt = datetime.datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
    elif '+' in iso_timestamp or '-' in iso_timestamp:
        # Handle the specific format with underscore used in filenames
        if '_' in iso_timestamp:
            # Replace underscore with colon for proper ISO parsing
            iso_timestamp = re.sub(r'(\d{2})_(\d{2})', r'\1:\2', iso_timestamp)
        dt = datetime.datetime.fromisoformat(iso_timestamp)
    else:
        dt = datetime.datetime.fromisoformat(iso_timestamp)
    
    # Convert to nanoseconds
    seconds_since_epoch = dt.timestamp()
    return int(seconds_since_epoch * 1_000_000_000)

def extract_bike_activities():
    """Extract all biking activities and their metadata."""
    bike_files = find_files("./**/Fit/**/BIKING.json") + find_files("./**/Fit/**/*_BIKING.json")
    activities = []
    
    for file_path in bike_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract basic metadata
            activity = {
                'file_path': file_path,
                'start_time': data.get('startTime', ''),
                'end_time': data.get('endTime', ''),
                'duration': data.get('duration', ''),
                'start_time_nanos': iso_to_nanos(data.get('startTime', '')),
                'end_time_nanos': iso_to_nanos(data.get('endTime', '')),
                'route': []
            }
            
            activities.append(activity)
            print(f"Processed biking activity: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return activities

def extract_location_data():
    """Extract location data from sample files."""
    location_files = find_files("./**/derived_com.google.location.sample*.json")
    location_data = []
    
    for file_path in location_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract location samples from "Data Points" array
            if 'Data Points' in data:
                for point in data['Data Points']:
                    if 'fitValue' in point and len(point['fitValue']) >= 2:
                        # Extract latitude from first fpVal
                        lat = point['fitValue'][0]['value']['fpVal'] if isinstance(point['fitValue'][0], dict) and 'value' in point['fitValue'][0] else point['fitValue'][0]
                        # Extract longitude from second fpVal
                        lng = point['fitValue'][1]['value']['fpVal'] if isinstance(point['fitValue'][1], dict) and 'value' in point['fitValue'][1] else point['fitValue'][1]
                        
                        # Get timestamps
                        start_time = int(point.get('startTimeNanos', 0))
                        end_time = int(point.get('endTimeNanos', 0))
                        
                        # Use the average of start and end time
                        timestamp = (start_time + end_time) // 2
                        
                        # Extract accuracy if available (usually third element)
                        accuracy = 0
                        if len(point['fitValue']) > 2 and isinstance(point['fitValue'][2], dict) and 'value' in point['fitValue'][2]:
                            accuracy = point['fitValue'][2]['value']['fpVal']
                        elif len(point['fitValue']) > 2:
                            accuracy = point['fitValue'][2]
                        
                        location_data.append({
                            'lat': lat,
                            'lng': lng,
                            'timestamp_nanos': timestamp,
                            'accuracy': accuracy
                        })
            
            # Extract location samples from legacy format
            if 'locations' in data:
                for location in data['locations']:
                    if 'latitudeE7' in location and 'longitudeE7' in location:
                        # Convert E7 format (degrees * 10^7) to standard format
                        lat = location['latitudeE7'] / 10000000
                        lng = location['longitudeE7'] / 10000000
                        timestamp = int(location.get('timestampMs', 0)) * 1000000  # ms to ns
                        
                        location_data.append({
                            'lat': lat,
                            'lng': lng,
                            'timestamp_nanos': timestamp,
                            'accuracy': location.get('accuracy', 0)
                        })
                    # Also check for activity data format with fitValue array
                    elif 'activity' in location and 'point' in location:
                        for point in location.get('point', []):
                            if 'fitValue' in point and len(point['fitValue']) >= 2:
                                lat = point['fitValue'][0]
                                lng = point['fitValue'][1]
                                start_time = int(point.get('startTimeNanos', 0))
                                end_time = int(point.get('endTimeNanos', 0))
                                
                                # Use the average of start and end time
                                timestamp = (start_time + end_time) // 2
                                
                                location_data.append({
                                    'lat': lat,
                                    'lng': lng,
                                    'timestamp_nanos': timestamp,
                                    'accuracy': point.get('accuracy', 0) if 'accuracy' in point else 
                                                (point['fitValue'][2] if len(point['fitValue']) > 2 else 0)
                                })
            
            # Alternative format check
            if 'data' in data and 'point' in data['data']:
                for point in data['data']['point']:
                    if 'fitValue' in point and len(point['fitValue']) >= 2:
                        lat = point['fitValue'][0]
                        lng = point['fitValue'][1]
                        start_time = int(point.get('startTimeNanos', 0))
                        end_time = int(point.get('endTimeNanos', 0))
                        
                        # Use the average of start and end time
                        timestamp = (start_time + end_time) // 2
                        
                        location_data.append({
                            'lat': lat,
                            'lng': lng,
                            'timestamp_nanos': timestamp,
                            'accuracy': point.get('accuracy', 0) if 'accuracy' in point else 
                                        (point['fitValue'][2] if len(point['fitValue']) > 2 else 0)
                        })
            
            print(f"Processed location file: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Sort by timestamp
    location_data.sort(key=lambda x: x['timestamp_nanos'])
    return location_data

def match_locations_to_activities(activities, location_data):
    """Match location data to biking activities based on timestamps."""
    for activity in activities:
        start_nanos = activity['start_time_nanos']
        end_nanos = activity['end_time_nanos']
        
        # Find location points within the activity time range
        route_points = [
            point for point in location_data 
            if start_nanos <= point['timestamp_nanos'] <= end_nanos
        ]
        
        # Sort by timestamp to ensure correct order
        route_points.sort(key=lambda x: x['timestamp_nanos'])
        
        # Add route to activity
        activity['route'] = route_points
        activity['point_count'] = len(route_points)
        
        print(f"Matched {len(route_points)} locations to activity {activity['file_path']}")
    
    return activities

def main():
    print("Extracting bike activities...")
    activities = extract_bike_activities()
    
    print(f"Found {len(activities)} biking activities.")
    
    print("Extracting location data...")
    location_data = extract_location_data()
    
    print(f"Found {len(location_data)} location data points.")
    
    print("Matching locations to activities...")
    activities_with_routes = match_locations_to_activities(activities, location_data)
    
    # Write to output file
    output_file = "bike_activities_with_routes.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(activities_with_routes, f, indent=2)
    
    print(f"Output written to {output_file}")
    
    # Print statistics
    activities_with_routes_count = sum(1 for a in activities_with_routes if a['route'])
    total_points = sum(len(a['route']) for a in activities_with_routes)
    print(f"Statistics:")
    print(f"- Total activities: {len(activities_with_routes)}")
    print(f"- Activities with routes: {activities_with_routes_count}")
    print(f"- Total route points: {total_points}")

if __name__ == "__main__":
    main()

