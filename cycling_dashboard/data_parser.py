"""
Data Parser Module for Cycling Activity Analysis Dashboard

This module provides functionality to parse TCX (Training Center XML) files
containing cycling activity data, handling missing data fields gracefully
and returning structured data for analysis.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import xml.etree.ElementTree as ET
import xmltodict
from tcxparser import TCXParser

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_tcx(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract activity data from TCX files with error handling.
    
    Args:
        file_path: Path to the TCX file
        
    Returns:
        Dictionary with structured activity metrics or None if parsing fails
    """
    try:
        # Check if file exists and has content
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
            
        if os.path.getsize(file_path) == 0:
            logger.error(f"Empty file: {file_path}")
            return None
        
        # Use TCXParser for basic data extraction
        try:
            tcx = TCXParser(file_path)
            
            # Check if this is a cycling activity
            activity_type = tcx.activity_type
            if activity_type and "bike" not in activity_type.lower() and "cycling" not in activity_type.lower():
                logger.info(f"Skipping non-cycling activity: {activity_type} in {file_path}")
                return None
                
        except Exception as e:
            logger.warning(f"TCXParser failed, falling back to manual parsing: {str(e)}")
            # If TCXParser fails, we'll continue with manual parsing below
            tcx = None
        
        # Parse XML for more detailed data extraction
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
                
            # Parse XML to dict for easier data extraction
            data_dict = xmltodict.parse(xml_content)
            
            # Fall back to ElementTree if xmltodict fails
            if not data_dict:
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # Extract namespaces if present
                namespaces = {'ns': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}
                
                # Use ElementTree to extract basic data
                # This is a fallback if other methods fail
        except Exception as e:
            logger.error(f"Failed to parse XML content: {str(e)}")
            return None
            
        # Initialize result dictionary with default values
        result = {
            'date': None,
            'duration': 0,
            'distance': 0,
            'elevation_gain': 0,
            'average_speed': 0,
            'max_speed': 0,
            'calories': 0,
            'coordinates': [],
            'speed_data': [],
            'elevation_data': [],
            'heart_rate_data': [],
            'cadence_data': [],
            'file_path': file_path,
            'activity_type': 'cycling'
        }
        
        # Extract basic information using TCXParser if available
        if tcx:
            try:
                result['date'] = tcx.started_at
            except:
                # Try to extract date from filename if available
                try:
                    filename = os.path.basename(file_path)
                    date_part = filename.split('_')[0]
                    result['date'] = datetime.strptime(date_part, '%Y-%m-%d')
                except:
                    logger.warning(f"Could not extract date from file: {file_path}")
            
            try:
                result['duration'] = tcx.duration
            except:
                logger.warning(f"Could not extract duration from file: {file_path}")
                
            try:
                result['distance'] = tcx.distance
            except:
                logger.warning(f"Could not extract distance from file: {file_path}")
                
            try:
                hr_data = tcx.hr_values
                if hr_data:
                    result['heart_rate_data'] = hr_data
            except:
                logger.warning(f"Could not extract heart rate data from file: {file_path}")
                
            try:
                if hasattr(tcx, 'altitude_points'):
                    result['elevation_data'] = tcx.altitude_points
            except:
                logger.warning(f"Could not extract elevation data from file: {file_path}")
                
            # Extract coordinates
            try:
                positions = tcx.position_values
                if positions:
                    result['coordinates'] = positions
            except:
                logger.warning(f"Could not extract GPS coordinates from file: {file_path}")
                
        # Process track points for detailed data
        try:
            trackpoints = _extract_trackpoints(data_dict)
            if trackpoints:
                # Process track points to extract time series data
                time_series_data = _process_trackpoints(trackpoints)
                
                # Extract elevation gain
                if time_series_data.get('elevation', []):
                    elevation_data = time_series_data['elevation']
                    result['elevation_data'] = elevation_data
                    result['elevation_gain'] = _calculate_elevation_gain(elevation_data)
                
                # Extract coordinates if not already set
                if not result['coordinates'] and time_series_data.get('coordinates', []):
                    result['coordinates'] = time_series_data['coordinates']
                    
                # Extract speed data
                if time_series_data.get('speed', []):
                    speed_data = time_series_data['speed']
                    result['speed_data'] = speed_data
                    
                    # Calculate average and max speed
                    valid_speeds = [s for s in speed_data if s is not None and s > 0]
                    if valid_speeds:
                        result['average_speed'] = sum(valid_speeds) / len(valid_speeds)
                        result['max_speed'] = max(valid_speeds)
                
                # Extract cadence data
                if time_series_data.get('cadence', []):
                    result['cadence_data'] = time_series_data['cadence']
                
                # Extract heart rate data if not already set
                if not result['heart_rate_data'] and time_series_data.get('heart_rate', []):
                    result['heart_rate_data'] = time_series_data['heart_rate']
        except Exception as e:
            logger.warning(f"Failed to process trackpoints: {str(e)}")
        
        # Clean up and validate result
        for key, value in result.items():
            if value is None:
                if key in ['distance', 'duration', 'elevation_gain', 'average_speed', 'max_speed', 'calories']:
                    result[key] = 0
                elif key in ['coordinates', 'speed_data', 'elevation_data', 'heart_rate_data', 'cadence_data']:
                    result[key] = []
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to parse TCX file: {file_path}, Error: {str(e)}")
        return None

def _extract_trackpoints(data_dict: Dict) -> List[Dict]:
    """Extract trackpoints from the TCX data structure"""
    try:
        # Navigate through the dictionary structure to find trackpoints
        # The structure might vary based on the TCX file format
        activities = data_dict.get('TrainingCenterDatabase', {}).get('Activities', {})
        
        if not activities:
            return []
            
        activity = activities.get('Activity', [])
        
        # Handle the case where there's only one activity (not in a list)
        if isinstance(activity, dict):
            activity = [activity]
            
        trackpoints = []
        
        for act in activity:
            laps = act.get('Lap', [])
            
            # Handle the case where there's only one lap (not in a list)
            if isinstance(laps, dict):
                laps = [laps]
                
            for lap in laps:
                lap_trackpoints = lap.get('Track', {}).get('Trackpoint', [])
                
                # Handle the case where there's only one trackpoint (not in a list)
                if isinstance(lap_trackpoints, dict):
                    lap_trackpoints = [lap_trackpoints]
                    
                trackpoints.extend(lap_trackpoints)
                
        return trackpoints
    except Exception as e:
        logger.warning(f"Failed to extract trackpoints: {str(e)}")
        return []

def _process_trackpoints(trackpoints: List[Dict]) -> Dict[str, List]:
    """Process trackpoints to extract time series data"""
    result = {
        'time': [],
        'coordinates': [],
        'elevation': [],
        'heart_rate': [],
        'cadence': [],
        'speed': []
    }
    
    for point in trackpoints:
        try:
            # Extract time
            if 'Time' in point:
                time_str = point['Time']
                try:
                    # Try to parse time string to datetime
                    time = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%SZ')
                    result['time'].append(time)
                except:
                    result['time'].append(None)
            else:
                result['time'].append(None)
                
            # Extract position
            if 'Position' in point:
                pos = point['Position']
                lat = float(pos.get('LatitudeDegrees', 0))
                lon = float(pos.get('LongitudeDegrees', 0))
                result['coordinates'].append((lat, lon))
            else:
                result['coordinates'].append(None)
                
            # Extract altitude
            if 'AltitudeMeters' in point:
                try:
                    alt = float(point['AltitudeMeters'])
                    result['elevation'].append(alt)
                except:
                    result['elevation'].append(None)
            else:
                result['elevation'].append(None)
                
            # Extract heart rate
            if 'HeartRateBpm' in point:
                try:
                    hr = int(point['HeartRateBpm'].get('Value', 0))
                    result['heart_rate'].append(hr)
                except:
                    result['heart_rate'].append(None)
            else:
                result['heart_rate'].append(None)
                
            # Extract cadence
            if 'Cadence' in point:
                try:
                    cadence = int(point['Cadence'])
                    result['cadence'].append(cadence)
                except:
                    result['cadence'].append(None)
            else:
                result['cadence'].append(None)
                
            # Extract speed (might be in Extensions)
            if 'Extensions' in point:
                try:
                    extensions = point['Extensions']
                    if 'TPX' in extensions:
                        tpx = extensions['TPX']
                        if 'Speed' in tpx:
                            speed = float(tpx['Speed'])
                            result['speed'].append(speed)
                        else:
                            result['speed'].append(None)
                    else:
                        result['speed'].append(None)
                except:
                    result['speed'].append(None)
            else:
                result['speed'].append(None)
                
        except Exception as e:
            logger.warning(f"Failed to process trackpoint: {str(e)}")
            # Append None to all lists to maintain data alignment
            for key in result:
                if len(result[key]) < len(trackpoints):
                    result[key].append(None)
    
    return result

def _calculate_elevation_gain(elevation_data: List[float]) -> float:
    """Calculate total elevation gain from elevation data points"""
    if not elevation_data or len(elevation_data) < 2:
        return 0
    
    # Filter out None values
    valid_elevations = [e for e in elevation_data if e is not None]
    
    if len(valid_elevations) < 2:
        return 0
    
    # Calculate elevation gain (only count positive changes)
    elevation_gain = 0
    for i in range(1, len(valid_elevations)):
        diff = valid_elevations[i] - valid_elevations[i-1]
        if diff > 0:
            elevation_gain += diff
    
    return elevation_gain

def get_activity_files(directory_path: str, ext: str = '.tcx') -> List[str]:
    """
    Scan a directory for TCX files.
    
    Args:
        directory_path: Path to the directory containing activity files
        ext: File extension to look for (default: '.tcx')
        
    Returns:
        List of paths to TCX files
    """
    tcx_files = []
    
    try:
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return []
            
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(ext.lower()):
                    tcx_files.append(os.path.join(root, file))
                    
        logger.info(f"Found {len(tcx_files)} {ext} files in {directory_path}")
        return tcx_files
    
    except Exception as e:
        logger.error(f"Error scanning directory {directory_path}: {str(e)}")
        return []

if __name__ == "__main__":
    # Example usage
    test_dir = "./Takeout 2/Takeout/Fit/Activités"
    files = get_activity_files(test_dir)
    
    if files:
        print(f"Testing parser with first file: {files[0]}")
        activity_data = parse_tcx(files[0])
        
        if activity_data:
            print("Successfully parsed activity data:")
            for key, value in activity_data.items():
                if isinstance(value, list):
                    print(f"{key}: {len(value)} data points")
                else:
                    print(f"{key}: {value}")
        else:
            print("Failed to parse activity data")
    else:
        print("No TCX files found for testing")

