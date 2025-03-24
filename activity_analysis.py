import json
import datetime
import glob

# Define activity type mappings based on common Google Fit values
activity_types = {
    0: "Still (not moving)",
    1: "Walking",
    2: "Running",
    3: "Cycling",
    7: "Walking",
    8: "Running",
    9: "In vehicle"
    # There are more types but these are common ones
}

# List to store cycling activities
cycling_activities = []

# Count activities by type
activity_counts = {}

for file_path in glob.glob("./Takeout/Takeout 2/Fit/Toutes les données/derived_com.google.activity.segment_com.google*.json"):
    print(f"Processing file: {file_path}")
    with open(file_path, "r") as file:
        try:
            data = json.load(file)
            if "Data Points" in data:
                data_points = data["Data Points"]
                print(f"  Found {len(data_points)} data points")
                
                # Inspect the first data point
                if data_points and len(data_points) > 0:
                    print(f"  First data point keys: {list(data_points[0].keys())}")
                    
                    # Count activities by type
                    for point in data_points:
                        if "activityType" in point:
                            activity_type = point["activityType"]
                            if activity_type not in activity_counts:
                                activity_counts[activity_type] = 0
                            activity_counts[activity_type] += 1
                        
                        # Also look for cycling activities specifically
                        if "activityType" in point and point["activityType"] == 3:  # 3 is usually cycling
                            start_time_ms = point.get("startTimeMillis", 0)
                            end_time_ms = point.get("endTimeMillis", 0)
                            
                            if start_time_ms and end_time_ms:
                                start_time = datetime.datetime.fromtimestamp(int(start_time_ms) / 1000)
                                end_time = datetime.datetime.fromtimestamp(int(end_time_ms) / 1000)
                                duration_mins = (int(end_time_ms) - int(start_time_ms)) / 1000 / 60
                                
                                cycling_activities.append({
                                    "start_time": start_time,
                                    "end_time": end_time,
                                    "duration_mins": duration_mins,
                                    "file": file_path
                                })
                                
                                print(f"  Found cycling activity: {start_time.strftime('%Y-%m-%d %H:%M:%S')} - Duration: {duration_mins:.2f} minutes")
        except json.JSONDecodeError:
            print(f"  Error parsing {file_path}")

# Print summary of activity types
print("\nActivity Type Summary:")
for activity_type, count in sorted(activity_counts.items()):
    activity_name = activity_types.get(activity_type, f"Unknown ({activity_type})")
    print(f"  {activity_name}: {count} activities")

# Print cycling activity summary
if cycling_activities:
    print("\nCycling Activities Summary:")
    print(f"  Total cycling activities: {len(cycling_activities)}")
    
    # Sort by start time
    cycling_activities.sort(key=lambda x: x["start_time"])
    
    # Print first 10
    print("\nFirst 10 cycling activities:")
    for i, activity in enumerate(cycling_activities[:10]):
        print(f"  {i+1}. {activity['start_time'].strftime('%Y-%m-%d %H:%M:%S')} - Duration: {activity['duration_mins']:.2f} minutes")
    
    # Calculate stats
    total_duration = sum(a["duration_mins"] for a in cycling_activities)
    avg_duration = total_duration / len(cycling_activities)
    print(f"\n  Total cycling time: {total_duration:.2f} minutes ({total_duration/60:.2f} hours)")
    print(f"  Average ride duration: {avg_duration:.2f} minutes")
    
    # Get date range
    earliest = cycling_activities[0]["start_time"].strftime('%Y-%m-%d')
    latest = cycling_activities[-1]["start_time"].strftime('%Y-%m-%d')
    print(f"  Date range: {earliest} to {latest}")
else:
    print("\nNo cycling activities found.")
