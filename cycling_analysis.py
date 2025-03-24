import json
import datetime
import glob

cycling_activities = []

for file_path in glob.glob("./Takeout/Takeout 2/Fit/Toutes les données/derived_com.google.activity.segment_com.google*.json"):
    with open(file_path, "r") as file:
        try:
            data = json.load(file)
            for entry in data:
                if "fitValue" in entry and entry["fitValue"][0]["value"].get("intVal") == 1:
                    start_time = datetime.datetime.fromtimestamp(entry["startTimeNanos"] / 1e9)
                    end_time = datetime.datetime.fromtimestamp(entry["endTimeNanos"] / 1e9)
                    duration_minutes = (entry["endTimeNanos"] - entry["startTimeNanos"]) / 1e9 / 60
                    
                    cycling_activities.append({
                        "file": file_path.split("/")[-1],
                        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "duration_minutes": round(duration_minutes, 2)
                    })
        except json.JSONDecodeError:
            print(f"Error parsing {file_path}")
            
# Sort by start time
cycling_activities.sort(key=lambda x: x["start_time"])

# Print summary
print(f"Total cycling activities found: {len(cycling_activities)}")
print("\nFirst 10 cycling activities:")
for i, activity in enumerate(cycling_activities[:10]):
    print(f"{i+1}. From {activity['start_time']} to {activity['end_time']} - Duration: {activity['duration_minutes']} mins")

# Print stats
if cycling_activities:
    total_duration = sum(activity["duration_minutes"] for activity in cycling_activities)
    avg_duration = total_duration / len(cycling_activities)
    print(f"\nTotal cycling time: {total_duration:.2f} minutes ({total_duration/60:.2f} hours)")
    print(f"Average ride duration: {avg_duration:.2f} minutes")
    
    # Get date range
    earliest = cycling_activities[0]["start_time"].split()[0]
    latest = cycling_activities[-1]["start_time"].split()[0]
    print(f"Date range: {earliest} to {latest}")
