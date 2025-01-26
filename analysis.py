import datetime

def analyze_frame_gaps(log_content):
    # Parse log lines into structured data
    frames = []
    for line in log_content.strip().split('\n'):
        parts = line.split(', ')
        frame_num = int(parts[0].split(' ')[1])
        timestamp = datetime.datetime.strptime(parts[1], '%Y-%m-%d %H:%M:%S')
        frames.append({
            'frame': frame_num,
            'time': timestamp,
            'ghost_protocol': int(parts[2].split(': ')[1]),
            'ghost_value': int(parts[3].split(': ')[1]),
            'or_count': int(parts[4].split(': ')[1]),
            'problem': int(parts[5].split(': ')[1])
        })

    # Find largest frame gap
    max_gap = {
        'start_frame': None,
        'end_frame': None,
        'gap_size': 0,
        'time_diff': None
    }

    for i in range(1, len(frames)):
        gap = frames[i]['frame'] - frames[i-1]['frame']
        time_diff = frames[i]['time'] - frames[i-1]['time']
        
        if gap > max_gap['gap_size']:
            max_gap = {
                'start_frame': frames[i-1]['frame'],
                'end_frame': frames[i]['frame'],
                'gap_size': gap,
                'time_diff': time_diff
            }

    return max_gap
with open("ack_stats.log", "r", encoding="utf-8") as f:
    log_data = f.read() 


result = analyze_frame_gaps(log_data)
print(f"Largest gap: {result['gap_size']} frames")
print(f"Between frames: {result['start_frame']} and {result['end_frame']}")
print(f"Time difference: {result['time_diff'].total_seconds()} seconds")
while True:
    False