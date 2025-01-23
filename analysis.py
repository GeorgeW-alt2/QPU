import matplotlib.pyplot as plt
import re

def plot_or_counts(log_file):
    frames = []
    or_counts = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Use regex to extract frame and OR count
            frame_match = re.search(r'Frame (\d+)', line)
            or_count_match = re.search(r'OR Count: (\d+)', line)
            
            if frame_match and or_count_match:
                frame = int(frame_match.group(1))
                or_count = int(or_count_match.group(1))
                
                frames.append(frame)
                or_counts.append(or_count)
    
    if not frames:
        print("No data extracted. Sample line:")
        with open(log_file, 'r') as f:
            print(f.readline().strip())
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(frames, or_counts, 'b-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Frame Number')
    plt.ylabel('OR Count')
    plt.title('OR Count vs Frame Number')
    
    plt.tight_layout()
    plt.show()

# Usage
plot_or_counts('ack_stats.log')