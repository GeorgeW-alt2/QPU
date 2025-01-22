import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import re

def extract_ack_refresh(log_line):
    """Extract ACK/Refresh value from a log line."""
    match = re.search(r'ACKs/Refresh: ([\d.]+)', log_line)
    if match:
        return float(match.group(1))
    return None

def calculate_entropy(values):
    """Calculate entropy from a list of values."""
    # Create histogram
    hist, _ = np.histogram(values, bins=30, density=True)
    
    # Normalize histogram to get probabilities
    hist_normalized = hist / np.sum(hist)
    
    # Calculate entropy (using base 2 for bits)
    return entropy(hist_normalized, base=2)

def analyze_ack_refresh_entropy(filename='ack_stats.log'):
    """Analyze entropy of ACK/Refresh values from log file."""
    # Read and extract ACK/Refresh values
    ack_values = []
    with open(filename, 'r') as f:
        for line in f:
            value = extract_ack_refresh(line)
            if value is not None:
                ack_values.append(value)
    
    ack_values = np.array(ack_values)
    
    # Calculate entropy
    entropy_value = calculate_entropy(ack_values)
    
    # Calculate basic statistics
    stats = {
        'entropy': entropy_value,
        'mean': np.mean(ack_values),
        'std': np.std(ack_values),
        'min': np.min(ack_values),
        'max': np.max(ack_values),
        'count': len(ack_values)
    }
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot histogram
    hist, bins, _ = plt.hist(ack_values, bins=30, density=True, alpha=0.7, 
                            color='skyblue', edgecolor='black')
    
    # Add title and labels
    plt.title(f"Distribution of ACK/Refresh Values\nEntropy: {entropy_value:.4f} bits", 
             pad=20)
    plt.xlabel("ACK/Refresh")
    plt.ylabel("Probability Density")
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = (f"Entropy: {stats['entropy']:.4f} bits\n"
                 f"Mean: {stats['mean']:.4f}\n"
                 f"Std Dev: {stats['std']:.4f}\n"
                 f"Min: {stats['min']:.4f}\n"
                 f"Max: {stats['max']:.4f}\n"
                 f"Sample Size: {stats['count']}")
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='white', alpha=0.8))
    
    # Save plot
    plt.savefig('ack_entropy_analysis.png')
    plt.close()
    
    return stats

if __name__ == "__main__":
    try:
        stats = analyze_ack_refresh_entropy()
        print("\nACK/Refresh Analysis Results:")
        print("-" * 30)
        print(f"Entropy: {stats['entropy']:.4f} bits")
        print(f"Mean: {stats['mean']:.4f}")
        print(f"Standard Deviation: {stats['std']:.4f}")
        print(f"Range: {stats['min']:.4f} - {stats['max']:.4f}")
        print(f"Sample Size: {stats['count']}")
        print("\nVisualization saved as 'ack_entropy_analysis.png'")
        
    except FileNotFoundError:
        print("Error: ack_stats.log file not found!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")