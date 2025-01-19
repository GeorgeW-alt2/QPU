import cv2
import numpy as np
import os
from datetime import datetime
from collections import deque
import random
import time
import matplotlib.pyplot as plt
PIN = 26000

class QuantumCommunicator:
    def __init__(self, sensitivity):
        # Camera and processing setup
        self.sensitivity = sensitivity
        self.data2 = None
        self.capture = cv2.VideoCapture(0)
        
        self.ack_data = []  # To store ACK/Refresh data
        self.ack_second_data = []  # To store ACK/Second data
        self.i = 0
        # Initialize quantum state variables
        self.Do = 1
        self.Do2 = 0
        self.qu = 0
        self.it = 0
        self.and_count = 0
        self.or_count = 0
        self.cyc = 0
        self.swi = 0
        self.longcyc = 3
        # Initialize the seed using the current time
        seed = int(time.time())
        random.seed(seed)
        self.numa = ",".join(str(np.random.randint(0, 2)) for _ in range(100000))
        self.corr = 3
        self.prime = 0
        self.ghostprotocol = 3000
        self.ghostprotocollast = 0
        self.GhostIterate = 0
        self.testchecknum = 5
        self.PIN = random.randint(5000, PIN) #Guess PIN, i.e max range 10000
        # ACK and status tracking
        self.ack = 0
        self.nul = 0
        self.last_ack_count = 0
        self.start_time = datetime.now()
        self.ack_history = []
        
        # Status tracking variables
        self.motion_frame_count = 0
        self.active_quadrants = set()
        self.last_status_update = datetime.now()
        self.status_update_interval = 0.5
        self.total_frames = 0
        
        # Ghost protocol variables
        self.ghost_messages = deque(maxlen=4)
        self.range = 10
        self.last_ghost_check = 0
        self.prime = 0
        self.prime_threshold = 3
        
        # OR state tracking
        self.or_state_duration = 0
        self.or_state_threshold = 3  # Number of consecutive seconds to trigger message
        self.last_or_state_time = None
        
        # AND state tracking
        self.and_state_duration = 0
        self.and_state_threshold = 3  # Number of consecutive seconds to trigger message
        self.last_and_state_time = None
    
    def analyze_ack_rate(self):
        """Calculate ACK rate statistics with baseline comparison"""
        current_time = datetime.now()
        elapsed_time = (current_time - self.start_time).total_seconds()
        
        # Calculate rates
        ack_delta = self.ack - self.last_ack_count
        refreshes = elapsed_time / self.status_update_interval
        current_acks_per_second = ack_delta / elapsed_time if elapsed_time > 0 else 0
        
        # Initialize or update baseline
        if not hasattr(self, 'baseline_ack_rate'):
            self.baseline_ack_rate = current_acks_per_second
            self.stable_periods = []
            self.current_stable_start = None
            self.baseline_buffer = []
        
        # Update moving baseline with decay
        decay = 0.55  # Slower baseline adaptation
        self.baseline_ack_rate = decay * self.baseline_ack_rate + (1 - decay) * current_acks_per_second
        
        # Track deviation from baseline
        deviation = abs(current_acks_per_second - self.baseline_ack_rate)
        stability_threshold = 0.2 * self.baseline_ack_rate  # 10% of baseline
        
        # Track stability periods
        if deviation <= stability_threshold:
            if self.current_stable_start is None:
                self.current_stable_start = self.i
        else:
            if self.current_stable_start is not None:
                if self.i - self.current_stable_start > 5:  # Minimum stable period
                    self.stable_periods.append((self.current_stable_start, self.i))
                self.current_stable_start = None
        
        stats = {
            'acks_per_refresh': round(ack_delta / refreshes if refreshes > 0 else 0, 2),
            'acks_per_second': round(current_acks_per_second, 2),
            'raw_acks_per_second': round(current_acks_per_second, 2),  # Added missing key
            'baseline_rate': round(self.baseline_ack_rate, 2),
            'deviation': round(deviation, 2),
            'total_acks': self.ack,
            'ack_delta': ack_delta,
            'elapsed_time': round(elapsed_time, 2)
        }
        
        self.last_ack_count = self.ack
        self.ack_history.append(stats)
        return stats

    def find_slope_stable_periods(self, data, window_size=3):
        """Find periods where rate changes with consistent slope"""
        stable_periods = []
        if len(data) < window_size:
            return stable_periods
        
        for i in range(len(data) - window_size):
            window = data[i:i + window_size]
            # Calculate slopes between consecutive points
            slopes = np.diff(window)
            # Check slope consistency (low variance in slopes)
            slope_variance = np.var(slopes)
            mean_slope = np.mean(slopes)
            
            # Detect consistent non-zero slope (steady increase/decrease)
            if slope_variance < 0.02 and abs(mean_slope) > 0.002:
                if not stable_periods or i > stable_periods[-1][1] + 2:
                    stable_periods.append([i, i + window_size, mean_slope])
                else:
                    # Extend existing period if slopes are similar
                    prev_slope = stable_periods[-1][2]
                    if abs(mean_slope - prev_slope) < 0.01:
                        stable_periods[-1][1] = i + window_size
                        stable_periods[-1][2] = (prev_slope + mean_slope) / 2
        
        return stable_periods

    def plot_ack_data(self):
        """Plot ACK data highlighting slope-stable regions"""
        plt.figure(figsize=(12, 8))
        
        # Get data for plotting
        refresh_data = [stats['acks_per_refresh'] for stats in self.ack_history]
        times = range(len(refresh_data))
        
        # Plot base data
        plt.plot(refresh_data, label="ACK/Refresh", color="gray", alpha=0.4)
        
        # Find and highlight slope-stable regions
        stable_periods = self.find_slope_stable_periods(refresh_data)
        
        # Color coding for slopes (red for increasing, blue for decreasing)
        for start, end, slope in stable_periods:
            color = 'red' if slope > 0 else 'blue'
            plt.axvspan(start, end, color=color, alpha=0.2)
            
            # Add slope annotation
            mid_point = (start + end) // 2
            plt.annotate(f'Slope: {slope:.3f}',
                        xy=(mid_point, refresh_data[mid_point]),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center',
                        bbox=dict(facecolor='white', alpha=0.7))
        
        plt.title("ACK/Refresh with Stable Slope Regions")
        plt.xlabel("Time Steps")
        plt.ylabel("ACK/Refresh Rate")
        plt.grid(True)
        plt.legend()
        
        # Add slope stability metrics
        if stable_periods:
            metrics = "Stable Slopes:\n"
            for i, (start, end, slope) in enumerate(stable_periods):
                direction = "increasing" if slope > 0 else "decreasing"
                metrics += f"Region {i+1}: {direction}\n"
                metrics += f"Length: {end-start}\n"
                metrics += f"Slope: {slope:.3f}\n"
            
            #plt.text(0.02, 0.98, metrics,
                    #transform=plt.gca().transAxes,
                    #verticalalignment='top',
                    #bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

    def log_ack_stats(self, stats):
        """Enhanced logging with both raw and smoothed ACK rates"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = (
            f"Frame: {self.i}, "
            f"{current_time}, "
            f"ACKs/Refresh: {stats['acks_per_refresh']}, "
            f"Raw ACKs/Second: {stats['raw_acks_per_second']}, "
            f"Smoothed ACKs/Second: {stats['acks_per_second']}, "
            f"Total ACKs: {stats['total_acks']}, "
            f"Delta: {stats['ack_delta']}, "
            f"Elapsed: {stats['elapsed_time']}s, "
            f"Ghost Protocol: {self.ghostprotocol}, "
            f"Ghost Value: {self.ghostprotocol * self.range}, "
            f"PIN: {self.PIN}"
        )
        
        self.i += 1
        
        if self.last_or_state_time:
            or_duration = (datetime.now() - self.last_or_state_time).total_seconds()
            log_entry += f", OR Duration: {or_duration:.2f}s"
        
        if self.last_and_state_time:
            and_duration = (datetime.now() - self.last_and_state_time).total_seconds()
            log_entry += f", AND Duration: {and_duration:.2f}s"
        
        log_entry += "\n"
        
        with open("ack_stats.log", "a") as f:
            f.write(log_entry)

    def clear_console(self):
        """Clear the console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def display_status(self):
        """Display current status information with ACK rate analysis"""
        current_time = datetime.now()
        if (current_time - self.last_status_update).total_seconds() < self.status_update_interval:
            return
            
        self.clear_console()
        ack_stats = self.analyze_ack_rate()

        print("=" * 50)
        print("QUANTUM COMMUNICATOR STATUS")
        print("=" * 50)
        print(f"Time: {current_time.strftime('%H:%M:%S')}")
        
        print(f"\nACK RATE ANALYSIS:")
        print(f"ACKs per Refresh: {ack_stats['acks_per_refresh']}")
        print(f"ACKs per Second: {ack_stats['acks_per_second']}")
        print(f"Total ACKs: {ack_stats['total_acks']}")
        print(f"Recent ACK Delta: {ack_stats['ack_delta']}")
        print(f"Elapsed Time: {ack_stats['elapsed_time']}s")
        
        print(f"\nQUANTUM STATES:")
        print(f"Current Quantum State (qu): {self.qu}")
        print(f"Cycle Position (cyc): {self.cyc}/{len(self.numa.split(','))}")
        print(f"Switch Counter (swi): {self.swi}/{self.longcyc}")
        
        print(f"\nDETECTION COUNTERS:")
        print(f"AND Gate Detections: {self.and_count}/{self.corr}")
        print(f"OR Gate Detections: {self.or_count}/{self.corr}")
        if self.last_or_state_time:
            or_duration = (current_time - self.last_or_state_time).total_seconds()
            print(f"Current OR State Duration: {or_duration:.2f}s")
        if self.last_and_state_time:
            and_duration = (current_time - self.last_and_state_time).total_seconds()
            print(f"Current AND State Duration: {and_duration:.2f}s")
        motion_percentage = (self.motion_frame_count / max(1, self.total_frames)) * 100
        print(f"Motion Detected: {self.motion_frame_count} frames ({motion_percentage:.1f}%)")
        
        print(f"\nPROTOCOL STATUS:")
        print(f"Ghost Protocol Value: {self.ghostprotocol * self.range}")
        print(f"Ghost Protocol State: {self.ghostprotocol * self.range}")
        print(f"Prime State: {self.prime}")
        print(f"Acknowledgments (ACK): {self.ack}")
        print(f"Nullifications (NUL): {self.nul}")
        
        print(f"\nGHOST PROTOCOL OUTPUT:")
        if self.ghost_messages:
            for msg in self.ghost_messages:
                print(msg)
        else:
            print("No ghost protocol messages yet")
        
        print("=" * 50)
        
        # Log ACK stats to file
        self.log_ack_stats(ack_stats)
        
        self.last_status_update = current_time
        self.active_quadrants.clear()

    def process_camera(self):
        """Process camera feed and detect motion in quadrants"""
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break
            
            self.total_frames += 1
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if self.data2 is None:
                self.data2 = gray
                continue
            
            self.process_motion(frame, gray)
            self.data2 = gray
            
            # Display status in command line
            self.display_status()
            
            # Display the resulting frame
            cv2.imshow('Motion Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.capture.release()
        cv2.destroyAllWindows()

    def process_motion(self, current_frame, gray_frame):
        """Process motion detection and quantum logic"""
        frame_delta = cv2.absdiff(self.data2, gray_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        motion_detected_in_frame = False
        
        height, width = thresh.shape
        quadrant_width = width // 16
        quadrant_height = height // 8
        
        for row in range(8):
            for col in range(16):
                x1 = col * quadrant_width
                y1 = row * quadrant_height
                x2 = (col + 1) * quadrant_width
                y2 = (row + 1) * quadrant_height
                
                quadrant = thresh[y1:y2, x1:x2]
                motion_detected = np.sum(quadrant > 10) > self.sensitivity
                
                if motion_detected:
                    motion_detected_in_frame = True
                    self.active_quadrants.add((row, col))
                    self.apply_quantum_logic(row, col)
                    self.highlight_quadrant(current_frame, x1, y1, x2, y2)
        
        if motion_detected_in_frame:
            self.motion_frame_count += 1

    def highlight_quadrant(self, frame, x1, y1, x2, y2):
        """Highlight a quadrant with motion"""
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    def apply_quantum_logic(self, b, bb):
        """Apply quantum state logic based on motion detection"""
        if self.Do == 1:
            self.Do2 = 1
            if self.qu == 1 and self.it == 0:
                self.qu = 0
                self.it += 1
            if self.qu == 0 and self.it == 0:
                self.qu = 1
                self.it += 1
            self.it = 0
            
        if 4 < b < 11 or 4 < bb < 11:
            self.or_count += 1
            
        if 4 < b < 11 and 4 < bb < 11:
            self.and_count += 1
            if self.Do == 1:
                self.toggle_quantum_state()
                
        self.check_quantum_states()

    def toggle_quantum_state(self):
        """Toggle quantum state based on current conditions"""
        if self.qu == 1 and self.it == 0:
            self.qu = 0
            self.it += 1
        elif self.qu == 0 and self.it == 0:
            self.qu = 1
            self.it += 1
        self.it = 0

    def check_quantum_states(self):
        """Check and process quantum states"""
        check = self.numa.split(",")
        current_time = datetime.now()
        
        # Process OR states
        if self.or_count > self.corr and self.cyc < len(check):
            if self.last_or_state_time is None:
                self.last_or_state_time = current_time
                self.ghost_messages.append(f"OR state initiated at {current_time.strftime('%H:%M:%S')}")
            
            or_duration = (current_time - self.last_or_state_time).total_seconds()
            
            if check[self.cyc] != str(self.qu):
                if self.swi == self.longcyc:
                    # Initialize the seed using the current time
                    seed = int(time.time())
                    random.seed(seed)
                    self.qu = np.random.randint(0, 2)
                    self.swi = 0
                self.swi += 1
                self.Do = 1
                self.nul += 1
                self.or_count = 0
                self.and_count = 0
                self.cyc += 1
                self.prime = min(self.prime + 1, self.prime_threshold)
                
                if or_duration >= self.or_state_threshold:
                    message = f"Prolonged OR state detected: Duration {or_duration:.2f}s, Value: {self.qu}"
                    self.ghost_messages.append(message)
                
                self.process_ghost_protocol()
        else:
            if self.last_or_state_time is not None:
                or_duration = (current_time - self.last_or_state_time).total_seconds()
                if or_duration >= self.or_state_threshold:
                    self.ghost_messages.append(f"OR state ended after {or_duration:.2f}s")
            self.last_or_state_time = None
        
        # Process AND states
        if self.and_count > self.corr and self.cyc < len(check):
            if self.last_and_state_time is None:
                self.last_and_state_time = current_time
                self.ghost_messages.append(f"AND state initiated at {current_time.strftime('%H:%M:%S')}")
            
            and_duration = (current_time - self.last_and_state_time).total_seconds()
            
            if check[self.cyc] == str(self.qu):
                if self.swi == self.longcyc:
                    # Initialize the seed using the current time
                    seed = int(time.time())
                    random.seed(seed)
                    self.qu = np.random.randint(0, 2)
                    self.swi = 0
                    self.prime = 0
                self.swi += 1
                self.Do = 1
                self.ack += 1
                self.and_count = 0
                self.cyc += 1
                if self.prime >= self.prime_threshold:
                    self.prime = 0
                else:
                    self.prime += 1
                
                if and_duration >= self.and_state_threshold:
                    message = f"Prolonged AND state detected: Duration {and_duration:.2f}s, Value: {self.qu}"
                    self.ghost_messages.append(message)
            else:
                self.last_and_state_time = None

    def process_ghost_protocol(self):
        """Process ghost protocol states"""
        current_value = self.ghostprotocol * self.range
        
        if self.prime < 1 and self.ghostprotocol > 3:
            if self.GhostIterate == 0:
                self.ghostprotocollast = current_value
                self.GhostIterate += 1
                current_time = datetime.now()
                message = f"Ghost Protocol Initiated: {self.ghostprotocol} (Value: {current_value}), Time: {current_time.strftime('%H:%M:%S')}"
                self.ghost_messages.append(message)
                self.last_ghost_check = current_value
                
                # Log initialization
                with open("ack_stats.log", "a") as f:
                    f.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')}, GHOST PROTOCOL INITIALIZED, Value: {current_value}\n")
            
            if current_value != self.ghostprotocollast:
                msg = f"Protocol state: {current_value}"
                self.ghost_messages.append(msg)
                self.ghostprotocollast = current_value
        self.ghostprotocol -= 1
        if self.ghostprotocol <= 0:
            exit()
    
def send_message(self):
        """Send a quantum message when conditions are met, could be a message or math."""
        problem = self.PIN # enter any math problem to solve
        if problem <= self.ghostprotocol * self.range :
            self.numa += ",".join('9' for _ in range(500)) #Paradox disruption

if __name__ == "__main__":
    try:
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        # Initialize the communicator
        communicator = QuantumCommunicator(sensitivity=1500)
        print("Quantum Communicator initialized. Starting camera feed...")
        
        # Start processing
        communicator.process_camera()
        
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Clean up
        if hasattr(communicator, 'capture'):
            communicator.capture.release()
        cv2.destroyAllWindows()
        communicator.plot_ack_data()

        print("Shutdown complete.")