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
    def plot_ack_data(self):
        """Plot the ACK and ACK/Second data."""
        plt.figure(figsize=(10, 6))

        # Plot ACK data
        plt.subplot(2, 1, 1)
        plt.plot(self.ack_data, label="ACK/Refresh Data", color="blue")
        plt.title("ACK/Refresh Data Over Time")
        plt.xlabel("Frame Number")
        plt.ylabel("Count")
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    def analyze_ack_rate(self):
        """Calculate ACK rate with quantum-inspired filtering"""
        current_time = datetime.now()
        elapsed_time = (current_time - self.start_time).total_seconds()
        
        # Calculate basic rates
        ack_delta = self.ack - self.last_ack_count
        refreshes = elapsed_time / self.status_update_interval
        
        # Initialize wave packet parameters if not exist
        if not hasattr(self, 'wave_history'):
            self.wave_history = deque(maxlen=20)  # Quantum well width
            self.uncertainty_factor = 0.3  # Heisenberg uncertainty parameter
            self.well_depth = 2.0  # Potential well depth
            
        current_acks_per_second = ack_delta / elapsed_time if elapsed_time > 0 else 0
        self.wave_history.append(current_acks_per_second)
        
        if len(self.wave_history) >= 3:
            # Apply quantum well filtering
            filtered_rate = self._quantum_filter(list(self.wave_history))
        else:
            filtered_rate = current_acks_per_second
        
        # Track oscillation phases for further filtering
        if not hasattr(self, 'phase_history'):
            self.phase_history = deque(maxlen=10)
        
        if len(self.wave_history) >= 2:
            phase = np.arctan2(current_acks_per_second - filtered_rate, 
                              self.uncertainty_factor)
            self.phase_history.append(phase)
        
        stats = {
            'acks_per_refresh': round(ack_delta / refreshes if refreshes > 0 else 0, 2),
            'acks_per_second': round(filtered_rate, 2),
            'raw_acks_per_second': round(current_acks_per_second, 2),
            'total_acks': self.ack,
            'ack_delta': ack_delta,
            'elapsed_time': round(elapsed_time, 2),
            'uncertainty': self.uncertainty_factor
        }
        
        self.last_ack_count = self.ack
        self.ack_history.append(stats)
        return stats

    def _quantum_filter(self, wave_data):
        """Apply quantum-inspired filtering to the signal"""
        if len(wave_data) < 3:
            return wave_data[-1]
        
        # Calculate wave packet spread
        mean = np.mean(wave_data)
        std = np.std(wave_data)
        
        # Adjust uncertainty based on signal variation
        self.uncertainty_factor = min(0.5, std / (mean + 1e-6))
        
        # Apply potential well damping
        weights = np.exp(-np.abs(np.array(range(len(wave_data))) - 
                                len(wave_data)) / self.well_depth)
        weights = weights / np.sum(weights)
        
        # Filter signal
        filtered_value = np.sum(weights * wave_data)
        
        return filtered_value

    def find_flattest_region(self, data, window_size=20):
        """Find the flattest contiguous region in the data"""
        if len(data) < window_size:
            return 0, len(data)
        
        min_variance = float('inf')
        start_idx = 0
        
        # Slide window through data to find minimum variance section
        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            variance = np.var(window)
            
            if variance < min_variance:
                min_variance = variance
                start_idx = i
                
        return start_idx, start_idx + window_size

    def plot_ack_data(self):
        """Plot ACK data with highlighted flattest region"""
        plt.figure(figsize=(12, 8))
        
        # Get data for plotting
        raw_data = [stats['raw_acks_per_second'] for stats in self.ack_history]
        filtered_data = [stats['acks_per_second'] for stats in self.ack_history]
        times = range(len(filtered_data))
        
        # Calculate uncertainty bands
        if hasattr(self, 'uncertainty_factor'):
            uncertainty = self.uncertainty_factor * np.array(filtered_data)
            upper_band = np.array(filtered_data) + uncertainty
            lower_band = np.array(filtered_data) - uncertainty
            
            # Find flattest region in the uncertainty band difference
            band_diff = upper_band - lower_band
            flat_start, flat_end = self.find_flattest_region(band_diff)
            
            # Plot base data
            plt.plot(raw_data, label="Raw Signal", color="gray", alpha=0.4)
            plt.fill_between(times, lower_band, upper_band, 
                            color="blue", alpha=0.2, label="Uncertainty Band")
            
            # Highlight flattest region
            plt.axvspan(flat_start, flat_end, color='green', alpha=0.2, 
                       label=f'Flattest Region (var={np.var(band_diff[flat_start:flat_end]):.6f})')
            
            # Add annotations
            plt.annotate(f'Start: {flat_start}', 
                        xy=(flat_start, np.mean([upper_band[flat_start], lower_band[flat_start]])),
                        xytext=(10, 30), textcoords='offset points',
                        arrowprops=dict(arrowstyle="->"))
            plt.annotate(f'End: {flat_end}', 
                        xy=(flat_end, np.mean([upper_band[flat_end-1], lower_band[flat_end-1]])),
                        xytext=(10, -30), textcoords='offset points',
                        arrowprops=dict(arrowstyle="->"))
        
        plt.title("ACK Rates with Flattest Region Highlighted")
        plt.xlabel("Time Steps")
        plt.ylabel("ACKs per Second")
        plt.legend(loc='upper left')
        plt.grid(True)
        
        # Add statistical information
        if hasattr(self, 'uncertainty_factor'):
            flat_region_stats = f"Flattest Region Stats:\nStart: {flat_start}\nEnd: {flat_end}\n" \
                               f"Length: {flat_end - flat_start}\n" \
                               f"Variance: {np.var(band_diff[flat_start:flat_end]):.6f}"
            plt.text(0.02, 0.98, flat_region_stats,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

    def clear_console(self):
        """Clear the console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def log_ack_stats(self, stats):
        """Log ACK statistics and ghost protocol messages to a file"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = (
            f"Frame: {self.i}, "  # Add frame number
            f"{current_time}, "
            f"ACKs/Refresh: {stats['acks_per_refresh']}, "
            f"ACKs/Second: {stats['acks_per_second']}, "
            f"Total ACKs: {stats['total_acks']}, "
            f"Delta: {stats['ack_delta']}, "
            f"Elapsed: {stats['elapsed_time']}s, "
            f"Ghost Protocol: {self.ghostprotocol}, "
            f"Ghost Value: {self.ghostprotocol * self.range}, "
            f"PIN: {self.PIN}"
        )
        self.i += 1
        self.ack_data.append(stats['acks_per_refresh'])
        self.ack_data.append(stats['acks_per_second'])
        if self.last_or_state_time:
            or_duration = (datetime.now() - self.last_or_state_time).total_seconds()
            log_entry += f", OR Duration: {or_duration:.2f}s"
        
        if self.last_and_state_time:
            and_duration = (datetime.now() - self.last_and_state_time).total_seconds()
            log_entry += f", AND Duration: {and_duration:.2f}s"
        
        log_entry += "\n"
        
        with open("ack_stats.log", "a") as f:
            f.write(log_entry)
        
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
            self.log_ack_stats(self.analyze_ack_rate())
            
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
        """Check and process quantum states with harmony tracking"""
        check = self.numa.split(",")
        current_time = datetime.now()
        
        # Initialize harmony tracking if not exists
        if not hasattr(self, 'harmony_periods'):
            self.harmony_periods = []
            self.harmony_start = None
            self.current_harmonies = 0
        
        # Check for quantum harmony
        if self.cyc < len(check):
            is_harmonized = check[self.cyc] == str(self.qu)
            
            if is_harmonized:
                if self.harmony_start is None:
                    self.harmony_start = self.cyc
                self.current_harmonies += 1
            else:
                if self.harmony_start is not None and self.current_harmonies > 7 and self.current_harmonies < 9:
                    # Record completed harmony period
                    self.harmony_periods.append({
                        'start': self.harmony_start,
                        'end': self.cyc - 1,
                        'duration': self.current_harmonies,
                        'time': current_time.strftime('%H:%M:%S')
                    })
                    print(f"Signature detected: {self.harmony_start*self.range}"
                          f"(Duration: {self.current_harmonies} cycles), PIN: {self.PIN}")
                self.harmony_start = None
                self.current_harmonies = 0
        
        # Process OR states
        if self.or_count > self.corr and self.cyc < len(check):
            if self.last_or_state_time is None:
                self.last_or_state_time = current_time
                self.ghost_messages.append(f"OR state initiated at {current_time.strftime('%H:%M:%S')}")
            
            or_duration = (current_time - self.last_or_state_time).total_seconds()
            
            if check[self.cyc] != str(self.qu):
                if self.swi == self.longcyc:
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
                    message = f"Prolonged OR state: Duration {or_duration:.2f}s, Value: {self.qu}"
                    self.ghost_messages.append(message)
                
                self.process_ghost_protocol()

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
        #communicator.plot_ack_data()
        input()
        print("Shutdown complete.")