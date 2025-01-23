import cv2
import numpy as np
import os
from datetime import datetime
import random
import time
import matplotlib.pyplot as plt
from collections import deque

PIN = random.randint(5000, 100000)
spin = 1  # 1 or -1

class QuantumCommunicator:
    def __init__(self, sensitivity=1500):
        self.sensitivity = sensitivity
        self.capture = cv2.VideoCapture(0)
        self.data2 = None
        self.initialize_quantum_vars()
        self.initialize_tracking_vars()
        with open("ack_stats.log", "w") as f:
            f.write("")
        with open("auto_text_log.txt", "w", encoding="utf-8") as f:
            f.write(f"=== New Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        
    def initialize_quantum_vars(self):
        self.qu = 0
        self.cyc = 0
        self.swi = 0
        self.longcyc = 3
        random.seed(int(time.time()))
        self.numa = ",".join(str(np.random.randint(0, 2)) for _ in range(100000))
        self.corr = 3
        self.ghostprotocol = 10000 if spin == -1 else 0
        self.PIN = PIN
        
    def initialize_tracking_vars(self):
        self.ack = 0
        self.last_ack = 0
        self.last_ack_time = time.time()
        self.ack_rates = deque(maxlen=50)
        self.nul = 0
        self.start_time = datetime.now()
        self.last_status_update = datetime.now()
        self.status_update_interval = 0.5
        self.ghost_messages = deque(maxlen=4)
        self.total_frames = 0
        self.motion_frame_count = 0
        self.prime = 0
        self.prime_threshold = 3
        self.and_count = 0
        self.or_count = 0
        self.last_ghost_value = 0
        self.range = 10
        self.i = 0
    def calculate_ack_rate(self):
        current_time = time.time()
        time_diff = max(0.1, current_time - self.last_ack_time)
        ack_diff = self.ack - self.last_ack
        rate = ack_diff / time_diff
        self.ack_rates.append(rate)
        self.last_ack = self.ack
        self.last_ack_time = current_time
        return rate

    def get_average_ack_rate(self):
        if len(self.ack_rates) > 0:
            return sum(self.ack_rates) / len(self.ack_rates)
        return 0

    def process_frame(self, frame):
        if frame is None:
            return False
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.data2 is None:
            self.data2 = gray
            return True
            
        frame_delta = cv2.absdiff(self.data2, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        self.process_quadrants(thresh, frame)
        self.data2 = gray
        
        return True

    def process_quadrants(self, thresh, frame):
        height, width = thresh.shape
        quad_w, quad_h = width // 16, height // 8
        
        for row in range(8):
            for col in range(16):
                x1, y1 = col * quad_w, row * quad_h
                x2, y2 = (col + 1) * quad_w, (row + 1) * quad_h
                
                if self.check_quadrant_motion(thresh[y1:y2, x1:x2]):
                    self.motion_frame_count += 1
                    self.apply_quantum_logic(row, col)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def check_quadrant_motion(self, quadrant):
        return np.sum(quadrant > 10) > self.sensitivity

    def apply_quantum_logic(self, row, col):
        if 4 < row < 11:
            self.or_count += 1
            return
            if 4 < col < 11:
                self.and_count += 1
                self.process_quantum_state()
                
        self.check_quantum_states()
        
    def process_quantum_state(self):
        random.seed(int(time.time()))
        self.qu = 1 - self.qu
        
    def check_quantum_states(self):
        check = self.numa.split(",")
        if self.cyc >= len(check):
            return
            
        if self.or_count > self.corr:
            if check[self.cyc] != str(self.qu):
                self.process_or_state()
            self.or_count = 0
                
        if self.and_count > self.corr:
            if check[self.cyc] == str(self.qu):
                self.process_and_state()
            self.and_count = 0
                
        self.update_ghost_protocol()
        
    def process_or_state(self):
        if self.swi >= self.longcyc:
            random.seed(int(time.time()))
            self.qu = np.random.randint(0, 2)
            self.swi = 0
        self.swi += 1
        self.nul += 1
        self.prime = min(self.prime + 1, self.prime_threshold)
        self.advance_cycle()
        
    def process_and_state(self):
        if self.swi >= self.longcyc:
            random.seed(int(time.time()))
            self.qu = np.random.randint(0, 2)
            self.swi = 0
        self.swi += 1
        self.ack += 1
        self.prime = 0 if self.prime >= self.prime_threshold else self.prime + 1
        self.advance_cycle()
        
    def advance_cycle(self):
        self.cyc += 1
        
    def update_ghost_protocol(self):
        current_value = self.ghostprotocol * self.range
        if self.prime < 1 and self.ghostprotocol > 3:
            if current_value != self.last_ghost_value:
                self.ghost_messages.append(f"Protocol state: {current_value}")
                self.last_ghost_value = current_value
                self.i = self.i if hasattr(self, 'i') else 0
            self.i += 1
        self.ghostprotocol -= -spin
        if (spin == -1 and self.ghostprotocol <= 0) or (spin == 1 and self.ghostprotocol >= 10000):
            self.cleanup()
            exit()
        
    def log_ack_stats(self):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Detailed CSV log
        log_entry = (
            f"Frame {self.i}, {current_time}, "
            f"{current_time}, "
            f"ACKs/Second: {self.calculate_ack_rate():.2f}, "
            f"Avg ACKs/Second: {self.get_average_ack_rate():.2f}, "
            f"Total ACKs: {self.ack}, "
            f"Ghost Protocol: {self.ghostprotocol}, "
            f"Ghost Value: {self.ghostprotocol * self.range}, "
            f"PIN: {self.PIN}, "
            f"AND Count: {self.and_count}, "
            f"OR Count: {self.or_count}, "
            f"Quantum State: {self.qu}, "
            f"Cycle/10: {self.cyc/10}, "
            f"PIN: {self.PIN}\n"
        )
        
        with open("ack_stats.log", "a") as f:
            f.write(log_entry)

        # Status log
        with open("auto_text_log.txt", "a", encoding="utf-8") as log:
            log.write(f"""
Frame {self.i}, {current_time}
Time: {current_time}
Quantum State: {self.qu}
Cycle: {self.cyc}/{len(self.numa.split(','))}
Ghost Protocol: {self.ghostprotocol * self.range}
PIN: {self.PIN}
ACK Rate: {self.calculate_ack_rate():.2f}/second
Total ACK: {self.ack} | NUL: {self.nul}
AND Count: {self.and_count} | OR Count: {self.or_count}

-------------------------
""")
            
    def run(self):
        try:
            while True:
                ret, frame = self.capture.read()
                if not self.process_frame(frame):
                    break
                    
                current_time = datetime.now()
                if (current_time - self.last_status_update).total_seconds() >= self.status_update_interval:
                    self.calculate_ack_rate()
                    self.display_status()
                    self.log_ack_stats()
                    self.last_status_update = current_time
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
        finally:
            self.cleanup()
            
    def cleanup(self):
        if hasattr(self, 'capture'):
            self.capture.release()
        cv2.destroyAllWindows()
        print("\nShutdown complete. Press Enter to view the plot...")
        input()
        self.plot_quantum_data()
        
    def plot_quantum_data(self):
        plt.figure(figsize=(12, 8))
        
        # Plot ghost protocol states
        plt.subplot(2, 1, 1)
        time_points = list(range(len(self.ghost_messages)))
        plt.plot(time_points, [int(msg.split(": ")[1]) for msg in self.ghost_messages], 'b-')
        plt.title("Ghost Protocol States")
        plt.xlabel("Time Steps")
        plt.ylabel("State Value")
        plt.grid(True)
        
        # Plot ACK rates
        plt.subplot(2, 1, 2)
        time_points = list(range(len(self.or_count)))
        plt.plot(time_points, list(self.or_count), 'g-')
        plt.title("ACK Rates Over Time")
        plt.xlabel("Time Steps")
        plt.ylabel("ACKs/Second")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def display_status(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        current_rate = self.calculate_ack_rate()
        avg_rate = self.get_average_ack_rate()
        print(f"""
Quantum Communicator Status
-------------------------
Time: {datetime.now().strftime('%H:%M:%S')}
Quantum State: {self.qu}
Cycle: {self.cyc}/{len(self.numa.split(','))}
Ghost Protocol: {self.ghostprotocol * self.range}
PIN: {self.PIN}

Performance Metrics
-----------------
ACK Rate: {current_rate:.2f}/second
Avg ACK Rate: {avg_rate:.2f}/second
Total ACK: {self.ack} | NUL: {self.nul}
AND Count: {self.and_count} | OR Count: {self.or_count}
Motion Frames: {self.motion_frame_count}/{self.total_frames}
-------------------------
""")

if __name__ == "__main__":
    communicator = QuantumCommunicator()
    communicator.run()
