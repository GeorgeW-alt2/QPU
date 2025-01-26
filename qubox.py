import cv2
import numpy as np
import os
from datetime import datetime
import random
import time
from collections import deque

spin = 1  # 1 or -1
problem = input("Enter math problem: ")
with open("data.csv", "r", encoding="utf-8") as f:
    data = f.readlines()  # Useful for sending messages
print("Solving: ", problem)

class QuantumCommunicator:
    def __init__(self, sensitivity=1500):
        self.sensitivity = sensitivity
        self.capture = cv2.VideoCapture(0)
        self.data2 = None
        self.initialize_quantum_vars()
        self.initialize_tracking_vars()
        self.logs = []
        self.or_count = 0
        self.or_count_per_second = 0
        self.last_or_count_time = time.time()
        self.start_time = datetime.now()
        self.prime_threshold = 0
        self.logged_or_counts = list()
        self.previous_motion_frame_count = 0
        with open("ack_stats.log", "w") as f:
            f.write("")

    def initialize_quantum_vars(self):
        self.qu = 0
        self.cyc = 0
        self.swi = 0
        self.longcyc = 200
        random.seed(int(time.time()))
        self.numa = ",".join(str(np.random.randint(0, 2)) for _ in range(100000))
        self.corr = 3
        self.ghostprotocol = 10000 if spin == -1 else 0

    def initialize_tracking_vars(self):
        self.ack = 0
        self.last_ack = 0
        self.last_ack_time = time.time()
        self.ack_rates = deque(maxlen=50)
        self.nul = 0
        self.motion_frame_count = 0
        self.prime = 0
        self.and_count = 0
        self.or_count = 0
        self.last_ghost_value = 0
        self.range = int(input("Range(e.g, 1 or 10): "))
        self.i = 0

    def calculate_or_count_per_second(self):
        current_time = time.time()
        if current_time - self.last_or_count_time >= 1:
            self.or_count_per_second = self.or_count
            self.or_count = 0
            self.last_or_count_time = current_time

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

            if 4 < col < 11:
                self.and_count += 1
                self.process_quantum_state()

        self.check_quantum_states()

    def process_quantum_state(self):
        random.seed(int(time.time()))
        if self.qu == np.random.randint(0, 2):
            self.qu = 0

        if self.qu == np.random.randint(0, 2):
            self.qu = 1

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
        if self.qu == 1:
            self.qu = 0

        if self.qu == 0:
            self.qu = 1

    def process_and_state(self):
        if self.swi >= self.longcyc:
            random.seed(int(time.time()))
            self.qu = np.random.randint(0, 2)
            self.swi = 0
        self.swi += 1
        self.ack += 1
        self.prime = 0 if self.prime >= self.prime_threshold else self.prime + 1
        self.advance_cycle()
        if self.qu == 0:
            self.qu = 0

        if self.qu == 1:
            self.qu = 1

    def advance_cycle(self):
        self.cyc += 1

    def update_ghost_protocol(self):
        current_value = self.ghostprotocol * self.range
        if self.prime < 1:
            if current_value != self.last_ghost_value:
                self.last_ghost_value = current_value
                self.i = self.i if hasattr(self, 'i') else 0
        self.i += 1
        self.ghostprotocol -= -spin
        if (spin == -1 and self.ghostprotocol <= 0) or (spin == 1 and self.ghostprotocol >= 10000):
            #self.print_all_logs()
            with open("ack_stats.log", "r", encoding="utf-8") as f:
                data = f.read()
            print(data)
            exit()


    def log_ack_stats(self):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = (
            f"Frame {self.i}, {current_time}, "
            f"Ghost Protocol: {self.ghostprotocol}, "
            f"Ghost Value: {self.ghostprotocol * self.range}, "
            f"OR Count: {self.or_count}, "
            f"Problem: {problem}\n"
        )
        self.logs.append(log_entry)
        self.logged_or_counts.append(self.or_count)
        if len(self.logged_or_counts) > 2:
            if self.logged_or_counts[-2] == 3 and self.logged_or_counts[-1] != 3 and self.logged_or_counts[-3] != 3:
                with open("ack_stats.log", "a") as f:
                    f.write(self.logs[-2])

    def print_all_logs(self):
        print("\nFull Session Log:")
        print("-" * 80)
        for log in self.logs:
            print(log.strip())
        print("-" * 80)

    def display_status(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"""
Quantum Communicator Status
-------------------------
Time: {datetime.now().strftime('%H:%M:%S')}
Quantum State: {self.qu}
Ghost Protocol: {self.ghostprotocol * self.range}

Performance Metrics
-----------------
OR Count: {self.or_count}
Motion Frames: {self.motion_frame_count}
Problem: {problem}
-------------------------
""")

    def run(self):
        try:
            while True:
                ret, frame = self.capture.read()
                if not self.process_frame(frame):
                    break

                self.log_ack_stats()
                self.display_status()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    #self.print_all_logs()
                    break
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
            #self.print_all_logs()

    def send_message(self):
        result = eval(problem)
        if result <= data[self.ghostprotocol * self.range]:
            self.numa += ",".join('9' for _ in range(500))

if __name__ == "__main__":
    communicator = QuantumCommunicator()
    communicator.run()
    communicator.capture.release()
    cv2.destroyAllWindows()
