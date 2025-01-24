import socket
import json
import time
import threading
import numpy as np
spin = 1  # 1 or -1
proof_of_work_size = 9
range_compute = 100000 #needs to be computable
_range =        100000 #simply add optimal ghost values to range then rerun
SHAmsg = "GeorgeW"
message ="" #leave empty
target = 0
class CryptoNight:
    def __init__(self):
        self.memory = bytearray(2097152)  # 2MB scratchpad
        self.aes_key = None
        
    def keccak_f(self, data):
        k = keccak.new(digest_bits=256)
        k.update(data)
        return k.digest()
        
    def aes_round(self, data, key):
        cipher = AES.new(key, AES.MODE_ECB)
        return cipher.encrypt(data)
        
    def memory_hard(self, state):
        if len(state) < 8:
            return state
            
        # Fill scratchpad with initial data
        self.memory[0:len(state)] = state
        for i in range(len(state), len(self.memory), 16):
            self.memory[i:i+16] = self.aes_round(self.memory[i-16:i], self.aes_key)
        
        # Memory-hard loop
        a = int.from_bytes(state[:8], 'little')
        for i in range(524288):
            addr = (a & 0x1FFFF0) >> 4
            if addr + 16 > len(self.memory):
                break
                
            chunk = self.memory[addr:addr+16]
            if a & 0x100000:
                b = self.aes_round(chunk, self.aes_key)
            else:
                b = bytes(x ^ y for x, y in zip(chunk, state))
                
            a = int.from_bytes(b[:8], 'little')
            self.memory[addr:addr+16] = b
        
        return state
        
    def hash(self, data):
        # Initial Keccak
        state = self.keccak_f(data)
        self.aes_key = state[:32]
        
        # Memory-hard function
        state = self.memory_hard(state)
        
        # Final Keccak
        return self.keccak_f(state)

class XMRMiner:
    def __init__(self, pool, port, wallet, worker_name="worker1"):
        self.pool = pool
        self.port = port
        self.wallet = wallet
        self.worker = worker_name
        self.cn = CryptoNight()
        self.socket = None
        self.running = False
        
    def connect(self):
        print(f"Connecting to {self.pool}:{self.port}")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.pool, self.port))
        
        login = {
            "method": "login",
            "params": {
                "login": self.wallet,
                "pass": self.worker,
                "agent": "xmr-miner-py/1.0"
            },
            "id": 1
        }
        self.socket.send(json.dumps(login).encode() + b'\n')
        response = self.socket.recv(8192).decode()
        print("Login response:", response)
        
    def process_job(self, job):
        try:
            target = int(job["target"], 16)
            blob = job["blob"]
            job_id = job["job_id"]
            
            print(f"New job received: {job_id}")
            print(f"Target: {target}")
            
            nonce = 0
            while self.running:
                hex_nonce = format(nonce, '08x').encode().hex()
                try:
                    input_data = bytes.fromhex(blob[:76] + hex_nonce + blob[84:])
                    hash_result = self.cn.hash(input_data).hex()
                    
                    if int(hash_result, 16) < target:
                        print(f"Share found! Nonce: {hex_nonce}")
                        self.submit_share(job_id, hex_nonce, hash_result)
                        
                    nonce += 1
                    if nonce % 100 == 0:
                        print(f"Hashes: {nonce}")
                    if nonce > 0xFFFFFFFF:
                        break
                        
                except ValueError as e:
                    print(f"Invalid hex data: {e}")
                    print(f"Blob length: {len(blob)}")
                    print(f"Nonce: {hex_nonce}")
                    break
                    
        except Exception as e:
            print(f"Job processing error: {e}")
            print(f"Job data: {job}")
                
    def submit_share(self, job_id, nonce, result):
        submit = {
            "method": "submit",
            "params": {
                "id": self.worker,
                "job_id": job_id,
                "nonce": nonce,
                "result": result
            },
            "id": 1
        }
        print("Submitting share:", submit)
        self.socket.send(json.dumps(submit).encode() + b'\n')
        response = self.socket.recv(8192).decode()
        print("Submit response:", response)
        
    def start_mining(self):
        self.running = True
        self.connect()
        
        while self.running:
            try:
                response = self.socket.recv(8192).decode()
                if not response:
                    time.sleep(1)
                    continue
                    
                for line in response.splitlines():
                    data = json.loads(line)
                    if "method" in data and data["method"] == "job":
                        threading.Thread(target=self.process_job, 
                                      args=(data["params"],),
                                      daemon=True).start()
                        
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5)
                print("Attempting reconnection...")
                self.connect()
                
    def stop(self):
        self.running = False
        if self.socket:
            self.socket.close()


import cv2
import numpy as np
import os
from datetime import datetime
import random
import time
from collections import deque
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from Crypto.Cipher import AES
from Crypto.Hash import keccak


def mine_chunk(start_nonce, chunk_size, target_zeros, thread_id):
    target = '0' * target_zeros
    end_nonce = start_nonce + chunk_size
    
    for nonce in range(start_nonce, end_nonce):
        message = f"{SHAmsg}{nonce}".encode()
        current_hash = hashlib.sha256(message).hexdigest()
        if current_hash.startswith(target):
            return nonce, current_hash, thread_id
    return None, None, None

class QuantumCommunicator:
    def __init__(self, sensitivity=1500):
        self.sensitivity = sensitivity
        self.capture = cv2.VideoCapture(0)
        self.data2 = None
        self.initialize_quantum_vars()
        self.initialize_tracking_vars()
        self.logs = []
        self.or_states = []
        self.should_mine = True
        with open("ack_stats.log", "w") as f:
            f.write("")
        
    def initialize_quantum_vars(self):
        self.qu = 0
        self.cyc = 0
        self.swi = 0
        self.longcyc = 3
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
        self.range = _range
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
        
        if self.data2 is None:
            self.data2 = gray
            return True
            
        frame_delta = cv2.absdiff(self.data2, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        self.process_quadrants(thresh, frame)
        self.data2 = gray
        self.total_frames += 1
        
        return True

    def process_quadrants(self, thresh, frame):
        height, width = thresh.shape
        quad_w, quad_h = width // 16, height // 8
        
        for row in range(16):
            for col in range(32):
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
        self.or_states.append(self.or_count)
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
        if self.prime < 1:
            if current_value != self.last_ghost_value:
                self.ghost_messages.append(f"Protocol state: {current_value}")
                self.last_ghost_value = current_value
                self.i = self.i if hasattr(self, 'i') else 0
            self.i += 1
        self.ghostprotocol -= -spin
        if (spin == -1 and self.ghostprotocol <= 0) or (spin == 1 and self.ghostprotocol >= range_compute):
            self.print_all_logs()
            self.should_mine = True
            input()
            exit()
        
    def log_ack_stats(self):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = (
            f"Frame {self.i}, {current_time}, "
            f"Ghost Protocol: {self.ghostprotocol}, "
            f"Ghost Value: {self.ghostprotocol * self.range}, "
            f"OR Count: {self.or_count}\n"
        )
        self.logs.append(log_entry)
        with open("ack_stats.log", "a") as f:
            f.write(log_entry)

    def print_all_logs(self):
        print("\nFull Session Log:")
        print("-" * 80)
        for log in self.logs:
            print(log.strip())
        print("-" * 80)
        if self.should_mine:
            self.start_mining()
            self.should_mine = False

    def sort_ghost_by_or(self, logs):
        data = {}
        for log in logs.splitlines():
            if "OR Count:" in log and "Ghost Value:" in log:
                try:
                    or_count = int(log.split("OR Count: ")[1].split(",")[0])
                    ghost_value = int(log.split("Ghost Value: ")[1].split(",")[0])
                    if ghost_value in data:
                        data[ghost_value] = max(data[ghost_value], or_count)
                    else:
                        data[ghost_value] = or_count
                except (ValueError, IndexError):
                    continue

        # Sort by OR count
        sorted_ghost = sorted(data.items(), key=lambda x: x[1], reverse=True)
        max_or = sorted_ghost[0][1] if sorted_ghost else 0
        ghost_values = []
        
        # Group values within 9 of max OR count
        for ghost_value, or_count in data.items():
            if max_or - 9 <= or_count:
                if ghost_value - 1 in data:
                    ghost_values.append(ghost_value - 1)
                ghost_values.append(ghost_value)
                if ghost_value + 1 in data:
                    ghost_values.append(ghost_value + 1)
        
        # Remove duplicates while maintaining order
        seen = set()
        return [x for x in ghost_values if not (x in seen or seen.add(x))]

    def print_progress_bar(self, iteration, total, length=50):
        percent = (iteration / float(total)) * 100
        filled_length = int(length * iteration // total)
        bar = '█' * filled_length + '-' * (length - filled_length)
        print(f'\rProgress: |{bar}| {percent:.1f}% Complete', end='\r')
        if iteration == total:
            print()

    def start_mining(self):
        print("\nStarting mining operation...")
        with open("ack_stats.log", "r") as f:
            log_content = f.read()
        
        ghost_values = self.sort_ghost_by_or(log_content)
        # Save to CSV
        with open('ghost_values.csv', 'w') as x:
            for value in ghost_values:
                x.write(f"{value}\n")
                x.flush()
        if ghost_values:
            print("Ghost values:",ghost_values)
            POOL = "xmr-us-east1.nanopool.org"
            PORT = 14444
            WALLET = "493hhZv9pApZsC7xKChhV5DMooBdaLz43cfAqDSHYsCiKxGi2Sfqk3n9Cnfwm4JYTXXR6TG3PpeTEK9qEyP3bBzcPXPAvGi"  # Replace with your XMR wallet address
            WORKER = "py_miner"
            
            miner = XMRMiner(POOL, PORT, WALLET, WORKER)
            try:
                print("Starting miner...")
                miner.start_mining()
            except KeyboardInterrupt:
                print("\nStopping miner...")
                miner.stop()
                print("Mining stopped")
            else:
                print("No ghost values found for mining")

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
                    self.print_all_logs()
                    break
                    
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
            self.print_all_logs()

            
    def plot_quantum_data(self):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        if self.ghost_messages:
            time_points = list(range(len(self.ghost_messages)))
            states = [int(msg.split(": ")[1]) for msg in self.ghost_messages]
            plt.plot(time_points, states, 'b-')
        plt.title("Ghost Protocol States")
        plt.xlabel("Time Steps")
        plt.ylabel("State Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def display_status(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        current_rate = self.calculate_ack_rate()
        avg_rate = self.get_average_ack_rate()
        print(f"""
        Quantum Miner Status
        -------------------------
        Time: {datetime.now().strftime('%H:%M:%S')}
        Quantum State: {self.qu}
        Ghost Protocol: {self.ghostprotocol * self.range}
        
        Performance Metrics
        -----------------
        OR Count: {self.or_count}
        Motion Frames: {self.motion_frame_count}/{self.total_frames}
        -------------------------
        """)
    def process_job(self, job):
        target = int(job["target"], 16)
        blob = job["blob"]
        job_id = job["job_id"]
        
        print(f"New job received: {job_id}")
        print(f"Target: {target}")
        
        nonce = 0
        while self.running:
            hex_nonce = format(nonce, '08x').encode().hex()
            try:
                input_data = bytes.fromhex(blob[:76] + hex_nonce + blob[84:])
                hash_result = self.cn.hash(input_data).hex()
                
                if int(hash_result, 16) < target:
                    print(f"Share found! Nonce: {hex_nonce}")
                    self.submit_share(job_id, hex_nonce, hash_result)
                    
                nonce += 1
                if nonce % 100 == 0:
                    print(f"Hashes: {nonce}")
                if nonce > 0xFFFFFFFF:
                    break
                    
            except ValueError as e:
                print(f"Invalid hex data: {e}")
                print(f"Blob length: {len(blob)}")
                print(f"Nonce: {hex_nonce}")
                break
    def send_message(self):
    
        hex_nonce = format(self.ghostprotocol * self.range, '08x').encode().hex()

        input_data = bytes.fromhex(blob[:76] + hex_nonce + blob[84:])
        hash_result = self.cn.hash(input_data).hex()
        
        if int(hash_result, 16) < target:
            result = int(hash_result, 16)
            self.numa += ",".join('9' for _ in range(500))
       
            
def main():
    try:
        communicator = QuantumCommunicator()
        communicator.run()
    except KeyboardInterrupt:
        print("\nStarting mining process...")
        communicator.start_mining()

if __name__ == "__main__":
    main()
    