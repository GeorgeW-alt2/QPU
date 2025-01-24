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
import json
import json
import socket
spin = 1  # 1 or -1
proof_of_work_size = 9
range_compute = 100000 #needs to be computable
_range =        100000 #simply add optimal ghost values to range then rerun
SHAmsg = ""
message ="" #leave empty
def shuffle_array(arr):
    n = len(arr)
    for i in range(n-1, 0, -1):
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]
    return arr
def load_ghost_values(filename='ghost_values.csv'):
    ghost_values = []
    with open(filename, 'r') as f:
        reader = f.readlines()
        for row in reader:
            ghost_values.append(int(row))
    return ghost_values
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
            
            ghost_values = load_ghost_values()
            ghost_values = shuffle_array(ghost_values)	
            nonce = ghost_values[0]
            while self.running:
                hex_nonce = format(nonce, '08x').encode().hex()
                try:
                    input_data = bytes.fromhex(blob[:76] + hex_nonce + blob[84:])
                    hash_result = self.cn.hash(input_data).hex()
                    print(int(hash_result, 16))
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


def main():
    POOL = "xmr-us-east1.nanopool.org"
    PORT = 14444
    WALLET = "493hhZv9pApZsC7xKChhV5DMooBdaLz43cfAqDSHYsCiKxGi2Sfqk3n9Cnfwm4JYTXXR6TG3PpeTEK9qEyP3bBzcPXPAvGi"  # Replace with your XMR wallet address
    random_number = random.randint(1000, 9999)
    WORKER = f"py_miner_{random_number}"    
    miner = XMRMiner(POOL, PORT, WALLET, WORKER)
    try:
        print("Starting miner...")
        miner.start_mining()
    except KeyboardInterrupt:
        print("\nStopping miner...")
        miner.stop()
        print("Mining stopped")
if __name__ == "__main__":
    main()
    
