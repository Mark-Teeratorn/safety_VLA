"""
CARLA Simulator Client and Camera Sensor Bridge over TCP Socket
"""
import time
import socket
import struct
import json
import threading
import cv2
import numpy as np
from typing import Optional, List, Dict

class CarlaBridge:
    def __init__(self, host: str = "10.100.16.130", port: int = 5555, width: int = 1280, height: int = 720):
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.sock = None
        self.connected = False
        self.latest_frame = None
        self.running = False
        self.lock = threading.Lock()
        self.recv_thread = None

    def connect(self) -> bool:
        self.running = True
        return self._try_connect()

    def _try_connect(self) -> bool:
        try:
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass
            print(f"[CARLA Bridge] Connecting to CARLA HIL Server at {self.host}:{self.port}...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)
            self.sock.connect((self.host, self.port))
            self.connected = True
            
            if self.recv_thread is None or not self.recv_thread.is_alive():
                self.recv_thread = threading.Thread(target=self._receiver_worker, daemon=True)
                self.recv_thread.start()
            print("[CARLA Bridge] Connected successfully to CARLA Simulator!")
            return True
        except Exception as e:
            print(f"[CARLA Bridge] Connection failed: {e}")
            self.connected = False
            return False

    def _receiver_worker(self):
        buffer = b""
        while self.running:
            if not self.connected:
                time.sleep(1.0)
                self._try_connect()
                continue
                
            try:
                while len(buffer) < 4:
                    chunk = self.sock.recv(4096)
                    if not chunk:
                        raise ConnectionResetError("Socket closed")
                    buffer += chunk
                    
                msg_len = struct.unpack(">I", buffer[:4])[0]
                buffer = buffer[4:]
                
                while len(buffer) < msg_len:
                    chunk = self.sock.recv(min(4096, msg_len - len(buffer)))
                    if not chunk:
                        raise ConnectionResetError("Socket closed")
                    buffer += chunk
                    
                frame_data = buffer[:msg_len]
                buffer = buffer[msg_len:]
                
                nparr = np.frombuffer(frame_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    with self.lock:
                        self.latest_frame = img
            except Exception:
                self.connected = False
                buffer = b""
                time.sleep(1.0)

    def apply_control(self, throttle: float, brake: float, steer: float,
                      recommended_speed: float = 40.0, risk_level: str = "LOW",
                      action_advisory: str = "", status_message: str = "",
                      reasoning: str = "", targets: Optional[List[Dict]] = None):
        if not self.connected or not self.sock:
            return
            
        try:
            clean_targets = []
            if targets:
                for t in targets:
                    clean_targets.append({
                        "class": t.get("class"),
                        "conf": float(t.get("conf", 0.0)),
                        "bbox": list(t.get("bbox")),
                        "in_path": bool(t.get("in_path", False))
                    })
                    
            payload = {
                "throttle": float(throttle),
                "brake": float(brake),
                "steer": float(steer),
                "rec_speed": float(recommended_speed),
                "risk": str(risk_level),
                "advisory": str(action_advisory),
                "status": str(status_message),
                "reasoning": str(reasoning),
                "targets": clean_targets
            }
            data = json.dumps(payload).encode("utf-8")
            header = struct.pack(">I", len(data))
            self.sock.sendall(header + data)
        except Exception:
            self.connected = False

    def generate_mock_frame(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return frame

    def destroy(self):
        self.running = False
        self.connected = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
