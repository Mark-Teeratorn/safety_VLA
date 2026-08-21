#!/usr/bin/env python3
"""
Autonomous Safe Driving HIL Bridge: BehaviorAgent + NVIDIA Jetson Safety Brain (AEB)
Streams Dashcam Video Frames to NVIDIA Jetson (tesla@100.106.22.30)
and applies Jetson Level-1 & Level-2 Safety Interventions in real-time
"""
import os
import sys
import time
import random
import socket
import struct
import json
import threading
import cv2
import numpy as np

# Ensure CARLA Python API is available
sys.path.append("/opt/carla-simulator/PythonAPI/carla")
sys.path.append("/opt/carla-simulator/PythonAPI/carla/agents")
sys.path.append("/opt/carla-simulator/PythonAPI")

import carla
from agents.navigation.behavior_agent import BehaviorAgent

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

HOST = "0.0.0.0"
PORT = 5555
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

class CarlaHILServer:
    def __init__(self, carla_host="localhost", carla_port=2000):
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.client = None
        self.world = None
        self.agent = None
        self.vehicle = None
        self.camera = None
        
        self.latest_raw_frame = None
        self.latest_control = {
            "throttle": 0.5,
            "brake": 0.0,
            "steer": 0.0,
            "hazard": False,
            "status": "Waiting for Jetson...",
            "reasoning": "",
            "targets": []
        }
        self.connected_client = None
        self.running = True
        self.lock = threading.Lock()
        self.spawn_points = []
        self.is_emergency_braking = False
        
    def setup_carla(self):
        print(f"[HIL Server] Connecting to CARLA at {self.carla_host}:{self.carla_port}...")
        self.client = carla.Client(self.carla_host, self.carla_port)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        
        # Clean existing actors with rolename ego_vehicle
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name") == "ego_vehicle":
                actor.destroy()
                
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        vehicle_bp.set_attribute("role_name", "ego_vehicle")
        
        self.spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = self.spawn_points[0] if self.spawn_points else carla.Transform(carla.Location(x=0, y=0, z=2))
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"[HIL Server] Spawned Ego Vehicle: {self.vehicle.type_id} (ID: {self.vehicle.id})")
        
        # Setup BehaviorAgent navigation
        self.agent = BehaviorAgent(self.vehicle, behavior="normal")
        dest = random.choice(self.spawn_points).location
        self.agent.set_destination(dest)
        print(f"[HIL Server] BehaviorAgent initialized, Route Target: {dest}")
        
        # Setup RGB Dashcam Camera
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
        cam_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
        cam_bp.set_attribute("fov", "90")
        
        # Mount on windshield (X=1.6, Z=1.7)
        cam_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.camera.listen(self._on_camera_image)
        print(f"[HIL Server] Dashcam Camera attached (ID: {self.camera.id})")
        
        # Spawn some background NPC vehicles
        self._spawn_traffic(bp_lib)
        
        # Start vehicle navigation & safety supervisory loop (30 Hz)
        safety_loop_thread = threading.Thread(target=self._navigation_and_safety_loop, daemon=True)
        safety_loop_thread.start()

    def _spawn_traffic(self, bp_lib):
        npc_bps = bp_lib.filter("vehicle.*")
        for i in range(1, min(20, len(self.spawn_points))):
            try:
                bp = np.random.choice(npc_bps)
                v = self.world.spawn_actor(bp, self.spawn_points[i])
                v.set_autopilot(True)
            except Exception:
                pass
        print("[HIL Server] Background NPC traffic initialized.")

    def _on_camera_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        bgr = array[:, :, :3]
        with self.lock:
            self.latest_raw_frame = bgr.copy()

    def _navigation_and_safety_loop(self):
        """Executes autonomous navigation and applies Jetson AEB overrides"""
        while self.running:
            try:
                if self.vehicle and self.vehicle.is_alive and self.agent:
                    # Pick new destination if reached
                    if self.agent.done():
                        dest = random.choice(self.spawn_points).location
                        self.agent.set_destination(dest)
                        
                    # Compute standard navigation control
                    base_control = self.agent.run_step()
                    
                    with self.lock:
                        ctrl = dict(self.latest_control)
                        
                    jetson_brake = float(ctrl.get("brake", 0.0))
                    hazard = bool(ctrl.get("hazard", False))
                    
                    if jetson_brake > 0.5 or hazard:
                        # JETSON AEB INTERVENTION: Trigger Emergency Braking
                        self.is_emergency_braking = True
                        final_control = carla.VehicleControl(
                            throttle=0.0,
                            brake=1.0,
                            steer=base_control.steer,
                            hand_brake=True
                        )
                    elif jetson_brake > 0.0:
                        # JETSON CAUTION: Decelerate
                        self.is_emergency_braking = False
                        final_control = carla.VehicleControl(
                            throttle=min(base_control.throttle, 0.25),
                            brake=max(base_control.brake, 0.2),
                            steer=base_control.steer,
                            hand_brake=False
                        )
                    else:
                        # JETSON CRUISE: Normal Driving
                        self.is_emergency_braking = False
                        final_control = base_control
                        
                    self.vehicle.apply_control(final_control)
            except Exception:
                pass
            time.sleep(0.033) # 30 Hz loop

    def socket_server_worker(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen(1)
        print(f"[HIL Server] TCP Bridge Server listening on {HOST}:{PORT} (Waiting for Tesla Jetson)...")
        
        while self.running:
            try:
                conn, addr = server_sock.accept()
                print(f"[HIL Server] >>> CONNECTED to Edge Device: {addr[0]}:{addr[1]} <<<")
                self.connected_client = conn
                self._handle_client(conn)
            except Exception as e:
                if self.running:
                    print(f"[HIL Server] Socket error: {e}")
                time.sleep(1)

    def _handle_client(self, conn):
        conn.settimeout(5.0)
        recv_thread = threading.Thread(target=self._client_receiver, args=(conn,), daemon=True)
        recv_thread.start()
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        while self.running and self.connected_client == conn:
            frame_to_send = None
            with self.lock:
                if self.latest_raw_frame is not None:
                    frame_to_send = self.latest_raw_frame.copy()
                    
            if frame_to_send is not None:
                result, encimg = cv2.imencode(".jpg", frame_to_send, encode_param)
                if result:
                    data = encimg.tobytes()
                    header = struct.pack(">I", len(data))
                    try:
                        conn.sendall(header + data)
                    except Exception:
                        break
            time.sleep(0.033)
            
        print("[HIL Server] Client disconnected.")
        self.connected_client = None

    def _client_receiver(self, conn):
        buffer = b""
        while self.running and self.connected_client == conn:
            try:
                while len(buffer) < 4:
                    chunk = conn.recv(4096)
                    if not chunk:
                        return
                    buffer += chunk
                msg_len = struct.unpack(">I", buffer[:4])[0]
                buffer = buffer[4:]
                
                while len(buffer) < msg_len:
                    chunk = conn.recv(min(4096, msg_len - len(buffer)))
                    if not chunk:
                        return
                    buffer += chunk
                    
                msg_data = buffer[:msg_len]
                buffer = buffer[msg_len:]
                
                ctrl_dict = json.loads(msg_data.decode("utf-8"))
                with self.lock:
                    self.latest_control = ctrl_dict
            except Exception:
                break

    def run_display(self):
        if not HAS_PYGAME:
            while self.running:
                time.sleep(1)
            return
            
        pygame.init()
        display = pygame.display.set_mode((IMAGE_WIDTH, IMAGE_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("CARLA <-> NVIDIA Jetson (tesla@100.106.22.30) Safe Driving HIL")
        clock = pygame.time.Clock()
        
        font_title = pygame.font.SysFont("monospace", 20, bold=True)
        font_main = pygame.font.SysFont("monospace", 18)
        font_small = pygame.font.SysFont("monospace", 15)
        
        while self.running:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    # Respawn vehicle on R
                    if self.spawn_points and self.vehicle:
                        sp = random.choice(self.spawn_points)
                        self.vehicle.set_transform(sp)
                        dest = random.choice(self.spawn_points).location
                        self.agent.set_destination(dest)
                    
            frame = None
            ctrl = {}
            with self.lock:
                if self.latest_raw_frame is not None:
                    frame = self.latest_raw_frame.copy()
                ctrl = dict(self.latest_control)
                
            if frame is not None:
                # Render bounding boxes received from Jetson
                targets = ctrl.get("targets", [])
                for t in targets:
                    bbox = t.get("bbox")
                    cls_name = t.get("class", "object")
                    conf = t.get("conf", 0.0)
                    in_path = t.get("in_path", False)
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        color = (0, 0, 255) if in_path else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(frame, f"{cls_name.upper()} {conf:.2f}", (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
                display.blit(surface, (0, 0))
                
                # Draw Semi-transparent HUD Header
                hud_overlay = pygame.Surface((IMAGE_WIDTH, 125))
                hud_overlay.set_alpha(210)
                hud_overlay.fill((15, 15, 25))
                display.blit(hud_overlay, (0, 0))
                
                # Header text
                is_online = (self.connected_client is not None)
                status_color = (50, 255, 100) if is_online else (255, 80, 80)
                status_str = "CONNECTED (NVIDIA Jetson tesla@100.106.22.30)" if is_online else "WAITING FOR JETSON CONNECTION..."
                
                t_title = font_title.render(f"[CARLA HIL AUTO-DRIVE] Edge Brain: {status_str}", True, status_color)
                display.blit(t_title, (20, 10))
                
                v_speed = 0.0
                if self.vehicle:
                    vel = self.vehicle.get_velocity()
                    v_speed = 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                    
                throttle = ctrl.get("throttle", 0.0)
                brake = ctrl.get("brake", 0.0)
                status_msg = ctrl.get("status", "Cruising safely")
                reasoning = ctrl.get("reasoning", "Road clear")
                
                aeb_text = "[ACTIVE INTERVENTION - EMERGENCY BRAKE]" if self.is_emergency_braking else "[AUTO-PILOT ACTIVE - JETSON MONITORING]"
                aeb_color = (255, 50, 50) if self.is_emergency_braking else (100, 255, 150)
                
                t_ctrl = font_main.render(f"Speed: {v_speed:4.1f} km/h | Mode: {aeb_text}", True, aeb_color)
                display.blit(t_ctrl, (20, 42))
                
                t_decision = font_small.render(f"Safety Decision : {status_msg[:65]}", True, (255, 220, 100))
                display.blit(t_decision, (20, 75))
                
                t_reason = font_small.render(f"Cognitive VLM   : {reasoning[:70]}", True, (150, 220, 255))
                display.blit(t_reason, (20, 98))
                
                pygame.display.flip()

    def destroy(self):
        self.running = False
        if self.camera:
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        if HAS_PYGAME:
            pygame.quit()
        print("[HIL Server] Cleaned up actors and closed bridge server.")

if __name__ == "__main__":
    server = CarlaHILServer()
    server.setup_carla()
    
    sock_thread = threading.Thread(target=server.socket_server_worker, daemon=True)
    sock_thread.start()
    
    try:
        server.run_display()
    except KeyboardInterrupt:
        pass
    finally:
        server.destroy()
