#!/usr/bin/env python3
"""
Interactive WASD Manual Driving (with Auto-Reverse on S) + NVIDIA Jetson AI Speed Advisory
"""
import os
import sys
import time
import math
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
import pygame

HOST = "0.0.0.0"
PORT = 5555
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

class KeyboardControl:
    def __init__(self):
        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self.reverse = False
        self.handbrake = False
        self.autopilot = False

    def parse_events(self, client, vehicle, spawn_points):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_p:
                    self.autopilot = not self.autopilot
                    vehicle.set_autopilot(self.autopilot)
                elif event.key == pygame.K_r:
                    if spawn_points:
                        sp = random.choice(spawn_points)
                        vehicle.set_transform(sp)
                        self.throttle = 0.0
                        self.brake = 0.0
                        self.steer = 0.0
                        self.reverse = False
                        
        keys = pygame.key.get_pressed()
        
        # Turn off autopilot if user drives manually
        if any([keys[pygame.K_w], keys[pygame.K_s], keys[pygame.K_a], keys[pygame.K_d],
                keys[pygame.K_UP], keys[pygame.K_DOWN], keys[pygame.K_LEFT], keys[pygame.K_RIGHT]]):
            if self.autopilot:
                self.autopilot = False
                vehicle.set_autopilot(False)

        if not self.autopilot:
            vel = vehicle.get_velocity()
            speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            
            # --- W KEY: FORWARD DRIVE / BRAKE IF IN REVERSE ---
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                if self.reverse:
                    # If reversing and pressing W: Brake first to stop
                    self.throttle = 0.0
                    self.brake = min(self.brake + 0.3, 1.0)
                    if speed_kmh < 1.0:
                        self.reverse = False # Switch to Forward
                        self.brake = 0.0
                else:
                    self.reverse = False
                    self.brake = 0.0
                    self.throttle = min(self.throttle + 0.15, 1.0)
            else:
                if not (keys[pygame.K_s] or keys[pygame.K_DOWN]):
                    self.throttle = max(self.throttle - 0.25, 0.0)

            # --- S KEY: BRAKE IF MOVING FORWARD / REVERSE IF STOPPED ---
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                if not self.reverse and speed_kmh > 2.0:
                    # Moving forward: S acts as powerful brake
                    self.throttle = 0.0
                    self.brake = min(self.brake + 0.35, 1.0)
                else:
                    # Stationary or already reversing: S acts as REVERSE GAS!
                    self.reverse = True
                    self.brake = 0.0
                    self.throttle = min(self.throttle + 0.15, 1.0)
            else:
                if not (keys[pygame.K_w] or keys[pygame.K_UP]):
                    self.brake = max(self.brake - 0.30, 0.0)

            # --- A / D KEYS: STEERING ---
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                self.steer = max(self.steer - 0.18, -1.0)
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                self.steer = min(self.steer + 0.18, 1.0)
            else:
                self.steer = self.steer * 0.6 # Quick center return

            # --- SPACEBAR: HANDBRAKE ---
            self.handbrake = bool(keys[pygame.K_SPACE])
            if self.handbrake:
                self.throttle = 0.0
                self.brake = 1.0
            
            ctrl = carla.VehicleControl(
                throttle=float(self.throttle),
                brake=float(self.brake),
                steer=float(self.steer),
                hand_brake=self.handbrake,
                reverse=self.reverse
            )
            vehicle.apply_control(ctrl)
            
        return True


class CarlaManualHILServer:
    def __init__(self, carla_host="localhost", carla_port=2000):
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        
        self.latest_raw_frame = None
        self.latest_advisory = {
            "rec_speed": 40.0,
            "risk": "LOW",
            "advisory": "Road clear. Safe to drive.",
            "status": "Cruising normally",
            "reasoning": "Road path is clear",
            "targets": []
        }
        self.connected_client = None
        self.running = True
        self.lock = threading.Lock()
        self.spawn_points = []
        self.keyboard = KeyboardControl()
        
    def setup_carla(self):
        print(f"[Manual HIL] Connecting to CARLA at {self.carla_host}:{self.carla_port}...")
        self.client = carla.Client(self.carla_host, self.carla_port)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        
        # Clean existing ego vehicles
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name") == "ego_vehicle":
                actor.destroy()
                
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        vehicle_bp.set_attribute("role_name", "ego_vehicle")
        
        self.spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = self.spawn_points[0] if self.spawn_points else carla.Transform(carla.Location(x=0, y=0, z=2))
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"[Manual HIL] Spawned Player Vehicle: {self.vehicle.type_id} (ID: {self.vehicle.id})")
        
        # Mount Dashcam Camera
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
        cam_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
        cam_bp.set_attribute("fov", "90")
        
        cam_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.camera.listen(self._on_camera_image)
        print(f"[Manual HIL] Dashcam Camera attached (ID: {self.camera.id})")
        
        # Spawn some NPC vehicles & walkers around town
        self._spawn_traffic(bp_lib)

    def _spawn_traffic(self, bp_lib):
        npc_bps = bp_lib.filter("vehicle.*")
        for i in range(1, min(25, len(self.spawn_points))):
            try:
                bp = np.random.choice(npc_bps)
                v = self.world.spawn_actor(bp, self.spawn_points[i])
                v.set_autopilot(True)
            except Exception:
                pass

    def _on_camera_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        bgr = array[:, :, :3]
        with self.lock:
            self.latest_raw_frame = bgr.copy()

    def socket_server_worker(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen(1)
        print(f"[Manual HIL] Socket Server listening on {HOST}:{PORT} (Waiting for Jetson Brain)...")
        
        while self.running:
            try:
                conn, addr = server_sock.accept()
                print(f"[Manual HIL] >>> CONNECTED to Jetson: {addr[0]}:{addr[1]} <<<")
                self.connected_client = conn
                self._handle_client(conn)
            except Exception as e:
                if self.running:
                    print(f"[Manual HIL] Socket error: {e}")
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
            
        print("[Manual HIL] Jetson disconnected.")
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
                
                advisory_dict = json.loads(msg_data.decode("utf-8"))
                with self.lock:
                    self.latest_advisory = advisory_dict
            except Exception:
                break

    def run_gui(self):
        pygame.init()
        display = pygame.display.set_mode((IMAGE_WIDTH, IMAGE_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("CARLA Manual WASD Drive + NVIDIA Jetson AI Advisory & Speed Predictor")
        clock = pygame.time.Clock()
        
        font_large = pygame.font.SysFont("monospace", 26, bold=True)
        font_title = pygame.font.SysFont("monospace", 20, bold=True)
        font_main = pygame.font.SysFont("monospace", 18, bold=True)
        font_small = pygame.font.SysFont("monospace", 15)
        
        while self.running:
            clock.tick(30)
            
            # Process WASD Keyboard Driving
            if not self.keyboard.parse_events(self.client, self.vehicle, self.spawn_points):
                self.running = False
                break
                
            frame = None
            adv = {}
            with self.lock:
                if self.latest_raw_frame is not None:
                    frame = self.latest_raw_frame.copy()
                adv = dict(self.latest_advisory)
                
            if frame is not None:
                # Draw YOLOv8 detections from Jetson
                targets = adv.get("targets", [])
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
                
                # --- TOP HUD: JETSON AI MODEL ADVISORY ---
                top_hud = pygame.Surface((IMAGE_WIDTH, 140))
                top_hud.set_alpha(220)
                top_hud.fill((12, 16, 28))
                display.blit(top_hud, (0, 0))
                
                is_online = (self.connected_client is not None)
                badge_col = (50, 255, 120) if is_online else (255, 70, 70)
                badge_str = "JETSON ONLINE (tesla@100.106.22.30)" if is_online else "WAITING FOR JETSON..."
                t_badge = font_title.render(f"[AI CO-PILOT] {badge_str}", True, badge_col)
                display.blit(t_badge, (20, 10))
                
                rec_spd = adv.get("rec_speed", 40.0)
                risk = adv.get("risk", "LOW")
                advisory = adv.get("advisory", "Safe to drive")
                reasoning = adv.get("reasoning", "Road is clear")
                
                if risk == "CRITICAL":
                    risk_col = (255, 40, 40)
                elif risk == "HIGH":
                    risk_col = (255, 140, 30)
                elif risk == "MEDIUM":
                    risk_col = (255, 220, 50)
                else:
                    risk_col = (50, 255, 100)
                    
                t_rec = font_large.render(f"MODEL SPEED ADVISORY: {rec_spd:4.1f} km/h | RISK: [{risk}]", True, risk_col)
                display.blit(t_rec, (20, 40))
                
                t_adv = font_main.render(f"Action: {advisory}", True, (255, 255, 255))
                display.blit(t_adv, (20, 78))
                
                t_reas = font_small.render(f"Cosmos VLM Brain: {reasoning[:85]}", True, (160, 220, 255))
                display.blit(t_reas, (20, 108))
                
                # --- BOTTOM HUD: VEHICLE TELEMETRY & CONTROLS ---
                bot_hud = pygame.Surface((IMAGE_WIDTH, 85))
                bot_hud.set_alpha(220)
                bot_hud.fill((12, 16, 28))
                display.blit(bot_hud, (0, IMAGE_HEIGHT - 85))
                
                v_speed = 0.0
                if self.vehicle:
                    vel = self.vehicle.get_velocity()
                    v_speed = 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                    
                gear_str = "REVERSE [R]" if self.keyboard.reverse else "DRIVE [D]"
                gear_col = (255, 120, 50) if self.keyboard.reverse else (100, 255, 150)
                mode_str = "[AUTOPILOT (P)]" if self.keyboard.autopilot else "[MANUAL WASD]"
                
                # Speedometer
                spd_col = (100, 255, 150) if v_speed <= (rec_spd + 5) else (255, 100, 100)
                t_spd = font_large.render(f"SPEED: {v_speed:5.1f} km/h | GEAR: {gear_str} {mode_str}", True, spd_col)
                display.blit(t_spd, (20, IMAGE_HEIGHT - 75))
                
                # Controls info
                t_ctrl_info = font_small.render(f"Throttle: {self.keyboard.throttle:.2f} | Brake: {self.keyboard.brake:.2f} | Steer: {self.keyboard.steer:+.2f} | Keys: [W] เดินหน้า [S] เบรก/ถอยหลัง [A/D] เลี้ยว [Space] เบรกมือ [R] รีเซ็ต", True, (220, 220, 220))
                display.blit(t_ctrl_info, (20, IMAGE_HEIGHT - 38))
                
                pygame.display.flip()

    def destroy(self):
        self.running = False
        if self.camera:
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        pygame.quit()
        print("[Manual HIL] Cleaned up actors.")

if __name__ == "__main__":
    server = CarlaManualHILServer()
    server.setup_carla()
    
    sock_thread = threading.Thread(target=server.socket_server_worker, daemon=True)
    sock_thread.start()
    
    try:
        server.run_gui()
    except KeyboardInterrupt:
        pass
    finally:
        server.destroy()
