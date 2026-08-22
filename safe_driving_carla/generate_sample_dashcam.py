"""
Generates a realistic synthetic driving dashcam video for testing the Safe Driving Pipeline.
"""
import cv2
import numpy as np

def create_sample_driving_video(output_path="/home/tesla/safe_driving_carla/sample_dashcam.mp4", duration_sec=10, fps=30):
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration_sec * fps
    print(f'[VideoGen] Generating {total_frames} frames of driving video at {output_path}...')
    
    for f in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 1. Sky & Horizon
        frame[:int(height * 0.45), :] = [220, 190, 150] # Daytime sky
        
        # 2. Road surface
        frame[int(height * 0.45):, :] = [55, 55, 55]    # Asphalt Road
        
        # 3. Sidewalks & Buildings
        frame[int(height * 0.35):int(height * 0.45), :int(width * 0.25)] = [120, 130, 140]
        frame[int(height * 0.35):int(height * 0.45), int(width * 0.75):] = [130, 140, 150]
        
        # 4. Perspective Lane Lines (Moving effect)
        dash_offset = (f * 15) % 80
        # Left Lane Marker
        cv2.line(frame, (int(width * 0.48), int(height * 0.45)), (int(width * 0.15), height), (255, 255, 255), 4)
        # Right Lane Marker
        cv2.line(frame, (int(width * 0.52), int(height * 0.45)), (int(width * 0.85), height), (255, 255, 255), 4)
        # Center dashed line
        for y in range(int(height * 0.45) + dash_offset, height, 80):
            scale = (y - height * 0.45) / (height * 0.55)
            x = int(width * 0.50)
            cv2.line(frame, (x, y), (x, min(height, y + 40)), (0, 220, 255), max(2, int(6 * scale)))
            
        # 5. Lead Vehicle ahead (Cruising)
        lead_scale = 0.8 + 0.2 * np.sin(f * 0.05)
        car_cx = int(width * 0.50 + 20 * np.sin(f * 0.03))
        car_cy = int(height * 0.55)
        cw, ch = int(120 * lead_scale), int(80 * lead_scale)
        cv2.rectangle(frame, (car_cx - cw//2, car_cy - ch), (car_cx + cw//2, car_cy), (180, 50, 40), -1)
        cv2.rectangle(frame, (car_cx - cw//2 + 10, car_cy - ch + 10), (car_cx + cw//2 - 10, car_cy - ch//2), (40, 40, 40), -1) # Window
        cv2.circle(frame, (car_cx - cw//3, car_cy - 10), 10, (0, 0, 255), -1) # Brake light
        cv2.circle(frame, (car_cx + cw//3, car_cy - 10), 10, (0, 0, 255), -1) # Brake light
        
        # 6. Event: Pedestrian crossing from right to left between frame 60 and 240
        if 60 <= f <= 240:
            progress = (f - 60) / 180.0
            ped_x = int(width * 0.85 - progress * (width * 0.45))
            ped_y = int(height * 0.65 + progress * 50)
            p_scale = 0.9 + 0.4 * progress
            
            # Draw pedestrian
            head_r = int(15 * p_scale)
            body_h = int(65 * p_scale)
            body_w = int(30 * p_scale)
            cv2.circle(frame, (ped_x, ped_y - body_h), head_r, (60, 140, 240), -1)
            cv2.rectangle(frame, (ped_x - body_w//2, ped_y - body_h + head_r), (ped_x + body_w//2, ped_y), (40, 90, 200), -1)
            
        out.write(frame)
        
    out.release()
    print(f'[VideoGen] Finished creating sample video: {output_path}')

if __name__ == '__main__':
    create_sample_driving_video()
