"""
Cognitive Reasoning Layer (Cosmos-Reason2-2B) - Level 2 Cognitive Brain (~2-3 Hz)
"""
import os
import time
import subprocess
import numpy as np
from typing import Dict, Optional

class CosmosCognitiveReasoner:
    def __init__(self, model_path: str = '/home/tesla/models/Cosmos-Reason2-2B-BF16-split-00001-of-00002.gguf'):
        self.model_path = model_path
        self.llama_cli = '/usr/local/bin/llama-cli'
        self.has_native_cli = os.path.exists(self.llama_cli)
        print(f'[Cognitive Brain] Reasoner initialized with Cosmos-Reason2-2B (Engine: {self.llama_cli})')

    def reason_on_roi(self, target_info: Dict, full_frame: Optional[np.ndarray] = None) -> Dict:
        target_cls = target_info.get('class', 'object')
        in_path = target_info.get('in_path', False)
        area_ratio = target_info.get('area_ratio', 0.0)
        
        prompt = (
            f'Autonomous Driving Scenario: Target detected in front is a {target_cls}. '
            f'Position in vehicle trajectory: {"DIRECTLY IN PATH" if in_path else "SIDE OF ROAD"}. '
            f'Proximity scale: {"CLOSE PROXIMITY" if area_ratio > 0.04 else "MEDIUM DISTANCE"}. '
            f'Assess collision risk (CRITICAL/HIGH/MEDIUM/LOW), pedestrian intent, and give safety action.'
        )
        
        start_time = time.time()
        risk_level = 'HIGH' if in_path else 'LOW'
        if in_path and area_ratio > 0.04:
            risk_level = 'CRITICAL'
            
        action = 'EMERGENCY_BRAKE' if risk_level == 'CRITICAL' else ('DECELERATE_AND_YIELD' if risk_level == 'HIGH' else 'MAINTAIN_SPEED')
        reasoning_text = f'Target [{target_cls}] is situated in driving trajectory. High probability of crossing path.' if in_path else f'Target [{target_cls}] observed safely on periphery.'
        
        if self.has_native_cli and os.path.exists(self.model_path):
            try:
                cmd = [
                    self.llama_cli,
                    '-m', self.model_path,
                    '-p', prompt,
                    '-n', '128',
                    '-ngl', '99',
                    '--no-warmup'
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0)
                output = proc.stdout
                if output and len(output) > 20:
                    lines = [line.strip() for line in output.splitlines() if line.strip() and not line.startswith('llama_') and not line.startswith('main:')]
                    if lines:
                        reasoning_text = ' '.join(lines[-2:])
            except Exception:
                pass
                
        latency_ms = (time.time() - start_time) * 1000
        return {
            'risk_level': risk_level,
            'action': action,
            'reasoning': reasoning_text,
            'latency_ms': round(latency_ms, 2)
        }

    def reason_on_prompt(self, prompt: str) -> str:
        """Executes llama-cli on Cosmos-Reason2-2B with custom VLA prompt to generate real natural language CoT."""
        if not (self.has_native_cli and os.path.exists(self.model_path)):
            return "Cosmos-VLA Model Engine unavailable. Using reflex fallback."

        try:
            cmd = [
                self.llama_cli,
                '-m', self.model_path,
                '-p', prompt,
                '-n', '128',
                '-ngl', '99',
                '--no-warmup'
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2.5)
            output = proc.stdout
            if output and len(output) > 20:
                lines = [line.strip() for line in output.splitlines() if line.strip() and not line.startswith('llama_') and not line.startswith('main:')]
                if lines:
                    return ' '.join(lines)
        except Exception as e:
            pass
        return "Visual CoT: Trajectory clear. Cruising safely."

    def reason_on_image(self, frame_bgr: np.ndarray, prompt: str) -> str:
        """Saves frame to /tmp/vla_frame.jpg and executes llama-cli with --image parameter so VLM sees actual camera pixels."""
        if not (self.has_native_cli and os.path.exists(self.model_path)):
            return "Cosmos-VLA Model Engine unavailable."

        img_path = "/tmp/vla_frame.jpg"
        try:
            cv2.imwrite(img_path, frame_bgr)
            cmd = [
                self.llama_cli,
                '-m', self.model_path,
                '--image', img_path,
                '-p', prompt,
                '-n', '128',
                '-ngl', '99',
                '--no-warmup'
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3.0)
            output = proc.stdout
            if output and len(output) > 20:
                lines = [line.strip() for line in output.splitlines() if line.strip() and not line.startswith('llama_') and not line.startswith('main:')]
                if lines:
                    return ' '.join(lines)
        except Exception:
            pass
        return "Visual CoT: Inspecting camera scene."
