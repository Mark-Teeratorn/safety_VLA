"""
System Configuration for Safe Driving Pipeline with CARLA Simulator
"""
from dataclasses import dataclass, field

@dataclass
class CarlaConfig:
    host: str = '10.100.16.130'
    port: int = 5555
    timeout: float = 10.0
    image_width: int = 1280
    image_height: int = 720
    fov: int = 90
    sensor_fps: int = 30

@dataclass
class PerceptionConfig:
    yolo_model_path: str = '/home/tesla/yolov8n.pt'
    conf_threshold: float = 0.40
    roi_expansion_ratio: float = 0.20
    critical_distance_threshold: float = 25.0

@dataclass
class ReasonerConfig:
    model_path: str = '/home/tesla/models/Cosmos-Reason2-2B-BF16-split-00001-of-00002.gguf'
    n_gpu_layers: int = 99
    max_tokens: int = 32
    temperature: float = 0.2
    vlm_query_interval: float = 0.35

@dataclass
class SystemConfig:
    carla: CarlaConfig = field(default_factory=CarlaConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    reasoner: ReasonerConfig = field(default_factory=ReasonerConfig)
