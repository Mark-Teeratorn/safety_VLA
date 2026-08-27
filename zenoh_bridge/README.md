# Zenoh ROS2 Bridge — Laptop → Orin → NUC

Pure-Python, **ROS2-version-independent** bridge using `eclipse-zenoh` + `rclpy.serialization`.  
No `zenoh-bridge-ros2dds` needed. Works across Humble ↔ Jazzy ↔ any distro.

## How It Works

Each script uses `rclpy` only to interface with its **local** ROS2, while Zenoh handles all cross-machine transport:

```
[Laptop — ROS2 Humble]            [AGX Orin — ROS2 Jazzy]         [Intel NUC]
 laptop_publisher.py               orin_bridge.py                   nuc_subscriber.py

 rclpy.subscribe(topic)            zenoh.subscribe(key)             zenoh.subscribe(key)
   → serialize_message(msg)          → deserialize_message(bytes)     → deserialize_message(bytes)
     → zenoh.put(bytes)                → rclpy.publish(topic)            → rclpy.publish(topic)
                                                ↕
                                     rclpy.subscribe(vla_topic)
                                       → serialize_message(msg)
                                         → zenoh.put(bytes) ──────────────→
```

> **Key insight**: CDR serialization (`rclpy.serialization`) is stable for standard messages (nav_msgs, sensor_msgs) across ROS2 versions — only the DDS transport layer differs between distros, which Zenoh replaces entirely.

## Network Topology

```
  LAPTOP (192.168.1.10) ──── ETH ────► ORIN (192.168.1.20) ◄──── ETH ──── NUC (192.168.1.30)
                                         Zenoh Router
```

The Orin always runs as the **Zenoh ROUTER** (hub). Both Laptop and NUC connect **to** the Orin.

---

## Zenoh Key Expressions

| Direction          | Key                                          | ROS2 Topic (source/dest)                                   | Msg Type       |
|--------------------|----------------------------------------------|------------------------------------------------------------|----------------|
| Laptop → Orin      | `ros2/laptop/localization/kinematic_state`   | `/localization/kinematic_state`                            | Odometry       |
| Laptop → Orin      | `ros2/laptop/perception/yolox/image_compressed` | `/perception/.../tensorrt_yolox_node/out/image`         | CompressedImage|
| Laptop → Orin      | `ros2/laptop/sensing/camera0/image_raw_compressed` | `/sensing/camera/camera0/image_raw/compressed`       | CompressedImage|
| Orin → NUC         | `ros2/vla/deceleration_cmd`                  | `/vla/deceleration_cmd`                                    | Float32 *      |
| Orin → NUC         | `ros2/vla/brake_cmd`                         | `/vla/brake_cmd`                                           | Float32 *      |
| Orin → NUC         | `ros2/vla/safety_status`                     | `/vla/safety_status`                                       | String *       |

\* Update VLA output msg types in `orin_bridge.py` and `nuc_subscriber.py` once confirmed.

---

## Step 1 — Set Static IPs on Ethernet Ports

### Laptop (connects to Orin)
```bash
# Find your Ethernet interface (currently enp129s0, state DOWN — plug the cable first)
sudo ip addr add 192.168.1.10/24 dev enp129s0
sudo ip link set enp129s0 up
ping 192.168.1.20   # verify Orin is reachable
```

### AGX Orin — port facing Laptop
```bash
sudo ip addr add 192.168.1.20/24 dev eth0   # adjust interface name
sudo ip link set eth0 up
```

### AGX Orin — port facing NUC
```bash
sudo ip addr add 192.168.1.20/24 dev eth1   # adjust interface name (or same port via switch)
sudo ip link set eth1 up
```

### Intel NUC
```bash
sudo ip addr add 192.168.1.30/24 dev eth0   # adjust interface name
sudo ip link set eth0 up
ping 192.168.1.20   # verify Orin is reachable
```

---

## Step 2 — Install on Each Machine

```bash
pip install eclipse-zenoh
```

No other installation needed. The scripts only require:
- `eclipse-zenoh` (pip)
- `rclpy` (already part of your ROS2 install)
- Standard ROS2 message packages (`nav_msgs`, `sensor_msgs`, `std_msgs`)

---

## Step 3 — Launch Order

**Always start Orin first** — it's the router.

### 1. On AGX Orin
```bash
source /opt/ros/jazzy/setup.bash
python3 orin_bridge.py
# Uses default port 7447, listens on 0.0.0.0
```

### 2. On Laptop
```bash
source /opt/ros/humble/setup.bash
# Make sure Autoware is running first
python3 laptop_publisher.py --orin-ip 192.168.1.20
```

### 3. On Intel NUC
```bash
source /opt/ros/<distro>/setup.bash
python3 nuc_subscriber.py --orin-ip 192.168.1.20
```

---

## Step 4 — Verify

### On Orin — should see incoming topics published locally:
```bash
ros2 topic hz /localization/kinematic_state
ros2 topic hz /sensing/camera/camera0/image_raw/compressed
```

### On NUC — should see VLA commands:
```bash
ros2 topic echo /vla/brake_cmd
ros2 topic echo /vla/deceleration_cmd
```

---

## Updating VLA Output Topics

When the VLA output topic names/types are confirmed, update these two files:

**`orin_bridge.py`** — change the subscription topic name and import:
```python
# Example: if VLA outputs autoware_control_msgs/Control
from autoware_control_msgs.msg import Control
self.create_subscription(Control, "/control/command/control_cmd", ...)
```

**`nuc_subscriber.py`** — match the same type:
```python
from autoware_control_msgs.msg import Control
msg = deserialize_message(bytes(sample.payload.to_bytes()), Control)
```

---

## File Reference
```
zenoh_bridge/
├── laptop_publisher.py    ← Run on Laptop (ROS2 Humble)
├── orin_bridge.py         ← Run on AGX Orin (ROS2 Jazzy) — Zenoh router
├── nuc_subscriber.py      ← Run on Intel NUC
├── requirements.txt       ← pip install eclipse-zenoh
└── config/                ← (unused in Python mode, kept for reference)
    ├── laptop_bridge.json5
    ├── orin_bridge.json5
    └── nuc_bridge.json5
```
