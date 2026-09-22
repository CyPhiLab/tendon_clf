# Raw per-motor feedback published by hardware_node, one message per motor.
# Units are SI (radians, rad/s) — converted from the AK motor's native
# degrees/RPM inside hardware_node before publishing.

std_msgs/Header header
int32   motor_id
float64 position        # rad
float64 velocity         # rad/s
float64 current          # A
float64 app_ctrl          # 
int32   fault_code       # AKFault enum value from protocol.py, 0 = none

