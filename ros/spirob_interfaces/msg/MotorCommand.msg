# Command sent from control_node -> hardware_node for all tendon motors at once.

std_msgs/Header header
int32[]   motor_ids
float64[] u              # controller output, aligned index-for-index with motor_ids
