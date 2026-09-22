# Fused/estimated full robot state published by state_estimation_node -> control_node & plot_node

std_msgs/Header header
float64[] q               # estimated joint/generalized positions (size = model.nq)
float64[] dq              # estimated joint/generalized velocities (size = model.nv)
float64[] task_pos        # estimated task-space position, size = robot.task_dim
float64[] task_vel        # estimated task-space velocity, size = robot.task_dim
bool      is_valid         # false until the estimator has produced a first valid estimate
