# Relevant information for the current robot action target.

string action_type
string target_name
string frame_id
geometry_msgs/Pose target_pose

# If object recogniotnion is used, the bounding box of the object has the following convention:
# [x_min, y_min, X_MAX, Y_MAX], where all data are pixels in the image.
int16[] bounding_box
