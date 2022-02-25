std_msgs/Header header
string category_str
int16 object_id

# If object recogniotnion is used, the bounding box of the object has the following convention:
# [x_min, y_min, X_MAX, Y_MAX], where all data are pixels in the image.
int16[] bounding_box
BoolMask mask

bool disambiguated
