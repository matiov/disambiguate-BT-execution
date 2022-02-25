"""Heuristic to compute the centroid of an object given its mask and bounding boxes."""

# Copyright (c) 2022, ABB
# All rights reserved.
#
# Redistribution and use in source and binary forms, with
# or without modification, are permitted provided that
# the following conditions are met:
#
#   * Redistributions of source code must retain the
#     above copyright notice, this list of conditions
#     and the following disclaimer.
#   * Redistributions in binary form must reproduce the
#     above copyright notice, this list of conditions
#     and the following disclaimer in the documentation
#     and/or other materials provided with the
#     distribution.
#   * Neither the name of ABB nor the names of its
#     contributors may be used to endorse or promote
#     products derived from this software without
#     specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from perception_utils.transformations import quaternion_from_euler


@dataclass
class Parameters():
    """Parameter settings."""

    normalize: bool = False
    debug: bool = False
    use_max: bool = False


def compute_mask_frame(
    classes: List[str],
    object_category: str,
    bounding_box: List[int],
    mask: List[List[bool]],
    params: Parameters
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the frame attached to the object given an heuristic.

    The Heuristic is implemented only for the object 'banana'.
    Object specific heuristic can be implemented as well.
    If not specified, the heuristic returns the center of the bounding box as point,
    and a rotation quaternion representing a 0 degrees rotation.

    Args
    ----
        object: the object class according to the COCO dataset.
        bounding_box: bounding box returned by the detection algorithm
                      (expected to be like [Xmin, Ymin, Xmax, Ymax]).
        mask: mask of the object returned by the detection algorithm.
        normalize: if to use the normalized center or not.
                   Set to True if the the mask has same dimension of the object box.
                   Set to False if the mask has the same dimension of the image.
        params: parameters choice from the class Parameters.

    Returns
    -------
        point: position of the object in the image space.
        rotation: rotation of the object in the image space.

    """
    # This check is not really necessary
    if object_category not in classes:
        raise ValueError('No heuristic available for the target object!')
    if bounding_box[2] < bounding_box[0] or\
       bounding_box[3] < bounding_box[1]:
        raise ValueError('Unsupported convention for bounding box, use [Xmin, Ymin, Xmax, Ymax]!')

    point = np.array([0, 0])
    rotation = np.array([0, 0, 0, 1])

    if object_category == 'banana':
        point, rotation = banana_heuristic(bounding_box, mask, params)
    elif object_category == 'bottle':
        point, rotation = bottle_heuristic(bounding_box)
    else:
        point = get_center(bounding_box)
        rotation = get_orientation(bounding_box)

    return point, rotation


def bottle_heuristic(bounding_box: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the heuristic for the bottle.

    Take a point in the lowest 8th of the bounding box.
    """
    rotation = np.array([0, 0, 0, 1])

    point = get_center(bounding_box)

    height = bounding_box[3] - bounding_box[1]
    point[1] = bounding_box[3] - height//8

    return point, rotation


def banana_heuristic(
    bounding_box: List[int],
    mask: List[List[bool]],
    params: Parameters
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the heuristic for the banana.

    Process the mask and look for a segment that cut through it.
    Then take the mid point of that segment.
    """
    segments = []
    directions = []
    bounds = []
    center = get_center(bounding_box)
    normalized_center = center
    if params.normalize:
        normalized_center = np.array([center[0]-bounding_box[0], center[1]-bounding_box[1]])
    x_right = np.abs(bounding_box[2] - center[0] - 1)
    x_left = np.abs(center[0] - bounding_box[0] - 1)
    y_below = np.abs(bounding_box[3] - center[1] - 1)
    y_above = np.abs(center[1] - bounding_box[1] - 1)
    if params.debug:
        print(f'\n ---- CENTER of banana: {center} ----')
        print(f'\n ---- Normalized center: {normalized_center} ----')
        print(f'\n ---- Directions: {x_right, x_left, y_above, y_below} ----')
    # scout in first drection (-1, -1)
    displacement = min(x_left, y_above)
    direction = [-1, -1]
    segment, pixels = scout_mesh(mask, normalized_center, displacement, direction, params)
    segments.append(segment)
    directions.append(direction)
    bounds.append(pixels)
    if params.debug:
        print(f'\n ---- SCOUTING in direction: {direction} ----')
        print(f' ---- Resulting segment: {segment} ----')
        print(f' ---- Resulting bounds: {pixels} ----')
        print(f' ---- Displacement: {displacement} ----')
    # scout in second drection (-1, 1)
    displacement = min(x_left, y_below)
    direction = [-1, 1]
    segment, pixels = scout_mesh(mask, normalized_center, displacement, direction, params)
    segments.append(segment)
    directions.append(direction)
    bounds.append(pixels)
    if params.debug:
        print(f'\n ---- SCOUTING in direction: {direction} ----')
        print(f' ---- Resulting segment: {segment} ----')
        print(f' ---- Resulting bounds: {pixels} ----')
        print(f' ---- Displacement: {displacement} ----')
    # scout in third drection (1, -1)
    displacement = min(x_right, y_above)
    direction = [1, -1]
    segment, pixels = scout_mesh(mask, normalized_center, displacement, direction, params)
    segments.append(segment)
    directions.append(direction)
    bounds.append(pixels)
    if params.debug:
        print(f'\n ---- SCOUTING in direction: {direction} ----')
        print(f' ---- Resulting segment: {segment} ----')
        print(f' ---- Resulting bounds: {pixels} ----')
        print(f' ---- Displacement: {displacement} ----')
    # scout in fourth drection (1, 1)
    displacement = min(x_right, y_below)
    direction = [1, 1]
    segment, pixels = scout_mesh(mask, normalized_center, displacement, direction, params)
    segments.append(segment)
    directions.append(direction)
    bounds.append(pixels)
    if params.debug:
        print(f'\n ---- SCOUTING in direction: {direction} ----')
        print(f' ---- Resulting segment: {segment} ----')
        print(f' ---- Resulting bounds: {pixels} ----')
        print(f' ---- Displacement: {displacement} ----')

    res_segment, direction, segment_bounds = get_banana_params(
        segments, directions, bounds, params)
    if params.debug:
        print(f'\n ---- BANANA PARAMETERS ----')
        print(f' ---- Resulting segment: {res_segment} ----')
        print(f' ---- Resulting bounds: {segment_bounds} ----')
        print(f' ---- Direction: {direction} ----')

    # point_x = (segment_bounds[0][0] + segment_bounds[1][0])//2
    # point_y = (segment_bounds[0][1] + segment_bounds[1][1])//2
    # point = np.array([point_x, point_y])
    point = translate_center(res_segment, direction, segment_bounds[0])
    euler_rotation = get_segment_rotation(segment_bounds)
    rotation = quaternion_from_euler(0, 0, euler_rotation)
    # translate the center in the correct position in the image
    if params.normalize:
        point[0] = point[0] + bounding_box[0]
        point[1] = point[1] + bounding_box[1]

    return point, rotation


def get_banana_params(
    segments: List[int],
    directions: List[List[int]],
    bounds: List[List[np.ndarray]],
    params: Parameters
) -> Tuple[int, List[int], List[np.ndarray]]:
    """Compute the relevant parameters of the BANANA heuristic to determine the centroid."""
    if params.use_max:
        # get max of list
        max_segment = max(segments)
        index = segments.index(max_segment)
        segments.pop(index)
        # get second max
        res_segment = max(segments)
        if res_segment > 0:
            # then it's valid
            directions.pop(index)
            bounds.pop(index)
            index = segments.index(res_segment)
        else:
            res_segment = max_segment
        direction = directions[index]
        segment_bounds = bounds[index]

    else:
        # get min of list
        min_segment = min(segments)
        index = segments.index(min_segment)
        segments.pop(index)
        # get second max
        res_segment = min(segments)
        if res_segment < 1e4:
            # then it's valid
            directions.pop(index)
            bounds.pop(index)
            index = segments.index(res_segment)
        else:
            res_segment = min_segment
        direction = directions[index]
        segment_bounds = bounds[index]

    return res_segment, direction, segment_bounds


def get_segment_rotation(segment_bounds: List[np.ndarray]) -> float:
    """Compute the rotation angle of the segment as euler angle in radians."""
    start_pixel = segment_bounds[0]
    end_pixel = segment_bounds[1]
    delta_x = end_pixel[0] - start_pixel[0]
    delta_y = end_pixel[1] - start_pixel[1]
    angle = np.arctan2(delta_y, delta_x)
    return angle


def get_center(bounding_box: List[int]) -> np.ndarray:
    """Get the center of the bounding box."""
    x_center = int((bounding_box[2] + bounding_box[0])//2)
    y_center = int((bounding_box[3] + bounding_box[1])//2)

    if type(x_center) is not int or type(y_center) is not int:
        raise TypeError('ERROR in the function: not able to perform integer division.')

    center = np.array([x_center, y_center])
    return center


def translate_center(
    segment: int,
    direction: List[int],
    start: List[int]
) -> np.ndarray:
    """
    Compute the centroid of the object given heuristic results.

    Args
    ----
        segment: length in pixels of the mask segment belonging to the object.
        direction: in which direction to apply the increment to compute the center.
        start: pixel where the mask segment starts.

    Returns
    -------
        point: center of the segment given the direction.

    """
    # x = x0 + iL where i = +1 or -1
    x = start[0] + direction[0]*(segment//2)
    # y = y0 + jL where j = +1 or -1
    y = start[1] + direction[1]*(segment//2)

    point = np.array([x, y])
    return point


def get_orientation(bounding_box: List[int]) -> np.ndarray:
    """
    Get the orientation of the bounding box.

    Orientation can either be 0 or +90 degrees, returned in quaternion notation.

    """
    dx = int(bounding_box[2] - bounding_box[0])
    dy = int(bounding_box[3] - bounding_box[1])

    # normally, no rotation to apply
    orientation = np.array([0, 0, 0, 1])
    if dx > dy:
        # rotate by +90 degrees.
        orientation = np.array([0, 0, 0.7071068, 0.7071068])

    return orientation


def scout_mesh(
    mask: List[List[bool]],
    center: np.ndarray,
    displacement: int,
    direction: List[int],
    params: Parameters
) -> Tuple[int, List[np.ndarray]]:
    """
    Compute the centroid of the object given an heuristic.

    Args
    ----
        mask: mask of the object returned by the detection algorithm.
        center: the center of the bounding box (must be normalized).
            To normalize the center, remove the values for Xmin and Ymin.
        displacement: number of pixels to scout.
        direction: dtermines the diagonal along which to scout the mesh.

    Returns
    -------
        segment: minimum segment of the mask along the diagonals.
        segment_start: pixel coordinate where the segment starts.

    """
    intersections = []
    n_intersections = 0
    segment = 0
    x = []
    y = []
    for k in range(displacement):
        i = center[0] + direction[0]*(k+1)
        j = center[1] + direction[1]*(k+1)
        if mask[j][i]:
            # then the pixel i,j belongs to the object.
            # check if it's the first pixel of a segment.
            if segment == 0:
                n_intersections += 1
                x.append(i)
                y.append(j)
            segment += 1
        else:
            # then the pixel i,j does not belong to the object so we reset.
            if segment > 0:
                intersections.append(segment)
            segment = 0
    # in case last point is still object point, append current segment.
    if segment > 0:
        intersections.append(segment)

    if n_intersections > 1 or n_intersections == 0:
        # then we are at the extremes of the object or there is no object.
        error_array = np.array([-1, -1])
        error_val = int(params.use_max)*(-1.0)*1e4 + int(not params.use_max)*1.0*1e4
        return error_val, [error_array, error_array]

    # there should be just 1 intersection, but better to be sure.
    # here, segment is a measure of length.
    segment = intersections[0]
    segment_start = np.array([min(x), min(y)])
    segment_end = np.array([
        segment_start[0] + direction[0]*segment,
        segment_start[1] + direction[1]*segment
    ])
    segment_bounds = [segment_start, segment_end]
    return segment, segment_bounds
