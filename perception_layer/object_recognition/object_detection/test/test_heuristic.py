"""Test routines for the mask_centroid_heuristic functions."""

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

import numpy as np

import object_detection.mask_centroid_heuristic as hr

CLASSES = [
    'banana',
    'bowl',
    'plate',
    'fork',
    'knife'
]


def test_get_center():
    """Test the get_center function."""
    # test the correct functioning in even case
    bounding_box = [700, 1000, 720, 1100]
    center = hr.get_center(bounding_box)
    oracle_center = np.array([710, 1050])
    assert np.array_equal(oracle_center, center) is True

    # test the correct functioning in odd case
    bounding_box = [700, 1000, 719, 1009]
    center = hr.get_center(bounding_box)
    oracle_center = np.array([709, 1004])
    assert np.array_equal(oracle_center, center) is True


def test_get_orientation():
    """Test the get_orientation function."""
    # test the correct functioning in even case
    bounding_box = [700, 1000, 720, 1100]
    orientation = hr.get_orientation(bounding_box)
    oracle_orientation = np.array([0, 0, 0, 1])
    assert np.array_equal(oracle_orientation, orientation) is True

    # test the correct functioning in odd case
    bounding_box = [700, 1000, 719, 1009]
    orientation = hr.get_orientation(bounding_box)
    oracle_orientation = np.array([0, 0, 0.7071068, 0.7071068])
    assert np.array_equal(oracle_orientation, orientation) is True


def test_banana_centroid():
    """Compute the centroid of the banana given its mask."""
    params = hr.Parameters()
    params.use_max = False
    mask = [
        [True, True, True, True, True, False, False, False, False, False, False, False, False],
        [True, True, True, True, True, True, True, False, False, False, False, False, False],
        [False, False, True, True, True, True, True, True, True, False, False, False, False],
        [False, False, False, False, True, True, True, True, True, True, False, False, False],
        [False, False, False, False, False, False, True, True, True, True, True, False, False],
        [False, False, False, False, False, False, False, False, True, True, True, True, False],
        [False, False, False, False, False, False, False, False, False, True, True, True, True],
        [False, False, False, False, False, False, False, False, False, False, True, True, True],
        [False, False, False, False, False, False, False, False, False, False, False, True, True],
        [False, False, False, False, False, False, False, False, False, False, False, True, True]
    ]
    mask = np.array(mask)
    # compute the center of the bounding box
    origin = [700, 1000]
    bounding_box = [origin[0], origin[1], origin[0] + mask.shape[1], origin[1] + mask.shape[0]]
    center = hr.get_center(bounding_box)
    normalized_center = np.array([center[0]-bounding_box[0], center[1]-bounding_box[1]])
    x_right = np.abs(bounding_box[2] - center[0] - 1)
    x_left = np.abs(center[0] - bounding_box[0] - 1)
    y_below = np.abs(bounding_box[3] - center[1] - 1)
    y_above = np.abs(center[1] - bounding_box[1] - 1)
    # scout in first drection (-1, -1)
    displacement = min(x_left, y_above)
    segment, bounds = hr.scout_mesh(mask, normalized_center, displacement, [-1, -1], params)
    segment_oracle = 3
    oracle_start = [4, 3]
    assert np.array_equal(segment, segment_oracle) is True
    assert np.array_equal(bounds[0], oracle_start) is True
    # scout in second drection (-1, 1)
    displacement = min(x_left, y_below)
    segment, bounds = hr.scout_mesh(mask, normalized_center, displacement, [-1, 1], params)
    segment_oracle = 1e4
    oracle_start = [-1, -1]
    assert np.array_equal(segment, segment_oracle) is True
    assert np.array_equal(bounds[0], oracle_start) is True
    # scout in third drection (1, -1)
    displacement = min(x_right, y_above)
    segment, bounds = hr.scout_mesh(mask, normalized_center, displacement, [1, -1], params)
    segment_oracle = 2
    oracle_start = [7, 4]
    assert np.array_equal(segment, segment_oracle) is True
    assert np.array_equal(bounds[0], oracle_start) is True
    # scout in fourth drection (1, 1)
    displacement = min(x_right, y_below)
    segment, bounds = hr.scout_mesh(mask, normalized_center, displacement, [1, 1], params)
    segment_oracle = 1e4
    oracle_start = [-1, -1]
    assert np.array_equal(segment, segment_oracle) is True
    assert np.array_equal(bounds[0], oracle_start) is True


def test_banana_heuristic():
    """Test the centroid heuristic for the banana."""
    params = hr.Parameters()
    params.use_max = False
    params.normalize = True
    mask = [
        [True, True, True, True, True, False, False, False, False, False, False, False, False],
        [True, True, True, True, True, True, True, False, False, False, False, False, False],
        [False, False, True, True, True, True, True, True, True, False, False, False, False],
        [False, False, False, False, True, True, True, True, True, True, False, False, False],
        [False, False, False, False, False, False, True, True, True, True, True, False, False],
        [False, False, False, False, False, False, False, False, True, True, True, True, False],
        [False, False, False, False, False, False, False, False, False, True, True, True, True],
        [False, False, False, False, False, False, False, False, False, False, True, True, True],
        [False, False, False, False, False, False, False, False, False, False, False, True, True],
        [False, False, False, False, False, False, False, False, False, False, False, True, True]
    ]
    mask = np.array(mask)
    # compute the center of the bounding box
    origin = [700, 1000]
    bounding_box = [origin[0], origin[1], origin[0] + mask.shape[1], origin[1] + mask.shape[0]]

    point, rotation = hr.compute_mask_frame(CLASSES, 'banana', bounding_box, mask, params)
    oracle_point = np.array([703, 1002])
    oracle_rotation = np.array([0, 0, -0.92387953,  0.38268343])
    assert np.array_equal(point, oracle_point)
    assert np.allclose(rotation, oracle_rotation, rtol=1e-3)


def test_bottle_heuristic():
    bounding_box = [700, 1000, 720, 1070]
    point, _ = hr.bottle_heuristic(bounding_box)

    dy = (bounding_box[3] - bounding_box[1])//8
    y_point = bounding_box[3] - dy
    oracle_point = np.array([point[0], y_point])

    assert np.array_equal(oracle_point, point) is True


def test_translate_center():
    """Test the translation function."""
    point = hr.translate_center(5, [1, 1], [10, 10])
    oracle_point = [12, 12]
    assert np.array_equal(oracle_point, point) is True
    point = hr.translate_center(5, [1, -1], [10, 10])
    oracle_point = [12, 8]
    assert np.array_equal(oracle_point, point) is True
    point = hr.translate_center(5, [-1, 1], [10, 10])
    oracle_point = [8, 12]
    assert np.array_equal(oracle_point, point) is True
    point = hr.translate_center(5, [-1, -1], [10, 10])
    oracle_point = [8, 8]
    assert np.array_equal(oracle_point, point) is True
