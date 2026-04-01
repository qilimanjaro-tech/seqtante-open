# Copyright 2023 Qilimanjaro Quantum Tech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


def decibels(s21: np.ndarray):
    """Convert result values from s21 into dB

    Args:
        s21 (np.ndarray): combination of I + iQ

    Returns:
        np.ndarray: dB conversion
    """

    return 20 * np.log10(np.abs(s21))


def rotate_iq(arr: np.ndarray):
    """
    Function to rotate the I vs Q matrix to synthesized the data obtained
    into a single dimension (Real component)
    leaving the other only with noise (Imaginary component)

    Args:
        arr (np.array(float)): array of complex numbers corresponding to IQ signal

    Returns:
        np.array(float): rotated values
    """
    # Compute the covariance matrix
    cov = np.cov(arr.real, arr.imag)
    # Get the eigenvalues and eigenvectors of the covariance matrix
    w, v = np.linalg.eig(cov)
    # Find the index of the max eigenvalue
    max_idx = np.argmax(w)
    # Compute the angle of rotation
    angle = np.arctan2(v[max_idx, 1], v[max_idx, 0])
    # Rotate the array
    rotated = arr * np.exp(1j * angle)
    return rotated


def two_tone_spectroscopy_map(S21: np.ndarray):
    """
    Rotates the data and subracts the minimum for each of the frequency traces
    of a 2D two tone spectroscopy map map.

    Args:
        S21 (np.array(float)): matrix of complex numbers corresponding to IQ, Axis 0 is a frequency sweep
        signals of a 2D two tone spectroscopy map. Second index should correspond to frequency

    Returns:
        np.array(float): rotated values

    """
    result = np.zeros(S21.shape)
    for ii in range(S21.shape[0]):
        rotated = rotate_iq(S21[ii, :]).real
        if np.mean(rotated) > (min(rotated) + max(rotated)) / 2:
            rotated = -rotated
        rotated = rotated - min(rotated)
        result[ii, :] = rotated
    return result


def subtract_mean(arr: np.ndarray):
    """
    Substract the mean from an array.

    Args:
        arr (np.array(float)): array to substract mean from

    Returns:
        np.array(float): array with mean substracted from values

    """
    return arr - np.mean(arr)


def normalize(arr: np.ndarray):
    """
    Normalize an array.

    Args:
        arr (np.array(float)): array to be normalized

    Returns:
        np.array(float): normalized array

    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def sss_from_center_span_npoints(center: float, span: float, npoints: int) -> tuple:
    """
    Get the start, stop and step values for a given center, span and number of points.

    Args:
        center (float): center value
        span (float): span value
        npoints (int): number of points

    Returns:
        tuple: start, stop and step values
    """
    start = center - span / 2
    stop = center + span / 2
    step = span / npoints
    return start, stop, step


def sss_from_array(arr: np.ndarray):
    """
    Get the start, stop and step values from a numpy array.

    Args:
        arr (np.array): numpy array
    Returns:
        tuple: start, stop and step values

    """
    start = arr[0]
    stop = arr[-1]
    step = arr[1] - arr[0]
    return start, stop, step


def array_from_center_span_npoints(center, span, npoints) -> np.ndarray:
    """
    Get the start, stop and step values for a given center, span and number of points.
    Args:
        center (float): center value
        span (float): span value
        npoints (int): number of points
    Returns:
        np.array: numpy array with start, stop and step values
    """
    start = center - span / 2
    stop = center + span / 2
    arr = np.linspace(start, stop, npoints)
    return arr


def center_span_npoints_from_array(arr):
    center = (arr[0] + arr[-1]) / 2
    span = arr[-1] - arr[0]
    npoints = len(arr)
    return center, span, npoints


def compensate_round(x, base=1):
    return base * round(x / base)