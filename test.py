import sys
import os
import SimpleITK as sitk
import glob
import logging.config
from argparse import ArgumentParser
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import inspect, copy
from scipy import ndimage
from importlib import import_module
import importlib, bisect
from enum import Enum
import logging, time
import collections
import pydicom as pyd
import functools
import json
import traceback
from torchvision.transforms import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import cv2
import PIL
# workspace_path = '/opt/libs'
# print("We add {} into python path for module lookup.".format(workspace_path))
# sys.path.append(workspace_path)

from models import CTSUNet

_SITK_INTERPOLATOR_DICT = {
    'nearest': sitk.sitkNearestNeighbor,
    'linear': sitk.sitkLinear,
    'gaussian': sitk.sitkGaussian,
    'label_gaussian': sitk.sitkLabelGaussian,
    'bspline': sitk.sitkBSpline,
    'hamming_sinc': sitk.sitkHammingWindowedSinc,
    'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
    'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
    'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
}

class ImageMetadata:
    special_fields = ('spacing', 'origin', 'direction')
    eps = 1e-5

    """Wrapper around the metadata of a medical image.

    Can store arbitrary data, but will report default values for three important fields:
     * spacing - defaults to 1mm spacing in all directions
     * origin - defaults to 0
     * direction - defaults to an identity matrix

    Parameters
    ----------
    ndim : int
        Number of dimensions of the image (>= 2)
    """
    def __init__(self, ndim, **kwargs):
        ndim = int(ndim)
        if not ndim > 1:
            raise ValueError(f'Invalid number of dimensions for an image, expected at least 2, got {ndim}')

        self.ndim = ndim
        self.spacing = np.array([1.0] * ndim, dtype=float)
        self.origin = np.array([0.0] * ndim, dtype=float)
        self.direction = np.identity(ndim, dtype=float)

        self.metadata = dict()
        for key, value in kwargs.items():
            self[key] = value

    @staticmethod
    def from_sitk(image):
        header = ImageMetadata(
            ndim=image.GetDimension(),
            spacing=image.GetSpacing(),
            origin=image.GetOrigin(),
            direction=image.GetDirection()
        )

        if image.HasMetaDataKey('SliceThickness'):
            header['slice_thickness'] = float(image.GetMetaData('SliceThickness'))
        if image.HasMetaDataKey('ConvolutionKernel'):
            header['convolution_kernel'] = image.GetMetaData('ConvolutionKernel')

        return header

    @staticmethod
    def from_file(filename):
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(filename))
        reader.ReadImageInformation()
        return ImageMetadata.from_sitk(reader)

    @staticmethod
    def from_dict(metadata):
        try:
            ndim = metadata['ndim']
        except KeyError:
            if 'spacing' in metadata:
                ndim = len(metadata['spacing'])
            elif 'origin' in metadata:
                ndim = len(metadata['origin'])
            elif 'direction' in metadata:
                ndim = len(metadata['direction'])**(1/2)
            else:
                raise ValueError('Could not determine dimensionality of the image')

        return ImageMetadata(ndim, **metadata)

    def to_dict(self):
        metadata = {
            'ndim': self.ndim,
            'spacing': self.spacing.astype(float).tolist(),
            'origin': self.origin.astype(float).tolist(),
            'direction': self.direction.flatten().astype(float).tolist()
        }

        for key, value in self.metadata.items():
            if isinstance(value, np.ndarray):
                metadata[key] = value.tolist()
            else:
                metadata[key] = value

        return metadata

    def __contains__(self, key):
        return key in self.metadata or key in self.special_fields

    def __getitem__(self, key):
        if key in self.special_fields:
            return getattr(self, key)
        else:
            return self.metadata[key]

    def __setitem__(self, key, value):
        if key == 'spacing':
            new_spacing = np.asarray(value, dtype=float).flatten()
            if new_spacing.size == 1:
                new_spacing = np.repeat(new_spacing, self.ndim)
            elif new_spacing.size != self.ndim:
                raise ValueError(
                    f'Pixel spacing of a {self.ndim}-dimensional image can only be a single value (isotropic) or {self.ndim} values')
            self.spacing = new_spacing.copy()
        elif key == 'origin':
            new_origin = np.asarray(value, dtype=float).flatten()
            if new_origin.size != self.ndim:
                raise ValueError(
                    f'Expected {self.ndim} values for coordinates of the origin of a {self.ndim}-dimensional image, '
                    f'got {new_origin.size} values')
            self.origin = new_origin.copy()
        elif key == 'direction':
            new_direction = np.asarray(value, dtype=float).flatten()

            # Check number of elements
            if new_direction.size != self.ndim ** 2:
                raise ValueError(
                    f'Expected {self.ndim ** 2} values for direction cosine matrix of a {self.ndim}-dimensional image, '
                    f'got {new_direction.size} values')
            new_direction = new_direction.reshape((self.ndim, self.ndim))

            # Check if matrix is orthogonal
            if not np.allclose(new_direction @ new_direction.T, np.identity(self.ndim), rtol=0, atol=self.eps):
                raise ValueError('Direction matrix needs to be orthogonal')

            self.direction = new_direction.copy()
        else:
            self.metadata[key] = value

    def __eq__(self, other):
        return np.allclose(self.spacing, other.spacing, rtol=0, atol=self.eps) and \
               np.allclose(self.origin, other.origin, rtol=0, atol=self.eps) and \
               np.allclose(self.direction, other.direction, rtol=0, atol=self.eps) and \
               self.metadata == other.metadata

    def __str__(self):
        summary = f'ndim={self.ndim}, ' \
            f'spacing={self.spacing.tolist()}, ' \
            f'origin={self.origin.tolist()}, ' \
            f'direction={self.direction.flatten().tolist()}'

        if len(self.metadata) > 0:
            summary += ', ' + ','.join(sorted(self.metadata.keys()))

        return f'ImageMetadata({summary})'

    def __repr__(self):
        return self.__str__() + super().__repr__()

    def __len__(self):
        return len(self.special_fields) + len(self.metadata)

    def __iter__(self):
        yield from self.keys()

    def keys(self):
        yield from self.special_fields
        yield from self.metadata.keys()

    def values(self):
        for key in self.keys():
            yield self[key]

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def copy(self):
        return copy.deepcopy(self)

    def has_default_direction(self, epsilon=0.01):
        """Tests whether the image's coordinate space has the standard orientation (no rotation)"""
        return bool(np.allclose(self.direction, np.identity(self.ndim), rtol=0, atol=epsilon))

    def has_regular_direction(self, epsilon=0.01):
        """Tests whether the image's coordinate space has a regular direction (swap/flips of axes, but no rotation, i.e. 0/1/-1 values only)"""
        return bool(np.all(
            np.isclose(self.direction, 0, rtol=0, atol=epsilon) | np.isclose(abs(self.direction), 1, rtol=0,
                                                                             atol=epsilon)
        ))

    def world_matrix(self):
        wm = np.zeros((self.ndim + 1, self.ndim + 1), dtype=float)
        wm[:self.ndim, :self.ndim] = self.direction * self.spacing
        wm[:self.ndim, -1] = self.origin
        wm[-1, -1] = 1
        return wm

    def physical_coordinates_to_indices(self, physical_coordinates, continuous=False):
        physical_coordinates = np.asanyarray(physical_coordinates).flatten()
        if physical_coordinates.size != self.ndim:
            raise ValueError(
                f'Expected a coordinate vector with {self.ndim} values, got {physical_coordinates.size} values')

        vector = np.concatenate((physical_coordinates, [1]))
        indices = np.linalg.pinv(self.world_matrix()) @ vector

        return indices[:-1] if continuous else np.around(indices[:-1]).astype(int)

    def indices_to_physical_coordinates(self, indices):
        indices = np.asanyarray(indices).flatten()
        if indices.size != self.ndim:
            raise ValueError(f'Expected an index vector with {self.ndim} values, got {indices.size} values')

        vector = np.concatenate((indices, [1]))
        coordinates = self.world_matrix() @ vector

        return coordinates[:-1]


def _reverse_axes(image):
    return np.transpose(image, tuple(reversed(range(image.ndim))))


def sitk_to_numpy(image, only_data=True):
    data = _reverse_axes(sitk.GetArrayFromImage(image))
    if only_data:
        return data
    else:
        return data, ImageMetadata.from_sitk(image)


def numpy_to_sitk(data, header=None):
    if data.ndim not in (2, 3, 4):
        raise ValueError(f'Cannot convert {data.ndim}D image to SimpleITK Image, only 2D, 3D and 4D are supported')

    if data.ndim == 4:
        # Turn data into series of 3D volumes and combine them into a 4D image
        if header:
            if not np.allclose(header['direction'][3, :].flatten(), (0, 0, 0, 1), rtol=0, atol=0.001) or \
                    not np.allclose(header['direction'][:, 3].flatten(), (0, 0, 0, 1), rtol=0, atol=0.001):
                raise ValueError('Cannot convert 4D array with rotation in 4th dimension to SimpleITK image')

            header3 = ImageMetadata(
                ndim=3,
                spacing=header['spacing'][:3],
                origin=header['origin'][:3],
                direction=header['direction'][:3, :3]
            )
            spacing4 = header['spacing'][3]
            origin4 = header['origin'][3]
        else:
            header3 = ImageMetadata(ndim=3)
            spacing4 = 1
            origin4 = 0

        image = sitk.JoinSeries(
            [numpy_to_sitk(data[:, :, :, i], header3) for i in range(data.shape[3])],
            origin4, spacing4
        )
    else:
        image = sitk.GetImageFromArray(_reverse_axes(np.asarray(data)))

        if header:
            # Copy standard header (spacing/origin/direction cosine matrix)
            if 'spacing' in header:
                image.SetSpacing([float(f) for f in header['spacing']])
            if 'origin' in header:
                image.SetOrigin([float(f) for f in header['origin']])
            if 'direction' in header:
                image.SetDirection([float(f) for f in np.asanyarray(header['direction']).flatten()])

    if header:
        # Copy additional metadata
        if 'slice_thickness' in header:
            image.SetMetaData('SliceThickness', str(header['slice_thickness']))
        if 'convolution_kernel' in header:
            image.SetMetaData('ConvolutionKernel', header['convolution_kernel'])

    return image


def change_direction(image, header=None, new_direction=(1, 0, 0, 0, 1, 0, 0, 0, 1), new_origin=None,
                     outside_val=0, only_nearest_neighbor_interpolation=False, eps=0.00001):
    """Changes the direction of an image, e.g., from coronal to axial slices"""
    # If a header was supplied, assume that the image is a numpy array (otherwise a simpleitk image)
    if header is not None:
        image = numpy_to_sitk(image, header)
    if np.allclose(image.GetDirection(), new_direction, rtol=0, atol=0.001):
        return image
    new_direction = np.asarray(new_direction).reshape(3, 3)
    if abs(np.linalg.det(new_direction) - 1) > eps:
        raise ValueError('Invalid direction cosine matrix specified, determinate != 1')

    # Compute new origin (didn't figure it out for switching to/from non-regular directions yet)
    if new_origin is None:
        transformed_base = new_direction.dot([1, 1, 1])
        if not np.allclose(np.abs(transformed_base), 1):
            raise ValueError('New origin can currently only be calculated for regular directions')

        lower_corner = image.TransformIndexToPhysicalPoint((0, 0, 0))
        upper_corner = image.TransformIndexToPhysicalPoint([s - 1 for s in image.GetSize()])

        min_coords = np.min([lower_corner, upper_corner], axis=0)
        max_coords = np.max([lower_corner, upper_corner], axis=0)

        new_origin = [
            min_coord if b >= 0 else max_coord
            for b, min_coord, max_coord in zip(transformed_base, min_coords, max_coords)
        ]

    new_origin = np.asarray(new_origin)

    # Compute new size
    shape = np.asarray(image.GetSize())
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    direction_transformation = np.matmul(new_direction.T, direction)  # undo original transform, apply new transform
    new_shape = np.abs(direction_transformation.dot(shape))

    # Compute new spacing
    spacing = np.asarray(image.GetSpacing())
    new_spacing = np.abs(direction_transformation.dot(spacing))

    # Convert from numpy data types to python data types that simpleitk can work with
    new_origin = [float(o) for o in new_origin]
    new_shape = [int(round(s)) for s in new_shape]
    new_spacing = [float(s) for s in new_spacing]
    new_direction = [float(s) for s in new_direction.reshape(-1)]

    # Choose a suitable interpolator
    interpolator = sitk.sitkNearestNeighbor
    if not only_nearest_neighbor_interpolation:
        for v in image.GetDirection():
            if abs(v) > eps and abs(v) - 1 > eps:
                interpolator = sitk.sitkLanczosWindowedSinc
                break

    # Resample image
    resampled = sitk.Resample(image,
                              new_shape,
                              sitk.Transform(),
                              interpolator,
                              new_origin,
                              new_spacing,
                              new_direction,
                              outside_val,
                              image.GetPixelID())

    # If the input was a numpy array, return also a numpy array (simpleitk image otherwise)
    if header is None:
        return resampled
    else:
        return sitk_to_numpy(resampled, only_data=False)

def resample_sitk_image(sitk_image, spacing=None, interpolator=None,
                        fill_value=0, new_size=None):
    """Resamples an ITK image to a new grid. If no spacing is given,
    the resampling is done isotropically to the smallest value in the current
    spacing. This is usually the in-plane resolution. If not given, the
    interpolation is derived from the input data type. Binary input
    (e.g., masks) are resampled with nearest neighbors, otherwise linear
    interpolation is chosen.
    Parameters
    ----------
    sitk_image : SimpleITK image or str
      Either a SimpleITK image or a path to a SimpleITK readable file.
    spacing : tuple
      Tuple of integers
    interpolator : str
      Either `nearest`, `linear` or None.
    fill_value : int
    Returns
    -------
    SimpleITK image.
    """
    _SITK_INTERPOLATOR_DICT = {
        'nearest': sitk.sitkNearestNeighbor,
        'linear': sitk.sitkLinear,
        'gaussian': sitk.sitkGaussian,
        'label_gaussian': sitk.sitkLabelGaussian,
        'bspline': sitk.sitkBSpline,
        'hamming_sinc': sitk.sitkHammingWindowedSinc,
        'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
        'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
        'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
    }
    if isinstance(sitk_image, str):
        sitk_image = sitk.ReadImage(sitk_image)
    num_dim = sitk_image.GetDimension()

    if not interpolator:
        interpolator = 'linear'
        pixelid = sitk_image.GetPixelIDValue()

        if pixelid not in [1, 2, 4]:
            raise NotImplementedError(
                'Set `interpolator` manually, '
                'can only infer for 8-bit unsigned or 16, '
                '32-bit signed integers')
        if pixelid == 1:  # 8-bit unsigned int
            interpolator = 'nearest'

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())
    orig_size = np.array(sitk_image.GetSize(), dtype=np.int)

    if not spacing:
        min_spacing = orig_spacing.min()
        new_spacing = [min_spacing] * num_dim
    else:
        new_spacing = [float(s) for s in spacing]

    assert interpolator in _SITK_INTERPOLATOR_DICT.keys(), \
        '`interpolator` should be one of {}'.format(
            _SITK_INTERPOLATOR_DICT.keys())

    sitk_interpolator = _SITK_INTERPOLATOR_DICT[interpolator]

    if new_size is None:
        new_size = orig_size * (orig_spacing / new_spacing)
        new_size = np.ceil(new_size).astype(
            np.int)  # Image dimensions are in integers
        new_size = [int(s) for s in
                    new_size]  # SimpleITK expects lists, not ndarrays

    resample_filter = sitk.ResampleImageFilter()

    resampled_sitk_image = resample_filter.Execute(sitk_image,
                                                   new_size,
                                                   sitk.Transform(),
                                                   sitk_interpolator,
                                                   orig_origin,
                                                   new_spacing,
                                                   orig_direction,
                                                   fill_value,
                                                   orig_pixelid)

    return resampled_sitk_image


def load_framed_3dCT_dcm(dicom_file):
    ds = pyd.dcmread(dicom_file)
    # padding = float(ds.PixelPaddingValue)
    frame_headers = ds.PerFrameFunctionalGroupsSequence
    pso_headers = np.asarray([(*fh.PlanePositionSequence[0].ImagePositionPatient,
                               *fh.PlaneOrientationSequence[0].ImageOrientationPatient,
                               *fh.PixelMeasuresSequence[0].PixelSpacing)
                              for fh in frame_headers], dtype=np.float32)
    s_idx = np.argsort(pso_headers[:, 2])
    origin = pso_headers[s_idx[0]][:3]
    if tuple(np.mean(pso_headers[:, 3:9], axis=0)) != tuple(pso_headers[:, 3:9][0]):
        raise NotImplementedError("Slices may have different orientations!")

    direction = np.asarray(pso_headers[:, 3:9][0].tolist() + [0.0, 0.0, 1.0]).reshape(3, 3)
    spacing = pso_headers[:, 9:][0].tolist() \
              + [np.mean(pso_headers[:, 2][s_idx][1:] - pso_headers[:, 2][s_idx][:-1])]
    scan_sitk = sitk.ReadImage(dicom_file)
    scan = sitk.GetArrayFromImage(scan_sitk).astype(np.int16)
    return scan[s_idx], origin[::-1].tolist(), direction[::-1].flatten().tolist(), \
           np.asarray(spacing[::-1], np.float64).tolist()


class MODEL_STATUS(Enum):
    UN_INIT = 0
    RANDOM_INITIALIZED = 1
    RELOAD_PRETRAINED = 2
    TRAINING = 3


def expand_dims(tensors, expected_dim):
    if tensors.dim() < expected_dim:
        for n in range(expected_dim - tensors.dim()):
            tensors = tensors.unsqueeze(0)

    return tensors


def squeeze_dims(tensors, expected_dim, squeeze_start_index=0):
    if tensors.dim() > expected_dim:
        for n in range(tensors.dim() - expected_dim):
            tensors = tensors.squeeze(squeeze_start_index)

    return tensors


def get_callable_by_name(module_name):
    cls = getattr(import_module(module_name.rpartition('.')[0]),
                  module_name.rpartition('.')[-1])
    return cls


class Settings:
    def __init__(self, settings_module_path, settings_name="settings"):

        # store the settings module in case someone later cares
        self.settings_module_path = settings_module_path
        spec = importlib.util.spec_from_file_location(settings_name, settings_module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        compulsory_settings = (
            "EXP_NAME",
            "MODEL_NAME",
        )

        self._explicit_settings = set()
        for setting in dir(mod):
            if setting.isupper():
                setting_value = getattr(mod, setting)
                if setting in compulsory_settings and setting is None:
                    raise AttributeError("The %s setting must be Not None. " % setting)
                setattr(self, setting, setting_value)
                self._explicit_settings.add(setting)

    def is_overridden(self, setting):
        return setting in self._explicit_settings

    def __str__(self):
        return "{}".format(self.__dict__)


def search_dict_key_recursively(dict_obj, trace_key, find_key):
    find_ = []

    def dict_traverse(dict_obj, trace_key, find_key):
        if not isinstance(dict_obj, dict):
            return
        if find_key in dict_obj.keys():
            find_.append(dict_obj[find_key])
        if trace_key in dict_obj.keys():
            dict_traverse(dict_obj[trace_key], trace_key, find_key)

    dict_traverse(dict_obj, trace_key, find_key)
    return find_


def get_value_recursively(search_dict, field):
    """
    Takes a dict with nested lists and dicts,
    and searches all dicts for a key of the field
    provided.
    """
    fields_found = []

    for key, value in search_dict.items():

        if key == field:
            fields_found.append(value)

        elif isinstance(value, dict):
            results = get_value_recursively(value, field)
            for result in results:
                fields_found.append(result)

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    more_results = get_value_recursively(item, field)
                    for another_result in more_results:
                        fields_found.append(another_result)

    return fields_found


def pad5d_ensure_division(t, division, **kwargs):
    assert (t.dim() == 5)

    def compute_pad(s, d):
        if s % d != 0:
            p = 0
            while True:
                if (s + p) % d == 0:
                    return p
                p += 1
        return 0

    padding = tuple(np.asarray([(compute_pad(s, division), 0) for s in t.shape[-3:][::-1]]).flatten().tolist())
    return F.pad(t, padding, **kwargs), padding


def load_pretrained_model(cpk_path, reload_objects, state_keys, device='cuda'):
    def reload_state(state, reload_dict, overwrite=False):
        current_dict = state.state_dict()
        if not overwrite:
            saved_dict = {k: v for k, v in reload_dict.items() if k in current_dict}

            # check in saved_dict, some tensors may not match in size.
            matched_dict = {}
            for k, v in saved_dict.items():
                cv = current_dict[k]
                if isinstance(cv, torch.Tensor) and v.size() != cv.size():
                    print(
                        "in {}, saved tensor size {} does not match current tensor size {}"
                            .format(k, v.size(), cv.size()))
                    continue
                matched_dict[k] = v
        else:
            matched_dict = {k: v for k, v in reload_dict.items()}
        current_dict.update(matched_dict)
        state.load_state_dict(current_dict)

    if device == 'cpu':
        saved_states = torch.load(cpk_path, map_location='cpu')
    else:
        saved_states = torch.load(cpk_path)
    min_len = min(len(reload_objects), len(state_keys))
    for n in range(min_len):
        if state_keys[n] in saved_states.keys():
            if state_keys[n] == "metric":
                reload_state(reload_objects[n], saved_states[state_keys[n]], True)
            else:
                reload_state(reload_objects[n], saved_states[state_keys[n]], False)
    return saved_states


class JobRunner:
    class ModelMetricState:

        def __init__(self, **kwargs):
            self._state_dict = copy.deepcopy(kwargs)

        def state_dict(self):
            return self._state_dict

        def load_state_dict(self, new_dict):
            self._state_dict.update(new_dict)

    def __init__(self, setting_module_file_path, settings_module=None, **kwargs):
        if setting_module_file_path is None:
            file_path = Path(inspect.getfile(self.__class__)).as_posix()
            setting_module_file_path = os.path.join(file_path.rpartition('/')[0], "settings.py")

        if settings_module is not None:
            self.settings = settings_module
        else:
            self.settings = Settings(setting_module_file_path)
        self.model_status = MODEL_STATUS.UN_INIT
        # config loggers
        logging.config.dictConfig(self.settings.LOGGING)
        self.logger = logging.getLogger(self.settings.EXP_NAME)

        def runner_excepthook(excType, excValue, traceback):
            self.logger.error("Logging an uncaught exception",
                              exc_info=(excType, excValue, traceback))

        self.model_metrics_save_dict = JobRunner.ModelMetricState()
        sys.excepthook = runner_excepthook

    def device_data(self, *args):
        raise NotImplementedError

    def print_model_parameters(self, iter):
        pass

    def init(self):
        # create model, initializer, optimizer, scheduler for training
        #  according to settings

        # cls = get_callable_by_name(self.settings.MODEL.pop('method'))
        self.model = CTSUNet(**self.settings.MODEL)
        self.is_cuda = self.settings.IS_CUDA & torch.cuda.is_available()
        if self.is_cuda:
            self.model.cuda()
            if torch.cuda.device_count() > 1:
                self.logger.info("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.model = torch.nn.DataParallel(self.model)
            self.model.is_cuda = True
        else:
            self.model.is_cuda = False

        self.model_status = MODEL_STATUS.RANDOM_INITIALIZED
        self.amp_module = None
        self.logger.debug("init finished, with full config = {}.".format(self.settings))
        self.current_iteration = 0
        self.epoch_n = 0
        self.saved_model_states = {}

    def generate_batches(self, *args):
        raise NotImplementedError

    def run(self, *args):
        raise NotImplementedError

    def run_job(self, *args):
        try:
            self.run(*args)
        except:
            self.logger.exception("training encounter exception.")

    def reload_model_from_cache(self):
        # check if we need to load pre-trained models. user can either specify a checkpoint by
        # giving an absolute path, or let the engine searching for the latest checkpoint in
        # model output path.
        if self.settings.RELOAD_CHECKPOINT:
            if self.settings.RELOAD_CHECKPOINT_PATH is not None:
                cpk_name = self.settings.RELOAD_CHECKPOINT_PATH
            else:
                # we find the checkpoints from the model output path, we reload whatever the newest.
                list_of_files = glob.glob(self.exp_path + '/*.pth')
                if len(list_of_files) == 0:
                    raise RuntimeError("{} has no checkpoint files with pth extensions."
                                       .format(self.exp_path))
                cpk_name = max(list_of_files, key=os.path.getctime)
            self.logger.debug("reloading model from {}.".format(cpk_name))
            reload_dicts = self.settings.RELOAD_DICT_LIST
            device = 'cuda' if self.model.is_cuda else 'cpu'
            self.saved_model_states = load_pretrained_model(cpk_name,
                                                            [self.model],
                                                            reload_dicts, device)
            self.current_iteration = self.saved_model_states['iteration'] + 1 \
                if 'iteration' in self.saved_model_states.keys() else 0
            self.epoch_n = self.saved_model_states['epoch_n'] \
                if 'epoch_n' in self.saved_model_states.keys() else 0
            self.logger.debug("reload model from {}.".format(cpk_name))
            self.model_status = MODEL_STATUS.RELOAD_PRETRAINED

    def update_model_state(self, **kwargs):
        self.saved_model_states['iteration'] = self.current_iteration
        self.saved_model_states['epoch_n'] = self.epoch_n
        self.saved_model_states['model_dict'] = self.model.state_dict()
        self.saved_model_states['optimizer_dict'] = self.optimizer.state_dict()
        self.saved_model_states['scheduler_dict'] = self.scheduler.state_dict()
        self.saved_model_states['metric'] = self.model_metrics_save_dict.state_dict()
        self.saved_model_states.update(kwargs)

    def save_model(self, **kwargs):
        self.update_model_state(**kwargs)
        cpk_name = os.path.join(self.exp_path, "{}.pth".format(self.current_iteration))
        time.sleep(10)
        torch.save(self.saved_model_states, cpk_name)
        self.logger.info("saved model into {}.".format(cpk_name))

    def archive_results(self, *args):
        raise NotImplementedError


def print_parameter(model):
    child_counter = 0
    for name, parameter in model.named_parameters():
        print("parameter_name:{}, grad_flag:{}".format(name, parameter.requires_grad))
        child_counter += 1


def find_shape_outter_boundary(mask, connectivity=1):
    mask = (mask > 0).astype(np.uint8)
    if np.sum(mask) == 0:
        return np.zeros_like(mask)

    template = ndimage.generate_binary_structure(mask.ndim, connectivity)
    # c_mask = ndimage.convolve(mask, template) > 0
    c_mask = ndimage.binary_dilation(mask, template)
    return c_mask & ~mask


def vote_region_based_on_neighbors(mask, voi, connectivity, vote_background=True):
    voi_slices = ndimage.find_objects(voi > 0)
    assert (len(voi_slices) == 1)
    voi_slices = voi_slices[0]
    # need to enlarge it for region of size 1.
    voi_slices = tuple([slice(max(ss.start - 1, 0), min(ss.stop + 1, sp))
                        for ss, sp in zip(voi_slices, mask.shape)])
    voi_r = voi[voi_slices]
    mask_r = mask[voi_slices]
    old_label = np.unique(mask_r[voi_r])
    b_edges = (find_shape_outter_boundary(voi_r, connectivity) * (mask_r > 0)) > 0
    b_edges_labels, b_edges_labels_num = np.unique(mask_r[b_edges], return_counts=True)
    if b_edges.sum() == 0:
        if vote_background:
            mask_r[voi_r] = 0
            return old_label, 0
        else:
            return old_label, old_label

    # add_size = (mask == b_edges_labels[np.argmax(b_edges_labels_num)]).sum()
    # old_size = (mask == old_label[0]).sum()
    new_label = b_edges_labels[np.argmax(b_edges_labels_num)]
    mask_r[voi_r] = new_label
    # assert (mask == old_label[0]).sum() == 0
    # assert old_size + add_size == (mask == b_edges_labels[np.argmax(b_edges_labels_num)]).sum()
    return old_label, new_label


def windowing(image, from_span=(-1150, 350), to_span=(0, 255)):
    image = np.copy(image)
    if from_span is None:
        min_input = np.min(image)
        max_input = np.max(image)
    else:
        min_input = from_span[0]
        max_input = from_span[1]
    image[image < min_input] = min_input
    image[image > max_input] = max_input
    image = ((image - min_input) / float(max_input - min_input)) * (to_span[1] - to_span[0]) + to_span[0]
    return image


def resample(narray, orig_spacing, factor=2, required_spacing=None, new_size=None, interpolator='linear'):
    if new_size is not None and narray.shape == new_size:
        print("size is equal not resampling!")
        return narray, orig_spacing
    s_image = sitk.GetImageFromArray(narray)
    s_image.SetSpacing(np.asarray(orig_spacing[::-1], dtype=np.float64).tolist())

    req_spacing = factor * np.asarray(orig_spacing)
    req_spacing = tuple([float(s) for s in req_spacing])
    if required_spacing:
        req_spacing = required_spacing
    if new_size:
        new_size = new_size[::-1]
    resampled_image = resample_sitk_image(s_image,
                                          spacing=req_spacing[::-1],
                                          interpolator=interpolator,
                                          fill_value=0, new_size=new_size)

    resampled = sitk.GetArrayFromImage(resampled_image)

    return resampled, req_spacing


class GaussianBlur:

    def __init__(self, sigma, mode='fixed'):
        self.sigma = sigma
        self.mode = mode

    # def _gaussian_blur(self, data, channel_id, meta):
    #     variance_v = np.random.uniform(self.sigma[0], self.sigma[1])
    #     data = ndimage.gaussian_filter(data, variance_v)
    #     meta[channel_id] = variance_v
    #     return data

    def gaussian_blur(self, data, meta):
        if self.mode == 'fixed':
            variance_v = self.sigma[0]
        else:
            variance_v = np.random.uniform(self.sigma[0], self.sigma[1])
        data = ndimage.gaussian_filter(data, variance_v)
        meta["v"] = variance_v
        return data

    def __call__(self, sample):
        meta = {}
        new_sample = {(k if "#" in k and "image" in k else k): (self.gaussian_blur(v.astype(np.float32), meta)
                                                                if "#" in k and "image" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        meta.update({"sigma": self.sigma})
        new_sample['meta'][self.__class__.__name__] = copy.deepcopy(meta)
        return new_sample


class Windowing(object):
    """Convert ndarrays in sample to Tensors if # sign and "image" tag in its keys."""

    def __init__(self, min=-1200, max=600, out_min=0, out_max=1):
        self.min = min
        self.max = max
        self.out_min = out_min
        self.out_max = out_max

    def __call__(self, sample):
        from_span = (self.min, self.max) if self.min is not None else None
        sample = {(k if "#" in k and "image" in k else k): (windowing(v.astype(np.float32),
                                                                      from_span=from_span,
                                                                      to_span=(self.out_min,
                                                                               self.out_max)) if "#" in k and "image" in k else v)
                  for k, v in sample.items()}
        return sample


class Resample(object):
    """Convert ndarrays in sample to Tensors if # sign and "image" tag in its keys."""

    def __init__(self, mode, factor, size=None):
        self.mode = mode
        self.factor = factor
        if size:
            self.size = list(size)

    def __call__(self, sample):
        new_sample = {"meta": {}}
        spacing = sample['meta']['spacing']
        if self.mode == 'random_spacing':
            factor = np.random.uniform(self.factor[0], self.factor[1])
            require_spacing = [factor] * len(spacing)
            new_size = None
        elif self.mode == 'fixed_factor':
            factor = self.factor
            require_spacing = None
            new_size = None
        elif self.mode == 'fixed_spacing':
            if isinstance(self.factor, (float, int)):
                factor = self.factor
                require_spacing = [factor] * len(spacing)
            elif isinstance(self.factor, (tuple, list)):
                require_spacing = self.factor
                factor = 2  # dummy number meaningless.
            new_size = None
        elif self.mode == "inplane_spacing_only":
            current_size = sample['meta']['size']
            assert (len(current_size) == 3)
            require_spacing = [spacing[0], self.factor[1],
                               self.factor[2]]
            new_size = None
            factor = 2
        elif self.mode == "inplane_resolution_only":
            current_size = sample['meta']['size']
            assert (len(current_size) == 3)
            require_spacing = [spacing[0], spacing[1] * current_size[1] / self.size[1],
                               spacing[2] * current_size[2] / self.size[2]]
            new_size = [current_size[0], self.size[1],
                        self.size[2]]
            factor = 2
        elif self.mode == "inplane_resolution_z_spacing":
            current_size = sample['meta']['size']
            assert (len(current_size) == 3)
            require_spacing = [self.factor[0], spacing[1] * current_size[1] / self.size[1],
                               spacing[2] * current_size[2] / self.size[2]]
            new_size = [int(round(current_size[0] * spacing[0] / self.factor[0])), self.size[1],
                        self.size[2]]
            factor = 2
        elif self.mode == "inplane_resolution_z_jittering":
            current_size = sample['meta']['size']
            assert (len(current_size) == 3)
            z_spacing_base = spacing[0]
            offset = np.random.uniform(-self.factor, self.factor)
            z_spacing = z_spacing_base + offset
            require_spacing = [z_spacing, spacing[1] * current_size[1] / self.size[1],
                               spacing[2] * current_size[2] / self.size[2]]
            new_size = [int(round(current_size[0] * spacing[0] / z_spacing)), self.size[1],
                        self.size[2]]
            factor = 2
        elif self.mode == "inplane_resolution_min_z_spacing":
            current_size = sample['meta']['size']
            assert (len(current_size) == 3)
            if spacing[0] < self.factor[0]:
                print("set spacing to {} from {}.".format(self.factor[0], spacing[0]))
                require_spacing = [self.factor[0], spacing[1] * current_size[1] / self.size[1],
                                   spacing[2] * current_size[2] / self.size[2]]
                new_size = [int(round(current_size[0] * spacing[0] / self.factor[0])), self.size[1],
                            self.size[2]]
            else:
                require_spacing = [spacing[0], spacing[1] * current_size[1] / self.size[1],
                                   spacing[2] * current_size[2] / self.size[2]]
                new_size = [current_size[0], self.size[1],
                            self.size[2]]
            factor = 2
        elif self.mode == "fixed_spacing_min_in_plane_resolution":
            current_size = sample['meta']['size']
            assert (len(current_size) == 3)
            if not isinstance(self.factor, (tuple, list)):
                factor = [self.factor] * 3
            else:
                factor = self.factor
            new_y_size = int(round(current_size[1] * spacing[1] / factor[1]))
            if new_y_size > self.size[1]:

                require_spacing = [spacing[0], spacing[1] * current_size[1] / self.size[1],
                                   spacing[2] * current_size[2] / self.size[2]]
                new_size = [current_size[0], self.size[1],
                            self.size[2]]
                print(
                    "new_size:{} > target_size {}. fixed_in_plane_resolution mode. {}.".format(new_y_size, self.size[1],
                                                                                               new_size))
            else:
                require_spacing = [spacing[0], factor[1],
                                   factor[2]]
                new_size = None
                print(
                    "new_size:{} <= target_size {}. fixed_spacing. {}.".format(new_y_size, self.size[1],
                                                                               require_spacing))
            factor = 2
        elif self.mode == "iso_minimal":
            factor = spacing[0]
            require_spacing = [np.min(spacing)] * len(spacing)
            new_size = None
        elif self.mode == "fixed_output_size":
            current_size = sample['meta']['size']
            ratio = current_size[-1] / self.size[-1]
            require_spacing = [spacing[-1] * ratio] * len(spacing)
            new_size = self.size[:]
            new_size[0] = int(round(current_size[0] * spacing[0] / require_spacing[0]))
            new_size[1] = int(round(current_size[1] * spacing[1] / require_spacing[1]))
            factor = 2
        elif self.mode == "spacing_size_match":
            require_spacing = self.factor[:]
            new_size = self.size[:]
            factor = 2
        else:
            raise NotImplementedError
        for k, v in sample.items():
            if "#" in k:
                if "reference" in k or 'weight_map' in k:
                    mode = 'nearest'
                else:
                    mode = 'linear'
                if v.ndim == 4:
                    r_results = [resample(vv, spacing, factor=factor,
                                          required_spacing=require_spacing, new_size=new_size, interpolator=mode) for vv
                                 in v]
                    new_spacing = r_results[0][-1]
                    nv = np.stack([r[0] for r in r_results], axis=0)
                elif v.ndim == 3:
                    nv, new_spacing = resample(v, spacing, factor=factor,
                                               required_spacing=require_spacing, new_size=new_size, interpolator=mode)
                else:
                    raise NotImplementedError
                new_sample[k] = nv
                new_size = nv.shape
            else:
                new_sample[k] = v
        old_size = sample['meta']['size']
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        new_sample['meta']['spacing'] = tuple(new_spacing)
        new_sample['meta']['size'] = new_size
        new_sample['meta']['size_before_resample'] = old_size
        new_sample['meta']['resample_factor'] = factor
        return new_sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors if # sign in its keys."""

    def __call__(self, sample, is_pin=False):
        if is_pin:
            sample = {(k if "#" in k else k): (torch.from_numpy(v.copy()).pin_memory() if "#" in k else v)
                      for k, v in sample.items()}
        else:
            sample = {(k if "#" in k else k): (torch.from_numpy(v.copy()) if "#" in k else v)
                      for k, v in sample.items()}
        return sample


import threading
import queue


def dist_func(in_q, out_q):
    mask, jid = in_q.get()
    dt = dist(mask).astype(np.float32)
    out_q.put((dt, jid))


def dist_slow(masks):
    dist_maps = []
    for mask in range(masks):
        dt = dist(mask).astype(np.float32)
        dist_maps.append(dt)
    return np.stack(dist_maps, axis=0)


def dist_threading(masks):
    in_q = queue.Queue()
    n_jobs = len(masks)
    for jid in range(n_jobs):
        in_q.put((masks[jid], jid))
    out_q = queue.Queue()
    workers = []
    for _ in range(n_jobs):
        t = threading.Thread(target=dist_func, args=(in_q, out_q))
        t.start()
        workers.append(t)

    for w in workers:
        w.join()

    dt_masks = np.zeros_like(np.asarray(masks), dtype=np.float32)
    while not out_q.empty():
        dt, jid = out_q.get()
        dt_masks[jid] = dt
    return dt_masks


from scipy.ndimage.morphology import distance_transform_edt as dist


class LobeSegmentationPostProcessing():

    def __init__(self, iter_no=2):
        super(LobeSegmentationPostProcessing, self).__init__()
        self.iter_no = iter_no

    @staticmethod
    def post_lobe(scan, pred, min_intensity, max_intensity, iter_no=5):
        # step.1 fix borders.
        post_pred = copy.deepcopy(pred)
        struct_e = ndimage.generate_binary_structure(3, 3)
        lobe_volumes = np.bincount(post_pred.flat)[1:]
        print("post_lobe volume: {}".format(lobe_volumes), flush=True)
        # step.1 make sure Left and right lung are single CC
        left_lung = np.logical_or(post_pred == 1, post_pred == 2)
        right_lung = np.logical_or(post_pred == 3, np.logical_or(post_pred == 4, post_pred == 5))
        for i, side_lung in enumerate([left_lung, right_lung]):
            debug_str = "left lung" if i == 0 else "right lung"
            sl_label, cc_num = ndimage.label(side_lung, struct_e)
            c_idx = np.bincount(sl_label.flat)[1:].argsort()  # sort largest except background
            if cc_num > 1:
                for non_max_idx in c_idx[:-1]:
                    non_max_b = sl_label == non_max_idx + 1
                    ol, nl = vote_region_based_on_neighbors(post_pred, non_max_b, 3)
                    print("non-max suppress {} {} -> {}.".format(debug_str, ol, nl), flush=True)

        # step.2 hole filling for left and right lung.
        left_lung = np.logical_or(post_pred == 1, post_pred == 2)
        right_lung = np.logical_or(post_pred == 3, np.logical_or(post_pred == 4, post_pred == 5))
        for i, side_lung in enumerate([left_lung, right_lung]):
            debug_str = "left lung" if i == 0 else "right lung"
            side_lung_filled = ndimage.binary_fill_holes(side_lung)
            side_lung_holes = np.logical_xor(side_lung, side_lung_filled)
            l_sl_holes, cc_num = ndimage.label(side_lung_holes, struct_e)
            for n in range(1, cc_num + 1):
                l_b = l_sl_holes == n
                ol, nl = vote_region_based_on_neighbors(post_pred, l_b, 3)
                print("filling holes {}, {} -> {}.".format(debug_str, ol, nl), flush=True)

        labels = np.unique(post_pred)[1:]
        if len(labels) == 0:
            return post_pred
        # idx_label_map = {}
        # l_cc_masks = np.zeros((len(labels),) + post_pred.shape, dtype=np.uint8)
        # # step.3 vote diff region and non-largest cc to nearest lobe labels.
        # all_largest_cc = np.zeros_like(post_pred, dtype=np.bool)
        # for idx, label in enumerate(labels):
        #     b_mask = post_pred == label
        #     b_mask_labeled, cc_num = ndimage.label(b_mask, struct_e)
        #     b_largest_cc = (b_mask_labeled == np.bincount(b_mask_labeled.flat)[1:].argmax() + 1)
        #     idx_label_map[idx] = label
        #     l_cc_masks[idx] = b_largest_cc == 0
        #     all_largest_cc = np.logical_or(all_largest_cc, b_largest_cc)

        # vote_region = np.logical_xor(all_largest_cc, post_pred > 0)

        # print("running dist vote cc.", flush=True)
        # # dist_map = dist_threading(l_cc_masks)
        # dist_map = dist_slow(l_cc_masks)
        # post_pred[vote_region] = np.vectorize((idx_label_map.__getitem__)) \
        #     (np.argmin(dist_map, axis=0)[vote_region])

        # step.4 vote non-largest cc again to nearest lobes
        print(f"vote non-largest cc again to nearest lobes iter:{iter_no}.", flush=True)
        labels = np.unique(post_pred)[1:]
        count = 0
        while count < iter_no + 1:
            all_notes = []
            for label in labels:
                debug_str = "Lobe_{}_iter_{}".format(label, count)
                lobe = post_pred == label
                l_lobe, cc_num = ndimage.label(lobe, struct_e)
                if cc_num > 1:
                    c_idx = np.bincount(l_lobe.flat)[1:].argsort()
                    for non_max_idx in c_idx[:-1]:
                        non_max_b = l_lobe == non_max_idx + 1
                        if count == iter_no:
                            post_pred[non_max_b] = 0
                            print("non-max suppress last round set {} -> 0.".format(debug_str, non_max_idx + 1),
                                  flush=True)
                        else:
                            ol, nl = vote_region_based_on_neighbors(post_pred, non_max_b, 3)
                            print("non-max suppress {} {} -> {}.".format(debug_str, ol, nl), flush=True)
                else:
                    print(f"single CC found for label:{label}", flush=True)

                all_notes.append(cc_num == 1)
            if np.alltrue(all_notes):
                break
            count += 1

        return post_pred


def write_array_to_mhd_itk(target_path, arrs, names, type=np.int16,
                           origin=[0.0, 0.0, 0.0],
                           direction=np.eye(3, dtype=np.float64).flatten().tolist(),
                           spacing=[1.0, 1.0, 1.0], orientation='RAI'):
    """ arr is z-y-x, spacing is z-y-x."""
    size = arrs[0].shape
    for arr, name in zip(arrs, names):
        # assert (arr.shape == size)
        simage = sitk.GetImageFromArray(arr.astype(type))
        simage.SetSpacing(np.asarray(spacing, np.float64).tolist())
        simage.SetDirection(direction)
        simage.SetOrigin(origin)
        fw = sitk.ImageFileWriter()
        fw.SetFileName(target_path + '/{}.mhd'.format(name))
        fw.SetDebug(False)
        fw.SetUseCompression(True)
        fw.SetGlobalDefaultDebug(False)
        fw.Execute(simage)
        with open(target_path + '/{}.mhd'.format(name), "rt") as fp:
            lines = fp.readlines()
            for idx, line in enumerate(lines):
                if "AnatomicalOrientation" in line:
                    header = line[:line.find("=")].strip()
                    newline = "{}={}".format(header, orientation) + os.linesep
                    lines[idx] = newline
                    break
        with open(target_path + '/{}.mhd'.format(name), "wt", newline='') as fp:
            fp.writelines(lines)


def write_array_to_mha_itk(target_path, arrs, names, type=np.int16,
                           origin=[0.0, 0.0, 0.0],
                           direction=np.eye(3, dtype=np.float64).flatten().tolist(),
                           spacing=[1.0, 1.0, 1.0]):
    """ arr is z-y-x, spacing is z-y-x."""
    # size = arrs[0].shape
    for arr, name in zip(arrs, names):
        # assert (arr.shape == size)
        simage = sitk.GetImageFromArray(arr.astype(type))
        simage.SetSpacing(np.asarray(spacing, np.float64).tolist())
        simage.SetDirection(direction)
        simage.SetOrigin(origin)
        fw = sitk.ImageFileWriter()
        fw.SetFileName(target_path + '/{}.mha'.format(name))
        fw.SetDebug(False)
        fw.SetUseCompression(True)
        fw.SetGlobalDefaultDebug(False)
        fw.Execute(simage)


def find_label_edges(mask):
    gz, gy, gx = np.gradient(mask.astype(np.uint8))
    edge = (gz ** 2 + gy ** 2 + gx ** 2) != 0
    return edge


def find_bbox_from_mask(mask):
    if np.sum(mask) == 0:
        tls, brs = np.vstack((np.zeros(mask.ndim, np.uint),
                              mask.shape))
        return tls, brs
    object_slices = ndimage.find_objects(mask > 0)[0]
    tls, brs = np.asarray([[ss.start, ss.stop] for ss in object_slices]).T
    # tls, brs = np.min(proposal_mask_coors, axis=1), np.max(proposal_mask_coors, axis=1) + 1
    return tls, brs


def sliding_window(bbox, output_resolution, overlaps, strides=None, append_spans=True):
    tls, brs = bbox
    if strides is not None:
        slices = tuple([slice(tl + o // 2, br - o + o // 2, stride)
                        for tl, br, o, stride in zip(tls, brs, output_resolution, strides)])
    else:
        slices = tuple([slice(tl + o // 2, br - o + o // 2, int(o * (1 - ol)))
                        for tl, br, o, ol in zip(tls, brs, output_resolution, overlaps)])
    if append_spans:
        spans = tuple(np.append(np.arange(ss.start, ss.stop, ss.step), ss.stop) for ss in slices)
    else:
        spans = tuple(np.arange(ss.start, ss.stop, ss.step) for ss in slices)
    g = np.meshgrid(*spans, sparse=False)
    grid = np.vstack([gg.ravel() for gg in g]).T
    return grid.astype(np.uint)


def cumsum(sequence):
    r, s = [], 0
    for e in sequence:
        l = len(e)
        r.append(l + s)
        s += l
    return r


class ShiftStitchTensorChunkSet(Dataset):
    """This is the same TensorChunkSet yet the resolutions are expected
    to be different, where the idea of shift and stitch mentioned
    in FCN paper is implemented.
    tensors_list are always 4d tensors.
    args:
        chunked: a list of tensors in batch, *spatial shapes. The stitches are applied to the last one.
        resolutions: a list of tuples, each defines the resolution of chunking.
        overlaps: the tuple defines the overlap of stitches in smallest resolution.
    """

    def __init__(self, chunked, resolutions, overlaps, strides=None, ref_idx=-1,
                 transforms=None, append_span=True, mask_bb=None, enlarge_bb=2, debug=False):
        super(ShiftStitchTensorChunkSet, self).__init__()

        self.tensors_list = chunked["chunked_list"]
        # tensors are in batch first dim ordering.
        self.meta = chunked["meta"]
        self.resolutions = resolutions
        assert (len(self.resolutions) == len(self.tensors_list))
        self.overlaps = overlaps
        self.transforms = transforms
        self.ref_idx = ref_idx
        if mask_bb is None:
            tls, brs = np.vstack((np.zeros(self.tensors_list[self.ref_idx][0].dim(), np.uint),
                                  self.tensors_list[self.ref_idx][0].shape))
        else:
            tls, brs = find_bbox_from_mask(mask_bb)
            if enlarge_bb > 0:
                tls = [max(0, tl - enlarge_bb) for tl in tls]
                brs = [min(ms, br + enlarge_bb) for br, ms in zip(brs, mask_bb.shape)]
        self.bb_box = (tls, brs)
        window_locations = sliding_window((tls, brs), resolutions[self.ref_idx],
                                          overlaps, strides, append_span).tolist()
        self.window_locations_list = [window_locations[:] for n
                                      in range(self.tensors_list[self.ref_idx].shape[0])]
        self.window_locations_type = [["sliding" for n in range(len(window_locations))]
                                      for window_locations in self.window_locations_list]

        # now we need pad inputs.
        output_res = self.resolutions[self.ref_idx]
        for tensor_idx, (res, input_tensor) in \
                enumerate(zip(resolutions[:self.ref_idx], self.tensors_list[:self.ref_idx])):
            pad_size = [(x // 2, x - x // 2) for x in (np.asarray(res) - output_res)]
            # assert (input_tensor.dim() == (len(res) + 1))
            pad_tensor = F.pad(input_tensor.unsqueeze(1),
                               tuple(np.asarray(pad_size[::-1]).flatten().tolist()), "constant", 0) \
                .squeeze(1)
            self.tensors_list[tensor_idx] = pad_tensor

        self.debug = debug

    def __len__(self):
        return sum([len(window_locations) for window_locations in self.window_locations_list])

    def __getitem__(self, idx):
        # the window_location_list is mutable and may be modified by augmenters. therefore,
        # we need to dynamically calculate indices.
        cumulative_sizes = cumsum(self.window_locations_list)
        tensor_idx = bisect.bisect_right(cumulative_sizes, idx)
        if tensor_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - cumulative_sizes[tensor_idx - 1]

        chunk_tensor_list = []
        chunk_slices = []

        target_center = self.window_locations_list[tensor_idx][sample_idx]
        for tensors, resolutions in zip(self.tensors_list, self.resolutions):

            t_size = tensors[tensor_idx].shape
            slices = []
            for rs, co, ips, ts in zip(self.resolutions[self.ref_idx], target_center, resolutions, t_size):
                start = co - rs // 2
                end = co + ips - rs // 2
                if start < 0:
                    diff = -start
                    start += diff
                    end += diff

                if end > ts:
                    diff = ts - end
                    end -= diff
                    start -= diff
                if not (start >= 0 and end <= ts):
                    print(start)
                    print(end)
                    print(ts)
                    print(target_center)
                    print(t_size)
                assert (start >= 0 and end <= ts)
                slices.append(slice(int(start), int(end)))
            slices = tuple(slices)
            chunk_slices.append(slices)
            target_tensor = tensors[tensor_idx][slices]
            # if target_tensor.shape != resolutions:
            #     print("")
            assert (target_tensor.shape == resolutions)
            chunk_tensor_list.append(target_tensor)

        def f(a):
            if isinstance(a, dict):
                return dict(zip(a, map(f, a.values())))
            else:
                return a[tensor_idx]

        old_dict = dict(zip(self.meta, map(f, self.meta.values())))

        ret = {
            "chunked_list": chunk_tensor_list,
            "meta": {
                "tensor_idx": tensor_idx,
                "sample_idx": sample_idx,
                "labels": tuple(torch.unique(chunk_tensor_list[self.ref_idx]).detach().cpu().numpy().tolist()),
                "slices": tuple(chunk_slices),
                "type": self.window_locations_type[tensor_idx][sample_idx],
                "location": tuple(self.window_locations_list[tensor_idx][sample_idx]),
                "meta": old_dict
            }
        }
        if self.transforms:
            ret = self.transforms(ret)
        return ret


def defaut_collate_func(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        # if _use_shared_memory:
        #     # If we're in a background process, concatenate directly into a
        #     # shared memory tensor to avoid an extra copy
        #     numel = sum([x.numel() for x in batch])
        #     storage = batch[0].storage()._new_shared(numel)
        #     out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif isinstance(batch[0], (str, bytes, int, float, tuple)):
        return batch
    elif isinstance(batch[0], list):
        transposed = zip(*batch)
        return [defaut_collate_func(samples) for samples in transposed]
    elif isinstance(batch[0], collections.Mapping):
        return {key: defaut_collate_func([d[key] for d in batch]) for key in batch[0]}
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ == 'ndarray':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    print(batch[0])
    raise TypeError((error_msg.format(type(batch[0]))))


class SimpleDataset(Dataset):

    def __init__(self, root_path, file_name_patterns, recursive=True, transforms=None):
        super(SimpleDataset, self).__init__()
        self.root_path = root_path
        self.file_name_patterns = file_name_patterns
        self.transforms = transforms
        file_list = functools.reduce(lambda x, y: x + y, [glob.glob('{}/{}'
                                                                    .format(root_path, file_name_pattern)
                                                                    , recursive=True)
                                                          for file_name_pattern
                                                          in file_name_patterns])

        self.uid_abs_map = {Path(file_abs).stem: file_abs for file_abs in file_list}
        self.series_uids = sorted(list(self.uid_abs_map.keys()))

    def __len__(self):
        return len(self.series_uids)

    def __getitem__(self, idx):
        uid = self.series_uids[idx]
        file_abs_path = self.uid_abs_map[uid]
        if file_abs_path.rpartition('.')[-1].lower() == 'dcm':
            scan, origin, direction, spacing = load_framed_3dCT_dcm(file_abs_path)
            sitk_scan = sitk.GetImageFromArray(scan)
            sitk_scan.SetSpacing(spacing[::-1])
            sitk_scan.SetOrigin(origin[::-1])
            direction = np.asarray(direction).reshape(3, 3)[::-1].flatten().tolist()
            sitk_scan.SetDirection(direction)
        else:
            # default as RAI
            sitk_scan = sitk.ReadImage(file_abs_path)
            # scan = sitk.GetArrayFromImage(sitk_scan)
            # origin = sitk_scan.GetOrigin()[::-1]
            # spacing = sitk_scan.GetSpacing()[::-1]
            # direction = np.asarray(sitk_scan.GetDirection()).reshape(3, 3)[::-1].flatten().tolist()
        print("Before loading, scan size:{}.".format(sitk_scan.GetSize()), flush=True)
        sitk_scan = change_direction(sitk_scan, header=None, new_direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
                                     new_origin=None,
                                     outside_val=0, only_nearest_neighbor_interpolation=False)
        scan = sitk.GetArrayFromImage(sitk_scan)
        origin = sitk_scan.GetOrigin()[::-1]
        spacing = sitk_scan.GetSpacing()[::-1]
        direction = np.asarray(sitk_scan.GetDirection()).reshape(3, 3)[::-1].flatten().tolist()
        ret = {
            "#image": scan,
            "meta": {"uid": uid,
                     "size": scan.shape,
                     "spacing": spacing,
                     "origin": origin,
                     "original_spacing": spacing,
                     "original_size": scan.shape,
                     "direction": direction}
        }
        print("after loading, scan meta:{}.".format(ret['meta']), flush=True)
        if self.transforms:
            ret = self.transforms(ret)
        return ret


class LobeSegmentationTSTestCOVID(JobRunner):

    def __init__(self, settings_module):
        super(LobeSegmentationTSTestCOVID, self).__init__(None, settings_module)
        self.settings.RELOAD_CHECKPOINT = True
        self.init()
        self.reload_model_from_cache()
        self.test_dump_num = 0
        self.nr_class = self.settings.NR_CLASS
        self.pad_num = self.settings.SCAN_PAD_NUM if hasattr(self.settings, "SCAN_PAD_NUM") else 0
        self.use_proposal = self.settings.USE_PROPOSAL
        self.logger.info("Test performs Regional Proposal? {}".format(self.use_proposal))
        print_parameter(self.model)

    def post_processing(self, scan, heatmap, pred):
        if hasattr(self.settings, "POST_METHOD") and self.settings.POST_METHOD is not None:
            self.logger.debug("found post method: we perform {}!"
                              .format(self.settings.POST_METHOD))
            if self.settings.POST_METHOD == "nn":
                post_time_s = time.time()
                try:
                    post_pred = LobeSegmentationPostProcessing.post_lobe(scan, pred,
                                                                         min_intensity=self.settings.WINDOWING_MIN,
                                                                         max_intensity=self.settings.WINDOWING_MAX)
                except Exception:
                    post_pred = pred
                finally:
                    post_time_e = time.time()
                    self.logger.info("post processing time:{}".format(post_time_e - post_time_s))

            else:
                return pred
        else:
            return pred

        return post_pred

    def run(self, scan_data):
        self.model.eval()
        with torch.no_grad():
            now = time.time()
            chunk_size = self.settings.TEST_STITCHES_PATCH_SIZE[0]
            chunk_overlap = self.settings.TEST_PATCH_OVERLAPS[0]
            chunk_batch_size = self.settings.TEST_BATCH_SIZE[1]
            scan = scan_data['#image']
            meta = scan_data['meta']
            uid = meta['uid']
            pad_scan, padding = pad5d_ensure_division(expand_dims(scan, 5), self.pad_num,
                                                      mode='constant', value=0)
            padding_np = ((padding[4], padding[5]),
                          (padding[2], padding[3]),
                          (padding[0], padding[1]))
            crop_slices = tuple([slice(pn[0], ps) for pn, ps in
                                 zip(padding_np, pad_scan.shape[-3:])])
            scan_level_inf = self.model.scan_level_inference(pad_scan).cpu().squeeze(0)
            heatmap1_np = scan_level_inf.numpy()
            proposal = np.argmax(heatmap1_np, 0).astype(np.uint8)
            heatmap_p = torch.zeros((self.nr_class,) + pad_scan.shape[-3:],
                                    dtype=torch.float32)
            overlay_counter_mask_p = torch.zeros(pad_scan.shape[-3:], dtype=torch.float32)
            if self.is_cuda and torch.cuda.is_available():
                heatmap_p = heatmap_p.cuda()
                overlay_counter_mask_p = overlay_counter_mask_p.cuda()
            chunked_loader = DataLoader(
                ShiftStitchTensorChunkSet({
                    "chunked_list": [pad_scan.squeeze(0), pad_scan.squeeze(0)],
                    "meta": {
                        "uid": [uid]
                    }
                }, chunk_size, chunk_overlap, ref_idx=-1,
                    mask_bb=proposal if self.use_proposal else None),
                batch_size=chunk_batch_size,
                num_workers=self.settings.NUM_WORKERS,
                collate_fn=defaut_collate_func,
                drop_last=False
            )
            bb_box = chunked_loader.dataset.bb_box
            bb_slices = tuple([slice(int(tl), int(br)) for tl, br in zip(*bb_box)])
            self.logger.info("start {}".format(meta))
            self.model.inference_cache.clear()
            for cube_idx, chunked_ in enumerate(chunked_loader):
                chunk_batch, *ignored = chunked_['chunked_list']
                chunk_batch = chunk_batch.unsqueeze(1)
                chunk_meta = chunked_['meta']
                chunk_slices = search_dict_key_recursively(chunk_meta, 'meta', 'slices')[0]
                if self.is_cuda and torch.cuda.is_available():
                    chunk_batch = chunk_batch.cuda()

                batch_inference = self.model.inference(pad_scan, chunk_batch, chunk_slices)
                for id, inference in enumerate(batch_inference):
                    slices = chunk_slices[id][-1]
                    slice_expand = (slice(None, None),) + slices
                    heatmap_p[slice_expand] += inference
                    overlay_counter_mask_p[slices] += 1
                self.model.zero_grad()

            assert (overlay_counter_mask_p[bb_slices].min() == 1)
            heatmap_p[(slice(None, None),) + bb_slices] \
                /= overlay_counter_mask_p[bb_slices]
            # heatmap_p[0][overlay_counter_mask_p == 0] = 1.0
            # heatmap_p = heatmap_p[(slice(None, None),) + crop_slices]
            # heatmap_np = heatmap_p.cpu().numpy()
            # prediction_np = np.argmax(heatmap_np, axis=0)
            # scan_np = scan.numpy()
            # prediction_np_post = self.post_processing(scan_np, heatmap_np, prediction_np)
            # original_size = meta['original_size']
            # original_spacing = meta['original_spacing']
            # spacing = meta['spacing']
            # prediction_np_post, _ = resample(prediction_np_post, spacing, factor=2, required_spacing=original_spacing,
            #                           new_size=original_size, interpolator='linear')
            heatmap_p = expand_dims(heatmap_p[(slice(None, None),) + crop_slices], 5)
            original_size = meta['original_size']
            original_spacing = meta['original_spacing']
            spacing = meta['spacing']
            heatmap_p = torch.stack([F.interpolate(heatmap_p[:, n, ::].unsqueeze(0), size=original_size,
                                                   mode='trilinear').squeeze(0).squeeze(0)
                                    .float().cpu() for n in range(heatmap_p.shape[1])])
            heatmap_np = heatmap_p.numpy().astype(np.float32)
            prediction_np = np.argmax(heatmap_np, 0).astype(np.uint8)
            heatmap_np = None
            scan_np, _ = resample(scan.numpy(), spacing, factor=2, required_spacing=original_spacing,
                                  new_size=original_size, interpolator='linear')
            prediction_np_post = self.post_processing(scan_np, heatmap_np, prediction_np)
            end = time.time()
            self.logger.info("Total Time:{}, {}.".format(end - now, uid))

        return prediction_np_post


def draw_2d(image_2d, masks_2d, colors, thickness, alpha=0.5):
    original = np.dstack((image_2d, image_2d, image_2d))
    blending = np.dstack((image_2d, image_2d, image_2d))

    for mask, color, thick in zip(masks_2d, colors, thickness):
        _, contours, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blending, contours, -1, color, thick)

    return original * (1 - alpha) + blending * alpha

def draw_mask_tile_single_view(image, masks_list, coord_mask, num_slices, output_path, colors,
                               thickness, ext='tif', alpha=0.5, flip_axis=0, zoom_size=360,
                               coord_axis=1, titles=None, title_offset=10, title_color=(0, 255, 0)):
    assert (all([image.shape == mask.shape for mask_list in masks_list for mask in mask_list]))
    if flip_axis is not None:
        image = np.flip(image, axis=flip_axis)
        coord_mask = np.flip(coord_mask, axis=flip_axis)
        m_shape = np.asarray(masks_list).shape
        masks_list = np.asarray([np.flip(mask, axis=flip_axis)
                                 for mask_list in masks_list for mask in mask_list]).reshape(m_shape)
    n_mask_list = len(masks_list)
    n_mask_per_list = len(masks_list[0])
    if zoom_size is not None:
        sp = [image.shape[s] for s in set(list(range(image.ndim))) - {coord_axis}]
        zoom_max_ratio = zoom_size / np.max(sp)
        zoom_ratio = [1.0 if n == coord_axis else zoom_max_ratio for n in range(image.ndim)]

        def zoom_and_pad(i, ratio, target_size, pad_ignore_axis, order):
            i_z = ndimage.zoom(i, ratio, order=order)
            crop_slices = tuple([slice(0, min(n, target_size)) if i != pad_ignore_axis else slice(None, None)
                                 for i, n in enumerate(i_z.shape)])
            i_z = i_z[crop_slices]
            pad_size = tuple([(0, 0) if n == pad_ignore_axis else (
                (target_size - zs) // 2, target_size - zs - (target_size - zs) // 2)
                              for n, zs in zip(range(i.ndim), i_z.shape)])
            i_z_p = np.pad(i_z, pad_size, mode='constant')

            assert (all(i_z_p.shape[n] == target_size for n in range(i.ndim) if n != pad_ignore_axis))
            return i_z_p

        image = zoom_and_pad(image, zoom_ratio, zoom_size, coord_axis, order=1)
        coord_mask = zoom_and_pad(coord_mask, zoom_ratio, zoom_size, coord_axis, order=0)
        masks_list = [zoom_and_pad(mask, zoom_ratio, zoom_size, coord_axis, order=0)
                      for mask_list in masks_list for mask in mask_list]

    if np.sum(coord_mask) > 0:
        foreground_slices = ndimage.find_objects(coord_mask)[0]
        s = foreground_slices[coord_axis].start
        e = foreground_slices[coord_axis].stop
        stride = (e - s) // num_slices
        if stride == 0:
            e = coord_mask.shape[coord_axis] - 1
            s = 0
            stride = (e - s) // num_slices
        slices_ids = list(range(s, e, stride))[:num_slices]
        assert (len(slices_ids) == num_slices)
    else:
        print("no object found!")
        return
    all_slice_tiles = []
    for slice_id in slices_ids:
        # form one slice source from image and masks.
        slice_image = np.take(image, slice_id, axis=coord_axis)
        slice_image_tiles = [np.dstack((slice_image, slice_image, slice_image))]
        for mask_list_id in range(n_mask_list):
            masks = masks_list[mask_list_id * n_mask_per_list: mask_list_id * n_mask_per_list + n_mask_per_list]
            mask_array = [np.take(mask, slice_id, axis=coord_axis) for mask in masks]
            rendered_image = draw_2d(slice_image, mask_array, colors, thickness, alpha=alpha)
            if titles:
                cv2.putText(rendered_image, titles[mask_list_id], (title_offset, title_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, title_color, 1, cv2.LINE_AA)
            slice_image_tiles.append(rendered_image)
        # put all sources into a tile
        slice_image_tiles = np.vstack(slice_image_tiles)
        all_slice_tiles.append(slice_image_tiles)

    draw_ = np.hstack(all_slice_tiles)
    pad_size = ((0, 0), ((1920 - draw_.shape[1]) // 2, (1920 - draw_.shape[1]) - (1920 - draw_.shape[1]) // 2), (0, 0))
    draw_ = np.pad(draw_, pad_size, mode="constant")
    if output_path:

        output_path = Path(output_path).absolute()
        if not os.path.exists(output_path.parent):
            os.makedirs(output_path.parent)
        fname = str(output_path) + '.{}'.format(ext)
        if ext == 'tif' or ext == 'tiff':
            draw_ = draw_.astype(np.uint8)
            rawtiff=PIL.Image.fromarray(draw_)
            rawtiff.save(fname, tiffinfo={
                282: (300,),
                283: (300,),
                256: (draw_.shape[1],),
                257: (draw_.shape[0],),
                296: (2,)
            })
        else:
            cv2.imwrite(fname, draw_)

def segment_lobes(input_path, output_path, out_ext='mha', list_uids=None):
    ''' this function load scans from a given path support multi-frame dicom, mhd, and mha formats.
    by calling segment_lobe after you initialize the handle'''
    name_patterns = ["*.dcm", "*.mhd", "*.mha"]
    scan_path = input_path + "/images/ct/"
    scan_set = SimpleDataset(scan_path, name_patterns, recursive=True,
                             transforms=None)
    if list_uids is not None:
        scan_set.series_uids = list_uids

    output_mask_path = output_path + "/images/pulmonary-lobes/"
    if not os.path.exists(output_mask_path):
        os.makedirs(output_mask_path)
    output_screenshots_path = output_path + "/images/pulmonary-lobes-screenshot/"
    if not os.path.exists(output_screenshots_path):
        os.makedirs(output_screenshots_path)
    old_n = len(scan_set.series_uids)
    existing_files = glob.glob(output_mask_path + '/*.{}'.format(out_ext), recursive=True)
    file_stems = [Path(ef).stem for ef in existing_files]
    scan_set.series_uids = list(set(scan_set.series_uids) - set(file_stems))
    new_n = len(scan_set.series_uids)
    print("we are running {} by excluding {} to {}."
          .format(new_n, old_n - new_n, output_mask_path), flush=True)
    handle = segment_lobe_init()
    total_n = len(scan_set)
    results = []
    for idx in range(total_n):
        metrics = {}
        start_time = time.time()
        error_messages = []
        uid = scan_set.series_uids[idx]
        try:
            data_dict = scan_set[idx]
            uid = data_dict['meta']['uid']
            print("processing {}.".format(uid), flush=True)
            pred = segment_lobe(handle, data_dict['#image'], data_dict['meta'])
            if out_ext == "mha":
                out_func = write_array_to_mha_itk
            elif out_ext == 'mhd':
                out_func = write_array_to_mhd_itk
            else:
                raise NotImplementedError
            out_func(output_mask_path, [pred], [data_dict['meta']['uid']], type=np.uint8,
                     origin=data_dict['meta']["origin"][::-1],
                     direction=np.asarray(data_dict['meta']["direction"]).reshape(3, 3)[
                               ::-1].flatten().tolist(),
                     spacing=data_dict['meta']["original_spacing"][::-1])
            scan = data_dict['#image']
            vis_table = handle.settings.VISUALIZATION_COLOR_TABLE
            # vis_sparseness = self.settings.VISUALIZATION_SPARSENESS
            vis_alpha = handle.settings.VISUALIZATION_ALPHA
            scan_screenshot_path = os.path.join(output_screenshots_path, uid)
            labels = list(range(1,6))
            draw_mask_tile_single_view(windowing(scan).astype(np.uint8),
                                       [[pred == label for label in labels]],
                                       pred > 0,
                                       5, scan_screenshot_path,
                                       colors=[vis_table[n] for n in range(len(labels))],
                                       thickness=[-1] * len(labels), flip_axis=0, coord_axis=1,
                                       titles=["Lobes"], alpha=vis_alpha)

        except:
            track = traceback.format_exc()
            error_msg = "Cannot process {}/{} test scans with uid {}, {}.".format(idx, total_n, uid, track)
            error_messages.append(error_msg)
        finally:
            metrics['runtime_seconds'] = round(time.time() - start_time, 1)
            results.append({
                'entity': uid,
                'metrics': metrics,
                'error_messages': error_messages
            })
    json_path = os.path.join(output_path, 'results.json')
    with open(json_path, 'w') as f:
        print('results:', results)
        j = json.dumps(results)
        f.write(j)


def segment_lobe_init():
    setting_module_path = os.path.join(os.path.dirname(__file__), 'settings.py')
    settings = Settings(setting_module_path)
    settings.RELOAD_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__) + '/best.pth')
    lobe_seg_instance = LobeSegmentationTSTestCOVID(settings)
    return lobe_seg_instance


def segment_lobe(handle, scan, scan_meta):
    # scan has to be a 3-d numpy array
    # scan_meta has to be like {"uid": uid,
    #                          "size": scan.shape,
    #                          "spacing": spacing,
    #                          "origin": origin,
    #                          "original_spacing": spacing,
    #                          "original_size": scan.shape,
    #                          "direction": direction}
    window_max = handle.settings.WINDOWING_MAX
    window_min = handle.settings.WINDOWING_MIN
    resample_size = handle.settings.RESAMPLE_SIZE
    resample_mode = handle.settings.RESAMPLE_MODE
    size_jittering = handle.settings.TEST_SPACING
    T = transforms.Compose([Windowing(min=window_min,
                                      max=window_max),
                            GaussianBlur((0.6, 0.601)),
                            Resample(mode=resample_mode,
                                     factor=size_jittering,
                                     size=resample_size),
                            ToTensor()
                            ])
    data_dict = {
        "#image": scan,
        "meta": scan_meta
    }
    transformed_data_dict = T(data_dict)

    pred = handle.run(transformed_data_dict)
    return pred


if __name__ == "__main__":
    print("Docker start running job.")
    parser = ArgumentParser()
    parser.add_argument('--out_ext', type=str, nargs='?',
                        default="mha",
                        help="set up scan path.")
    args = parser.parse_args()
    # a = r"D:\workspace\datasets\covidra\derived_s3\original\scan/"
    # b = r"D:\workspace\datasets\covidra\derived_s3\original\testout/"
    segment_lobes('/input/', '/output/', args.out_ext)
    # segment_lobes(a, b, args.out_ext)
