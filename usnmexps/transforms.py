# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-04-21 13:46:14
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-05-17 14:05:10

import numpy as np
import re
from scipy.spatial.transform import Rotation

from instrulink import logger

from .constants import FLOAT_REGEXP, DEG_TO_RAD

''' 3D Transformation utilities '''

class MultiTransform:

    ''' Interface to perform multiple spatial transformations in succession '''

    INV_SUFFIX = '^-1'  # Inverse suffix
    TRANSLATION_RGXP = f'T\(({FLOAT_REGEXP}), ({FLOAT_REGEXP}), ({FLOAT_REGEXP})\)'  # Translation regexp
    ROTATION_RGXP = f'R([xyz])\(({FLOAT_REGEXP})°\)' # Rotation regexp
    SYMETTRY_RGXP = f'S([xyz])' # Symmetry regexp

    def __init__(self, transform_dict=None):
        '''
        Initialize
        
        :param transform_dict (optional): dictionary of (descriptor: matrix) pairs
            for each input transform
        '''
        self.transforms = []
        self.descriptors = []
        if transform_dict is not None:
            for desc, M in transform_dict.items():
                self.add_transform(M, desc)

    def __repr__(self):
        ''' String representation '''
        if len(self.descriptors) == 0:
            return 'IdentityTransform'
        chained_desc = ', '.join(self.descriptors)
        return f'{self.__class__.__name__}({chained_desc})'
    
    def topology(self):
        slist = []
        for desc, M in zip(self.descriptors, self.transforms):
            slist.append(f'{desc}:\n{M.round(3)}')
        s = '\n\n'.join(slist)
        return s

    def add_transform(self, M, desc):
        ''' 
        Add transform matrix to list of transforms
        
        :param 
        '''
        m, n = M.shape
        if m != n:
            raise ValueError(f'{m}-by-{n} matrix is not square')
        self.transforms.append(M)
        self.descriptors.append(desc)
    
    @staticmethod
    def rotation_angle_to_vector(axkey, theta):
        '''
        Translate a rotation angle around a given axis to a rotation vector
        
        :param theta: rotation angle (in radians)
        :param axkey: axis key (one of "X", "Y" or "Z")
        :return: rotation vector
        '''
        # Identify rotation axis index
        rotaxis = 'XYZ'.index(axkey.upper())
        # Construct rotation vector
        r = np.zeros(3)
        r[rotaxis] = theta
        # Return
        return r

    @classmethod
    def get_rotation_matrix(cls, axkey, theta):
        '''
        Get matrix transform for 3D rotation around a specific axis
        
        :param axkey: axis key ("X", "Y", or "Z")
        :param theta: rotation angle
        :return: rotation matrix
        '''
        vrot = cls.rotation_angle_to_vector(axkey, theta)
        return Rotation.from_rotvec(vrot).as_matrix()
    
    @staticmethod
    def get_translation_matrix(v):
        '''
        Get matrix transform for 3D translation by a specific vector
        
        :param v: 3D translation vector
        :return: translation matrix
        '''
        T = np.identity(v.size + 1)
        T[:-1, -1] = v
        return T

    @staticmethod    
    def get_symmetry_matrix(axkey):
        '''
        Get matrix transform for symmetry across specific axis
        
        :param axkey: axis key ("X", "Y", or "Z")
        :return: translation matrix
        '''
        iax = 'XYZ'.index(axkey.upper())
        S = np.identity(3)
        S[iax, iax] = -1
        return S

    @staticmethod
    def get_rotation_desc(axkey, theta):
        return f'R{axkey.lower()}({theta / DEG_TO_RAD:.0f}°)'

    def add_rotation(self, axkey, theta, origin=None):
        ''' Add rotation transform '''
        if origin is not None:
            self.add_translation(-origin)
        R = self.get_rotation_matrix(axkey, theta)
        self.add_transform(R, self.get_rotation_desc(axkey, theta))
        if origin is not None:
            self.add_translation(origin)
    
    def add_rotations(self, rot_dict, origin=None):
        ''' Add rotation transform(s) specified in a dictionary '''
        for axkey, theta in rot_dict.items():
            self.add_rotation(axkey, theta, origin=origin)
    
    @staticmethod
    def get_translation_desc(v):
        vstr = ', '.join([f'{vv}' for vv in v])
        return f'T({vstr})'
    
    def add_translation(self, v):
        ''' Add translation transform '''
        T = self.get_translation_matrix(v)
        self.add_transform(T, self.get_translation_desc(v))
    
    @staticmethod
    def get_symmetry_desc(axkey):
        return f'S{axkey.lower()}'
    
    def add_symmetry(self, axkey):
        ''' Add symmetry transform '''
        S = self.get_symmetry_matrix(axkey)
        self.add_transform(S, self.get_symmetry_desc(axkey))
    
    def add_symmetries(self, vbasis):
        ''' Add symmetry transform(s) based on vector base '''
        for k, x in zip('XYZ', vbasis):
            if x == -1:
                self.add_symmetry(k)
    
    @classmethod
    def apply_transform(cls, M, v):
        '''
        Apply multi-transform to vector(s)
        
        :param M: multi-transform matrix
        :param v: 3D vector or N-by-3 array of vectors
        :param: transformed vector(s) with same dimension as input
        '''
        if v.ndim > 1:
            return np.array([cls.apply_transform(M, vv) for vv in v])
        reduce = False
        if M.shape[0] == 4:
            reduce = True
            v = np.append(v, 1)
        vout = M.dot(v)
        if reduce:
            vout = vout[:-1]
        return vout
    
    def apply(self, v, verbose=False):
        ''' Apply all transforms successifvely to vector(s) '''
        for desc, M in zip(self.descriptors, self.transforms):
            if verbose:
                logger.info(f'applying {desc} transform')
            v = self.apply_transform(M, v)
        return v
    
    def invert_descriptor(self, desc):
        ''' Invert a specific transform descriptor '''
        # If match for translation descriptor, invert sign of vector
        mo = re.match(self.TRANSLATION_RGXP, desc)
        if mo is not None:
            v = np.array([float(mo.group(1)), float(mo.group(2)), float(mo.group(3))])
            return self.get_translation_desc(-v)
        
        # If match for rotation descriptor, invert sign of angle
        mo = re.match(self.ROTATION_RGXP, desc)
        if mo is not None:
            axkey = mo.group(1)
            theta = float(mo.group(2)) * DEG_TO_RAD
            return self.get_rotation_desc(axkey, -theta)
        
        # If match for symmetry descriptor, do nothing
        mo = re.match(self.SYMETTRY_RGXP, desc)
        if mo is not None:
            return desc
        
        # If no match (i.e. unknown transform type), add/remove inversion suffix
        if desc.endswith(self.INV_SUFFIX):
            return desc[:-len(self.INV_SUFFIX)]
        else:
            return f'{desc}{self.INV_SUFFIX}'
    
    def inverse(self):
        ''' Get inverse multi-transform '''
        inv_transforms = [np.linalg.inv(M) for M in self.transforms[::-1]]
        inv_descriptors = [self.invert_descriptor(d) for d in self.descriptors[::-1]]
        return self.__class__(dict(zip(inv_descriptors, inv_transforms)))


def transform_func(func, M):
    ''' 
    Modify a function to transform coordinates prior to evaluation
    
    :param func: evaluation function
    :param M: multi-transform object
    :return: modified function
    '''
    def transformed_func(x, y, z):
        # Assemble X, Y, Z into 2D coordinates array
        coords = np.array([x, y, z])
        # If array is 2 dimensional (i.e., x, y and z are vectors), set it as n-by-3 array
        if coords.ndim == 2:
            coords = coords.T
        # Transform coordinates
        trans_coords = np.squeeze(M.apply(coords)).T
        # Evaluate function on transformed coordinates
        return func(*trans_coords)
    return transformed_func