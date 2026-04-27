# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-05-03 08:32:56
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-06-12 15:48:11

import numpy as np
import pandas as pd
import time

from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from instrulink.logger import logger

from .utils import *
from .constants import MM_TO_UM
from .transforms import MultiTransform
from .adaptive_mappers import GPRFieldMapper


class Scanner:
    ''' Generic scanner '''

    def __init__(self, mp=None, vbasis=None, theta=None, ax=None, canvas=None):
        '''
        Initialization
        
        :param mp (optiional): micro-manipulator object
        :param vbasis: vector base of coordinate system (default = (1, 1, 1))
        :param theta (optional): dictionary of rotation angles (in radians) around X, Y, and Z axes
            used to define the scanning coordinate system
        :param ax (optional): matplotlib axis object on which ot plot the scanning process
        :param canvas (optional): canvas object on which to plot the scanning process
        '''
        # Assign input arguments as class attributes
        self.mp = mp
        self.theta = theta
        self.ax = ax
        self.vbasis = vbasis
        self.canvas = canvas

        # Set up empty data attribute
        self.data = None
        
        # Get initial position
        self.p0 = self.get_position()
    
    @property
    def theta(self):
        return self._theta
    
    @theta.setter
    def theta(self, value):
        # If value is not None
        if value is not None:
            # Check validity of input theta dictionary
            if not isinstance(value, dict):
                raise ValueError('rotation angles must be provided as a dictionary of (axis: angle) pairs')
            for k, v in value.items():
                if k.upper() not in 'XYZ':
                    raise ValueError(f'invalid axis key: {k} (must be one of "X", "Y" or "Z")')
                if not isinstance(v, (int, float)):
                    raise ValueError(f'invalid {k} rotation angle: {v} (must be float typed)')
            
            # Cast keys to upper case and values to float
            value = {k.upper(): float(v) for k, v in value.items()}

        # Otherwise, cast as empty dictionary
        else:
            value = {}

        # assign as class attribute
        self._theta = value

    @property
    def vbasis(self):
        ''' Getter for coordinates vector basis '''
        return self._vbasis

    @vbasis.setter
    def vbasis(self, value):
        ''' Setter for coordinates vector basis '''
        # If no vector provided, default to normal basis (1, 1, 1)
        if value is None:
            value = np.array([1., 1., 1.])
        
        # Cast input as array
        value = np.asarray(value)
        
        # Check that array dimensions are correct
        if value.ndim != 1 or value.size != 3:
            raise ValueError('vbasis must be a 3D vector')

        # Check that array values are correct
        if not all(np.abs(value) == 1):
            raise ValueError('all vector values must equal -1 or 1')
        
        # Assign
        self._vbasis = value
    
    @property
    def ax(self):
        ''' Getter for matplotlib axis '''
        return self._ax
    
    @ax.setter
    def ax(self, val):
        ''' Setter for matplotlib axis '''
        self._ax = val
        # Extract underlying matplotlib figure
        if val is not None:
            self.fig = val.get_figure()

    def get_position(self):
        ''' Get 3D position '''
        if self.mp is None:
            return np.zeros(3)
        else:
            return self.mp.get_position()

    def set_position(self, x, **kwargs):
        ''' Set 3D position '''
        if self.mp is not None:
            self.mp.set_position(x, **kwargs)
    
    def translate(self, v, **kwargs):
        ''' Translate position by some vector '''
        if self.mp is not None:
            self.mp.translate(v, **kwargs)
    
    def pos_str(self, pos):
        ''' Represent position as string '''
        pos_str = ', '.join([f'{x:.2f}' for x in pos])
        return f'[{pos_str}] um'
    
    def get_transform(self, org=None):
        ''' Get multi-transform object '''
        # Create multi-transform object
        M = MultiTransform()
        # Add any specified rotations
        M.add_rotations(self.theta, origin=org)
        # Add any specified symmetries
        M.add_symmetries(self.vbasis)
        # Return
        return M
    
    def get_normal_vector(self):
        ''' Get vector normal to scanning base plane (i.e. tank base surface) '''
        # Define reference translation vector: upward direction
        vup = np.array([0, 0, 1])
        # Grab system transform and apply it on normal vector
        vup = self.get_transform().apply(vup)
        # Return
        return vup
        
    def move_along_normal_vector(self, d):
        ''' 
        Move along vector normal to reference plane passing by current position
        
        :param d: distance to move (um, positive means away from reference plane)
        '''
        # If distance is 0, do nothing
        if d == 0:
            return
        # Get normal vector to reference plane, scaled by input distance
        tvec = self.get_normal_vector() * d
        # Log process
        s = 'away from' if d > 0 else 'towards'
        logger.info(f'moving {d:.2f} um {s} base surface (translation vector = {self.pos_str(tvec)})')
        # Translate
        self.translate(tvec)
    
    def move_updown(self, d):
        '''
        Move up or down along the Z axis by some distance

        :param d: distance to move (um, positive means up)
        '''
        # If distance is 0, do nothing
        if d == 0:
            return
        # Get vertical translation vector in scanner's basis coordinate system, scaled by input distance
        tvec = np.array([0, 0, d]) * self.vbasis  # um
        # Log process
        s = 'up' if d > 0 else 'down'
        logger.info(f'moving {d:.2f} um {s} (translation vector = {self.pos_str(tvec)})')
        # Translate
        self.translate(tvec)
    
    def check_coordinate(self, k, v):
        ''' Check micro-manipulator coordinate for a particular dimension '''
        if self.mp is not None:
            self.mp.check_coordinate(k, v)
    
    def check_coordinates(self, d):
        ''' Check dictionary of micro-manipulator coordinates '''
        if self.mp is not None:
            self.mp.check_coordinates(d)
    
    @staticmethod
    def expand_coordinate(axkey, coord):
        '''
        Expand 1 axis coordinate(s) into 3D coordinate vector(s)
        
        :param axkey: expansion axis key
        :param coord: 1D coordinate(s) along axis
        :return: 3D coordinate(s)
        '''
        coord = np.asarray(coord)
        squeeze = coord.ndim == 0
        i = 'XYZ'.index(axkey.upper())
        exp_coords = np.zeros((coord.size, 3))
        exp_coords[:, i] = coord
        if squeeze:
            exp_coords = np.squeeze(exp_coords)
        return exp_coords
    
    @staticmethod
    def get_coords_per_dim(x=None, y=None, z=None, check_odd=False, order='XYZ'):
        '''
        Determine coordinates per dimension
        
        :param x: vector of relative X coordinates (um)
        :param y: vector of relative Y coordinates (um)
        :param z: vector of relative Y coordinates (um)
        :param check_odd (optional): whether to check that dimension vectors are odd-numbered
        :param order (optional): string indicating the order in which the dimensions are scanned
        '''
        # Assemble dictionary of coordinate vector for each provided dimension
        coords_per_dim = dict(filter(lambda x: x[1] is not None, zip('XYZ', [x, y, z])))
        
        # Check valifity of coordinate vectors: dictionary of 1D, odd-sized vectors 
        if len(coords_per_dim) == 0:
            raise ValueError('at least 1 scan dimension must be provided')
        coords_per_dim = {k: np.atleast_1d(v) for k, v in coords_per_dim.items()}
        for v in coords_per_dim.values():
            if v.ndim != 1:
                raise ValueError('all input arrays must be 1-dimensional')
            if check_odd and v.size % 2 != 1:
                raise ValueError('all input arrays must contain an odd number of elements')
        
        # Re-arrange dictionary according to coordinates order
        ordered_keys = [k for k in order if k in coords_per_dim.keys()]
        coords_per_dim = {k: coords_per_dim[k] for k in ordered_keys}
        
        # Return dictionary
        return coords_per_dim

    def init_scan_data(self, coords, scankey):
        ''' Initialize scanning data as a dataframe '''
        # Create empty dataframe with number of scanned coordinates
        data = pd.DataFrame(index=range(coords.shape[0]))
        
        # For each scanned dimension, add position sequence (in mm) to dataframe
        for i, k in enumerate('xyz'):
            if k in scankey.lower():
                data[f'{k} (mm)'] = coords[:, i] / MM_TO_UM
        
        # Add "out" column to store outputs as they are evaluated during the sequence
        data['out'] = np.nan

        # Assign as class attribute
        self.data = data
    
    def update_scan_data(self, val):
        ''' Update scanning data '''
        # Get index of first NaN in "out" column
        idx = self.data['out'].isna().idxmax()
        # Update value at that index
        self.data.loc[idx, 'out'] = val

    def get_scan_plot(self, scankey, scantype=None, fig=None, subplot=111):
        '''
        Create scan axis backbone
        
        :param scankey: string indicating the axes on which the scan is performed
        :param scantype (optional): string representing the type of scan
        :param fig (optional): existing figure object
        :param subplot (optional): subplot position on figure
        :return: scan figure object
        '''
        # Sort scan key
        scankey = ''.join(sorted(scankey))
        
        # Create figure if not provided
        if fig is None:
            fig = plt.figure(figsize=(6, 5))
        
        # Initialize new axis with appropriate projection
        ndims = len(scankey)
        subplot_kwargs = {}
        if ndims == 3:
            subplot_kwargs = {'projection': '3d'}
        ax = fig.add_subplot(subplot, **subplot_kwargs)
        if ndims == 3 and self.vbasis[2] == -1:
            ax.view_init(azim=-110, elev=-160)
        
        # Set axis title with scanning type and covered dimensions
        prefix = scankey
        if scantype is not None:
            prefix  = f'{prefix} {scantype}'
        ax.set_title(f'{prefix} scan')
        
        # Set axis labels with correct dimensions
        for axname, dimname in zip('xyz', scankey):
            getattr(ax, f'set_{axname}label')(f'{dimname} (mm)')
        
        self.plot_geom_config(scankey, ax=ax)
        
        # Adjust layout and return
        fig.tight_layout()
        return ax
    
    def init_scan_plot(self, scankey):
        '''
        Initialize a scan plot
        
        :param scankey: string indicating the axes on which the scan is performed
        '''
        # Turn on interactive mode
        if self.canvas is None:
            plt.ion()

        # Extract matplotlib axis object
        if self.ax is None:
            self.ax = self.get_scan_plot(scankey)

        # Set empty handlers for matplotlib objects 
        self.mplobjs = []

        # Call update scan plot function to draw initial data 
        self.update_scan_plot()

    @staticmethod    
    def get_base_plane(org, vnorm):
        ''' 
        Get base plane from origin and normal vector
        
        :param org: origin point
        :param vnorm: normal vector
        '''
        # Compute plane origin
        d = -org.dot(vnorm)

        # Create XY meshgrid
        x = np.linspace(-1, 1, 3) + org[0]
        y = np.linspace(-1, 1, 3) + org[1]
        X, Y = np.meshgrid(x, y)

        # Calculate z surface
        Z = (-vnorm[0] * X - vnorm[1] * Y - d) / vnorm[2]
        
        # Return
        return X, Y, Z        
    
    def plot_geom_config(self, scankey, ax=None):
        ''' 
        Plot scanner geometrical configuration (base plane and normal vector)
        
        :param scankey: scan key
        :param ax (optinal): axis object
        :return: figure handle
        '''
        if ax is None:
            ax = self.get_scan_plot('XYZ')
            ax.set_title('scan configuration')
        fig = ax.get_figure()

        # Compute normal vector, and reduce it if needed
        vnorm = self.get_normal_vector()
        if len(scankey) == 2:
            vnorm = np.array([vnorm[0], vnorm[-1]])

        # Define origin
        org = self.p0 / MM_TO_UM
        
        # Plot normal vector
        c = 'C0'
        ax.plot(*zip(org, org + vnorm), c=c)
        ax.scatter(*(org + vnorm), c=c, marker='x')

        # Compute plane origin
        d = -org.dot(vnorm)

        # 3D case
        if len(scankey) == 3:
            # Plot surface
            ax.plot_surface(
                *self.get_base_plane(org, vnorm),
                edgecolor='royalblue', lw=0.5, rstride=1, cstride=1, alpha=0.5)
        
        else:
            # Create x vector
            x = np.linspace(-1, 1, 3)
            # Calculate y vector
            y = (d - vnorm[0] * x)
            # Plot line
            ax.plot(x, y, c='C0', lw=0.5)

        return fig
    
    def transform_and_check(self, coords):
        ''' 
        Transform coordinates to physical coordinate system and check that they are within bounds
        of the micro-manipulator

        :param coords: array of XYZ coordinates
        :return: transformed coordinates
        '''
        # Transform coordinates to align them with scanner system
        coords = self.get_transform().apply(coords)

        # Add initial position to relative coordinates, to get absolute coordinates
        coords += self.p0
        
        # Check that all positions fall within XYZ bounds of the micro-manipulator
        if coords.ndim == 1:
            self.check_coordinates(dict(zip('XYZ', coords)))
        else:
            for v in coords:
                self.check_coordinates(dict(zip('XYZ', v)))
        
        # Return transformed coordinates
        return coords
    
    def scan(self, coords, scankey, finit=None, feval=None, fclose=None, reset=False):
        '''
        Generic scanning function
        
        :param coords: vector of relative XYZ coordinates to scan sequentially
        :param scankey: string indicating the axes on which the scan is performed
        :param finit: initialization function called before starting the scan
        :param feval: evaluation function called at each location
        :param fclose: termination function called upon scan completion
        :param reset: whether to go back to the original position at the end of the scan
        :return: array of function evaluation results at all locations
        '''
        # Get initial position
        self.p0 = self.get_position()

        # Transform and check coordinates
        coords = self.transform_and_check(coords)
        
        # Initialize scanning data
        self.init_scan_data(coords, scankey)
        
        # Call initialization function, if any
        if finit is not None:
            finit()

        # Log scan start 
        logger.info(f'starting {len(coords)}-points {scankey} scan')
        
        # Initialize scan outputs vector
        out = np.zeros(coords.shape[0], dtype=float)
        
        # Initialize continuation flag to True
        self.continue_scan = True
        
        # For each scan location
        for ipos, v in enumerate(coords):
            # Log coordinates and move to position
            logger.info(f'position {ipos + 1}/{coords.shape[0]}: {self.pos_str(v)}')
            self.set_position(v, silent=True)
        
            # Call evaluation function (if any) and append to outputs,
            # and update continuation flag 
            if feval is not None:
                out[ipos], self.continue_scan = feval(ipos, *v)
            # Otherwise, append position index to outputs
            else:
                out[ipos] = ipos
            
            # Update scanning data with current evaluation output
            self.update_scan_data(out[ipos])

            # If continuation flag was set to False, abort scan
            if not self.continue_scan:
                break
        
        # Call termination function, if any
        if fclose is not None:
            fclose()

        # Log scan completion
        logger.info(f'{scankey} scan completed')
        
        # If specified, go back to original position
        if reset:
            logger.info(f'moving back to {self.pos_str(self.p0)}')
            self.set_position(self.p0)
        
        # Return output
        return out


class GridScanner(Scanner):
    ''' Rectilinear grid scanner '''

    @classmethod
    def get_scan_sequence(cls, **kwargs):
        '''
        Generate a zig-zag scan sequence of XYZ positions covering a
        1, 2, or 3 dimensional grid.

        :return: 2-tuple with:
            - coords: 2D array representing the XYZ scan sequence
            - coords_per_dim: dictionary of coordinates per dimension
        '''
        # Determine scan coordinates per dimension
        coords_per_dim = cls.get_coords_per_dim(**kwargs)

        # Compute scan dimensionality and number of coordinates per dimension
        ndims = len(coords_per_dim)
        nperdim = [v.size for v in coords_per_dim.values()]

        # Identify axes covered by scan sequence
        scanaxes = ['XYZ'.index(k) for k in coords_per_dim.keys()]

        # Initialize 3D positions array with zeros
        coords = np.zeros((np.prod(nperdim), 3))
        
        # 1D case: fill relevant axis of 3D position array with scan coordinates
        if ndims == 1:
            (k, v), = coords_per_dim.items()
            coords[:, 'XYZ'.index(k)] = v

        # 2D-3D case
        else:
            # Generate meshgrid of first 2 dimensions
            mesh = np.meshgrid(*list(coords_per_dim.values())[:2])

            # Invert every 2nd row of first dimension to generate 2D zig-zag
            mesh[0][::2] = mesh[0][::2, ::-1]
            
            # Serialize meshgrid vectors 
            smesh = [m.ravel() for m in mesh]
            
            # 2D case: fill relevant axes of 3D position array with serialized vectors
            if ndims == 2:
                for i, sm in zip(scanaxes, smesh):
                    coords[:, i] = sm
            
            # 3D case
            else:
                # Repeat serialized meshgrid for each value of last coordinate,
                # generating a 2D array of positions for each vector
                smesh = np.array([[sm] * nperdim[-1] for sm in smesh])
                
                # Invert 2D positions order on every 2nd value of last coordinate 
                # to generate 3D zig-zag
                for i in [0, 1]:
                    smesh[i, ::2] = smesh[i, ::2][:, ::-1]
                
                # Re-serialize vectors of first 2 dimensions,
                # and add them to 3D position array
                smesh = [m.ravel() for m in smesh]
                for i, sm in zip(scanaxes, smesh):
                    coords[:, i] = sm
                
                # Add repeated vector of last dimension to 3D position array
                coords[:, scanaxes[-1]] = np.repeat(
                    list(coords_per_dim.values())[-1], nperdim[0] * nperdim[1])
        
        # Return serialized coordinates sequence and coordinates per dimension
        return coords, coords_per_dim
    
    @classmethod
    def get_position_index(cls, xyz, **kwargs):
        ''' 
        Get index of XYZ position within scan sequence
        
        :param xyz: XYZ position vector
        :return: 3D index of position vector within scan sequence
        '''
        # Generate 3D scan sequence from coordinates per dimension vectors
        coords_seq, _ = cls.get_scan_sequence(**kwargs)
        # Compute array of distances from each scan location to input position
        dists = np.linalg.norm(coords_seq - xyz, axis=1)
        # Return index of minimal distance to position within the sequence
        return dists.argmin()

    @staticmethod
    def unwrap_scan_output(out, coords_per_dim):
        '''
        Unwrap the output of a scan sequence
        
        :param out: scan sequence output
        :param coords_per_dim: vector of scan coordinates per dimension
        :return unwrapped output array, in XYZ order
        '''
        # Extract scan dimensions from coordinates per dimension vectors
        dims = [v.size for v in coords_per_dim.values()]
        ndims = len(dims)

        # 1D case: return plain output
        if ndims == 1:
            return out

        # 2D-3D case
        else:
            # Reshape output as 2D array where 1st dimension is the scan's last dimension
            out = np.reshape(out, (dims[-1], -1))

            # Re-invert every 2nd row of first dimension to compensate zig-zag scan
            # in this dimension
            out[::2, :] = out[::2, ::-1]
            
            # 3D case 
            if ndims == 3:
                # Reshape output as 3D array
                out = np.reshape(out, dims[::-1])
                # Re-invert every 2nd row of first dimension
                out[:, ::2, :] = out[:, ::2, ::-1]
            
            # Transpose output matrix to recover proper dimensions order
            out = out.T

            # Re-organize output matrix to match XYZ order, by swapping unmatched
            # axes in reversed order
            dimkeys = list(coords_per_dim.keys())
            idxs = [dimkeys.index(k) for k in 'XYZ' if k in dimkeys]
            idxs_to_check = list(range(len(idxs)))[1:][::-1]
            for i in idxs_to_check:
                if idxs[i] != i:
                    out = np.swapaxes(out, idxs[i], i)

            # Return unwrapped and re-ordered output   
            return out
    
    def get_scan_plot(self, scankey, **kwargs):
        ''' Get grid scan plot '''
        return super().get_scan_plot(scankey, scantype='adaptive', **kwargs)

    def update_scan_plot(self):
        ''' Update an adaptive-scan plot '''
        # Remove existing handles and clear handles list
        for mplobj in self.mplobjs:
            if mplobj is not None:
                mplobj.remove()

        # Extract names of coordinates columns
        coordkeys = [k for k in self.data.columns if k.lower().startswith(tuple('xyz'))]

        # Extract and plot scan path
        scanpath = self.data[coordkeys]
        pathline, = self.ax.plot(*scanpath.values.T, 'k', alpha=0.1)
        
        # Extract and plot data from evaluated positions
        eval_data = self.data.dropna()
        evalcoords = eval_data[coordkeys]
        evalvals = eval_data['out']
        evalcollection = self.ax.scatter(
            *evalcoords.values.T,
            c=evalvals,
            s=5,
        )

        # Add handles to handles list
        self.mplobjs = [pathline, evalcollection]

        # Update figure
        redraw(self.fig)
        if self.canvas is not None:
            self.canvas.draw()

    def scan(self, x=None, y=None, z=None, feval=None, plot=False, reset=False, **kwargs):
        '''
        Scan 1, 2 or 3-dimensional rectilinear grid.
        
        :param x: vector of relative X coordinates (um)
        :param y: vector of relative Y coordinates (um)
        :param z: vector of relative Z coordinates (um)
        :param feval (optional): callback function to execute at each grid position
        :param plot (optional): whether to plot the grid scanning procedure
        :param reset (optional): whether to go back to original position after scan
        :return: 2-tuple with:
            - dictionary of coordinates per dimension (um)
            - multi-dimensional array of evaluation function results along the grid 
        '''
        # Get scan sequence and scan key from x, y and z input vectors
        coords, coords_per_dim = self.get_scan_sequence(x=x, y=y, z=z, **kwargs)
        scankey = ''.join(coords_per_dim.keys())
        npts = coords.shape[0]

        # Define empty initialization, update wrapper, and termination functions 
        finit, fclose, evalwrapper = None, None, feval

        # If plot requested, enrich these functions
        if plot:
            # Initialize plot upon scan start
            def finit():
                self.init_scan_plot(scankey)
            
            # Update plot at each position (or every nth position if high number of points)
            def evalwrapper(ipos, *args, **kwargs):
                if npts < LARGE_SCAN_THR_NPOINTS or ipos % LARGE_SCAN_REFRESH_EVERY == 0:
                    self.update_scan_plot()
                return feval(ipos, *args, **kwargs)

            # Update plot upon scan completion
            def fclose():
                self.update_scan_plot()
                if self.canvas is None:
                    plt.ioff()

        # Call generic scanning function
        scanvals = super().scan(
            coords, scankey, finit=finit, feval=evalwrapper,
            fclose=fclose, reset=reset)
        
        # Unwrap output of scan sequence
        scanvals = self.unwrap_scan_output(scanvals, coords_per_dim)
        
        # Return output
        return coords_per_dim, scanvals


class CrossSearchScanner(Scanner):
    ''' Cross-search scanner '''
    
    def get_scan_plot(self, scankey, **kwargs):
        ''' Get cross scan plot '''
        return super().get_scan_plot(scankey, scantype='cross', **kwargs)

    def update_scan_plot(self):
        ''' Update cross-scan plot '''
        # Remove existing handles and clear handles list
        for mplobj in self.mplobjs:
            if mplobj is not None:
                mplobj.remove()
        self.mplobjs = []

        # If data already exists
        if self.data is not None:
            # Extract names of coordinates columns
            coordkeys = [k for k in self.data.columns if k.lower().startswith(tuple('xyz'))] 

            # For each scanning "cycle"
            for _, gdata in self.data.groupby('cycle'):
                # Extract and plot scan path
                scanpath = gdata[coordkeys]
                pathline, = self.ax.plot(*scanpath.values.T, 'k')

                # Extract and plot data from evaluated positions
                eval_data = gdata.dropna()
                evalcoords = eval_data[coordkeys]
                evalvals = eval_data['out']
                evalcollection = self.ax.scatter(
                    *evalcoords.values.T,
                    c=evalvals,
                    s=5,
                )

                # Add handles to handles list
                self.mplobjs = self.mplobjs + [pathline, evalcollection]
                
        # Update figure
        redraw(self.fig)
        if self.canvas is not None:
            self.canvas.draw()
    
    def init_scan_data(self, coords, scankey):
        ''' Initialize scanning data as a dataframe '''
        # If reset flag ON
        if self.reset_scan_data:
            # Initialize new dataframe, and set flag to False
            super().init_scan_data(coords, scankey)
            self.data['cycle'] = 0
            self.reset_scan_data = False

        # Otherwise
        else:
            # Create new dataframe based on current data with same columns
            # and consecutive index
            newdata = pd.DataFrame(
                index=np.arange(coords.shape[0]) + len(self.data), columns=self.data.columns)
            
            # Fill in new dataframe with current coordinates
            for col in newdata.columns:
                if col == 'out':
                    newdata[col] = np.nan
                elif col == 'cycle':
                    newdata[col] = self.data[col].max() + 1
                else:
                    newdata[col] = coords[:, 'xyz'.index(col[0])] / MM_TO_UM
            
            # Append to existing data
            self.data = pd.concat([self.data, newdata])

    def _scan(self, x=None, y=None, z=None, feval=None, order='XYZ', plot=False):
        '''
        Scan 1, 2 or 3-dimensional space in a cross-search fashion.
        
        :param iter: iteration index
        :param x: vector of relative X coordinates (um)
        :param y: vector of relative Y coordinates (um)
        :param z: vector of relative Y coordinates (um)
        :param feval (optional): callback function to execute at each grid position
        :param order (optional): string indicating the order in which the dimensions are scanned
        :param plot (optional): whether to plot the grid scanning procedure
        :return: 2-tuple with:
            - array of XYZ scanning locations
            - array of evaluation function results along the locations
        '''
        # Get scanning sequence and scan key from x, y and z input vectors
        coords_per_dim = self.get_coords_per_dim(
            x=x, y=y, z=z, check_odd=True, order=order)
        scankey = ''.join(coords_per_dim.keys())
        
        # Determine problem dimensionality
        ndims = len(scankey)
        if ndims == 1:  # Turn off plot for 1D case
            plot = False
        
        # Check that all positions fall within XYZ bounds
        for k, v in coords_per_dim.items():
            for vext in bounds(v):
                self.check_coordinate(k, vext)

        # Initialize scan plot if specified
        if plot:
            self.init_scan_plot(scankey)
        
        # Compute total number of coordinates 
        ncoords = sum([c.size for c in coords_per_dim.values()])

        # Initialize global output vector
        out = []

        # Initialize position index and continue flag
        ipos = 0
        self.continue_scan = True

        # Generate expanded coordinates dictionary
        exp_coords_dict = {k: self.expand_coordinate(k, v) for k, v in coords_per_dim.items()}
        # Transform expand coordinates
        M = self.get_transform()
        exp_coords_dict = {k: M.apply(v) for k, v in exp_coords_dict.items()}
            
        # Log scan start
        logger.info(f'starting {scankey} cross-scan')

        # For each scanned dimension
        for k, exp_coords in exp_coords_dict.items():
            # Extract dimension index
            i = 'XYZ'.index(k)

            # Add current position offset
            exp_coords += self.ref_pos

            # Initialize scan data
            self.init_scan_data(exp_coords, scankey)

            # Initialize vector to store outputs along current scanning axis
            dimout = np.zeros(exp_coords.shape[0], dtype=float)

            # For each XYZ coordinate along scanning axis
            for j, xyz in enumerate(exp_coords):

                # Move to XYZ position
                self.set_position(xyz, silent=plot)
                                
                # Evaluate function, and fill out output vector
                if feval is not None:
                    dimout[j], self.continue_scan = feval(ipos, *xyz)
                else:
                    dimout[j], self.continue_scan = ipos, True
                
                logger.info(f'position {ipos + 1}/{ncoords}: {self.pos_str(xyz)}, output = {dimout[j]:.4f}')
                
                # Update scan data with current evaluation output
                self.update_scan_data(dimout[j])

                # Update scan plot if specified
                if plot:
                    self.update_scan_plot()

                # Append to global output
                out.append((xyz, dimout[j]))
                
                # Increment position index
                ipos += 1

                if not self.continue_scan:
                    break
            
            if self.continue_scan:
                # Find location giving max output along dimension and set as new reference
                jmax, max_along_dim = np.argmax(dimout), dimout.max()
                self.ref_pos = exp_coords[jmax]
                logger.info(
                    f'found max of {max_along_dim:.3f} along {k} axis at {self.pos_str(self.ref_pos)}')
            
                # Move to that location
                self.set_position(self.ref_pos, silent=plot)
            else:
                break
        
        # Turn off interactive plot mode
        if plot and self.canvas is None:
            plt.ioff()
        
        # Log scan completion
        logger.info(f'{scankey} scan completed')

        # Unwrap outputs
        scanlocs, scanvals = [np.array(o) for o in zip(*out)]

        # Find location giving max output across dimensions
        imax, max_across_dims = np.argmax(scanvals), scanvals.max()

        if self.continue_scan:
            # If global max across dimensions is higher than local max from last dimension,
            # reset reference to global max and move to that location
            if max_across_dims > max_along_dim:
                logger.warning(
                    f'global max of {max_across_dims:.3f} at {self.pos_str(self.ref_pos)} higher than local max of {max_along_dim:.3f} -> moving there')
                self.ref_pos = scanlocs[imax]
                self.set_position(self.ref_pos, silent=plot)
        
        # Return outputs
        return scanlocs, scanvals

    def scan(self, x=None, y=None, z=None, niters=3, ref_factor=3, endmode='gotomaxproj', **kwargs):
        '''
        Recursive cross-scan, dividing search domain by a given factor at each iteration

        :param x: vector of relative X coordinates (um)
        :param y: vector of relative Y coordinates (um)
        :param z: vector of relative Y coordinates (um)
        :param niters: number of cross-search iterations
        :param ref_factor: search refinement factor at each iteration
        :param endmode: string specifying what to do at the end of the search (see SCAN_ENDMODES):
        :return: 2-tuple with:
            - array of XYZ scanning locations
            - array of evaluation function results along the locations
        '''
        # Check end mode validity
        if endmode not in SCAN_ENDMODES:
            raise ValueError(f'invalid endmode "{endmode}". Candidates are {SCAN_ENDMODES}')

        # Get initial position vector and set it as reference position
        self.p0 = self.get_position()
        self.ref_pos = self.p0.copy()

        # Set "reset scan data" flag
        self.reset_scan_data = True

        # Perform initial cross-search
        scanlocs, scanvals = self._scan(x=x, y=y, z=z, **kwargs)
        
        # For each additional iteration
        for i in range(1, niters):
            if self.continue_scan:
                logger.info(f'refinement {i}')
                # Determine new relative coordinate vectors
                if x is not None:
                    x = refine_vec(x, ref_factor)
                if y is not None:
                    y = refine_vec(y, ref_factor)
                if z is not None:
                    z = refine_vec(z, ref_factor)
                # Perform new cross-search on refined coordinates
                scanlocs, scanvals = self._scan(x=x, y=y, z=z, **kwargs)

        # If specified, go back to reference (i.e. original) position
        if endmode == 'gotoref':
            logger.info(f'moving back to {self.pos_str(self.p0)}')
            self.set_position(self.p0)
        
        # If specified, go to position of max value
        elif endmode == 'gotomax':
            logger.info(f'staying at max value position: {self.pos_str(self.ref_pos)}')
        
        # If specified, go to max value position projected onto reference plane
        # of initial position
        elif endmode == 'gotomaxproj':
            logger.info(f'projecting {self.pos_str(self.ref_pos)} onto {self.pos_str(self.p0)} plane')
            pproj = project_onto_plane(self.ref_pos, self.get_normal_vector(), self.p0)
            logger.info(f'going to projected location: {self.pos_str(pproj)}')
            self.set_position(pproj)
             
        # Return outputs of last cross-search
        return scanlocs, scanvals
    

class AdaptiveScanner(Scanner):
    ''' Interface for adaptive scanning procedures '''

    def init_scan_plot(self, scankey):
        '''
        Initialize a scan plot
        
        :param scankey: string indicating the axes on which the scan is performed
        '''
        # Turn on interactive mode
        if self.canvas is None:
            plt.ion()

        # Extract matplotlib axis object
        if self.ax is None:
            self.ax = self.get_scan_plot(scankey)

        # Set empty handlers for matplotlib object
        self.mplobj = None

        # Call update scan plot function to draw initial data 
        self.update_scan_plot()
    
    def get_scan_plot(self, scankey, **kwargs):
        ''' Get grid scan plot '''
        return super().get_scan_plot(scankey, scantype='adaptive', **kwargs)
    
    def update_scan_plot(self):
        ''' Update an adaptive-scan plot '''
        # Remove existing handles and clear handles list
        if self.mplobj is not None:
            self.mplobj.remove()
            self.mplobj = None

        # Update scanned locations and store matplotlib object
        _, self.mplobj = self.mapper.scan_plot(
            ax=self.ax, 
            add_cbar=False,
            title=None,
            transform=lambda x: self.transform_and_check(x * MM_TO_UM) / MM_TO_UM
        )

        # Update figure
        redraw(self.fig)
        if self.canvas is not None:
            self.canvas.draw()
    
    @property
    def continue_scan(self):
        ''' Get continuation flag '''
        return self._continue_scan
    
    @continue_scan.setter
    def continue_scan(self, value):
        ''' Set continuation flag '''
        self._continue_scan = value
        if not value and self.mapper is not None:
            self.mapper.stop()
    
    def scan(self, x=None, y=None, z=None, feval=None, plot=False, reset=False, **kwargs):
        '''
        Scan 1, 2 or 3-dimensional rectilinear grid.
        
        :param x: vector of relative X coordinates (um)
        :param y: vector of relative Y coordinates (um)
        :param z: vector of relative Z coordinates (um)
        :param feval (optional): callback function to execute at each grid position
        :param plot (optional): whether to plot the grid scanning procedure
        :param reset (optional): whether to go back to original position after scan
        :return: 2-tuple with:
            - dictionary of coordinates per dimension (um)
            - multi-dimensional array of evaluation function results along the grid 
        '''
        # Initialize empty placeholder for mapper object
        self.mapper = None
        
        # Assemble dictionary of coordinate vector for each provided dimension
        coords_per_dim = {
            'X': np.asarray(x),  # um
            'Y': np.asarray(y),  # um
            'Z': np.asarray(z),  # um
        }
        scankey = ''.join(coords_per_dim.keys())

        # Remove order key from kwargs
        if 'order' in kwargs:
            kwargs.pop('order')

        # Get initial position
        self.p0 = self.get_position()

        # Get bounds per dimension
        bounds_per_dim = {k : np.array([v.min(), v.max()]) for k, v in coords_per_dim.items()}

        # Get 3D coordinates of volume corner points by taking all possible combinations of bounds
        corner_points = np.array(list(product(*list(bounds_per_dim.values()))))

        # Transform and check volume corners coordinates in physical coordinate system 
        self.transform_and_check(corner_points)

        # Define evaluation function wrapper
        def feval_wrapper(x):
            # Rescale to um, transform and check coordinates in physical system
            xt = self.transform_and_check(x * MM_TO_UM)
            # Move to position
            self.set_position(xt, silent=True)
            # Increment position index
            self.ipos += 1 
            # Call evaluation function
            yout, self.continue_scan = feval(self.ipos, *xt)
            logger.info(f'position {self.ipos}: {self.pos_str(xt)}, output = {yout:.4f}')            
            # If requested, update scan plot
            if plot:
                self.update_scan_plot()
            # Return evaluation result
            return yout

        # Create mapper object
        self.mapper = GPRFieldMapper(
            feval_wrapper,
            {k: v / MM_TO_UM for k, v in bounds_per_dim.items()},  # mm
            unit='mm',
            projkey='XYZ', # 'RZΘ',
        )

        # If requested, initialize scan plot
        if plot:
            self.init_scan_plot(scankey)

        # Log scan start 
        logger.info(f'starting adaptive {scankey} scan')
        self.ipos = 0
        
        # Construct evaluation grid
        Xgrid = np.meshgrid(*coords_per_dim.values(), indexing='ij')  # um
        Xgrid_mm = tuple(x / MM_TO_UM for x in Xgrid)  # mm

        # Initialize continuation flag to True
        self.continue_scan = True
        
        # Initialize mapper
        self.mapper.initialize(n=3000)

        # Initialize optimizer
        self.mapper.optimize_init(Xgrid_mm, max_pts=3500, **kwargs)

        # Refine GPR model iteratively
        logger.info('optimizing field predictor iteratively...')

        # Get reference time
        tstart = time.perf_counter()

        # While continuation flag is True
        while self.continue_scan:
            # Update mapper underlying GPR model
            conv = self.mapper.update()
            # Compue elapsed time for iteration
            tstop = time.perf_counter()
            tcomp = tstop - tstart
            tstart = tstop
            # Log iteration and convergence score
            logger.info(f'iteration {self.mapper.iiter} ({tcomp:.1f} s): convergence score = {self.mapper.conv_score:.4f}')
            # Update continuation flag, if not manually set to False
            if self.continue_scan:
                self.continue_scan = not conv
            # If scan manually stopped during execution, log warning
            else:
                logger.warning('scan manually stopped during execution')
        
        # Log scan completion
        logger.info(f'{scankey} scan completed. Fitted GPR model: kernel = {self.mapper.kernel}')
        
        # If plot requested, turn-off interactive mode
        if plot:
            if self.canvas is None:
                plt.ioff()
        
        # If specified, go back to original position
        if reset:
            logger.info(f'moving back to {self.pos_str(self.p0)}')
            self.set_position(self.p0)

        # Predict over evaluation grid
        scanvals = self.mapper.predict(Xgrid_mm)
        
        # Return output
        return coords_per_dim, scanvals
