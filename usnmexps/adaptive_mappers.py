# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2025-02-25 00:08:49
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-06-13 15:54:47

import pickle
import textwrap
from functools import wraps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn.metrics import r2_score
from scipy.stats import qmc
from scipy.spatial.distance import cdist

from .utils import logger, apply_rolling_window, cylindrical_to_cartesian, cartesian_to_cylindrical, find_shortest_path


class AdaptiveFieldMapper:
    ''' Generic class for adaptive field mapping. '''

    # Possible projection keys
    XYZ_KEYS = ['X', 'Y', 'Z']
    PROJ_KEYS = [  
        'XYZ',  # Cartesian coordinates
        'RZ',    # Cylindrical coordinates without azimuthal angle
        'RZΘ',  # Cylindrical coordinates
    ]

    NAVG_METRICS = 5  # Number of iterations over which to average metrics

    # Convergence thresholds
    CONV_THRS = {
        'rel. param change': (.05, 'ub'),  # Relative change in kernel parameter threshold (upper bound)
        'LML change': (None, 'ub'),  # Change in LML threshold (upper bound)
        'rel. max uncertainty': (0.05, 'ub'),  # Relative max uncertainty threshold (upper bound, in MPa)
        'r2': (0.95, 'lb'),  # R^2 threshold (lower bound)
        'max. rel. error': (.1, 'ub'),  # Max relative error absolute value threshold (upper bound)
    }

    def __init__(self, feval, bounds_per_dim, varscale=None, unit=None, projkey=None):
        '''
        Constructor.

        :param feval: spatial evaluation function
        :param bounds_per_dim: dictionary of bounds per dimension
        :param varscale: typical variation scale broadcasted for each dimension. If None, set to 1/3 of the 
            smallest domain range across all dimensions.
        :param unit (optional): unit of the physical domain used for plotting purposes
        :param projkey: projection key specifying the projected coordinate system where the GPR is evaluated (default: None).
        '''
        # Assign attributes
        self.feval = feval
        self.projkey = projkey
        self.bounds_per_dim = bounds_per_dim
        self.varscale = varscale
        self.unit = unit

        # Initialize training data
        self.df_train = None
        self.init_mode = None
        self.prev_lml = None
        self.prev_kernel_params = None
        self.nperiter = None
        self.xnext = None
        self.ynext_prior = None

        # Initialize underlying model
        self.init_model()

        # Initialize can_eval flag
        self.can_eval = True
        
    def __repr__(self):
        ''' String representation. '''
        bounds_str = ', '.join([f'{k}={np.round(v, 2)}' for k, v in self.bounds_per_dim.items()])
        kwargs = {
            'varscale': np.round(self.varscale, 2),
            'proj': self.projkey,
        }
        if self.ninit > 0:
            kwargs['ninit'] = self.ninit
            kwargs['init_mode'] = self.init_mode
        if self.nperiter is not None:
            kwargs['nperiter'] = self.nperiter
        if self.iiter is not None:
            kwargs['niters'] = self.iiter
        kwargs_str = ', '.join([f'{k}={v}' for k, v in kwargs.items()])
        return f'{self.__class__.__name__}({bounds_str}, {kwargs_str})'
    
    def wrappped_str(self, txt):
        ''' Precede text with object representation and constrain it to 100 characters max per line. '''
        return textwrap.fill(f'{repr(self)}: {txt}', 100)
    
    def stop(self):
        self.can_eval = False

    @property
    def use_cylindrical(self):
        return self.projkey.startswith('RZ')
    
    @property
    def feval(self):
        ''' Spatial evaluation function. '''
        return self._feval
    
    @feval.setter
    def feval(self, func):
        ''' Set spatial evaluation function. '''
        # If None, return
        if func is None:
            return
        
        # Check if input is callable
        if not callable(func):
            raise ValueError(f'input {func} is not callable')

        # Define wrapper around evaluation function
        @wraps(func)
        def wrapper(X):
            # Extract iteration number
            iiter = self.iiter + 1
            # If multi-point input
            if X.ndim > 1:
                # Re-organize to yield shortest "evaluation path" 
                ishort, _ = find_shortest_path(X)
                X = X[ishort]
                # Evaluate function at each point, and add to training data
                y = np.zeros(X.shape[0])
                for i, x in enumerate(X):
                    if not self.can_eval:
                        break
                    y[i] = func(x)
                    self.add_data_row(x, y[i], iiter)
                # Return array of evaluated values
                return y
            
            # Default single-point input: evaluate function, add to training data,
            # and return output
            y = func(X)
            self.add_data_row(x, y, iiter)
            return y
        
        # Assign wrapped function
        self._feval = wrapper
    
    @property
    def projkey(self):
        ''' Projection key. '''
        return self._projkey
    
    @projkey.setter
    def projkey(self, val):
        ''' Set projection key. '''
        if val is None:
            val = 'XYZ'
        if val not in self.PROJ_KEYS:
            raise ValueError(f'invalid projection key: {val}')
        self._projkey = val

    @property
    def dimkeys(self):
        ''' Dimension keys. '''
        return list(self.bounds_per_dim.keys())

    @property
    def range_per_dim(self):
        ''' Range per dimension. '''
        return {k: np.diff(v)[0] for k, v in self.bounds_per_dim.items()}
    
    @property
    def minrange(self):
        ''' Minimum range across all dimensions. '''
        return np.min(list(self.range_per_dim.values()))

    @property
    def volume(self):
        ''' Volume of the domain. '''
        return np.prod(list(self.range_per_dim.values()))
    
    @property
    def varscale(self):
        ''' Variation scale. '''
        return self._varscale
    
    @varscale.setter
    def varscale(self, val):
        ''' Set variation scale. '''
        if val is None:
            val = self.minrange / 3
        ratio = val / self.minrange
        if ratio < .05:
            raise ValueError(f'variation scale ({val}) is too small compared to minimal domain range ({self.minrange})')
        elif ratio >= 1:
            raise ValueError(f'variation scale ({val}) exceeds minimal domain range ({self.minrange})')
        self._varscale = val
    
    def kernel(self):
        ''' Underlying model kernel '''
        raise NotImplementedError
    
    def init_model(self):
        ''' Initialize unerlying model '''
        raise NotImplementedError
    
    def parse_parameter(self, x):
        ''' Parse kernel parameter. '''
        # If input is a list, parse each element
        if isinstance(x, dict):
            d = {}
            for k, v in x.items():
                try:
                    d[k] = self.parse_parameter(v)
                except ValueError as e:
                    raise ValueError(f'error parsing {k}: {e}')
                
                # If multi-level dictionary, flatten it by concatenating keys
                if isinstance(d[k], dict):
                    d.update({f'{k}_{kk}': vv for kk, vv in d[k].items()})
                    del d[k] 
            return d

        # If input is an array 
        if isinstance(x, np.ndarray):
            # If 0D, return scalar
            if x.ndim == 0:
                return x.item()
            # If 1-element, return extracted element 
            elif x.size == 1:
                return x[0]
            # Otherwise, parse along projection keys
            else:
                if x.size != len(self.projkey):
                    raise ValueError(f'array size ({x.size}) does not match {self.projkey} domain dimensionality')
                return dict(zip(self.projkey, x))
        
        # Otherwise, return input as is
        return x
    
    def get_kernel_params(self, *args, **kwargs):
        raise NotImplementedError

    def create_grid(self, nperdim, bounds_per_dim):
        '''
        Generate grid
        
        :param nperdim: number of samples per dimension on grid
        :param bounds_per_dim: dictionary of bounds per dimension
        :return: n-dimensional rectilinear grid coordinates matrix
        '''
        ntot = np.prod(list(nperdim.values()))
        nstr = '-by-'.join([str(n) for n in nperdim.values()])
        logger.info(f'{self}: creating {ntot} ({nstr}) points rectilinear grid')
        coords_per_dim = {k: np.linspace(*v, nperdim[k]) for k, v in bounds_per_dim.items()}
        return np.meshgrid(*coords_per_dim.values(), indexing='ij')
    
    def serialize(self, X):
        '''
        Serialize n-dimensional grid coordinates matrix into 2D matrix.
        
        :param X: n-dimensional rectilinear grid coordinates matrix
        :return: 2D matrix of serialized coordinates (npoints, ndim)
        '''
        return np.vstack([xx.ravel() for xx in X]).T
    
    def project(self, x):
        ''' 
        Project input coordinates to coordinate system where underlying GPR gets evaluated.

        :param x: input cartesian coordinates
        :return: projected coordinates 
        '''
        if self.use_cylindrical:
            x = cartesian_to_cylindrical(x)
            if self.projkey == 'RZ':
                x = x[:, :-1]
        return x
    
    def inverse_project(self, x):
        ''' 
        Inverse-project input coordinates to original coordinate system.

        :param x: input projected coordinates
        :return: original cartesian coordinates
        '''
        if self.use_cylindrical:
            x = cylindrical_to_cartesian(x)
        return x

    def get_nperdnim(self, n, bounds_per_dim):
        ''' 
        Determine number of initial samples per dimension for a target
        number of samples, based on domain dimensions.
        
        :param n: target number of samples
        :param bounds_per_dim: dictionary of bounds per dimension
        :return: dictionary of number of samples per dimension
        '''
        # Determine number of initial samples per dimension based on domain dimensions
        ranges = np.array([b[1] - b[0] for b in bounds_per_dim.values()])
        factors = ranges / np.max(ranges)
        fprod = np.prod(factors)
        nmax = np.round(np.power(n / fprod, 1 / factors.size))
        n = np.round(factors * nmax).astype(int)
        return dict(zip(bounds_per_dim.keys(), n))
    
    def sample_rectilinear(self, n, bpd=None):
        '''
        Sample n points over a rectilinear grid.

        :param n: number of samples
        :param bpd: bounds per dimension
        :return: n-by-m matrix of samples
        '''
        # If no bounds per dimension are provided, use default
        if bpd is None: 
            bpd = self.bounds_per_dim.copy()

        # Determine number of samples per dimension based on 
        npd = self.get_nperdnim(n, bpd)

        # Create rectilinear grid
        X = self.create_grid(npd, bpd)

        # Serialize grid and return
        return self.serialize(X)
    
    def sample_lhc(self, n, bpd=None):
        '''
        Sample n points using Latin hypercube sampling.

        :param n: number of samples
        :param bpd: bounds per dimension
        :return: n-by-m matrix of samples
        '''
        # If no bounds per dimension are provided, use default
        if bpd is None:
            bpd = self.bounds_per_dim.copy()

        # Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=len(bpd))
        sample = sampler.random(n=n)

        # Scale to domain bounds and return
        lb = np.array([bpd[k][0] for k in bpd.keys()])
        ub = np.array([bpd[k][1] for k in bpd.keys()])
        return qmc.scale(sample, lb, ub)
    
    @property
    def xtrain(self):
        ''' Access training data locations '''
        if self.df_train is None:
            return None
        return self.df_train[self.XYZ_KEYS].values

    @property
    def ytrain(self):
        ''' Access training data values '''
        if self.df_train is None:
            return None
        return self.df_train['value'].values

    @property
    def itrain(self):
        ''' Access training data iteration '''
        if self.df_train is None:
            return None
        if 'iteration' not in self.df_train.columns:
            return None
        return self.df_train['iteration']
    
    @property
    def iiter(self):
        if self.itrain is None:
            return None
        if len(self.itrain) == 0:
            return -1
        return self.itrain.max()
    
    @property
    def ninit(self):
        ''' Number of samples in initial population '''
        if self.itrain is None:
            return 0
        return (self.itrain == 0).sum()
    
    def init_df_train(self):
        ''' Initialize training dataframe '''
        self.df_train = pd.DataFrame(
            columns=[*self.XYZ_KEYS, 'iteration', 'value'])
        self.df_train.index.name = 'sample'
    
    def add_data_row(self, x, y, i):
        '''
        Add row to training data.

        :param x: input cartesian coordinates
        :param y: output value
        :param i: iteration number
        '''
        irow = len(self.df_train)
        self.df_train.loc[irow] = {**dict(zip(self.XYZ_KEYS, x)), 'value': y, 'iteration': i}
        if irow == 0:
            self.df_train['iteration'] = self.df_train['iteration'].astype(int) 
 
    def initialize(self, n=None, mode='random'):
        ''' 
        Populate initial cartesian evaluation grid and fit GPR model.
        
        :param n (optional): number of initial samples. If not provided, n is chosen based on the volume size and expected variation scale
        :param mode (optional): sampling mode, one of 'rect' (for rectilinear) or 'random' (for Latin hypercube). Default: 'rect'
        '''
        # If n is not provided, determine it based on the volume size and expected variation scale
        if n is None:
            npervarscale = 2
            pitch = self.varscale / npervarscale  # target grid pitch
            logger.info(f'{self}: no initial number of samples provided -> aiming for {1 / npervarscale}*varscale ({pitch}) pitch')
            ndim = len(self.bounds_per_dim)
            density = np.power(pitch, -ndim)  # target volumetric density
            n = int(np.round(self.volume * density))  # target number of samples
        
        # Sample initial points
        if mode == 'rect':
            xinit = self.sample_rectilinear(n)
        elif mode == 'random':
            xinit = self.sample_lhc(n)
        else:
            raise ValueError(f'invalid sampling mode: {mode}')
        logger.info(f'{self}: generated initial population of {n} {mode} points')

        # Initialize dataframe
        self.init_df_train()

        # Evaluate function at initial points
        self.feval(xinit)
        logger.info(f'{self}: evaluated function at initial points')

        # Fit GPR model
        self.fit()
        logger.info(f'{self}: fitted GPR model: kernel = {self.kernel}')

        # Update class attributes
        self.init_mode = mode
        self.nperiter = None
    
    def fit(self, *args, **kwargs):
        ''' Fit underlying model to training data. '''
        raise NotImplementedError
    
    def _predict(self, x, **kwargs):
        '''
        Predict values over coordinates vector
        
        :param x: n-by-m matrix of cartesian coordinates
        :return: n-dimensional vector of predicted values (and optionally uncertainties)
        '''
        raise NotImplementedError
    
    def predict(self, x, return_std=False):
        ''' 
        Predict outputs over a rectilinear grid.
        
        :param x: n-dimensional meshgrid of cartesian grid coordinates,
            or n-by m matrix of cartesian coordinates
        :param return_std: whether to also return uncertainty
        :return: n-dimensional matrix of predicted values, 
            or n-by m matrix of predicted values
        '''
        # If meshgrid input, extract dimensions and serialize it
        is_meshgrid = isinstance(x, tuple)
        if is_meshgrid:
            nperdim = x[0].shape
            x = self.serialize(x)
        
        # If single point input, reshape it to 2D
        is_singleton = x.ndim == 1
        if is_singleton:
            x = x.reshape(1, -1)
                
        # Predict values over serialized grid and unpack output
        ypred = self._predict(self.project(x), return_std=return_std)
        if return_std:
            ypred, ystd = ypred
        
        # If input was singleton, reshape output to 0D
        if is_singleton:
            ypred = ypred[0]
            if return_std:
                ystd = ystd[0]
        
        # If meshgrid input, reshape output(s) to match its shape
        if is_meshgrid:
            ypred = ypred.reshape(*nperdim)
            if return_std:
                ystd = ystd.reshape(*nperdim)
        
        # Return output(s)
        if return_std:
            return ypred, ystd
        else:
            return ypred        
    
    def select_sampling_points(self, X, Ystd, n, dmin=None):
        '''
        Select N new sampling points based on GP uncertainty and spatial diversity.

        :param X: m-dimensional meshgrid of cartesian grid coordinates
        :param Ystd: n-dimensional matrix of uncertainty values
        :param n: number of new points to select.
        :param dmin: minimum allowed distance between selected points. If none, the distance is set to the default value.
        :return: array of shape (n, m) with selected next sampling points.
        '''
        # Serialize grid coordinates and uncertainty
        x = self.serialize(X)
        ystd = Ystd.ravel()
        
        # Sort points by decreasing uncertainty
        x = x[np.argsort(-ystd)]
        
        # Start with the most uncertain point
        xout = [x[0]]  

        # If only one point is requested, return it
        if n == 1:
            return np.array(xout)
        
        # If no minimum distance is provided, set it to default variation scale
        if dmin is None:
            dmin = self.varscale
        
        # Greedily select points ensuring spatial diversity
        for candidate in x[1:]:
            # Check if candidate is far enough from previously selected points
            if np.all(cdist([candidate], xout) > dmin):
                xout.append(candidate)
                # Stop once we have enough points
                if len(xout) == n:
                    break  
        
        # Return as array
        return np.array(xout)
    
    def get_lml(self):
        ''' Return log marginal likelihood. '''
        raise NotImplementedError
    
    def get_lml_change(self):
        ''' Compute change in log marginal likelihood. '''
        lml = self.get_lml()
        dlml = lml - self.prev_lml
        self.prev_lml = lml
        return dlml
    
    def get_kernel_params_relative_change(self):
        ''' Compute absolute value of relative change in kernel parameters. '''
        params = self.get_kernel_params()
        dparams = {k: abs((v - self.prev_kernel_params[k]) / self.prev_kernel_params[k]) for k, v in params.items()}
        self.prev_kernel_params = params
        return dparams
    
    def evaluate(self, Xgrid, nnext=1, Ygrid=None):
        '''
        Predict values over test grid and compute accuracy metrics
        
        :param Xgrid: n-dimensional meshgrid of cartesian XYZ coordinates where 
            to evaluate the function
        :param nnext: number of next points to sample (default: 1)  
        :param Ygrid: n-dimensional matrix of true values (optional)
        :return: 2-tuple with:
            - location(s) of next point(s) to sample
            - series of convergence metrics
        '''
        # Initialize metrics dictionary
        metrics = {}
        
        # Extract changes in kernel parameters (if available)
        if self.prev_kernel_params is not None:
            for k, v in self.get_kernel_params_relative_change().items():
                metrics[f'{k} rel. change'] = v

        # Compute change in log marginal likelihood from previous iteration (if available)
        if self.prev_lml is not None:
            metrics['LML change'] = self.get_lml_change()

        # Predict over the entire domain, along with uncertainty
        Ypred, Ypredstd = self.predict(Xgrid, return_std=True)

        # Compute relative max uncertainty w.r.t. max predicted value
        metrics['rel. max uncertainty'] = abs(Ypredstd.max() / Ypred.max())

        # If ground truth is provided, compute R^2 and max relative error
        if Ygrid is not None:
            metrics['r2'] = r2_score(Ygrid.ravel(), Ypred.ravel())
            rel_yerr = (Ygrid - Ypred) / Ygrid
            metrics['max. rel. error'] = np.abs(rel_yerr).max()
        # Otherwise, if naive prediction at new points is stored from previous run, 
        # re-predict values function at same points after fit and evaluate
        # accuracy of naive prediction
        elif self.ynext_prior is not None:
            ynext_post = self.predict(self.xnext)
            metrics['r2'] = r2_score(ynext_post, self.ynext_prior)
            rel_yerr = (ynext_post - self.ynext_prior) / ynext_post
            metrics['max. rel. error'] = np.abs(rel_yerr).max()
        # Otherwise, set metrics to NaN
        else:
            metrics['r2'] = np.nan
            metrics['max. rel. error'] = np.nan
        
        # Determine next point(s) to sample
        self.xnext = self.select_sampling_points(Xgrid, Ypredstd, nnext)

        # Perform naive prediction at next point(s)
        self.ynext_prior = self.predict(self.xnext)

        # Convert metrics to pandas Series
        metrics = pd.Series(metrics)
        metrics.index.name = 'metric'
        
        # Return outputs
        return self.xnext, metrics
    
    def extract_threshold_params(self, k):
        ''' Extract threshold value and type for a given metric. '''
        if 'rel. change' in k:
            return self.CONV_THRS['rel. param change']
        else:
            return self.CONV_THRS[k]

    def has_converged(self, metrics, k=None):
        ''' Check metric(s) has(ve) passed convergence threshold(s). '''
        # If no metric is specified, return dictionary of convergence flags for all metrics
        if k is None:
            convs = {k: self.has_converged(metrics, k) for k in metrics.keys()}
            s = pd.Series(convs, name='converged')
            s.index.name = 'metric'
            return s
        
        # Extract threshold value and type
        thr, ttype = self.extract_threshold_params(k)
        if thr is None:
            return True

        # Check if metric has converged
        if ttype == 'ub' and metrics[k] <= thr:
            return True
        if ttype == 'lb' and metrics[k] >= thr:
            return True
        return False
    
    def convergence_score(self, metrics):
        '''
        Compute convergence score from metrics.
        
        :param metrics: dictionary of accuracy metrics
        :return: convergence score
        '''
        scores = []
        for k, v in metrics.items():
            thr, ttype = self.extract_threshold_params(k)
            if thr is not None:
                x = 1 - v / thr
                if ttype == 'lb':
                    x = -x
                scores.append(min(1, np.exp(x)))
        return np.mean(scores)
    
    @property
    def conv_score(self):
        ''' Return convergence score from last metrics '''
        return self.convergence_score(self.metrics.iloc[-1])
    
    def update(self):
        ''' 
        Update GPR model.
        
        :return: boolean indicating whether convergence has been reached
        '''
        # Fit model
        self.fit()
        
        # Evaluate model, and extract next sampling location and accuracy metrics
        xnext, current_metrics = self.evaluate(self.Xgrid, nnext=self.nperiter)

        # Append metrics to global dataframe
        if self.metrics is None:
            self.metrics = current_metrics.to_frame().T
        else:
            self.metrics = pd.concat([
                self.metrics, current_metrics.to_frame().T], ignore_index=True)
            self.metrics.index.name = 'iteration'
        
        # If possible, compute average of accuracy metrics over last NAVG_METRICS iterations
        # and compute convergence flags on them
        if self.iiter >= self.NAVG_METRICS:
            avg_metrics = self.metrics.iloc[-self.NAVG_METRICS:].mean()
            conv = self.has_converged(avg_metrics)
            if self.has_conv_once is None:
                self.has_conv_once = pd.Series(index=conv.index, data=False)
            for k, v in conv.items():
                if v and not self.has_conv_once[k]:
                    logger.info(f'{self}: {k} has converged after {self.iiter} iterations')
            self.has_conv_once = self.has_conv_once | conv
            
            # Return True if all convergence criteria are met
            # (only after minimum number of iterations has been reached)
            if all(conv) and self.iiter >= self.min_iter:
                logger.info(f'{self}: overall convergence reached after {self.iiter} iterations')
                return True
        
        # Log warning and return True if maximum number of iterations is reached
        if self.iiter == self.max_iter:
            logger.warning(f'{self}: maximum number of iterations ({self.max_iter}) reached')
            return True
        
        # Evaluate function at next sample(s) and add to training data
        self.feval(xnext)

        # Return False if convergence not reached
        return False
    
    def optimize_init(self, Xgrid, min_pts=500, max_pts=2000, nperiter=50):
        '''
        Optimize field predictor by iteratevely sampling inside defined volume.
        
        :param Xgrid: n-dimensional meshgrid of cartesian XYZ coordinates where to evaluate the function        
        :param min_pts: minimal overall number of training points, including initialization (default: 500)
        :param max_pts: maximal overall number of training points, including initialization (default: 2000)
        :param nperiter: number of samples to add per iteration (default: 50)
        '''
        # Assign parameters to object
        self.nperiter = nperiter
        self.Xgrid = Xgrid

        # Compute bounds for number of optimization points and number of optimization iterations
        nptstot_bounds = np.array([min_pts, max_pts])
        nptsopt_bounds = nptstot_bounds - self.ninit
        niter_bounds = np.round(nptsopt_bounds / nperiter).astype(int)
        self.min_iter = max(niter_bounds[0], 0)
        self.max_iter = max(niter_bounds[1], 0)
        
        # Initialize algorithm variables
        self.prev_kernel_params = self.get_kernel_params()
        self.prev_lml = self.get_lml()
        self.metrics = None
        self.has_conv_once = None

    def optimize(self, *args, **kwargs):
        '''
        Optimize field predictor by iteratevely sampling inside defined volume.
        
        :return: dictionary of convergence metrics over iterations
        '''
        self.optimize_init(*args, **kwargs)

        if self.max_iter == 0:
            logger.warning('no optimization iterations to perform')
            return None

        # Refine GPR model iteratively
        logger.info(f'{self}: optimizing field predictor iteratively...')
        conv = False
        with tqdm(total=self.max_iter) as pbar:
            while not conv:
                conv = self.update()
                pbar.update(1)

        # Log final kernel parameters
        logger.info(f'{self}: fitted GPR model: kernel = {self.kernel}')
        
        # Return metrics over iterations
        return self.metrics
    
    def xtrain_path_length(self):
        ''' Compute path length of scan. '''
        if self.xtrain is None:
            return 0
        return np.sum(np.linalg.norm(np.diff(self.xtrain, axis=0), axis=1))
    
    def plot_metrics_evolution(self, metrics, height=2):
        ''' Plot metrics evolution over iterations. '''
        # Project all metrics into single column
        df = metrics.melt(
            var_name='metric',
            value_name='value',
            ignore_index=False
        ).reset_index()

        # Create facet grid with 1 axis per metric
        g = sns.FacetGrid(
            data=df, 
            col='metric',
            sharey=False,
            col_wrap=5,
            height=height,
        )
        g.set_titles('{col_name}')

        # Plot each metric over iterations
        g.map(sns.lineplot, 'iteration', 'value', label='raw')
        legend_data = g._legend_data.copy()

        # Add smoothed metrics to plot
        smoothed_metrics = apply_rolling_window(metrics, self.NAVG_METRICS)
        for k, ax in g.axes_dict.items():
            lh = ax.plot(smoothed_metrics[k], color='C1', label='smoothed')
        legend_data['smoothed'] = lh[0]
        
        # Add convergence thresholds and adjust axis limits if needed
        for k, ax in g.axes_dict.items():
            thr, ttype = self.extract_threshold_params(k)
            if thr is not None:
                lh = ax.axhline(thr, color='r', linestyle='--', label=f'threshold')
            if ttype == 'ub':
                ax.set_ylim(0, ax.get_ylim()[1])
            ax.set_ylabel(k)
            sns.despine(ax=ax)
        legend_data['threshold'] = lh
        
        # Add legend
        g.add_legend(
            legend_data=legend_data,
            bbox_to_anchor=(1.05, .5),
            loc='center left'
        )
        
        # Extract figure object
        fig = g.figure

        # Add subtitle and adjust layout
        fig.suptitle(self.wrappped_str('metrics convergence'))
        fig.tight_layout()

        # Return figure
        return fig
    
    def scan(self, *args, **kwargs):
        ''' Initialize and optimize GPR model '''
        init_kwargs = {}
        if 'ninit' in kwargs:
            init_kwargs['n'] = kwargs.pop('ninit')
        if 'initmode' in kwargs:
            init_kwargs['mode'] = kwargs.pop('initmode')
        self.initialize(**init_kwargs)
        return self.optimize(*args, **kwargs)
    
    def get_iter_cmap(self, n, cmap='viridis'):
        ''' Get custom colormap for iterations '''
        cmap = plt.get_cmap(cmap)
        irange = np.linspace(0, 1, n)
        colors = cmap(irange)  # Generate n colors from default cmap
        colors = np.vstack(([1, 0, 0, 1], colors))  # Prepend red (RGBA)
        return mcolors.ListedColormap(colors)  # return as ListedColormap
    
    def scan_plot(self, ax=None, cmode='val', cmap='viridis', marker='.', s=30, add_cbar=True,
                  title='sampled locations', transform=None):
        '''
        Plot sampled locations
        
        :param ax: axis object (optional)
        :param cmode: color mode, one of:
            - 'val' (default, to color by value)
            - 'iter' (to color by iteration)
        :param cmap: colormap (default: 'viridis')
        :param marker: marker style (default: '.')
        :param s: marker size (default: 50)
        :param add_cbar: whether to add colorbar (default: True)
        :param transform (optional): transformation function to apply to to x coordinates before plotting 
        :return: figure object and scatter object
        '''
        # Get data dimensionality
        if self.xtrain is None:
            logger.warning('no sampling grid -> assuming 3D')
            ndims = 3
        else:
            ndims = self.xtrain.shape[1]
        
        # Create/retrieve figure and axis
        if ax is None:
            if ndims == 3:
                fig = plt.figure(figsize=(8, 4))
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig, ax = plt.subplots(figsize=(8, 4))
        else:
            if ndims == 3:
                if not isinstance(ax, Axes3D):
                    raise ValueError('3D plot requires Axes3D object')
            fig = ax.get_figure()
        
        # Format axis labels
        axsuffix = '' if self.unit is None else f' ({self.unit})'
        for k, axk in zip(self.dimkeys, 'xyz'):
            getattr(ax, f'set_{axk}label')(f'{k}{axsuffix}')

        # Add axis title if provided
        if title is not None and len(title) > 0:
            ax.set_title(self.wrappped_str(title))
        
        # Check if data has been sampled
        if self.ytrain is None or len(self.ytrain) == 0: 
            logger.warning('no sampled data to plot')
            return fig, None
    
        # Plot sampled locations
        xplt = self.xtrain if transform is None else transform(self.xtrain)
        if cmode == 'val':
            cmap = plt.get_cmap(cmap)
            scobj = ax.scatter(
                *xplt.T, c=self.ytrain, cmap=cmap,
                marker=marker, s=s, label='sampled points')
            norm = plt.Normalize(vmin=self.ytrain.min(), vmax=self.ytrain.max())
        elif cmode == 'iter':
            cmap = self.get_iter_cmap(self.iiter)
            scobj = ax.scatter(
                *xplt.T, c=self.itrain, cmap=cmap,
                marker=marker, s=s, label='sampled points')
            norm = plt.Normalize(vmin=self.itrain.min(), vmax=self.itrain.max())
        else:
            raise ValueError(f'invalid color mode: {cmode}')

        # Add colorbar if requested
        if add_cbar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            if len(fig.axes) == 1:
                fig.subplots_adjust(bottom=.1, right=.8, top=.9)
                cax = fig.add_axes([.85, .1, 0.075, .8])
                cbar = fig.colorbar(sm, cax=cax, ticks=sm.get_clim())
                cbar.set_label(cmode, labelpad=-20)
            else:
                cax = fig.axes[1]

        # Return figure and scatter object 
        return fig, scobj
    
    def to_pickle(self, path):
        ''' Save to pickle file. '''
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def from_pickle(path):
        ''' Load from pickle file. '''
        with open(path, 'rb') as f:
            return pickle.load(f)


class GPRFieldMapper(AdaptiveFieldMapper):
    ''' Adaptive field mapping using Gaussian Process Regression. '''

    def init_model(self):
        ''' Initialize Gaussian Process Regressor object. '''
        # Assemble underlying kernels
        R = RBF(
            length_scale=[self.varscale] * len(self.projkey), 
            length_scale_bounds=(.1, 6))
        C0 = ConstantKernel(
            constant_value=.2,
            constant_value_bounds=(1e-6, 1e0))
        C1 = ConstantKernel(
            constant_value=.2,
            constant_value_bounds=(1e-6, 1e0))
        W = WhiteKernel(
            noise_level=.05,
            noise_level_bounds=(1e-10, 1e-1))
        K = C1 * R + W + C0
        logger.info(f'{self}: naive kernel = {K}')

        # Construct mapper to facilitate extraction of kernel parameters 
        self.kmapper = {
            'k1__k1__k1': 'C1',
            'k1__k1__k2': 'RBF',
            'k1__k2': 'W',
            'k2': 'C0',
        }

        # Instantiate GPR model
        logger.info(f'{self}: instantiating GPR model')
        self.gpr = GaussianProcessRegressor(K)
    
    @property
    def kernel(self):
        ''' Underlying GPR kernel. '''
        return self.gpr.kernel_
    
    def is_composite_kernel(self, k):
        ''' Check if kernel is composite. '''
        if not isinstance(k, Kernel):
            raise ValueError(f'input {k} is not a kernel')
        return hasattr(k, 'k1') and hasattr(k, 'k2')
    
    def get_kernel_params(self, add_hyperparams=False):
        '''
        Get kernel parameters (and optionally hyperparameters).
        
        :param add_hyperparams: whether to include hyperparameters in output
        :return: dictionary of kernel parameters, or 2-tuple with parameters
            and hyperparameters dictionaries
        '''
        # Extract raw kernel parameters dictionary
        raw_params = self.kernel.get_params(deep=True)

        # Separate kernel objects from non-kernel objects
        kernels = {k: v for k, v in raw_params.items() if isinstance(v, Kernel)}
        nonkernels = {k: v for k, v in raw_params.items() if not isinstance(v, Kernel)}
        
        # Extract subkernels and re-order them by name
        subkernels = {k: v for k, v in kernels.items() if not self.is_composite_kernel(v)}
        subkernels = {k: subkernels[k] for k in sorted(subkernels.keys())}

        # Initialize parameters dictionary
        params = {}
        
        # For each subkernel
        for k in subkernels.keys():
            # Extract associated keys from nonkernels dictionary
            associated_keys = [kk for kk in nonkernels.keys() if kk.startswith(f'{k}_')]
            # Restrict to non-bounds keys (should be only 1)
            pkeys = [kk for kk in associated_keys if 'bounds' not in kk]
            if len(pkeys) != 1:
                raise ValueError(f'more than one associated parameter for kernel {k}: {pkeys}')
            # Extract associated parameter(s)
            params[k] = nonkernels[pkeys[0]]

        # Map to more readable names, and parse array parameters
        params = self.parse_parameter({self.kmapper[k]: v for k, v in params.items()})

        # If requested, extract hyperparameters dictionary
        if add_hyperparams:
            hyperparams = {k: v.theta for k, v in subkernels.items()}
            hyperparams = self.parse_parameter({self.kmapper[k]: v for k, v in hyperparams.items()})

        # Return outputs
        if add_hyperparams:
            return params, hyperparams
        return params
    
    def get_lml(self):
        ''' Return log marginal likelihood. '''
        return self.gpr.log_marginal_likelihood_value_

    def fit(self, ignore_warnings=True):
        ''' Fit GPR model to training data. '''
        # If no training data is available, raise error
        if self.ytrain is None:
            raise ValueError('training data not initialized')
        
        # Fit GPR model
        with warnings.catch_warnings():
            if ignore_warnings:
                warnings.simplefilter('ignore')
            self.gpr.fit(self.project(self.xtrain), self.ytrain)
    
    def _predict(self, x, **kwargs):
        '''
        Predict values over coordinates vector
        
        :param x: n-by-m matrix of cartesian coordinates
        :return: n-dimensional vector of predicted values (and optionally uncertainties)
        '''
        return self.gpr.predict(x, **kwargs)