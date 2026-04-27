# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-09-02 13:06:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-06-13 13:25:25

import re
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from cycler import cycler
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from instrulink.logger import logger
from tqdm import tqdm
from datetime import datetime
from scipy.ndimage import gaussian_filter

from .constants import *
from .calib_utils import parse_outputs_by_date, compute_amplifier_gain_over_time, vratio_to_gain
from .utils import rescale, is_within, pressure_to_intensity, intensity_to_pressure, idxmax, get_mux_slice
from .scanners import GridScanner
from .calibrators import Calibrator

# Expanded library of distinct, high-contrast, accessible colors
distinct_colors = [
    # Primary vibrant colors
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', 
    '#bcbd22', '#17becf', '#7f7f7f',
    # Additional high-contrast colors
    '#FF1744', '#00E676', '#2979FF', '#FF6D00', '#9C27B0', '#00BCD4', '#4CAF50',
    '#FF5722', '#3F51B5', '#009688', '#795548', '#607D8B', '#E91E63', '#CDDC39',
    # Extended palette for many traces
    '#B71C1C', '#1B5E20', '#0D47A1', '#E65100', '#4A148C', '#006064', '#33691E',
    '#BF360C', '#1A237E', '#004D40', '#3E2723', '#263238', '#880E4F', '#827717',
    '#FF8A65', '#81C784', '#64B5F6', '#FFB74D', '#BA68C8', '#4DB6AC', '#AED581',
    '#FFAB91', '#90A4AE', '#F48FB1', '#DCE775', '#A1887F', '#B0BEC5', '#CE93D8'
]

# List of sequential cmaps and markers (backup)
cmaps = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
cmap_offset = 0.3  # offset on colormaps 0-1 range
markers = ['o', 'v', '^', 's', '*', 'D', 'p', 'P', 'd', 'H', '+', 'x']

# Line styles for additional differentiation
line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (3, 5, 1, 5)), 
               (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (5, 5)), (0, (3, 3)), (0, (1, 5))]

VALIDATION_PATTERNS = {
    P_KEY: CALIBRATION_POUT_PATTERN,
    CPLRATIODB_KEY: CALIBRATION_CPL_PATTERN,
    VOUT_KEY: CALIBRATION_VOUT_PATTERN,
    GAIN_KEY: CALIBRATION_VOUT_PATTERN,
}


def identity(x, y):
    return y


def organize_traces_for_colors(pouts_by_date):
    '''
    Organize traces to maximize color distinction between similar measurement runs.
    
    :param pouts_by_date: dictionary of date keys to column lists
    :return: flattened list of (date_key, col) tuples optimized for color assignment
    '''
    organized_traces = []
    
    # First, add one trace from each date to ensure date separation
    max_cols_per_date = max(len(cols) for cols in pouts_by_date.values())
    
    for col_index in range(max_cols_per_date):
        for date_key, cols in pouts_by_date.items():
            if col_index < len(cols):
                organized_traces.append((date_key, cols[col_index]))
    
    return organized_traces


def select_representative_dates(pouts_by_date, data, convfunc, x, dissimilarity_threshold=0.05):
    '''
    Select a representative subset of calibration dates to display, based on
    how different each date's mean curve is from its neighbors.
    
    Always keeps: first date, last date, and any date whose mean curve 
    deviates significantly from its predecessor (sequential jump detection).
    
    :param pouts_by_date: dict mapping date keys to lists of column names
    :param data: DataFrame of calibration output values (columns = traces)
    :param convfunc: output conversion function
    :param x: input vector (for convfunc)
    :param dissimilarity_threshold: normalized RMSE threshold (fraction of data range)
        above which a date is considered "different enough" to display
    :return: list of date keys to display, list of date keys to hide
    '''
    date_keys = list(pouts_by_date.keys())
    if len(date_keys) <= 5:
        # Few enough dates - show all
        return date_keys, []
    
    # Compute mean curve per date
    date_means = {}
    for dk, cols in pouts_by_date.items():
        valid_cols = [c for c in cols if c in data.columns]
        if valid_cols:
            date_means[dk] = convfunc(x, data[valid_cols].mean(axis=1)).values
    
    if not date_means:
        return date_keys, []
    
    # Compute the overall data range for normalization
    all_values = np.concatenate(list(date_means.values()))
    data_range = np.nanmax(all_values) - np.nanmin(all_values)
    if data_range == 0 or np.isnan(data_range):
        return date_keys, []
    
    # Always include first and last dates
    selected = {date_keys[0], date_keys[-1]}
    
    # Sequential jump detection: compare each date to its predecessor
    # and keep dates that represent a meaningful change
    for i in range(1, len(date_keys)):
        dk = date_keys[i]
        dk_prev = date_keys[i - 1]
        if dk in date_means and dk_prev in date_means:
            rmse = np.sqrt(np.nanmean((date_means[dk] - date_means[dk_prev]) ** 2))
            nrmse = rmse / data_range
            if nrmse > dissimilarity_threshold:
                selected.add(dk)
                selected.add(dk_prev)  # keep both sides of a jump
    
    # Also keep the overall median date as a reference midpoint
    mid_idx = len(date_keys) // 2
    selected.add(date_keys[mid_idx])
    
    # Ensure we have at least ~5 dates by filling in evenly spaced ones
    if len(selected) < 5 and len(date_keys) > 5:
        step = max(1, len(date_keys) // 5)
        for i in range(0, len(date_keys), step):
            selected.add(date_keys[i])
    
    # Preserve original ordering
    shown = [dk for dk in date_keys if dk in selected]
    hidden = [dk for dk in date_keys if dk not in selected]
    
    logger.info(f'selected {len(shown)}/{len(date_keys)} representative dates '
                f'(hiding {len(hidden)} similar dates)')
    
    return shown, hidden


def plot_calibration_data(data, title=None, prefix=None, details=False, mode=None, logx=False, logy=False, ylabel=P_KEY,
                          convfunc=None, groundy='auto', yref=None, ax=None, cmap_iterator=None, marker='o',
                          yerr='sd', color=None):
    '''
    Plot calibration output curves
    
    :param data: dataframe of calibration curves
    :param title (optional): plot title
    :param prefix (optional): prefix for calibration curves labels
    :param details: whether to plot individual traces (legacy, use mode instead)
    :param mode: display mode - 'summary' (mean per date), 'details' (all traces), 'mean' (single mean)
                 If not specified, inferred from details flag for backward compatibility
    :param ylabel: y-axis label
    :param convfunc (optional): output conversion function
    :param groundy (optional): whether to plot set y-axis lower bound to 0
    :param yref (optional): reference value(s) (either constant or input dependent) for y-axis
    :param ax (optional): axis object
    :param cmap_iterator (optional): iterator over colormap names
    :param marker: marker style for data points (default: 'o')
    :param yerr: error shading type ('sd' or 'se')
    :param color: color for mean trace (if mode='mean')
    :return: figure handle
    '''
    # Resolve mode from legacy details flag if not explicitly set
    if mode is None:
        mode = 'details' if details else 'mean'
    # If ylabel is "all", plot all available output columns types 
    if ylabel == 'all':
        ylabel = [k for k, p in VALIDATION_PATTERNS.items() if any([col for col in data.columns if re.match(p, col)])]
    
    if isinstance(ylabel, (tuple, list)) and len(ylabel) == 1:
        ylabel = ylabel[0]

    # If no axis provided
    if ax is None:
        if isinstance(ylabel, (tuple, list)):
            naxes = len(ylabel)
            fig, axes = plt.subplots(naxes, 1, figsize=(12, 6 * naxes), dpi=100)
            fig.subplots_adjust(hspace=0.4)
        else:
            fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
            naxes = 1
    
    # If axis(es) provided
    else:
        if isinstance(ylabel, (tuple, list)):
            if not isinstance(ax, (tuple, list, np.ndarray)):
                raise ValueError('multiple ylabels require multiple axes')
            if len(ax) != len(ylabel):
                raise ValueError(f'number of ylabels {len(ylabel)} must match number of axes ({len(ax)})')
            axes = ax
            fig = axes[0].get_figure()
        else:
            fig = ax.get_figure()
            naxes = 1
    
    # If multiple ylabels are provided, plot each one in a separate axis
    if naxes > 1:
        for ax, yl in zip(axes, ylabel):
            plot_calibration_data(
                data.copy(), 
                title=title, 
                prefix=prefix, 
                mode=mode,
                logx=logx, 
                logy=logy, 
                ylabel=yl, 
                convfunc=convfunc, 
                groundy=groundy, 
                yref=yref, 
                ax=ax, 
                cmap_iterator=cmap_iterator, 
                marker=marker, 
                yerr=yerr, 
                color=color)
        return fig

    if convfunc is None:
        convfunc = identity

    # Extract input vector and format axes
    xkey = data.columns[0]  # xkey: name of first column
    x = data.pop(xkey)
    if xkey == VIN_KEY:
        ax.set_xlabel('Vin (Vpp)')
        x = x * MV_TO_V
    else:
        ax.set_xlabel(xkey)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    
    # Improve plot aesthetics for crisp, accessible appearance
    for sk in ['top', 'right']:
        ax.spines[sk].set_visible(False)
    for sk in ['bottom', 'left']:
        ax.spines[sk].set_linewidth(1.2)  # Slightly thicker spines
        ax.spines[sk].set_color('#2E2E2E')  # Darker, more readable color
    
    # Enhance grid and ticks for maximum readability
    ax.grid(True, alpha=0.4, linewidth=0.8, linestyle='-', color='#CCCCCC')
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.0, length=5, 
                   colors='#333333')
    ax.tick_params(axis='both', which='minor', width=0.8, length=3, colors='#666666')
    
    # Improve axis labels for better readability
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    
    if title is not None:
        logger.info(f'plotting {title} calibration data')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=15)

    # Get iterator over colormaps, if not provided
    if cmap_iterator is None:
        cmap_iterator = iter(cmaps)

    # Remove columns with all NaNs
    data = data.dropna(axis=1, how='all')

    # Select only output columns match the expected pattern
    validation_pattern = VALIDATION_PATTERNS[ylabel]
    output_cols = [k for k in data.columns if re.match(validation_pattern, k)]
    if not output_cols:
        logger.error('no valid output columns found')
        return
    data = data[output_cols]

    # Coupling ratio case: compute ratio from CPLFWD and CPLREV columns, and convert to dB
    if ylabel == CPLRATIODB_KEY:
        fwdkeys = [k for k in data.columns if 'CPLFWD' in k]
        revkeys = [k for k in data.columns if 'CPLREV' in k]
        assert len(fwdkeys) == len(revkeys), 'mismatched number of forward and reverse coupling columns'
        with np.errstate(invalid='ignore', divide='ignore'):
            ratios = data[revkeys].values / data[fwdkeys].values
        data = pd.DataFrame(ratios, index=data.index, columns=revkeys)
        data = vratio_to_gain(data)  # dB
    
    # Initialize trace counter
    ntraces = 0

    # Summary mode: one mean line per date with error band
    if mode == 'summary':
        pouts_by_date = parse_outputs_by_date(data.columns, rgxp=validation_pattern)
        
        # Select representative dates using dissimilarity metric
        shown_dates, hidden_dates = select_representative_dates(
            pouts_by_date, data, convfunc, x)
        
        # Use a clean, limited color palette (tab10) for maximum distinction
        summary_colors = list(plt.get_cmap('tab10').colors)
        summary_markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'p', 'h']
        
        # Plot shown dates as solid lines with error bands
        for i, date_key in enumerate(shown_dates):
            cols = pouts_by_date[date_key]
            valid_cols = [c for c in cols if c in data.columns]
            if not valid_cols:
                continue
            c = summary_colors[i % len(summary_colors)]
            m = summary_markers[i % len(summary_markers)]
            date_data = data[valid_cols]
            ymean = convfunc(x, date_data.mean(axis=1))
            
            lbl = date_key
            if prefix is not None:
                lbl = f'{prefix} {lbl}'
            
            ax.plot(x, ymean, marker=m, markersize=4, color=c,
                    linewidth=2.0, alpha=0.95, label=lbl,
                    markeredgewidth=0.5, markeredgecolor='white')
            
            # Add error band if multiple traces exist for this date
            if len(valid_cols) > 1:
                if yerr == 'sd':
                    yerr_vals = convfunc(x, date_data.std(axis=1))
                elif yerr == 'se':
                    yerr_vals = convfunc(x, date_data.sem(axis=1))
                else:
                    yerr_vals = convfunc(x, date_data.std(axis=1))
                ax.fill_between(x, ymean - yerr_vals, ymean + yerr_vals,
                               alpha=0.15, color=c, linewidth=0)
            ntraces += 1
        
        # Plot hidden (similar) dates as thin gray lines in background
        for date_key in hidden_dates:
            cols = pouts_by_date[date_key]
            valid_cols = [c for c in cols if c in data.columns]
            if not valid_cols:
                continue
            date_data = data[valid_cols]
            ymean = convfunc(x, date_data.mean(axis=1))
            # First hidden line gets a legend label, rest are unlabeled
            lbl = f'other dates ({len(hidden_dates)})' if date_key == hidden_dates[0] else None
            ax.plot(x, ymean, color='#CCCCCC', linewidth=0.8, alpha=0.5,
                    label=lbl, zorder=1)
            ntraces += 1

    # Detail mode: plot each trace with distinct colors, line styles, and enhanced readability
    elif mode == 'details':
        # Plot traces with distinct colors by date
        pouts_by_date = parse_outputs_by_date(data.columns, rgxp=validation_pattern)
        
        # Identify the most recent date to highlight it
        date_keys = list(pouts_by_date.keys())
        latest_date = date_keys[-1] if date_keys else None
        
        # Bold colors reserved for the latest date's traces
        latest_colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#e377c2']
        
        # Use our expanded color library for older dates
        all_colors = distinct_colors
        
        # Store lines and labels for hover functionality and legend management
        lines = []
        labels = []
        date_groups = {}  # Group by date for organized legend
        
        # Use distinct colors and line styles for maximum differentiation
        color_index = 0
        style_index = 0
        
        for date_key, cols in pouts_by_date.items():
            date_groups[date_key] = []
            is_latest = (date_key == latest_date)
            
            # Assign distinct colors and line styles to each measurement within a date
            for i, k in enumerate(cols):
                if is_latest:
                    # Latest date: bold colors, thick lines, high zorder
                    color = latest_colors[i % len(latest_colors)]
                    line_style = '-'  # Always solid for latest
                    lw = 3.5
                    alpha = 1.0
                    ms = 6
                    zorder = 50
                    mew = 1.0
                    mec = 'black'
                else:
                    # Older dates: use regular colors but slightly dimmed
                    color = all_colors[color_index % len(all_colors)]
                    line_style = line_styles[style_index % len(line_styles)]
                    lw = 1.8
                    alpha = 0.6
                    ms = 3
                    zorder = 5
                    mew = 0.5
                    mec = 'white'
                    color_index += 1
                    style_index += 1
                
                # Create more readable label with date grouping
                if ylabel == P_KEY:
                    base_lbl = k.replace('Pout', '').replace('(MPa)', '').strip()
                elif ylabel == CPLRATIODB_KEY:
                    base_lbl = k.replace('CPLREV', '').replace('(Vpp)', '').strip()
                elif ylabel in (VOUT_KEY, GAIN_KEY):
                    base_lbl = k.replace('Vout', '').replace('(Vpp)', '').strip()
                else:
                    base_lbl = k
                
                # Create clear, hierarchical label
                lbl = f"{date_key}: {base_lbl}" if len(pouts_by_date) > 1 else base_lbl
                if prefix is not None:
                    lbl = f'{prefix} {lbl}'
                
                # Plot with visual properties based on recency
                line, = ax.plot(x, convfunc(x, data[k]), 
                               marker=marker, markersize=ms, 
                               label=lbl, color=color, 
                               linewidth=lw,
                               alpha=alpha,
                               linestyle=line_style,
                               zorder=zorder,
                               markeredgewidth=mew, 
                               markerfacecolor=color, 
                               markeredgecolor=mec)
                lines.append(line)
                labels.append(lbl)
                date_groups[date_key].append((line, lbl))
                ntraces += 1
        
        # Store default visual properties for hover reset
        default_props = []
        for line in lines:
            default_props.append({
                'alpha': line.get_alpha(),
                'linewidth': line.get_linewidth(),
                'markersize': line.get_markersize(),
                'zorder': line.get_zorder(),
            })

        # Enhanced hover functionality with better visual feedback
        def on_hover(event):
            if event.inaxes == ax:
                found_line = False
                for line, label in zip(lines, labels):
                    if line.contains(event)[0]:
                        # Highlight this line more prominently
                        for j, other_line in enumerate(lines):
                            if other_line == line:
                                other_line.set_alpha(1.0)
                                other_line.set_linewidth(4.5)
                                other_line.set_markersize(8)
                                other_line.set_zorder(100)
                            else:
                                other_line.set_alpha(0.15)
                                other_line.set_zorder(1)
                        
                        # Update the plot title to show current line info
                        title_text = f'{title} - Highlighted: {label}' if title else f'Highlighted: {label}'
                        ax.set_title(title_text, fontsize=12, fontweight='bold', color='darkblue')
                        fig.canvas.draw_idle()
                        found_line = True
                        return
                
                # Reset to default properties if not hovering over any line  
                if not found_line:
                    for j, line in enumerate(lines):
                        line.set_alpha(default_props[j]['alpha'])
                        line.set_linewidth(default_props[j]['linewidth'])
                        line.set_markersize(default_props[j]['markersize'])
                        line.set_zorder(default_props[j]['zorder'])
                    ax.set_title(title if title else '', fontsize=11, fontweight='bold', color='black')
                    fig.canvas.draw_idle()
            ax.set_title(title if title else '', fontsize=11, fontweight='bold', color='black')
            fig.canvas.draw_idle()
        
        # Connect enhanced hover event
        fig.canvas.mpl_connect('motion_notify_event', on_hover)

    # Aggregate (mean) mode: plot single mean trace with std shading
    else:
        ymean = convfunc(x, data.mean(axis=1))
        ax.plot(x, ymean, marker='o', markersize=5, label=prefix, color=color, 
                linewidth=3.0,  # Thicker lines for better visibility
                markeredgewidth=0.8, 
                markerfacecolor=color, 
                markeredgecolor='white',
                alpha=0.95)
        if data.shape[1] > 1:
            if yerr == 'sd':
                yerr = convfunc(x, data.std(axis=1))
            elif yerr == 'se':
                yerr = convfunc(x, data.sem(axis=1))
            ax.fill_between(x, ymean - yerr, ymean + yerr, alpha=0.25, color=color, 
                           edgecolor=color, linewidth=0.5)
        ntraces += 1
    
    # Add reference y-values, if specified
    if yref is not None:
        yref = yref(x) if callable(yref) else np.full(x.size, yref)
        ax.plot(x, yref, 'k--', label='reference')
        ntraces += 1

    # Add legend with adaptive formatting based on number of entries
    if ntraces > 1 or mode in ('details', 'summary'):
        # Determine legend columns: use 2 columns if many entries, 1 otherwise
        ncol = 2 if ntraces > 12 else 1
        fontsize = 7 if ntraces > 15 else (8 if ntraces > 8 else 9)
        
        # Choose legend title based on mode
        if mode == 'summary':
            legend_title = 'Calibration Dates'
        elif mode == 'details':
            legend_title = 'Measurement Runs'
        else:
            legend_title = 'Data Series'
        
        legend = ax.legend(
            bbox_to_anchor=(1.02, 1), 
            loc='upper left',
            frameon=True,
            fancybox=True,
            shadow=False,
            ncol=ncol,
            fontsize=fontsize,
            title=legend_title,
            title_fontsize=fontsize + 1,
            columnspacing=0.8,
            handlelength=2.0,
            handletextpad=0.5,
            labelspacing=0.4,
            borderpad=0.4,
        )
        # Style the legend
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_edgecolor('#CCCCCC')
        legend.get_frame().set_linewidth(0.5)
        
        # Make sure plot doesn't get cut off by legend
        plt.tight_layout()
        if hasattr(fig, 'subplots_adjust'):
            fig.subplots_adjust(right=0.78)
    
    # Ground y-axis lower bounds, if specified
    if groundy == 'auto':
        groundy = ylabel in (CPLRATIODB_KEY, GAIN_KEY)
    if groundy:
        if ylabel == CPLRATIODB_KEY:
            ax.set_ylim(1.1 * ax.get_ylim()[0], 0)
        else:
            ax.set_ylim(0, 1.1 * ax.get_ylim()[1])
    
    return fig


def plot_calibration_curves(fpaths, groundy='auto', yref=None, details=False, mode=None, ax=None, **kwargs):
    '''
    Plot calibration output curves
    
    :param fpaths: list of full paths to calibration files
    :param details: whether to plot individual traces (legacy, use mode instead)
    :param mode: display mode - 'summary' (default), 'details', or 'mean'
    :param ylabel: y-axis label
    :param convfunc (optional): output conversion function
    :param groundy (optional): whether to plot set y-axis lower bound to 0
    :param yref (optional): reference value(s) (either constant or input dependent) for y-axis
    :param ax (optional): axis object
    :return: figure handle
    '''
    # Resolve mode from legacy details flag if not explicitly set
    if mode is None:
        mode = 'details' if details else 'mean'

    # If multiple ylabels are provided, plot each one in a separate subplot
    if 'ylabel' in kwargs and isinstance(kwargs['ylabel'], (list, tuple)):
        ylabels = kwargs.pop('ylabel')
        naxes = len(ylabels)
        fig, axes = plt.subplots(naxes, 1, figsize=(12, 6 * naxes), dpi=100)
        if naxes == 1:
            axes = [axes]
        fig.subplots_adjust(hspace=0.4)
        for ax, ylabel in zip(axes, ylabels):
            plot_calibration_curves(
                fpaths, 
                groundy=groundy, 
                yref=yref, 
                mode=mode, 
                ax=ax, 
                ylabel=ylabel, 
                **kwargs)
        return fig
    
    # Create/retrieve figure and axis with enhanced size and layout
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7), dpi=100)  # Larger figure for better readability
    else:
        fig = ax.get_figure()

    # Get number of files and iterator over colormaps
    nfiles = len(fpaths)

    # Construct colormap/color iterator depending on mode
    if mode == 'details':
        cmap_iterator = iter(cmaps)
    elif mode == 'mean':
        color_iterator = iter(plt.get_cmap('tab10').colors + plt.get_cmap('Dark2').colors)

    # Loop through calibration files
    for ifile, fpath in enumerate(fpaths):
        # Load calibration data table from Excel file
        fname = os.path.basename(fpath)
        fcode = os.path.splitext(fname)[0]
        if nfiles == 1:
            ax.set_title(f'{os.path.splitext(fname)[0]} calibration data')
        logger.info(f'loading calibration data from {fname}...')
        data = pd.read_excel(fpath, engine='openpyxl')

        # Check if this is the last file iterated on
        islastfile = ifile == nfiles - 1
        
        # Plot calibration data from file
        fig = plot_calibration_data(
            data, 
            title=f'{fcode} calibration data' if nfiles == 1 else None,
            prefix=fcode if nfiles > 1 else None, 
            groundy=groundy if islastfile else False, 
            yref=yref if islastfile else None,
            ax=ax,
            cmap_iterator=cmap_iterator if mode == 'details' else None,
            marker=markers[ifile] if mode == 'details' else None,
            mode=mode,
            color=next(color_iterator) if mode == 'mean' else None,
            **kwargs
        )
    
    if nfiles > 1:
        ax.set_title('calibration data')

    # Add legend if required
    if ax.get_legend() is None and nfiles > 1:
        ax.legend()

    # Return figure handle
    return fig


def plot_amplifier_gain_over_time(fpaths, refgain=None):
    '''
    Plot amplifier(s) gain(s) over time, extracted from calibration curves
    
    :param fpaths: list of full paths to calibration files
    :param refgain: reference gain (optional)
    :return: figure handle
    '''
    # Create figure backbone
    fig, ax = plt.subplots()
    ax.set_xlabel('date')
    ax.set_ylabel('gain (dB)')
    sns.despine(ax=ax)

    # Get number of files
    nfiles = len(fpaths)

    # Loop through calibration files
    for ifile, fpath in enumerate(fpaths):
        # Load transducer calibration data table from Excel file
        fname = os.path.basename(fpath)
        fcode = os.path.splitext(fname)[0]
        if nfiles == 1:
            ax.set_title(f'{os.path.splitext(fname)[0]} calibration data')
        logger.info(f'loading calibration data from {fname}...')
        # Compute gain over time
        data = pd.read_excel(fpath, engine='openpyxl')
        gain_by_date = compute_amplifier_gain_over_time(data)
        # Plot gain over time
        logger.info('plotting data...')
        c = f'C{ifile}'
        ax.plot(
            gain_by_date.index, 
            gain_by_date['mean'], 
            marker='o', 
            label=fcode if nfiles > 1 else 'measured gain', 
            color=c)
        ax.fill_between(
            gain_by_date.index, 
            gain_by_date['mean'] - gain_by_date['sem'], 
            gain_by_date['mean'] + gain_by_date['sem'], 
            fc=c, ec=None, alpha=0.3)

    # Auto format dates on x-axis 
    fig.autofmt_xdate()
    
    # Add reference gain, if specified
    if refgain is not None:
        ax.axhline(refgain, color='k', linestyle='--', label='reference')
    
    # Add legend if appropriate
    if nfiles > 1 or refgain is not None:
        ax.legend()
    
    # Return figure handle
    return fig


def add_cbar_to_fig(fig, sm, label, fs=8, bottom=0.1, right=0.8, top=0.9):
    ''' Add colorbar to right hand side of figure '''
    fig.subplots_adjust(bottom=bottom, right=right, top=top)
    cax = fig.add_axes([right + 0.05, bottom, 0.075, top - bottom])
    cbar = fig.colorbar(sm, cax=cax, ticks=sm.get_clim())
    cbar.set_label(label, fontsize=fs, labelpad=-20)
    cax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
    for item in cax.get_yticklabels():
        item.set_fontsize(fs)


def plot_traces(xyz, x, y, z, traces_dir):
    '''
    Plot traces associated to a particular XYZ scanning location
    
    :param xyz: XYZ scanning location (um)
    :param x: vector of scanned x coordinates (um)
    :param y: vector of scanned y coordinates (um)
    :param z: vector of scanned z coordinates (um)
    :param traces_dir: path to directory containing traces files
    '''
    xyz_str = ', '.join([f'{k:.2f}' for k in xyz])

    # Derive scanning position index from XYZ coordinates
    ipos = GridScanner.get_position_index(
        xyz, x=x, y=y, z=z)

    # Derive traces file path
    npos = x.size * y.size * z.size
    npad = int(np.floor(np.log10(npos))) + 1
    traces_fpath = os.path.join(traces_dir, f'traces_pos{ipos:0{npad}}.csv')
    
    # Load traces data and compute envelope and its peak
    traces_data = pd.read_csv(traces_fpath, index_col=TIME_US)
    yenv = Calibrator.extract_envelope(
        traces_data['corr'], navg=NAVG_PENV, ntrim=NTRIM_PENV)
    ypeak = np.nanmax(yenv)
    yp2p = 2 * ypeak
    logger.info(f'extracting data for position [{xyz_str}]: index = {ipos}, amplitude = {yp2p:.2f} MPa')
    
    # Create figure
    fig, ax = plt.subplots()
    sns.despine(ax=ax)
    xyz_str = ', '.join([f'{k} = {v:.2f} um' for k, v in zip('XYZ', xyz)])
    ax.set_title(f'file {ipos} ({xyz_str})\nwaveform data')
    ax.set_xlabel(TIME_US)
    ax.set_ylabel(P_KEY)
    
    # Plot traces
    traces_data.plot(ax=ax)
    ax.plot(traces_data.index, -yenv, 'k--', label='envelope')
    ax.plot(traces_data.index, yenv, 'k--')
    ax.axhline(-ypeak, c='r', ls='--', label='peak')
    ax.axhline(ypeak, c='r', ls='--')
    ax.text(0.5, 0.9, f'A = {yp2p:.2f} MPa', transform=ax.transAxes)
    ax.legend()
    
    # Render figure
    plt.show()


def compute_mesh_edges(x, scale='lin'):
    ''' 
    Compute the appropriate edges of a mesh that quads a linear or logarihtmic distribution.

    :param x: the input vector
    :param scale: the type of distribution ('lin' for linear, 'log' for logarihtmic)
    :return: the edges vector
    '''
    if scale == 'log':
        x = np.log10(x)
        range_func = np.logspace
    else:
        range_func = np.linspace
    dx = x[1] - x[0]
    n = x.size + 1
    return range_func(x[0] - dx / 2, x[-1] + dx / 2, n)


def plot_heatmap(x, y, data, ax=None, norm=None, cmap='viridis', shading='flat',
                 clevels=None, xlabel=None, ylabel=None, fs=8, title=None):
    ''' 
    Plot a heatmap
    
    :param x: x coordinates vector
    :param y: y coordinates vector
    :param data: 2D data array
    :param ax: axis object
    :param norm: normalizer object to determine colors
    :param cmap: colormap name
    :param gouraud: whether or not to smooth out map with Gouraud shading 
    '''
    # Initialize figure and axis, if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Set axis labels if provided
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fs)
    else:
        ax.set_xticks([])
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fs)
    else:
        ax.set_yticks([])
    
    # Ensure 1:1 physical aspect
    ax.set_aspect(1.)

    # If needed, compute vectors of x and y edges
    if shading == 'flat':
        x = compute_mesh_edges(x)
        y = compute_mesh_edges(y)
    
    # Plot heatmap with appropriate shading
    ax.pcolormesh(
        x, y, data.T, cmap=cmap, norm=norm, shading=shading)
    
    # Plot contour levels if provided
    if clevels is not None:
        ax.contour(x, y, data.T, levels=clevels, colors='w')
    
    # Set fontsize
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    
    # Add title if provided
    if title is not None:
        ax.set_title(title, fontsize=fs)
    
    # Return
    return fig


def get_xintercepts(x, y, yref):
    intcpts = np.where(np.diff(np.sign(y - yref)))[0]
    a = (y[intcpts + 1] - y[intcpts]) / (x[intcpts + 1] - x[intcpts])
    b = y[intcpts] - a * x[intcpts]
    return (yref - b) / a


def plot_profile(ax, x, y, xkey, xunit, ykey, fs, c='C0'):
    # Set title and labels
    ax.set_title(f'{xkey} profile', fontsize=fs + 2)
    ax.set_xlabel(f'{xkey} ({xunit})', fontsize=fs)
    ax.set_yticks([0, y.max() / 2, y.max()])
    if ykey is not None:
        ax.set_ylabel(ykey, fontsize=fs)
    else:
        ax.set_yticklabels([])
    # Plot profile and set y-bounds
    ax.plot(x, y, c=c)
    if y.min() > 0:
        ax.set_ylim(0, y.max())
    # Extract halfmax crossings on each side of peak
    yref = 0.5 * y.max()
    halfmax_crossings = get_xintercepts(x, y, yref)
    isleft = halfmax_crossings <= x[y.argmax()]
    lcross, rcross = halfmax_crossings[isleft], halfmax_crossings[~isleft]
    # Compute focal range along dimension
    focus_xbounds = [x[0], x[-1]]
    focus_ybounds = [y[0], y[-1]]
    if lcross.size == 0:
        logger.warning(f'{xkey} profile: found no crossing on left side of peak')
    else:
        focus_xbounds[0] = lcross[-1]
        focus_ybounds[0] = yref
        ax.axvline(lcross[-1], c='k', ls='--')
        if lcross.size > 1:
            logger.warning(f'{xkey} profile: found {lcross.size} crossings on left side of peak')
    if rcross.size == 0:
        logger.warning(f'{xkey} profile: found no crossing on right side of peak')
    else:
        focus_xbounds[1] = rcross[0]
        focus_ybounds[1] = yref
        ax.axvline(rcross[0], c='k', ls='--')
        if rcross.size > 1:
            logger.warning(f'{xkey} profile: found {rcross.size} crossings on right side of peak')
    # Plot area under curve for focal range
    focus_idxs = np.where(is_within(x, focus_xbounds))[0]
    xin = np.hstack(([focus_xbounds[0]], x[focus_idxs], [focus_xbounds[1]]))
    yin = np.hstack(([focus_ybounds[0]], y[focus_idxs], [focus_ybounds[1]]))
    ax.fill_between(xin, yin, fc=c, ec='none', alpha=0.5)
    # Annotate FWHM info if possible
    if lcross.size > 0 and rcross.size > 0:
        fwhm = rcross[0] - lcross[-1]
        ax.text(0.6, 0.9, f'FWHM = {fwhm:.2f} {xunit}',
                transform=ax.transAxes, fontsize=fs)


def get_axes_dims(n):
    if n <= 3:
        return n, 1
    xmax = int(np.floor(np.sqrt(n)))
    xrange = np.arange(1, xmax + 1)[::-1]
    x = next(filter(lambda i: n % i == 0, xrange), None)
    if x > 1:
        return n // x, x
    else:
        return xmax, int(np.ceil(n / xmax))


def log_scalar(xyz, x, y, z, data, out_key):
    ''' Lofg info relative to data extraction '''
    xyz_str = ', '.join([f'{k:.2f}' for k in xyz])
    ix = np.where(x == xyz[0])[0][0]
    iy = np.where(y == xyz[1])[0][0]
    iz = np.where(z == xyz[2])[0][0]
    idx = (ix, iy, iz)
    out = data[ix, iy, iz]
    logger.info(
        f'extracting scalar for position [{xyz_str}]: index = {idx}, {out_key} = {out:.2f}')


def plot_slices(x, y, z, data, xyz_unit, out_key, out_unit, sliceaxis=0,
                title=None, fs=8, cmap='viridis', ax=None,  
                interactive=False, traces_dir=None, bounds=None, **kwargs):
    ''' 
    Visualize an XYZ 3D field a series of slices
    
    :param x: vector of X coordinates
    :param y: vector of Y coordinates
    :param z: vector of Z coordinates
    :param data: 3D (X, Y, Z) data matrix
    :param title (optional): figure title
    :param sliceaxis (optional): index of the axis to be sliced
    :param fs (optional): labels font size
    :param contour_dBs: decibel levels at which to plot contour lines
    :param cmap (optional): colormap
    :param unit (optional): coordinates unit
    :param outkey (optional): name of the output variable
    :param ax: axis object (for single slice only)
    :return: figure handle
    '''
    # Check that enough dimensions are provided
    ndims = sum(k is not None for k in [x, y, z])
    if ndims < 2:
        raise ValueError(f'not enough input dimenions ({ndims})')
    
    # If only 2 dimensions provided, provide placeholder vector, adapt data,
    # and modify slice axis if necessary
    if ndims == 2:
        placeholdervec = np.array([0])
        if x is None:
            sliceaxis, x = 0, placeholdervec
        elif y is None:
            sliceaxis, y = 1, placeholdervec
        if z is None:
            sliceaxis, z = 2, placeholdervec
        data = np.expand_dims(data, axis=sliceaxis)

    # Determine slice-axis dependent parameters
    plotaxes = {0: ['Y', 'Z'], 1: ['X', 'Z'], 2: ['X', 'Y']}[sliceaxis]
    plotvecs = {0: [y, z], 1: [x, z], 2: [x, y]}[sliceaxis]
    refax = {0: 'X', 1: 'Y', 2: 'Z'}[sliceaxis]
    refvec = {0: x, 1: y, 2: z}[sliceaxis]

    # If axis object is provided, make sure only a single slice is required
    if ax is not None and refvec.size > 1:
        raise ValueError(f'cannot fit {refvec.size} slices onto single provided axis')
    
    # Determine data bounds and set up normalizer and scalar mappable accordingly
    if bounds is None:
        bounds = (data.min(), data.max())    
    norm = plt.Normalize(*bounds)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Initialize figure and axes accorsding to number of slices to plot
    if ax is None:
        ncols, nrows = get_axes_dims(refvec.size)
        factor = 1.6
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(max(7, factor * ncols), max(5, factor * nrows)))
        add_cbar = True
    else:
        fig = ax.get_figure()
        axes = [ax]
        add_cbar = False
    axes = np.atleast_2d(axes)
    
    # Loop through axes and plot slices
    for i, (ax, v) in enumerate(zip(axes.ravel(), refvec)):
        idx = np.unravel_index(i, axes.shape)
        ax.set_title(f'{refax} = {v:.2f} {xyz_unit}', fontsize=fs)
        xlabel, ylabel = None, None
        if idx[0] == axes.shape[0] - 1:
            xlabel = f'{plotaxes[0]} ({xyz_unit})'
        if idx[1] == 0:
            ylabel = f'{plotaxes[1]} ({xyz_unit})'
        plot_heatmap(
            *plotvecs, np.take(data, i, axis=sliceaxis), ax=ax, norm=norm, 
            cmap=cmap, xlabel=xlabel, ylabel=ylabel, fs=fs, **kwargs)

    # Hide remaining axes
    for ax in axes.ravel()[i + 1:]:
        ax.remove()

    # Add colorbar
    if add_cbar:
        out_desc = out_key
        if out_unit is not None:
            out_desc = f'{out_desc} ({out_unit})'
        add_cbar_to_fig(fig, sm, out_desc)

    # Add title if specified
    slicekey = 'XYZ'[sliceaxis]
    stitle = f'{slicekey} slice'
    if axes.size > 1:
        stitle = f'{stitle}s'
    if title is not None:
        stitle = f'{title} - {stitle}'
    if axes.size > 1:
        fig.suptitle(stitle, fontsize=fs + 2)
    else:
        axes[0, 0].set_title(stitle, fontsize=fs + 2)
    
    # If specified, add interactivity    
    if interactive:
        def on_click(event):
            if event.inaxes in axes:
                # Get the relevant axis index
                iax = np.where(axes.ravel() == event.inaxes)[0][0]
                # Resolve XYZ data point
                xyz_resolved = np.insert(
                    resolve_xy(event, plotvecs),
                    sliceaxis, refvec[iax])
                # Plot associated traces
                log_scalar(xyz_resolved, x, y, z, data, out_key)
                plot_traces(xyz_resolved, x, y, z, traces_dir)
        fig.canvas.mpl_connect('button_press_event', on_click)

    # Return figure
    return fig


def plot_focus_slices(x, y, z, data, xyz_unit, out_key, out_unit, mark_focus,
                      title=None, cmap='viridis', fs=8, interactive=False,
                      traces_dir=None, bounds=None, focus_color='r', zinterp=None, **kwargs):
    '''
    Plot 1 slice across focus for each dimension    
    '''
    # Find index and XYZ position of acoustic focus
    focus_idx = idxmax(data)
    coords = [x, y, z]
    focus_xyz = [coords[i][idx] for i, idx in enumerate(focus_idx)]

    # Determine bounds from data, if not provided
    if bounds is None:
        bounds = (data.min(), data.max())
    
    # Initialize figure and axes
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    if title is not None:
        fig.suptitle(title)
    
    # For each XYZ dimension
    for iaxis, (idx, axcol) in enumerate(zip(focus_idx, axes.T)):
        
        # Plot projected 2D field across focus in complementary plane
        projcoords = coords.copy()
        projcoords[iaxis] = np.atleast_1d(projcoords[iaxis][idx])
        projdata = np.take(data, indices=[idx], axis=iaxis)
        ax = axcol[0]
        plot_slices(*projcoords, projdata, xyz_unit, out_key, out_unit,
                    sliceaxis=iaxis, fs=fs, cmap=cmap, ax=ax, bounds=bounds,
                    **kwargs)
        
        # If specified, mark focus on 2D projection
        if mark_focus is not None:
            if mark_focus == 'center':
                focus2d = focus_xyz.copy()
                del focus2d[iaxis]
                for linefunc, lineval in zip([ax.axvline, ax.axhline], focus2d):
                    linefunc(lineval, ls='--', color=focus_color, zorder=80)
            elif mark_focus == 'contour':
                xycontour = [proj for i, proj in enumerate(projcoords) if i != iaxis]
                ax.contour(
                    *xycontour, np.squeeze(projdata).T, levels=[0.5], 
                    colors=[focus_color])
        
        # Plot projected 1D profile across focus in that dimension 
        projvec = data[get_mux_slice(focus_idx, iaxis)]
        ax = axcol[1]
        plot_profile(ax, coords[iaxis], projvec, 'XYZ'[iaxis], xyz_unit, 
                     out_key if iaxis == 0 else None, fs, c=f'C{iaxis}')        
        
        # If specified, interpolate pressure at specific z-value and plot associated lines
        if iaxis == 2 and zinterp is not None:
            amp_at_z = np.interp(zinterp, coords[iaxis], projvec)
            ax.axvline(zinterp, ls='--', color='dimgray')
            ax.axhline(amp_at_z, ls='--', color='dimgray')
            ax.text(
                0.5, 0.7, f'P(z={zinterp:.2f} mm) = {amp_at_z:.2f} {"" if out_unit is None else out_unit}', 
                color='dimgray', fontsize=fs, transform=ax.transAxes)

        # Clean up axis and labels
        sns.despine(ax=ax)
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fs)
    
    # Add colorbar
    norm = plt.Normalize(*bounds)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.tight_layout()
    add_cbar_to_fig(fig, sm, out_key)
    
    # If specified, add interactivity
    if interactive:
        mapaxes = axes[0]
        def on_click(event):
            # If click event is on a map axis
            if event.inaxes in mapaxes:
                # Get the relevant axis index
                iax = np.where(mapaxes == event.inaxes)[0][0]
                # Resolve slice coordinate and get associated XY grid
                xygrid = coords.copy()
                slicevec = xygrid.pop(iax)
                # Resolve XYZ data point
                xyz_resolved = np.insert(
                    resolve_xy(event, xygrid),
                    iax, slicevec[focus_idx[iax]])
                # Plot associated traces
                log_scalar(xyz_resolved, x, y, z, data, out_key)
                plot_traces(xyz_resolved, x, y, z, traces_dir)
        fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Return figure
    return fig


def resolve_xy(event, xygrid):
    ''' Resolve XY data point on a 2D grid from click event coordinates '''
    ix = np.abs(xygrid[0] - event.xdata).argmin()
    iy = np.abs(xygrid[1] - event.ydata).argmin()
    return np.array([xygrid[0][ix], xygrid[1][iy]])


def plot_acoustic_field(coords_per_dim, Pmat, sliceaxis, xyz_unit='um',
                        gaussian_sigma=None, out_mode='amp', norm=False, zoffset=500., 
                        mark_focus=None, **kwargs):
    '''
    Load mapping data and plot transducer acoustic field
    
    :param coords_per_dim: dictionary of coordinates per dimension (um)
    :param Pmat: multi-dimensional array of pressure values along the grid 
    :param sliceaxis: axis across which 2D planes are "sliced". One of:
        - 'X'/'Y'/'Z'
        - 'auto': automatically select axis with smallest dimension
        - 'focus': to slice 1 plane across focus for each dimension
    :param xyz_unit: unit of coordinates (defaults to "um")
    :param gaussian_sigma (optional): standard deviation (in xyz_unit) of gaussian kernel
        for gaussian filter based denoising of pressure field prior to plotting (useful
        to remove spatial jitter noise or other "short-scale" sources of noise). Can be provided
        either as a scalar (broadcasted across all dimensions) or as a 3-tuple.
    :param out_mode: output field type to plot, one of:
        - 'amp' (default): for pressure amplitude
        - 'int': for acoustic intensity
    :param norm: whether to normalize outptut field prior to plotting (defaults to False)
    :param zoffset: offset (in um) added to axial coordinates vector prior to plotting,
        meant to represent distance from transducer surface to mapping "base plane"
    :param mark_focus: whether and how to materialize "focus" (for focus slicing only)
    :return: figure handle
    '''
    # Rescale coordinates (and z-offset) to appropriate unit, if needed
    factor = {
        'um': 1e0,
        'mm': 1e-3
    }[xyz_unit]
    coords_per_dim = {k: v * factor for k, v in coords_per_dim.items()}
    zoffset *= factor

    # If gaussian filter kernel sigma is specified, apply gaussian filtering to smooth pressure matrix 
    if gaussian_sigma is not None:

        # If "auto-scaling" required, set kernel std to 1.5 indexes across all dimensions
        if gaussian_sigma == 'auto':
            gaussian_sigma_idx = 1.5

        else:
            # Check validity of gaussian_sigma, and cast as float
            is_valid = True
            if isinstance(gaussian_sigma, (tuple, list, np.ndarray)):
                if isinstance(gaussian_sigma, np.ndarray) and gaussian_sigma.ndim > 1:
                    is_valid = False
                if len(gaussian_sigma) != 3:
                    is_valid = False
                if not all(isinstance(s, (int, float)) for s in gaussian_sigma):
                    is_valid = False
                if not any(s < 0 for s in gaussian_sigma):
                    is_valid = False
                if is_valid:
                    gaussian_sigma = np.array(gaussian_sigma, dtype=float)
            elif isinstance(gaussian_sigma, (int, float)):
                if gaussian_sigma < 0:
                    is_valid = False
                if isinstance(gaussian_sigma, int):
                    gaussian_sigma = float(gaussian_sigma)
            else:
                is_valid = False
            if not is_valid:
                raise ValueError(f'invalid gaussian_sigma value: {gaussian_sigma} (must be either a float or 3-tuple, >=0)')
            
            # Convert sigma from xyz unit to index unit based on spatial resolution across each axis
            dx = np.round(np.array([v[1] - v[0] for v in coords_per_dim.values()]), 6)
            if all(np.isclose(item, dx[0]) for item in dx):
                dx = dx[0]
            gaussian_sigma_idx = np.round(gaussian_sigma / dx, 6)
        
        # Apply gaussian filter to pressure matrix
        logger.info(f'smoothing pressure matrix with gaussian filter with kernel sigma = {gaussian_sigma} {xyz_unit} ({gaussian_sigma_idx} indexes)')
        Pmat = gaussian_filter(Pmat, sigma=gaussian_sigma_idx)

    # If specified, convert field to intensity
    out_key = 'pressure'
    out_unit = 'MPa'
    if out_mode == 'int':
        Pmat = pressure_to_intensity(Pmat / PA_TO_MPA) / M2_TO_CM2
        out_key = 'intensity'
        out_unit = 'W/cm2'

    # If specified, normalize field
    if norm:
        Pmat = rescale(Pmat, 0, 1)
        # Pmat /= Pmat.max()
        out_key = f'normalized {out_key}'
        out_unit = None
    
    # Add z-offset
    coords_per_dim['Z'] += zoffset

    # Plot slices and return figure
    args = [*coords_per_dim.values(), Pmat, xyz_unit, out_key, out_unit]
    if sliceaxis == 'focus':
        logger.info('plotting focus slices across each dimension')
        return plot_focus_slices(*args, mark_focus, **kwargs)
    else:
        # If "auto" slice axis specified, select axis with smallest dimension
        if sliceaxis == 'auto':
            nperdim = {k: len(v) for k, v in coords_per_dim.items()}
            nvals = np.array(list(nperdim.values()))
            sliceaxis = list(nperdim.keys())[np.argmin(nvals)]

        # Check that slice axis is valid
        if not isinstance(sliceaxis, str) or sliceaxis.upper() not in 'XYZ':
            raise ValueError(f'invalid slice axis: {sliceaxis} (must be one of {"focus", "auto", "X", "Y", "Z"}')
        
        # Remove zinterp if still in keyword arguments (only for focus plots)
        if 'zinterp' in kwargs:
            del kwargs['zinterp']
        
        # Identify slice plane   
        sliceaxis = sliceaxis.upper()
        sliceplane = 'XYZ'.replace(sliceaxis, '')

        # Plot plane slices across identified axis
        logger.info(f'plotting field {sliceplane} slices across {sliceaxis} dimension')
        return plot_slices(*args, sliceaxis='XYZ'.index(sliceaxis), **kwargs)


def plot_transformation(M, p0, plane='XZ', details=False):
    '''
    Plot transform process
    
    :param origin: rotation origin
    :param M: Multi-transform object
    :param p0: original point(s)
    :param plane: 2D plane on which to project 3D points for plotting purposes
    :param details: whether to plot details of sub-transformations
    :return: figure handle
    '''
    plane = plane.upper()
    # Extract indexes of dimensions to be plotted
    ix = 'XYZ'.index(plane[0])
    iy = 'XYZ'.index(plane[1])

    # Cast input to 2D
    p0 = np.atleast_2d(p0)

    # Create figure backbone
    fig, ax = plt.subplots()
    ax.set_title(M)
    for sk in ['top', 'right']:
        ax.spines[sk].set_visible(False)
    ax.set_xlabel(plane[0])
    ax.set_ylabel(plane[1])
    ax.set_aspect(1.)

    # Label origin
    ax.axhline(0, c='k', ls='--')
    ax.axvline(0, c='k', ls='--')
    ax.scatter(0, 0, c='k')
    
    # Plot initial point(s)
    ax.scatter(p0[:, ix], p0[:, iy], label='original')
    
    # Apply transform(s) and plot transformed point(s)
    if details:
        p1 = p0
        for desc, transform in zip(M.descriptors, M.transforms):
            p1 = M.apply_transform(transform, p1)
            ax.scatter(p1[:, ix], p1[:, iy], label=desc)
    else:
        p1 = M.apply(p0, verbose=True)
        ax.scatter(p1[:, ix], p1[:, iy], c=f'C{len(M.transforms)}', label='final')
    
    # Apply inverse transformation and plot
    p0 = M.inverse().apply(p1)
    ax.scatter(p0[:, ix], p0[:, iy], label='recovered', c='silver', marker='x')

    # Add legend
    ax.legend()

    # Return figure
    return fig


def plot_P_DC_protocol(P, DC):
    ''' Plot sonication protocol in the DC - pressure space '''

    # Compute time average intensities over P - DC grid
    nperax = 100
    Prange = np.linspace(0, 1.25 * P.max(), nperax)  # MPa
    DCrange = np.linspace(0, 100, nperax)  # % 
    Isppa = pressure_to_intensity(Prange / PA_TO_MPA) / M2_TO_CM2  # W/cm2
    Ispta = np.dot(np.atleast_2d(Isppa).T, np.atleast_2d(DCrange)) * 1e-2  # W/cm2

    # Create figure
    fs = 12
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_title('Ispta (W/cm2)', fontsize=fs)
    ax.set_xlabel('duty cycle (%)', fontsize=fs)
    ax.set_ylabel('peak pressure (MPa)', fontsize=fs)
    sns.despine(ax=ax)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)

    # Plot Ispta colormap over DC - DC space
    cmap = sns.color_palette('rocket', as_cmap=True).reversed()
    ax.pcolormesh(
        DCrange, Prange, Ispta, shading='gouraud', rasterized=True, cmap=cmap)

    # Plot contours for characteristic Ispta values
    Ispta_levels = np.array([.01, .2, 1, 2, 5, 10, 20])
    labels_DC = 80
    labels_Ispta = Ispta_levels / labels_DC * 1e2  # W/cm2
    labels_P = intensity_to_pressure(labels_Ispta * M2_TO_CM2) * PA_TO_MPA 
    labels_locs = [(labels_DC, p) for p in labels_P]
    CS = ax.contour(DCrange, Prange, Ispta, levels=Ispta_levels, colors='k')
    ax.clabel(CS, fontsize=fs, inline=True, fmt='%.2g', manual=labels_locs)

    # Plot sampled DC - P combinations
    ax.scatter(DC, P, c='deepskyblue', edgecolors='k', zorder=80)

    # Finalize figure layout
    fig.tight_layout()

    # Return figure
    return fig


def save_figs_book(dirpath, figs, name='figs'):
    '''
    Save figures dictionary as consecutive pages in single PDF document.

    :param dirpath: directory path where to save the PDF file
    :param figs: dictionary of figure objects to save
    :param name: base name of the PDF file (default is 'figs')
    '''
    # Check that directory path is valid
    if not os.path.isdir(dirpath):
        raise ValueError(f'Invalid directory path: "{dirpath}"')

    # Check that there is at least one figure to save 
    if len(figs) == 0:
        logger.warning('no figures to save')
        return
    
    # Assemble file name with base name and current date
    today = datetime.now().strftime('%Y.%m.%d')
    fname = f'{name}_{today}.pdf'

    # Save figures in PDF file
    fpath = os.path.join(dirpath, fname)
    logger.info(f'saving figures in {fpath}:')
    file = PdfPages(fpath)
    for v in tqdm(figs.values()):
        file.savefig(v, transparent=True, bbox_inches='tight')
    file.close()


def plot_focal_distances(cbydate=False):
    '''
    Plot focal distances for each transducer

    :param cbydate (optional): whether to use color-coding by date (default is False)
    :return: figure handle
    '''
    # Load focal distances data table
    if not os.path.isfile(FOCAL_DISTANCES_FPATH):
        raise ValueError(f'focal distances file not found: {FOCAL_DISTANCES_FPATH}')
    df = pd.read_excel(FOCAL_DISTANCES_FPATH, engine='openpyxl')
    df = df.set_index('date')

    # Merge columns and create extra index level called 'transducer'
    df.columns.name = 'transducer'
    distances = (df.stack() / 1e3).rename('focal distance (mm)')
    distances.index = distances.index.swaplevel(0, 1)
    distances = distances.sort_index()

    # Create figure backbone
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.despine(ax=ax)

    # Plot focal distances
    kwargs = dict(
        data=distances.reset_index(),
        x='focal distance (mm)',
        y='transducer',
        ax=ax,
    )
    if cbydate:
        sns.stripplot(hue='date', **kwargs)
    else:
        sns.violinplot(inner='points', **kwargs)
    ax.set_xlim(0, 5)

    # Add line for global average across transducers
    avgdist = distances.groupby('transducer').mean().mean()
    ax.axvline(avgdist, color='k', linestyle='--', label=f'global avg. = {avgdist:.2f} mm')
    ax.legend()
    
    # Finalize figure layout
    fig.tight_layout()

    # Return figure
    return fig