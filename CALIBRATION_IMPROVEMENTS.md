# Calibration Graph Readability Improvements

## Summary of Changes Made

The calibration graph has been completely redesigned to address your accessibility and readability concerns. Here are the key improvements:

## 🔍 **Major Visibility Improvements**

### 1. **Separate Plots Instead of Subplots**
- **BEFORE**: Two graphs cramped in one figure 
- **AFTER**: Each measurement type gets its own dedicated window
- **Benefit**: Much more space, easier to focus on each dataset

### 2. **Expanded Color Library**
- **BEFORE**: Limited colors that repeated and mixed up
- **AFTER**: 40+ distinct, high-contrast, accessible colors
- **Benefit**: Each trajectory has a unique, easily distinguishable color

### 3. **Enhanced Line Styles**
- **BEFORE**: Thin, hard-to-see lines with poor contrast
- **AFTER**: Thicker lines (2.5px), multiple line styles (solid, dashed, dotted)
- **Benefit**: Lines are much more visible and distinguishable even when colors are similar

### 4. **Improved Markers and Visual Elements**
- **BEFORE**: Small, unclear markers
- **AFTER**: Larger markers (5px) with white edges for better contrast
- **Benefit**: Easier to see individual data points

## 🏷️ **Legend and Label Improvements**

### 5. **Better Legend Positioning and Formatting**
- **BEFORE**: Overlapping, unclear legend
- **AFTER**: Positioned outside plot area, organized by date, shadow and frame
- **Benefit**: Clear identification of which trajectory belongs to which run

### 6. **Hierarchical Labeling**
- **BEFORE**: Confusing, abbreviated labels
- **AFTER**: Clear format: "Date: Run#" (e.g., "2025.04.01: #1")
- **Benefit**: Immediately know which run and date each line represents

## 🎯 **Interactive Features**

### 7. **Enhanced Hover Highlighting**
- **NEW**: Hover over any line to highlight it and fade others
- **NEW**: Title updates to show which trajectory you're viewing
- **NEW**: Lines get thicker and more prominent when hovered
- **Benefit**: Easy to isolate and examine individual trajectories

### 8. **Smart Color Organization**
- **NEW**: Algorithm distributes colors to maximize distinction between similar runs
- **Benefit**: Similar measurement runs don't get similar colors

## 📊 **Overall Graph Quality**

### 9. **Better Figure Sizing**
- **BEFORE**: Small, cramped plots
- **AFTER**: Large (12" x 8") high-DPI figures
- **Benefit**: Everything is bigger and clearer

### 10. **Improved Grid and Axes**
- **BEFORE**: Faint grid, thin axes
- **AFTER**: Darker, more visible grid and axes, bold labels
- **Benefit**: Easier to read values and follow trends

## 🚀 **How to Use the Improvements**

### Option 1: Use the Enhanced Script
```bash
python scripts/plot_improved_calibration.py --details --yout all
```

### Option 2: Update Your Existing Scripts
The existing `plot_transducer_calibration.py` script has been updated with all improvements.

### Key Parameters:
- `--details`: Shows individual traces (RECOMMENDED)
- `--yout all`: Creates separate plots for P and coupling ratio
- `--yout P`: Shows only pressure plot
- `--yout cpl`: Shows only coupling ratio plot



