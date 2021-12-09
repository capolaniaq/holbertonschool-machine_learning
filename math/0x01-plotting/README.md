# 0x01. Plotting

By Alexa Orrico, Software Engineer at Holberton School

Author Carlos Polania (capolaniaq@correo.udistrital.edu.co)

## Learning Objectives

### General

-   What is a plot?
-   What is a scatter plot? line graph? bar graph? histogram?
-   What is  `matplotlib`?
-   How to plot data with  `matplotlib`
-   How to label a plot
-   How to scale an axis
-   How to plot multiple sets of data at the same time
## Tasks
### 0. Line Graph
Complete the following source code to plot `y` as a line graph:
-   `y`  should be plotted as a solid red line
-   The x-axis should range from 0 to 10
```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

# your code here
```

### 1. Scatter
Complete the following source code to plot `x ↦ y` as a scatter plot

-   The x-axis should be labeled  `Height (in)`
-   The y-axis should be labeled  `Weight (lbs)`
-   The title should be  `Men's Height vs Weight`
-   The data should be plotted as magenta points

### 2. Change of scale

Complete the following source code to plot  `x ↦ y`  as a line graph:

-   The x-axis should be labeled  `Time (years)`
-   The y-axis should be labeled  `Fraction Remaining`
-   The title should be  `Exponential Decay of C-14`
-   The y-axis should be logarithmically scaled
-   The x-axis should range from 0 to 28650

### 3. Two is better than one

Complete the following source code to plot  `x ↦ y1`  and  `x ↦ y2`  as line graphs:

-   The x-axis should be labeled  `Time (years)`
-   The y-axis should be labeled  `Fraction Remaining`
-   The title should be  `Exponential Decay of Radioactive Elements`
-   The x-axis should range from 0 to 20,000
-   The y-axis should range from 0 to 1
-   `x ↦ y1`  should be plotted with a dashed red line
-   `x ↦ y2`  should be plotted with a solid green line
-   A legend labeling  `x ↦ y1`  as  `C-14`  and  `x ↦ y2`  as  `Ra-226`  should be placed in the upper right hand corner of the plot

### 4. Frequency

Complete the following source code to plot a histogram of student scores for a project:

-   The x-axis should be labeled  `Grades`
-   The y-axis should be labeled  `Number of Students`
-   The x-axis should have bins every 10 units
-   The title should be  `Project A`
-   The bars should be outlined in black

### 5. All in One

Complete the following source code to plot all 5 previous graphs in one figure:

-   All axis labels and plot titles should have a font size of  `x-small`  (to fit nicely in one figure)
-   The plots should make a 3 x 2 grid
-   The last plot should take up two column widths (see below)
-   The title of the figure should be  `All in One`

### 6. Stacking Bars

Complete the following source code to plot a stacked bar graph