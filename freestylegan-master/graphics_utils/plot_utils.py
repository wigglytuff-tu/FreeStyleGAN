import numpy as np
import matplotlib.pyplot as plt
import cv2

#=================================================

# point scattering into a figure
def scatterPlot(
    points,
    size=None,
    grid=None, 
    ranges=None, 
    pointSize=4,
    flipH=None,
    flipV=None,
    tight=True,
    captions=None,
    showAxes=None,
    filename=None, 
    show=True,
    close=True):
    if not isinstance(points, list):
        points = [points]
    plotCount = len(points)
    if ranges is not None and not isinstance(ranges, list):
        ranges = [ranges] * plotCount
    fig = plt.figure(figsize=size)
    for n, p in enumerate(points):
        ax = fig.add_subplot(plotCount, 1, n+1)
        if ranges is not None:
            ax.set_aspect(1)
            ax.set_xlim(*ranges[n][0])
            ax.set_ylim(*ranges[n][1])
        if grid is not None:
            ax.set_xticks(grid[0]) 
            ax.set_yticks(grid[1])
            ax.grid()
        if showAxes is not None:
            ax.get_xaxis().set_visible(showAxes[n][0])
            ax.get_yaxis().set_visible(showAxes[n][1])
        if flipH is not None and flipH[n]:
            ax.invert_yaxis()
        if flipV is not None and flipV[n]:
            ax.invert_xaxis()
        if captions:
            ax.set_title(captions[n])
        if p is not None:
            if (p.shape[1] < 2):
                p = np.tile(p, (1, 2))
                p[:,1] = 0.5
            ax.scatter(p[:,0], p[:,1], c='black', s=pointSize)
    if tight:
        plt.tight_layout()
    if show:
        plt.show()
    if filename is not None:
        plt.savefig(filename)
    if close:    
        plt.close('all')
    return fig

#=================================================

# overlay a plot with curves
def linePlotOverlay(fcts, ranges, sampleCount=100, linewidth=0.5, axis=None):
    curveSamplesX = np.linspace(*ranges, sampleCount)
    for f in fcts:
        curveSamplesY = f(curveSamplesX)
        ax = plt if axis is None else axis
        ax.plot(curveSamplesX, curveSamplesY, lw=linewidth)

#=================================================

# rasterize matplotlib figure to numpy array
def rasterizeFigure(fig, res=None):
    fig.canvas.draw()
    canvas = np.array(fig.canvas.renderer.buffer_rgba()) / 255
    plt.close()
    if res is not None:
        canvas = cv2.resize(canvas, (res, res), interpolation=cv2.INTER_LINEAR)
    return canvas