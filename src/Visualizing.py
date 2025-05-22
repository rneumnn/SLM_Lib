import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from statsmodels.robust.robust_linear_model import RLMResults

C0 = 299792458  #m/s

"""
    Package to visualize reconstructed electric fields
    """

def plotReconstruction3D(data:np.ndarray, surfaceCount = 1, minRel =.6, maxRel =.8, downscale = 1, opacity = 0.6):
    data = data[:,::downscale,::downscale]
    z_len, y_len, x_len = data.shape
    z = np.linspace(0, data.shape[0] - 1, data.shape[0])
    y = np.linspace(0, data.shape[1] - 1, data.shape[1])
    x = np.linspace(0, data.shape[2] - 1, data.shape[2])
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

    # Plotly-Figur erzeugen
    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=data.flatten(),
        isomin=np.max(data) * 0.4,
        isomax=np.max(data) * 0.6,
        surface_count=surfaceCount,
        opacity=0.6,
        colorscale='Plasma',
        caps=dict(x_show=False, y_show=False, z_show=False),
        skip_invalid = True,
        render_mode="SVG"
        )
    )

    #shadows
    # Projektionen (along axes)
    threshhold = .6
    proj_xy = np.max(data, axis=0)  # shape: (y, x)
    proj_xz = np.max(data, axis=1)  # shape: (z, x)
    proj_yz = np.max(data, axis=2)  # shape: (z, y)
    proj_xy[~np.any(data>data.max()*threshhold, axis=0)] = 0
    proj_xz[~np.any(data>data.max()*threshhold, axis=1)] = 0
    proj_yz[~np.any(data>data.max()*threshhold, axis=2)] = 0
    # Mittlere Schnitte
    z_mid = z_len // 2
    y_mid = y_len // 2
    x_mid = x_len // 2
    slice_xy = data[z_mid, :, :]   # shape: (y, x)
    slice_xz = data[:, y_mid, :]   # shape: (z, x)
    slice_yz = data[:, :, x_mid]   # shape: (z, y)
    color = 'Blues'
    # XY-Projektion (Z = z_min)
    fig.add_trace(go.Surface(
        z=np.full_like(slice_xy, z[0]),  # bottom plane
        x=np.tile(x, (y_len, 1)),
        y=np.tile(y[:, None], (1, x_len)),
        surfacecolor=proj_xy,
        colorscale=color,
        showscale=False,
        opacity=0.5
    ))

    # XZ-Projektion (Y = y_max)
    fig.add_trace(go.Surface(
        y=np.full_like(slice_xz, y[0]),  # back plane
        x=np.tile(x, (z_len, 1)),
        z=np.tile(z[:, None], (1, x_len)),
        surfacecolor=proj_xz,
        colorscale=color,
        showscale=False,
        opacity=0.5
    ))

    # YZ-Projektion (X = x_max)
    fig.add_trace(go.Surface(
        x=np.full_like(slice_yz, x[0]),  # side plane
        y=np.tile(y, (z_len, 1)),
        z=np.tile(z[:, None], (1, y_len)),
        surfacecolor=proj_yz,
        colorscale=color,
        showscale=False,
        opacity=0.5
    ))

    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=2.25, y=2.25, z=1.25)
    )   

    fig.update_layout(scene_camera=camera, title="Surfaceplot", scene =dict(aspectratio=dict(x=1, y=1, z=1.95)))
    # fig.update_layout(
    #     scene = dict(xaxis = dict(range=[min(x), max(x)]),
    #                 yaxis = dict(range=[min(y), max(y)],),
    #                 zaxis = dict( range=[min(z),max(z)],)))

    fig.show()

def showImage(data:np.ndarray):
    if data.dtype == complex:
        data = np.abs(data)
    plt.figure()
    plt.title("ccd image")
    plt.imshow(data, origin="lower")
    plt.colorbar()
    #calculate x,y in mm!
    return plt.gcf()

def showAngleFit(
    angleData: list, 
    timings: list, 
    angleFit: list, 
    intensityData: list = None, 
    cmap=mpl.colormaps.get_cmap("inferno"), 
    labels: list[str] = None
):
    """
    Plot multiple angle datasets and their robust linear fits over time.

    This function visualizes a list of angle measurements (as lists or arrays) against their corresponding time points,
    overlaying each with its robust linear fit. Optionally, a secondary y-axis can display intensity data for each dataset.
    Each dataset is plotted with a unique color and marker for clarity.

    Parameters
    ----------
    angleData : list of lists or 1D arrays
        Each entry is a sequence of angle values for one dataset.
    timings : list of lists or 1D arrays
        Each entry is a sequence of time points corresponding to the angle data.
    angleFit : list of statsmodels RLMResults
        Each entry is a robust linear model fit for the corresponding dataset.
    intensityData : list of lists or 1D arrays, optional
        Each entry is a sequence of intensity values for one dataset (plotted on a secondary y-axis if provided).
    cmap : matplotlib.colors.Colormap, optional
        Colormap for distinguishing datasets (default: "inferno").
    labels : list of str, optional
        Custom labels for each dataset (default: numbered).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plot.
    """
    #for case that only 1d arrays are given
    if not type(angleData) == list:
        angleData = [angleData]
    if not type(timings) == list:
        timings = [timings]
    if not type(angleFit) == list:
        angleFit = [angleFit]
    if not type(intensityData) == list:
        intensityData = [intensityData]

    n = len(angleData)
    if cmap is None:
        cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, n))
    fig, ax1 = plt.subplots()
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>','+', '.']  # extend if needed

    if intensityData is not None:
        ax2 = ax1.twinx()
    else:
        ax2 = None

    if labels is None:
        labels = [str(x) for x in range(1,len(timings)+1)]

    for i in range(n):
        marker = markers[i % len(markers)]
        # Plot angle data with dashed line and marker
        ax1.plot(timings[i], np.unwrap(angleData[i]), ls='--', marker=marker, color=colors[i], label=f'Dataset {labels[i]}')
        # Plot fit with solid line and same marker
        fit_params = angleFit[i].params
        fit_line = fit_params[0] + fit_params[1] * timings[i]
        ax1.plot(timings[i], fit_line, ls='-', marker=marker, color=colors[i], label=f'Fit {i+1}')
        # Plot intensity max if provided
        if intensityData is not None:
            ax2.plot(timings[i], intensityData[i], ls=':', marker="", color=colors[i], label=f'Intensity {labels[i]}')

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Angle")
    if ax2:
        ax2.set_ylabel("Intensity (max)")

    # Combine legends from both axes if needed
    handles1, labels1 = ax1.get_legend_handles_labels()
    if ax2:
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(handles1 + handles2, labels1 + labels2, loc='best')
    else:
        ax1.legend(loc='best')

    plt.title("Angle Data with Robust Linear Fits and Intensity Max")
    plt.tight_layout()
    plt.show()
    return fig

def showOGV(
        OAM: list, 
        angleFit: list, 
        cmap=mpl.colormaps.get_cmap("inferno"), 
        labels: list[str] = None
    ):
    """
    Plot OGV as a function of OAM.

    Parameters
    ----------
    OAM : list
        Each entry is the OAM from the data.
    angleFit : list of statsmodels RLMResults
        Each entry is a robust linear model fit for the corresponding dataset.
    cmap : matplotlib.colors.Colormap, optional
        Colormap for distinguishing datasets (default: "inferno").

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plot.
    """
    #for case that only 1d arrays are given
    if not type(angleFit) == list:
        angleFit = [angleFit]

    n = len(OAM)
    if cmap is None:
        cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, n))
    fig, ax = plt.subplots()
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', '<', '>','+', '.']  # extend if needed

    #Theoretical OGV
    OGV_theory = (2*np.pi*C0/778e-9 - 2*np.pi*C0/806e-9)/9

    for i in range(n):
        marker = markers[i % len(markers)]
        # Plot OGV
        fit_params = angleFit[i].params
        ax.plot(OAM[i], fit_params[1], marker=marker, color=colors[i], label=f'Experimental OGV {OAM[i]}')

    #Plot Theoretical OGV
    ax.plot(OAM[0], -OGV_theory, marker="X", color="r", label="Theoretical OGV")

    ax.set_xlabel("OAM")
    ax.set_ylabel("OGV (rad/s)")

    plt.title("Theoretical and Exerimental OGV")
    plt.tight_layout()
    plt.grid(True, "both")
    plt.legend()
    plt.show()
    return fig

def showOGV(
        OAM: list, 
        angleFit: list,
        radius: list, 
        cmap=mpl.colormaps.get_cmap("inferno"), 
        labels: list[str] = None
    ):
    """
    Plot LOGV as a function of OAM.

    Parameters
    ----------
    OAM : list 
        Each entry is a sequence of time points corresponding to the angle data.
    angleFit : list of statsmodels RLMResults
        Each entry is a robust linear model fit for the corresponding dataset.
    intensityData : list of lists or 1D arrays, optional
        Each entry is a sequence of intensity values for one dataset (plotted on a secondary y-axis if provided).
    cmap : matplotlib.colors.Colormap, optional
        Colormap for distinguishing datasets (default: "inferno").
    labels : list of str, optional
        Custom labels for each dataset (default: numbered).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plot.
    """
    #for case that only 1d arrays are given
    if not type(angleFit) == list:
        angleFit = [angleFit]

    n = len(OAM)
    if cmap is None:
        cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, n))
    fig, ax = plt.subplots()
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', '<', '>','+', '.']  # extend if needed

    for i in range(n):
        marker = markers[i % len(markers)]
        # Plot OGV
        fit_params = angleFit[i].params
        ax.plot(OAM[i], fit_params[1], marker=marker, color=colors[i], label=f'Experimental OGV {OAM[i]}')

    #Plot Theoretical OGV
    ax.plot(OAM[0], -OGV_theory, marker="X", color="r", label="Theoretical OGV")

    ax.set_xlabel("OAM")
    ax.set_ylabel("OGV (rad/s)")

    plt.title("Theoretical and Exerimental OGV")
    plt.tight_layout()
    plt.grid(True, "both")
    plt.legend()
    plt.show()
    return fig

def showOGV(
        OAM: list, 
        angleFit: list, 
        cmap=mpl.colormaps.get_cmap("inferno"), 
        labels: list[str] = None
    ):
    """
    Plot OGV as a function of OAM.

    Parameters
    ----------
    OAM : list
        Each entry is the OAM from the data.
    angleFit : list of statsmodels RLMResults
        Each entry is a robust linear model fit for the corresponding dataset.
    cmap : matplotlib.colors.Colormap, optional
        Colormap for distinguishing datasets (default: "inferno").

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plot.
    """
    #for case that only 1d arrays are given
    if not type(angleFit) == list:
        angleFit = [angleFit]

    n = len(OAM)
    if cmap is None:
        cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, n))
    fig, ax = plt.subplots()
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', '<', '>','+', '.']  # extend if needed

    #Theoretical OGV
    OGV_theory = (2*np.pi*C0/778e-9 - 2*np.pi*C0/806e-9)/9

    for i in range(n):
        marker = markers[i % len(markers)]
        # Plot OGV
        fit_params = angleFit[i].params
        ax.plot(OAM[i], fit_params[1], marker=marker, color=colors[i], label=f'Experimental OGV {OAM[i]}')

    #Plot Theoretical OGV
    ax.hlines(-OGV_theory, 0, 200, ls="-", color="r", label="Theoretical OGV")

    ax.set_xlabel("OAM")
    ax.set_ylabel("OGV (rad/s)")

    plt.title("Theoretical and Experimental OGV")
    plt.tight_layout()
    plt.grid(True, "both")
    plt.legend()
    plt.show()
    return fig

def showLOGV(
        OAM: list, 
        expLOGV: list,
        theoryLOGV: list, 
        cmap=mpl.colormaps.get_cmap("inferno"), 
        labels: list[str] = None
    ):
    """
    Plot LOGV as a function of OAM.

    Parameters
    ----------
    OAM : list 
        Each entry is a sequence of time points corresponding to the angle data.
    angleFit : list of statsmodels RLMResults
        Each entry is a robust linear model fit for the corresponding dataset.
    intensityData : list of lists or 1D arrays, optional
        Each entry is a sequence of intensity values for one dataset (plotted on a secondary y-axis if provided).
    cmap : matplotlib.colors.Colormap, optional
        Colormap for distinguishing datasets (default: "inferno").
    labels : list of str, optional
        Custom labels for each dataset (default: numbered).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plot.
    """
    #for case that only 1d arrays are given
    if not type(angleFit) == list:
        angleFit = [angleFit]

    n = len(OAM)
    if cmap is None:
        cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, n))
    fig, ax = plt.subplots()
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', '<', '>','+', '.']  # extend if needed

    for i in range(n):
        marker = markers[i % len(markers)]
        # Plot LOGV
        fit_params = angleFit[i].params
        ax.plot(OAM[i], expLOGV[i], marker=marker, color=colors[i], label=f'Experimental LOGV {OAM[i]}')
        ax.plot(OAM[i], theoryLOGV[i], marker="X", color=colors[i], label=f'Theoretical LOGV {OAM[i]}')

    ax.set_xlabel("OAM")
    ax.set_ylabel("LOGV (/c)")

    plt.title("Theoretical and Experimental LOGV")
    plt.tight_layout()
    plt.grid(True, "both")
    plt.legend()
    plt.show()
    return fig   