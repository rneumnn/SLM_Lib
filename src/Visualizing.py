import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from statsmodels.robust.robust_linear_model import RLMResults

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

def showAngleFit(angleData: np.ndarray, timings: np.ndarray, angleFit: list[RLMResults], intensityData: np.ndarray = None, cmap=mpl.colormaps.get_cmap("perula")):
    """
    Plots each dataset in angleData with the corresponding robust linear fit against the given timings.
    Optionally, plots the maximum of an intensity dataset on a secondary y-axis for each angle dataset.

    Parameters
    ----------
    angleData : np.ndarray
        2D array of angle data, shape (n, t), where n is the number of datasets and t is the number of time points.
    timings : np.ndarray
        1D array of timestamps, shape (t,).
    angleFit : list[RLMResults]
        List of fitted robust linear model results, one for each dataset.
    intensityData : np.ndarray, optional
        2D array of intensity data, shape (n, t), where each row corresponds to the intensity for one angle dataset.
    cmap : matplotlib.colors.Colormap, optional
        Colormap to use for plotting different datasets.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    """
    import matplotlib.pyplot as plt

    n = angleData.shape[0]
    if cmap is None:
        cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, n))
    fig, ax1 = plt.subplots()
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>','+', '.']  # extend if needed

    if intensityData is not None:
        ax2 = ax1.twinx()
    else:
        ax2 = None

    for i in range(n):
        marker = markers[i % len(markers)]
        # Plot angle data with dashed line and marker
        ax1.plot(timings, angleData[i], ls='--', marker=marker, color=colors[i], label=f'Dataset {i+1}')
        # Plot fit with solid line and same marker
        fit_params = angleFit[i].params
        fit_line = fit_params[0] + fit_params[1] * timings
        ax1.plot(timings, fit_line, ls='-', marker=marker, color=colors[i], label=f'Fit {i+1}')
        # Plot intensity max if provided
        if intensityData is not None:
            ax2.plot(timings, intensityData[i], ls=':', marker=marker, color=colors[i], label=f'Intensity {i+1}')

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