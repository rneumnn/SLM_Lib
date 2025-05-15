import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

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