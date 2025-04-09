import numpy as np
from functools import partial

import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib import animation, rc
from matplotlib.colors import to_rgba_array
from matplotlib.colors import ListedColormap

from flbench.bdec import credint


###########
# General #
###########
# TODO: create another function to plot 3d scatter
def plot_bivar(fn, box=[[-1,1],[-1,1]], n=50, type='mesh', **kwargs):
    """
    Plot bivariate function fn, that should be vectorized first.
    type can be 'mesh', 'contour', 'contourf', 'surf'.

    Example
    --------
    ```
    # pdf = lambda x, y: np.exp( -(x**2 + y**2) / 2) / (2 * np.pi)
    pdf = lambda x: np.exp( -(x**2).sum(-1) / 2) / (2 * np.pi)
    plt.subplot(121, projection="3d")
    plot_bivar(pdf, type='surf')
    ```
    """
    if isinstance(box, (int, float)):
        box = [[-box, box], [-box, box]]

    xs, ys = np.linspace(*box[0], n), np.linspace(*box[1], n)
    xx, yy = np.meshgrid(xs, ys)
    # zz = fn(xx.reshape(-1), yy.reshape(-1)).reshape(n, n)
    zz = fn(np.stack((xx, yy), -1).reshape(-1, 2)).reshape(n, n)

    if type=='surf':
        out = plt.gca().plot_surface(xx, yy, zz, **kwargs)
    elif type=='mesh':
        out = plt.pcolormesh(xx, yy, zz, **kwargs)
    elif type=='contour':
        out = plt.contour(xx, yy, zz, **kwargs)
    elif type=='contourf':
        out = plt.contourf(xx, yy, zz, **kwargs)
    return out


#############
# 3D Meshes #
#############
def mean_slice(mesh, sli:int | float | slice=None, axis=-1):
    """
    Return a 2D mean projected (along given axis) slice from a 3D mesh.
    """
    mesh_shape = np.array(mesh.shape)
    if sli is None:
        sli = slice(None)
    elif isinstance(sli, int):
        sli = slice(None, sli)
    elif isinstance(sli, float):
        sli = slice(None, round(sli*mesh_shape[axis]))
    return np.moveaxis(mesh, axis, -1)[...,sli].mean(-1)


def plot_mesh(mesh, box_shape=None, sli:int | float | slice=None, axis=-1, vlim:float | tuple[float,float]=1e-4, cmap='viridis'):
    """
    Plot a 2D mean projected slice (along given axis) from a 3D mesh.

    Parameters
    ----------
    mesh : ndarray
        The 3D mesh to be plotted.
    box_shape : tuple of int, optional
        The shape of the mesh physical box in Mpc/h. If None, it defaults to mesh shape.
    sli : int, float or slice, optional
        The slice to be averaged along the last axis of the mesh. 
        * If None, entire axis is used. 
        * If integer, specifies the number of slices used starting from 0. 
        * If float, specifies the proportion of axis used starting from 0.
    vlim : float or tuple of float, optional
        The limit values for colormap. 
        * If float, specifies the proportion of values discarded bilateraly. 
        * If tuple, specifies (vmin, vmax).
    cmap : str, optional
        The colormap used for plotting. Default is 'viridis'.

    Returns
    -------
    quad : QuadMesh
        The QuadMesh object created by `plt.pcolormesh`.
    """
    mesh_shape = np.array(mesh.shape)
    if box_shape is None:
        box_shape = mesh_shape
    else:
        plt.xlabel("$x$ [Mpc/$h$]"), plt.ylabel("$y$ [Mpc/$h$]")

    mesh2d = mean_slice(mesh, sli, axis)

    if vlim is None:
        vlim = None, None
    elif isinstance(vlim, float):
        vlim = np.quantile(mesh2d, [vlim/2, 1-vlim/2])
    vmin, vmax = vlim

    # xx, yy = np.indices(mesh_shape[:2]) * (box_shape/mesh_shape)[:2,None,None]
    xs, ys = np.linspace(0, box_shape[0], mesh_shape[0]), np.linspace(0, box_shape[1], mesh_shape[1])
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    quad = plt.pcolormesh(xx, yy, mesh2d, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.gca().set_aspect(1)
    return quad


def anim_meshes(meshes, box_shape=None, vlim:float | tuple[float,float]=1e-4, 
                cmap='viridis', pause=10):
    """
    Animate a list of 2D meshes.
    """
    rc('animation', html='html5')
    meshes = np.asarray(meshes)
    assert meshes.ndim == 3, "meshes must be a list of 2D arrays"

    if vlim is None:
        vlim = np.quantile(meshes, [0, 1])
    elif isinstance(vlim, float):
        vlim = np.quantile(meshes, [vlim/2, 1-vlim/2])

    quad = plot_mesh(meshes[0,...,None], box_shape, None, vlim, cmap)
    plt.colorbar()

    def update(i):
        if i < len(meshes):
            quad.set_array(meshes[i])
        return quad,

    anim = animation.FuncAnimation(plt.gcf(), update, frames=len(meshes)+pause, interval=100, blit=True)
    plt.close()
    display(anim)




def circ_conv(a, b, axis=-1):
    """
    Circular convolution of two arrays along a given axis.
    Returned array has the maximum length of the two arrays along axis.
    """
    a, b = np.moveaxis(a, axis, -1), np.moveaxis(b, axis, -1)
    n = max(a.shape[-1], b.shape[-1])
    ab = np.fft.rfft(a, n) * np.fft.rfft(b, n)
    ab = np.fft.irfft(ab, n)
    return np.moveaxis(ab, -1, axis)


def circ_mean(a, n=1, axis=-1):
    """
    Circular mean of array along a given axis.
    """
    a = np.moveaxis(a, axis, -1)
    out = circ_conv(a, np.ones(n)/n, axis=-1)
    return np.moveaxis(out, -1, axis)




def scan_mesh3d(mesh, sli:int | float=1/16):
    """
    Return a list of 2D mesh as the mean projected slices from a 3D mesh.
    """
    mesh_shape = np.array(mesh.shape)
    if isinstance(sli, float):
        sli = round(sli*mesh_shape[-1])
    return np.moveaxis(circ_mean(mesh, sli, axis=-1), -1, 0)


def anim_scan(mesh, box_shape=None, sli:int | float=1/16, vlim:float | tuple[float,float]=1e-4, 
              cmap='viridis', pause=0):
    scan = scan_mesh3d(mesh, sli)
    anim_meshes(scan, box_shape, vlim=vlim, cmap=cmap, pause=pause)   
    



##################
# Power Spectrum #
##################
def plot_pow(ks, pow, *args, ell=None, log=False, fill=None, **kwargs):
    if ell is None:
        sub = ""
    else:
        i_ell = [0, None, 1, None, 2][ell]
        sub = f"_{ell}"
        pow = pow[...,i_ell,:]

    if log:
        if fill is None:
            out = plt.loglog(ks, pow, *args, **kwargs)
        else:
            scis = credint(pow, fill, axis=0)
            out = plt.fill_between(ks[0], *scis.T, *args, alpha=(1-fill)**.5, **kwargs)
            plt.xscale('log'), plt.yscale('log')
        plt.ylabel("$P"+sub+"(k)$ [Mpc/$h$]$^3$")
    else:
        if fill is None:
            out = plt.plot(ks, ks * pow, *args, **kwargs)
        else:
            scis = credint(pow, fill, axis=0)
            out = plt.fill_between(ks[0], *(ks[0] * scis.T), *args, alpha=(1-fill)**.5, **kwargs)
        plt.ylabel("$k P"+sub+"(k)$ [Mpc/$h$]$^2$")
    plt.xlabel("$k$ [$h$/Mpc]")
    return out


def plot_trans(ks, trans, *args, log=False, fill=None, **kwargs):
    if log:
        if fill is None:
            out = plt.loglog(ks, trans, *args, **kwargs)
        else:
            scis = credint(trans, fill, axis=0)
            out = plt.fill_between(ks[0], *scis.T, *args, alpha=(1-fill)**.5, **kwargs)
            plt.xscale('log'), plt.yscale('log')
    else:
        if fill is None:
            out = plt.semilogy(ks, trans, *args, **kwargs)
        else:
            scis = credint(trans, fill, axis=0)
            out = plt.fill_between(ks[0], *scis.T, *args, alpha=(1-fill)**.5, **kwargs)
            plt.yscale('log')
    plt.xlabel("$k$ [$h$/Mpc]"), plt.ylabel("transfer")
    return out


def plot_coh(ks, coh, *args, log=False, fill=None, **kwargs):
    if log:
        if fill is None:
            out = plt.loglog(ks, coh, *args, **kwargs)
        else:
            scis = credint(coh, fill, axis=0)
            out = plt.fill_between(ks[0], *scis.T, *args, alpha=(1-fill)**.5, **kwargs)
            plt.xscale('log'), plt.yscale('log')
    else:
        if fill is None:
            out = plt.semilogy(ks, coh, *args, **kwargs)
        else:
            scis = credint(coh, fill, axis=0)
            out = plt.fill_between(ks[0], *scis.T, *args, alpha=(1-fill)**.5, **kwargs)
            plt.yscale('log')
    plt.xlabel("$k$ [$h$/Mpc]"), plt.ylabel("coherence")
    return out


def plot_powtranscoh(ks, pow1, trans, coh, *args, 
                     log=False, fill:float=None, axes:list=None, **kwargs):
    outs = []

    plt.subplot(131) if axes is None else plt.sca(axes[0])
    out = plot_pow(ks, pow1, *args, log=log, fill=fill, **kwargs)
    outs.append(out)
    
    plt.subplot(132) if axes is None else plt.sca(axes[1])
    out = plot_trans(ks, trans, *args, log=log, fill=fill, **kwargs)
    outs.append(out)

    plt.subplot(133) if axes is None else plt.sca(axes[2])
    out = plot_coh(ks, coh, *args, log=log, fill=fill, **kwargs)
    outs.append(out)
    return outs



##########
# Colors #
##########
def alternate(a, b, axis=0):
    a, b = np.moveaxis(a, axis, 0), np.moveaxis(b, axis, 0)
    assert a.shape == b.shape
    mid = a.shape[0]
    out = np.zeros((2*mid, *a.shape[1:]))
    out[:mid:2] = a[::2]
    out[mid::2] = b[::2]
    out[1:mid:2] = b[1::2]
    out[mid+1::2] = a[1::2]
    return np.moveaxis(out, 0, axis)

def interlace(a, b, axis=0):
    a, b = np.moveaxis(a, axis, 0), np.moveaxis(b, axis, 0)
    assert a.shape == b.shape
    out = np.empty((a.shape[0] + b.shape[0], *a.shape[1:]))
    out[0::2] = a
    out[1::2] = b
    return np.moveaxis(out, 0, axis)

c1 = plt.get_cmap('Dark2').colors
c2 = plt.get_cmap('Set2').colors

SetDark2 = ListedColormap(alternate(c2, c1))
DarkSet2 = ListedColormap(alternate(c1, c2))



def color_switch(color, reverse=False):
    """
    Select between color an its negative, or colormap and its reversed.
    Typically used to switch between light theme and dark theme. 

    `color` must be Matpotlib color, or array of colors, or colormap.

    No need to switch the default color cycle `f'C{i}'`, Matplotlib handles it already.
    """
    try:
        color = to_rgba_array(color)
    except:
        if isinstance(color, str): # handle cmap
            if reverse:
                if color.endswith('_r'): # in case provided cmap is alreday reversed
                    return color[:-2]
                else:
                    return color+'_r'# reverse cmap
            else:
                return color
        else:
            raise TypeError("`color` must be Matpotlib color, or array of colors, or colormap.")

    if reverse:
        color[...,:-1] = 1-color[...,:-1] # take color negative, does not affect alpha
    return color


def set_plotting_options(usetex=False, font_size=10):
    params = {'text.usetex': usetex,
            #   'ps.useafm': True,
            #   'pdf.use14corefonts': True,
              'font.family': 'roman' if usetex else 'sans-serif',
              'font.size':font_size,} 
            # NOTE: 'ps.useafm' and 'pdf.use14corefonts' for PS and PDF font comptatibiliies
    plt.rcParams.update(params)
    # import matplotlib as mpl
    # mpl.rcParams.update(mpl.rcParamsDefault)


def theme(dark=False, usetex=False, font_size=10, cmap='SetDark2'):
    """
    Set Matplotlib theme and return an adequate color switching function.
    """
    if dark: 
        plt.style.use('dark_background')
    else: 
        plt.style.use('default')

    if cmap is None:
        cmap = plt.get_cmap('tab10') # default cmap
    elif cmap == 'SetDark2':
        cmap = SetDark2
    else:
        cmap = plt.get_cmap(cmap) # cmap can be plt.get_cmap('viridis', 10) for instance
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cmap.colors)

    rc('animation', html='html5') # handle Matplotlib animations
    set_plotting_options(usetex, font_size)
    theme = partial(color_switch, reverse=dark)
    return theme