from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.colors import ListedColormap
import matplotlib.tri as mtri
import numpy as np
import matplotlib.pyplot as plt


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)


def setup_axes(ax, view_angles=(30, 120)):
    ax.view_init(view_angles[0], view_angles[1])
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([-0.7, 0.7])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def find_levels(A, level):
    integral = 0
    n = -1

    while integral < level:
        n = n + 1
        integral = integral + A[n]

    return 0.5 * (A[n - 1] + A[n])


def sphere(res):
    theta = np.linspace(0, 2 * np.pi, num=res, endpoint=False)
    phi = np.linspace(np.pi * (-0.5 + 1. / (res + 1)), np.pi * 0.5, num=res, endpoint=False) + np.pi
    theta, phi = np.meshgrid(theta, phi)
    theta, phi = theta.ravel(), phi.ravel()

    mesh_x, mesh_y = ((np.pi * 0.5 - phi) * np.cos(theta), (np.pi * 0.5 - phi) * np.sin(theta))
    x, z, y = -np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)

    triangles = mtri.Triangulation(mesh_x, mesh_y).triangles
    triang = mtri.Triangulation(x, y, triangles)

    return triang, z, triangles, x, y


def axes(ax, alt=False):

    if alt is False:
        # X-AXIS:
        xcolor = plt.get_cmap("Dark2")(0)
        ax.add_artist(Arrow3D([0, -1], [0, 0], [0, 0], mutation_scale=20, lw=3, arrowstyle="-", color=xcolor, zorder=6,
                              alpha=0.3))  # in sphere
        ax.add_artist(Arrow3D([-1, -1.35], [0, 0], [0, 0], mutation_scale=20, lw=3, arrowstyle="-|>", color=xcolor,
                              zorder=6))  # out of sphere
        ax.text(-1.25, 0, 0.1, '$\hat{X}$', fontsize=34, color=xcolor)

    # Y-AXIS:
    ycolor = plt.get_cmap("Dark2")(3)
    ax.add_artist(
        Arrow3D([0, 0], [0, 0], [0, 1], mutation_scale=20, lw=3, arrowstyle="-", color=ycolor, zorder=6, alpha=0.3))
    ax.add_artist(Arrow3D([0, 0], [0, 0], [1, 1.4], mutation_scale=20, lw=3, arrowstyle="-|>", color=ycolor, zorder=6))
    ax.text(-0.03, 0, 1.3, '$\hat{Y}$', fontsize=34, color=ycolor)

    # Z-AXIS:

    if alt:
        offset = 1.25
    else:
        offset = 1.7

    zcolor = plt.get_cmap("Dark2")(2)
    ax.add_artist(
        Arrow3D([0, 0], [0, 1], [0, 0], mutation_scale=20, lw=3, arrowstyle="-", color=zcolor, zorder=6, alpha=0.3))
    ax.add_artist(Arrow3D([0, 0], [1, offset], [0, 0], mutation_scale=20, lw=3, arrowstyle="-|>", color=zcolor, zorder=6))

    if alt:
        ax.text(0, 1.3, 0.15, '$\hat{Z}$', fontsize=34, color=zcolor)
    else:
        ax.text(0, 1.8, 0.15, '$\hat{Z}$', fontsize=34, color=zcolor)


def vectors(ax, n_o, n_s, i_o):
    # n_o:
    ax.add_artist(
        Arrow3D([0, -n_o[0]], [0, n_o[2]], [0, n_o[1]], mutation_scale=20, lw=5, arrowstyle="-", color="cyan", zorder=6,
                alpha=0.2))
    ax.add_artist(
        Arrow3D([-n_o[0], -n_o[0] * 1.3], [n_o[2], n_o[2] * 1.3], [n_o[1], n_o[1] * 1.3], mutation_scale=20, lw=5,
                arrowstyle="-|>", color="cyan", zorder=11))
    ax.text(-n_o[0] * 1.2 + 0.2, n_o[2] * 1.2, n_o[1] * 1.2, '$n_o$', fontsize=35, color='cyan', weight="bold")

    # n_s:
    ax.add_artist(
        Arrow3D([0, -n_s[0]], [0, n_s[2]], [0, n_s[1]], mutation_scale=20, lw=5, arrowstyle="-", color=(0.8, 0.8, 0.8),
                zorder=6, alpha=0.2))
    ax.add_artist(
        Arrow3D([-n_s[0], -n_s[0] * 1.4], [n_s[2], n_s[2] * 1.4], [n_s[1], n_s[1] * 1.4], mutation_scale=20, lw=5,
                arrowstyle="-|>", color=(0.8, 0.8, 0.8), zorder=11))
    ax.text(-n_s[0] * 1.3 - 0.1, n_s[2] * 1.3 + 0.1, n_s[1] * 1.3, '$n_s$', fontsize=35, color=(0.8, 0.8, 0.8), zorder=11,
            weight="bold")

    if i_o != 90:
        # n_s2:
        ax.add_artist(
            Arrow3D([0, -n_s[0]], [0, -n_s[2]], [0, n_s[1]], mutation_scale=20, lw=5, arrowstyle="-", color=(0.8, 0.8, 0.8),
                    zorder=6, alpha=0.2))
        ax.add_artist(
            Arrow3D([-n_s[0], -n_s[0] * 1.4], [-n_s[2], -n_s[2] * 1.4], [n_s[1], n_s[1] * 1.4], mutation_scale=20, lw=5,
                    arrowstyle="-|>", color=(0.8, 0.8, 0.8), zorder=11))
        ax.text(-n_s[0] * 1.3 - 0.1, -n_s[2] * 1.3 + 0.1, n_s[1] * 1.3, '$n_s$', fontsize=35, color=(0.8, 0.8, 0.8),
                zorder=11,
                weight="bold")


def angles(ax, i_o, n_o, n_s):
    # i_o:
    r = 1.02
    theta = np.linspace(0, i_o * np.pi / 180, 100)
    phi = np.pi / 2

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    ax.plot(-x, z, y, linewidth=5, zorder=10, c='C0')
    ax.text(-x[50] * 1.05 + 0.1, z[50] * 1.05 + 0.05, y[50] * 1.05 - 0.06, '$i_o$', fontsize=45, c='C0', zorder=12,
            weight="bold")

    from scipy.spatial import geometric_slerp

    # psi: (NOT THE CORRECT INTERPOLATION)
    """
    theta = np.linspace(np.arccos(n_o[2]),np.arccos(n_s[2]),100)
    phi = np.linspace(np.pi/2,(np.arctan2(n_s[1],n_s[0])),100)

    P0 = np.array([np.arccos(n_o[2]),np.pi/2])
    P1 = np.array([np.arccos(n_s[2]),np.arctan2(n_s[1],n_s[0])])

    psi = np.arccos(np.dot(n_o,n_s))
    P = P0*np.sin(psi*(1-t))/np.sin(psi) + P1*np.sin(psi * t)/np.sin(psi)

    """
    theta0 = np.arccos(n_o[2])
    phi0 = np.pi / 2
    x0 = np.sin(theta0) * np.cos(phi0)
    y0 = np.sin(theta0) * np.sin(phi0)
    z0 = np.cos(theta0)

    theta1 = np.arccos(n_s[2])
    phi1 = np.arctan2(n_s[1], n_s[0])
    x1 = np.sin(theta1) * np.cos(phi1)
    y1 = np.sin(theta1) * np.sin(phi1)
    z1 = np.cos(theta1)

    start = np.array([x0, y0, z0])
    end = np.array([x1, y1, z1])
    t_vals = np.linspace(0, 1, 100)
    result = geometric_slerp(start, end, t_vals)

    # ax.plot([-x0,-x1],[z0,z1],[y0,y1],linewidth = 10,zorder = 10,c='C1')
    result = result * r
    ax.plot(-result[..., 0], result[..., 2], result[..., 1], linewidth=5, zorder=10, c='orange')
    # ax.plot(-result[0],result[2],result[1],linewidth = 5,zorder = 10,c='C1')
    ax.text(-result[50, 0] + 0.05, result[50, 2], result[50, 1] + 0.03, '$\psi$', fontsize=45, c='orange', zorder=12,
            weight="bold")

    # lambda:
    theta = np.pi / 2 * np.ones(100)
    phi = np.linspace(np.pi / 2, (np.arctan2(n_s[1], n_s[0])), 100)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    ax.plot(-x, z, y, linewidth=5, zorder=10, c='C2')
    ax.text(-x[50] * 1.05 - 0.02, z[50], y[50] * 1.05, '$\lambda$', fontsize=45, c='C2', zorder=12, weight="bold")
    ax.add_artist(
        Arrow3D([0, -x[-1]], [0, z[-1]], [0, y[-2]], mutation_scale=20, lw=5, arrowstyle="-", color="C2", zorder=5,
                alpha=0.5))

    # i_s:
    theta = np.linspace(0, np.arccos(n_s[2]), 100)
    phi = np.arctan2(n_s[1], n_s[0])

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    ax.plot(-x, z, y, linewidth=5, zorder=10, c='C3')
    ax.text(-x[50] * 1.05 - 0.1, z[50] * 1.05, y[50] * 1.05 - 0.06, '$i_s$', fontsize=45, c='C3', zorder=12,
            weight="bold")


def psi_contours(i_o, ax, view_angles, cmap):
    # psi_contour(i_o,2,cmap(0),ax,view_angles,cmap)
    psi_contour(i_o, 10, cmap(0), ax, view_angles, cmap)
    psi_contour(i_o, 20, cmap((20 - 10) / 90), ax, view_angles, cmap)
    psi_contour(i_o, 30, cmap((30 - 10) / 90), ax, view_angles, cmap)
    psi_contour(i_o, 40, cmap((40 - 10) / 90), ax, view_angles, cmap)
    psi_contour(i_o, 50, cmap((50 - 10) / 90), ax, view_angles, cmap)
    psi_contour(i_o, 60, cmap((60 - 10) / 90), ax, view_angles, cmap)
    psi_contour(i_o, 70, cmap((70 - 10) / 90), ax, view_angles, cmap)
    psi_contour(i_o, 80, cmap((80 - 10) / 90), ax, view_angles, cmap)
    psi_contour(i_o, 90, cmap((90 - 10) / 90), ax, view_angles, cmap)
    psi_contour(i_o, 100, cmap((80 - 10) / 90), ax, view_angles, cmap)
    psi_contour(i_o, 110, cmap((70 - 10) / 90), ax, view_angles, cmap)
    psi_contour(i_o, 120, cmap((60 - 10) / 90), ax, view_angles, cmap)
    psi_contour(i_o, 130, cmap((50 - 10) / 90), ax, view_angles, cmap)
    psi_contour(i_o, 140, cmap((40 - 10) / 90), ax, view_angles, cmap)
    psi_contour(i_o, 150, cmap((30 - 10) / 90), ax, view_angles, cmap)
    psi_contour(i_o, 160, cmap((20 - 10) / 90), ax, view_angles, cmap)
    psi_contour(i_o, 170, cmap((10 - 10) / 90), ax, view_angles, cmap)


def psi_contour(i_o, psi, c, ax, view_angles, cmap=None):
    if psi > 90:
        frac = (180 - psi) / 90
    else:
        frac = psi / 90

    c = cmap(frac)
    # this is some hacky stuff:
    N = 600

    i_o = i_o * np.pi / 180
    i_s = np.linspace(-179.99 * np.pi / 180, 179.99 * np.pi / 180, N)

    inarccos = (np.cos(psi * np.pi / 180) - np.cos(i_s) * np.cos(i_o)) / (np.sin(i_s) * np.sin(i_o))
    mask = np.logical_and(inarccos < 1, inarccos > -1)

    inarccos = inarccos[mask]
    lmbda = np.arccos(inarccos)

    i_s = i_s[mask]
    n_s = [np.sin(i_s) * np.sin(lmbda), np.sin(i_s) * np.cos(lmbda), np.cos(i_s)]

    ### to not show contours on the backside:
    a = view_angles[1] * np.pi / 180. - np.pi
    e = view_angles[0] * np.pi / 180. - np.pi / 2.
    X = [np.sin(e) * np.cos(a), np.sin(e) * np.sin(a), np.cos(e)]
    Z = np.c_[-n_s[0], n_s[2], n_s[1]]

    cond = (np.dot(Z, X) >= 0)
    x = -n_s[0][cond]
    y = n_s[2][cond]
    z = n_s[1][cond]

    if cond.sum() > 0:
        J = 0
        K = np.argmax(~cond)
        ax.plot(x[J:K], y[J:K], z[J:K], zorder=6, color=c, linewidth=3, alpha=0.7)

        J = np.argmax(~cond)
        K = N * 2
        ax.plot(x[J:K], y[J:K], z[J:K], zorder=6, color=c, linewidth=3, alpha=0.7)


def density_contour(contour, ax, a):
    phi = contour[:, 0]

    theta = contour[:, 1]

    x = 1.01 * np.sin(theta) * np.cos(phi)
    y = 1.01 * np.sin(theta) * np.sin(phi)
    z = 1.01 * np.cos(theta)

    ax.plot(-x, z, y, zorder=12, c='k', alpha=a, linewidth=2, linestyle='--')


def modify_cmap(cmap):
    background_c = cmap(0)

    background_c = (
    np.min([1, background_c[0] * 1.5]), np.min([1, background_c[1] * 1.5]), np.min([1, background_c[2] * 1.5]), 1)

    newcolors = cmap(np.linspace(0, 1, 256))
    N = 20
    mask = np.arange(0, N)
    vals = np.ones((N, 4))
    vals[:, 0] = cmap(mask)[0:N, 0]
    vals[:, 1] = cmap(mask)[0:N, 1]
    vals[:, 2] = cmap(mask)[0:N, 2]
    vals[:, 3] = np.linspace(0, 1, N)
    newcolors[:N, :] = vals
    newcmp = ListedColormap(newcolors)

    return newcmp, background_c


def density(i_s_samples, lmbda_samples, ax, res, cmap, view_angles):
    # angles:
    i_s_samples = i_s_samples * np.pi / 180
    lmbda_samples = lmbda_samples * np.pi / 180
    nss = [np.sin(i_s_samples) * np.sin(lmbda_samples), np.sin(i_s_samples) * np.cos(lmbda_samples),
           np.cos(i_s_samples)]

    a = view_angles[1] * np.pi / 180. - np.pi
    e = view_angles[0] * np.pi / 180. - np.pi / 2.
    X = [np.sin(e) * np.cos(a), np.sin(e) * np.sin(a), np.cos(e)]
    Z = np.c_[-nss[0], nss[2], nss[1]]

    cond = (np.dot(Z, X) >= 0)
    #nss = [nss[0][cond], nss[1][cond], nss[2][cond]]

    phis = np.arctan2(nss[1][cond], nss[0][cond])
    thetas = i_s_samples[cond]

    # 2D histogram:
    heatmap, xedges, yedges = np.histogram2d(phis, thetas, bins=res, range=([-np.pi, np.pi], [0, np.pi]))

    xedges = xedges[1:] - 0.5 * (xedges[1] - xedges[0])
    yedges = yedges[1:] - 0.5 * (yedges[1] - yedges[0])

    heatmap = heatmap / np.sin(yedges)

    # generate sphere:
    triang, z, triangles, x, y = sphere(res)

    # shaded background:
    newcmp, background_c = modify_cmap(cmap)
    ax.plot_trisurf(triang, z, color=background_c, shade=True, linewidth=0, antialiased=False)

    # heatmap on sphere:

    vals = heatmap.flatten("F")
    A = np.sort(vals, axis=None)
    vmax = A[-20]
    colors = np.mean(vals[triangles], axis=1)
    collec = ax.plot_trisurf(triang, z, cmap=newcmp, vmax=vmax, linewidth=0, antialiased=False)
    collec.set_array(colors)

    return xedges, yedges, heatmap, vals


def density_contours(x, y, heatmap, vals, ax):
    normconst = np.trapz(vals)

    B = -np.sort(-vals / normconst, axis=None)

    level1 = find_levels(B, 0.683)
    level2 = find_levels(B, 0.95)

    CS = plt.contour(x, y, heatmap.T / normconst, colors='k', levels=[level2, level1], alpha=0)

    density_contour(CS.allsegs[0][0], ax, 0.7)
    density_contour(CS.allsegs[1][0], ax, 1)


def obliquity_plot(ax, i_o, i_s, lmbda, i_s_samples=None, lmbda_samples=None, alt=False,
                   view_angles=(30, 120), resolution=250,
                   draw_axes=True, draw_angles=True, draw_vectors=True,
                   draw_psi_contours=True, draw_sphere=True,
                   sphere_cmap=plt.get_cmap('viridis', 256), contour_cmap=plt.get_cmap('rainbow')):

    n_o = [0, np.sin(i_o * np.pi / 180), np.cos(i_o * np.pi / 180)]
    n_s = [np.sin(i_s * np.pi / 180) * np.sin(lmbda * np.pi / 180),
           np.sin(i_s * np.pi / 180) * np.cos(lmbda * np.pi / 180), np.cos(i_s * np.pi / 180)]

    setup_axes(ax, view_angles)

    if i_s_samples is None and draw_sphere is True:
        triang, z, t, x, y = sphere(resolution)
        ax.plot_trisurf(triang, z, color=(1, 1, 1), shade=True, linewidth=0, antialiased=False)

    if draw_axes:
        axes(ax, alt)

    if draw_vectors:
        vectors(ax, n_o, n_s, i_o)

    if draw_angles:
        angles(ax, i_o, n_o, n_s)

    if draw_psi_contours:
        psi_contours(i_o, ax, view_angles, contour_cmap)

    if i_s_samples is not None:
        xedges, yedges, heatmap, vals = density(i_s_samples, lmbda_samples, ax, resolution, sphere_cmap, view_angles)
        # density_contours(xedges,yedges,heatmap,vals,ax)

