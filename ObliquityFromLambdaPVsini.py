import numpy as np
import matplotlib.pyplot as plt
from twopiece.scale import tpnorm
import obliquityplot as op
import matplotlib as mpl


def gaussian(mu, sigma, x):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def calc_i(s, lcc, v, v_p, bins):

    P = lcc.found_rotation_period
    Perr = lcc.found_rotation_period_err
    vsinierr = np.max([0.1, 0.02*s.vsini, s.vsini_err])
    v_first_est = 2 * np.pi * s.R * 695700 / (P * 24 * 3600)

    PRbuffer = np.max([Perr + s.R_err, 1]) * 60
    vsinibuffer = np.max([vsinierr, 1]) * 60
    minx = np.min([v_first_est - PRbuffer, s.vsini - vsinibuffer])
    minx = np.max([minx, 0])
    minx = np.max([minx, v.min() / 2])
    maxx = np.max([v_first_est + PRbuffer, s.vsini + vsinibuffer])
    maxx = np.min([maxx, v.max() * 2])

    vs = np.linspace(minx, maxx, bins*2)

    cosfirst = 1 - np.linspace(0, 1, bins) ** 3
    cosfirst = -cosfirst[::]
    cossec = 1 - np.linspace(0, 1, bins) ** 3
    cossec = cossec[::-1]
    cosi = np.append(cosfirst, cossec)

    Lv = np.interp(vs, v, v_p)

    if len(Lv.shape) > 1:
        Lv = Lv[:, 0]
        vs = vs[:, 0]
        Lv = Lv / np.trapz(Lv, vs, axis=0)
    else:
        Lv = Lv / np.trapz(Lv, vs)

    Prior_v = np.ones(bins*2)
    Prior_cosi = 1

    posterior_data_given_cosi = np.ones(bins*2)

    for n in range(len(cosi)):
        Lu = gaussian(s.vsini, vsinierr, vs * np.sqrt(1 - cosi[n] ** 2))

        in_integral = Lv * Lu * Prior_v
        posterior_data_given_cosi[n] = Prior_cosi * np.trapz(in_integral, vs)

    norm_const = np.trapz(posterior_data_given_cosi, cosi)
    posterior_data_given_cosi = posterior_data_given_cosi / norm_const

    i_s = np.arccos(cosi)
    i_sp = posterior_data_given_cosi * np.sin(i_s)

    return [cosi, posterior_data_given_cosi, i_s, i_sp]


def calc_v(s, lcc, N, bins):

    P = lcc.found_rotation_period
    Perr = lcc.found_rotation_period_err

    Ps = np.random.normal(P, Perr, 2*N)*24*3600
    Ps = Ps[Ps > 0]

    Rs = np.random.normal(s.R, s.R_err, 2 * len(Ps)) * 695700  # km
    Rs = Rs[Rs > 0.1]
    Rs = Rs[:len(Ps)]

    v = 2 * np.pi * Rs / Ps

    vmax = 4 * (1 + s.R_err) * 2 * np.pi * s.R * 695700 / (P * 24 * 3600)

    v = v[v < vmax]
    v = v[v > 0]
    v = v[:N]

    h = np.histogram(v, bins)
    y = h[0]
    x = h[1]
    v = x[1:] - 0.5 * (x[1] - x[0])

    return v, y


def calc_psi(s, cosi, cosip, N, bins):

    nmax = cosip.max()
    cosi_samples = []

    while len(cosi_samples) < N:
        x = np.random.uniform(-1, 1, int(N/2))
        y = np.interp(x, cosi, cosip) / nmax
        p = np.random.random(int(N/2))

        mask = np.less(p, y)
        cosi_samples = np.append(x[mask], cosi_samples)

    cosi_samples = cosi_samples[:N]
    i_samples = np.arccos(np.array(cosi_samples))

    if s.lmbda is None:
        lmbda_samples = np.random.uniform(-np.pi, np.pi, len(i_samples))
    else:
        dist = tpnorm(loc=s.lmbda, sigma1=s.lmbda_err1, sigma2=s.lmbda_err2)
        lmbda_samples = dist.random_sample(size=len(i_samples))

    io_s = np.random.normal(s.i_o, s.i_o_err, len(i_samples))
    oy = np.sin(io_s)
    oz = np.cos(io_s)

    psi_samples = []
    for h in range(len(i_samples)):
        psi = np.dot([0, oy[h], oz[h]], [np.sin(i_samples[h]) * np.sin(lmbda_samples[h]), np.sin(i_samples[h]) * np.cos(lmbda_samples[h]), cosi_samples[h]])
        psi_samples.append(np.arccos(psi))

    psi_mean = np.mean(psi_samples)
    psi_err = np.std(psi_samples)

    counts, xbins = np.histogram(psi_samples, bins)
    x = xbins + 0.5 * (xbins[1] - xbins[0])
    x = x[:-1]
    counts = counts / np.trapz(counts, x)

    return x, counts, psi_mean, psi_err, i_samples, lmbda_samples

def make_plot(s, lcc, bins, v, v_p, i_s, i_sp, psi, psip, psi_mean, psi_err, i_s_samples, lmbda_samples):
    fig = plt.figure()
    fig.tight_layout()

    gs = fig.add_gridspec(4, 6, hspace=0.4, wspace=0.02)
    ax1 = fig.add_subplot(gs[0:1, 0:2])
    ax2 = fig.add_subplot(gs[0:1, 2:4])
    ax3 = fig.add_subplot(gs[0:1, 4:6])
    ax4 = fig.add_subplot(gs[1:2, :])
    ax7 = fig.add_subplot(gs[2:4, 0:3], projection='3d', proj_type='ortho')
    ax8 = fig.add_subplot(gs[2:4, 3:], projection='3d', proj_type='ortho')
    fig.set_size_inches(18, 24)

    i_s_mean = np.median(i_s_samples[i_s_samples < np.pi / 2])
    i_s_err = np.std(i_s_samples[i_s_samples < np.pi / 2])

    ######################### AXIS 7 ##############################

    res = 150
    azimuth = 120
    elev = 30

    if s.lmbda is not None and (s.lmbda > np.pi/2 or s.lmbda < -np.pi/2):
        elev = -30

    if s.lmbda is None:
        op.obliquity_plot(ax7, s.i_o*180/np.pi, i_s_mean*180/np.pi, 0, view_angles=(elev, azimuth),
                          i_s_samples=i_s_samples * 180 / np.pi, lmbda_samples=lmbda_samples * 180 / np.pi, resolution=res, draw_vectors=False, draw_angles=False)
    else:
        op.obliquity_plot(ax7, s.i_o * 180 / np.pi, i_s_mean*180/np.pi,
                          s.lmbda * 180 / np.pi, view_angles=(elev, azimuth),
                          i_s_samples=i_s_samples * 180 / np.pi, lmbda_samples=lmbda_samples * 180 / np.pi,
                          resolution=res, draw_vectors=True, draw_angles=True)

    ax7.axis(False)

    ######################### AXIS 8 ##############################

    azimuth = 180
    elev = 0

    if s.lmbda is not None and s.lmbda < 0:
        azimuth = 0

    if s.lmbda is None:
        op.obliquity_plot(ax8, s.i_o * 180 / np.pi, np.mean(i_s_samples[i_s_samples < np.pi/2]) * 180 / np.pi, 0,
                          view_angles=(elev, azimuth),
                          i_s_samples=i_s_samples * 180 / np.pi, lmbda_samples=lmbda_samples * 180 / np.pi, resolution=res,
                          draw_vectors=False, draw_angles=False, alt=True)
    else:
        op.obliquity_plot(ax8, s.i_o * 180 / np.pi, np.mean(i_s_samples[i_s_samples < np.pi / 2]) * 180 / np.pi,
                          s.lmbda * 180 / np.pi,
                          view_angles=(elev, azimuth),
                          i_s_samples=i_s_samples * 180 / np.pi, lmbda_samples=lmbda_samples * 180 / np.pi,
                          resolution=res,
                          draw_vectors=True, draw_angles=False, alt=True)

    ax8.axis(False)

    ######################### AXIS 1 ##############################

    ax = ax1
    P = np.linspace(lcc.found_rotation_period - 4 * lcc.found_rotation_period_err,
                    lcc.found_rotation_period + 4 * lcc.found_rotation_period_err, 100)
    ax.plot(P, gaussian(lcc.found_rotation_period, lcc.found_rotation_period_err, P), 'k-', lw=6)
    ax.set_xlabel('Rotation period [d]')
    ax.set_yticks([])
    ax.set_ylabel('Probability density')

    ######################### AXIS 2 ##############################

    ax = ax2
    R = np.linspace(s.R - 4*s.R_err, s.R + 4*s.R_err, 100)
    ax.plot(R, gaussian(s.R, s.R_err, R), c='pink', lw=6)
    ax.set_xlabel('$R$ [$R_\odot$]', fontsize=20)
    ax.set_yticks([])
    ax.set_title(s.name)

    ######################### AXIS 3 ##############################

    ax = ax3
    vsini = np.linspace(np.max([0, s.vsini - 4*s.vsini_err]), s.vsini + 4*s.vsini_err, 100)
    ax.plot(vsini, np.abs(gaussian(s.vsini, s.vsini_err, vsini)), c='purple', lw=6)
    ax.plot(v, v_p/np.abs(np.trapz(v_p, v)), c='brown', lw=6)
    ax.set_xlabel('Rotation speed [km/s]', fontsize=20)
    ax.set_yticks([])
    ax.legend(['vsin$i_s$', 'v'], fontsize=20)

    ######################### AXIS 4 ##############################

    ax = ax4
    ###################### I_S ##############################

    x = i_s[i_s < np.pi/2]*180/np.pi
    y = i_sp[i_s < np.pi/2]
    y = y/np.abs(np.trapz(y, x))

    i_s_handle, = ax.plot(x, y, c='C3', lw=6)

    ######################### LAMBDA ##############################
    counts, xbins = np.histogram(lmbda_samples, bins)
    x = xbins + 0.5 * (xbins[1] - xbins[0])
    lmbda = x[:-1]
    
    if s.lmbda is not None and s.lmbda < 0:
        x = -lmbda*180/np.pi
        y = np.abs(counts / np.trapz(counts, x))
        
        mask = x > 0
        x = x[mask]
        y = y[mask]
    else:
        x = lmbda*180/np.pi
        
        y = np.abs(counts / np.trapz(counts, x))

    lmbda_handle, = ax.plot(x, y, c='C2', lw=6)
    maxy = ax.get_ylim()[1]

    ###################### I_O ##############################

    i_o = np.linspace(s.i_o - 4 * s.i_o_err, s.i_o + 4 * s.i_o_err, 100)

    x = i_o * 180 / np.pi
    y = gaussian(s.i_o, s.i_o_err, i_o)
    # y = y/np.trapz(y, x)
    y = y / y.max() * maxy

    i_o_handle, = ax.plot(x, y, c='C0', lw=6)

    ######################### PSI ##############################

    psi = psi*180/np.pi
    psip = psip/np.trapz(psip, psi)

    path = mpl.path.Path(np.column_stack([psi, psip]))
    verts = path.interpolated(steps=3).vertices
    x, y = verts[:, 0], verts[:, 1]
    z = np.linspace(psi.min(), psi.max(), len(x))
    z[z > 90] = 180 - z[z > 90]
    z = np.asarray(z)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = mpl.collections.LineCollection(segments, array=z, cmap=plt.get_cmap('rainbow'), norm=plt.Normalize(0, 90),
                              linewidth=8)

    ax.add_collection(lc)

    ######################### SETUP AXIS 4 ##############################

    ax.plot([90, 90], [0, maxy], 'k--', lw=2)

    legend_inputs = ['$i_o = %1.1f^\circ\pm%1.1f$' % (s.i_o*180/np.pi, s.i_o_err*180/np.pi),
                     '$i_s = %1.1f^\circ\pm%1.1f$' % (i_s_mean*180/np.pi, i_s_err*180/np.pi),
                     '$\psi = %1.1f^\circ\pm%1.1f$' % (psi_mean*180/np.pi, psi_err*180/np.pi)]

    if s.lmbda is not None:
        legend_inputs.append('$\lambda = %1.1f^{+%1.1f^\circ}_{-%1.1f^\circ}$' % (s.lmbda*180/np.pi, s.lmbda_err2*180/np.pi, s.lmbda_err1*180/np.pi))
    else:
        legend_inputs.append('$\lambda$ isotropic')


    ############################

    from matplotlib.patches import Rectangle
    from matplotlib.legend_handler import HandlerBase

    class HandlerColormap(HandlerBase):
        def __init__(self, cmap2, num_stripes=8, **kw):
            HandlerBase.__init__(self, **kw)
            self.cmap = cmap2
            self.num_stripes = num_stripes

        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            stripes = []
            for ij in range(self.num_stripes):
                s2 = Rectangle([xdescent + ij * width * 1.2/ self.num_stripes, ydescent],
                               1.2 * width / self.num_stripes,
                               height*0.5, fc=self.cmap((2 * ij + 1) / (2 * self.num_stripes)))
                stripes.append(s2)
            return stripes

    cmaps = [plt.cm.rainbow]  # set of colormaps
    cmap_handles = [Rectangle((0, 0), 1, 1) for _ in cmaps]
    handler_map = dict(zip(cmap_handles,
                           [HandlerColormap(cm, num_stripes=48) for cm in cmaps]))
    ax.legend(handles=[i_o_handle, i_s_handle, cmap_handles[0], lmbda_handle], labels=legend_inputs, handler_map=handler_map)

    ######## LEGEND #############
    #ax.legend([i_o_handle, i_s_handle, lmbda_handle, c], legend_inputs, fontsize=20)
    ax.set_xlim([0, 180])
    ax.set_ylim([0, maxy*1.1])
    ax.set_xlabel('Angles [deg]', fontsize=20)
    ax.set_ylabel('Probability density')
    ax.set_yticks([])

    ######################### colorbars ##############################

    cbaxes = fig.add_axes([0.1, 0.15, 0.02, 0.3])
    cmap = plt.get_cmap('rainbow')

    B = 9
    newcolors = cmap(np.linspace(0.08, 1, B))
    newcolors2 = cmap(np.linspace(1, 0.08, B))
    newcolors2 = newcolors2[1:]
    vals = np.ones((B + B - 1, 4))
    vals[:B, 0] = newcolors[:, 0]
    vals[B:, 0] = newcolors2[:, 0]
    vals[:B, 1] = newcolors[:, 1]
    vals[B:, 1] = newcolors2[:, 1]
    vals[:B, 2] = newcolors[:, 2]
    vals[B:, 2] = newcolors2[:, 2]

    marcus_cmap = mpl.colors.ListedColormap(vals)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=plt.Normalize(5, 175), cmap=marcus_cmap), pad=0.01, cax=cbaxes,
                        ticks=10 + 10 * np.arange(17))
    cbar.set_label('$\psi$ [degrees]', fontsize=20, labelpad=-100)

    cbaxes = fig.add_axes([0.9, 0.15, 0.02, 0.3])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=plt.get_cmap("viridis")), pad=0.01, cax=cbaxes)
    cbar.set_label('Probability density', fontsize=20, labelpad=10)

    plt.show()

    return fig


def check_params(s, lcc):

    params = [s.lmbda, s.lmbda_err1, s.lmbda_err2, s.i_o, s.i_o_err, s.vsini, s.vsini_err, s.R, s.R_err,
              lcc.found_rotation_period, lcc.found_rotation_period_err]
    params_text = ['lambda', 'lambda_err1', 'lambda_err2', 'i_o', 'i_o_err', 'vsini', 'vsini_err', 'R', 'R_err', 'P',
                   'P_err']
    needed = []

    for c, p in enumerate(params):
        if str(p) == 'nan' or p is None:
            needed.append(params_text[c])

    if len(needed) > 0:
        print('The following parameters are not defined: ' + ', '.join(needed))

        if "i_o" in needed:
            print('Assuming i_o = 90 degrees')
        if 'vsini' in needed or 'R' in needed or 'P' in needed:
            print('Need at minimum vsini, R, and P')
            return False

    if lcc.found_rotation_period == 0:
        print('Did not find a rotation period')
        return False

    return True


def obliquity(s, lcc, N=400000, bins=80):
    fig = None

    if check_params(s, lcc):
        print("found period")
        print(lcc.found_rotation_period)
        v, v_p = calc_v(s, lcc, N, bins)
        cosi, cosip, i_s, i_sp = calc_i(s, lcc, v, v_p, bins)
        psi, psip, psi_mean, psi_err, i_s_samples, lmbda_samples = calc_psi(s, cosi, cosip, N, bins)
        fig = make_plot(s, lcc, bins, v, v_p, i_s, i_sp, psi, psip, psi_mean, psi_err, i_s_samples, lmbda_samples)

    return fig



