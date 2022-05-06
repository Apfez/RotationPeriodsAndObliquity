import numpy as np
from scipy.ndimage import gaussian_filter
import lightkurve as lk
import matplotlib.pyplot as plt
import copy
import matplotlib as mpl
import pandas as pd
"""
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

sys.path.insert(0, currentdir) 
"""
import ObliquityFromLambdaPVsini as obl
import time

SMALL_SIZE = 20
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


class System(object):

    def __init__(self, name, T_eff=None, vsini=None, vsini_err=None, R=None, R_err=None,
                 transit_first_time=None, transit_duration=None, orbital_period=None, split_lc=True, override_period=None,
                 lmbda=None, lmbda_err1=None, lmbda_err2=None, i_o=np.pi/2, i_o_err=2*np.pi/180):

        self.name = name
        self.T_eff = T_eff
        self.vsini = vsini

        if str(vsini_err) == "nan" and vsini is not None:
            self.vsini_err = 0.1*vsini
        else:
            self.vsini_err = vsini_err

        self.R = R
        self.R_err = R_err
        self.transit_first_time = transit_first_time
        self.transit_duration = transit_duration
        self.orbital_period = orbital_period

        self.split_lc = split_lc
        self.lccs = None
        self.title = None
        self.exptime = None
        self.override_period = override_period
        self.handles = []
        self.names = []
        self.lmbda = lmbda
        self.lmbda_err1 = lmbda_err1
        self.lmbda_err2 = lmbda_err2
        self.i_o = i_o
        self.i_o_err = i_o_err

        if None not in (self.vsini, self.vsini_err, self.R, self.R_err) and "nan" not in (str(self.vsini), str(self.vsini_err), str(self.R), str(self.R_err)):
            self.max_rotation_period = self.calculate_max_period()

            if self.max_rotation_period < 0:
                self.max_rotation_period = np.inf
        else:
            self.max_rotation_period = np.inf

    def calculate_max_period(self):

        R_max = self.R + self.R_err * 2
        vsini_min = self.vsini - self.vsini_err * 2

        # Simple calculation to take into account uncertainty in vsini and R when finding the maximum rotation period. That is, P if v = vsini.
        return 2 * np.pi * R_max * 695700 / vsini_min  * 1 / (24 * 3600)


class lc_container(object):

    def __init__(self, lc=None):
        self.lc = lc
        self.ac_x = None
        self.ac_y = None
        self.ac_y_smoothed = None
        self.indeces = None
        self.all_periods = None
        self.x_HWHM = None
        self.y_HWHM = None

        self.found_rotation_period = None
        self.found_rotation_period_err = None
        self.gaussian_kernel_width_hours = 0


def fill_gaps(lc):

    dt = lc.time.value - np.median(np.diff(lc.time.value)) * lc.cadenceno.value

    mask = np.all([lc.cadenceno.value < lc.cadenceno.value.max(), lc.cadenceno.value < lc.cadenceno.value[0]], axis=0)
    lc.cadenceno.value[mask] = lc.cadenceno.value.max() + lc.cadenceno.value[
        mask]  # sometimes cadenceno reaches a too high value and overflows.

    ncad = np.arange(lc.cadenceno.value[0], lc.cadenceno.value[-1] + 1, 1)
    in_original = np.in1d(ncad, lc.cadenceno.value)

    if len(lc.flux) > len(in_original):
        print('could not fill gaps')
        return lc

    ncad = ncad[~in_original]
    ndt = np.interp(ncad, lc.cadenceno.value, dt)

    ncad = np.append(ncad, lc.cadenceno.value)
    ndt = np.append(ndt, dt)
    ncad, ndt = ncad[np.argsort(ncad)], ndt[np.argsort(ncad)]
    ntime = ndt + np.median(np.diff(lc.time.value)) * ncad

    f = np.zeros(len(ntime))

    f[in_original] = np.copy(lc.flux) / np.mean(np.copy(lc.flux))
    f[~in_original] = 1

    lc = lc.fill_gaps()
    lc.time = ntime
    lc.flux = f

    return lc


def ACF(ss, lcc, t, f, timestep=2):
    dt = np.nanmedian(t[1:] - t[:-1])
    N = len(f)

    maxK = np.min([int(70 / dt), int(np.floor(N / (2 * timestep)))])

    ac_x = np.zeros(maxK)
    ac_y = np.zeros(maxK)

    # could also be minus np.mean(f), but if properly normalized it should not matter
    x = f - 1

    for k in range(1, maxK + 1):
        ac_x[k - 1] = k * dt * timestep

        top = np.dot(x[0:N - k * timestep], x[k * timestep:N])
        bot = np.dot(x[0:N], x[0:N])

        ac_y[k - 1] = top / bot

    if lcc.gaussian_kernel_width_hours == 0:
        lcc.gaussian_kernel_width_hours = estimate_kernel_width(ss, ac_x, ac_y, timestep)

    expt = 3600 * lcc.gaussian_kernel_width_hours / (timestep * ss.exptime)  # gaussian kernel width in units of hours

    ac_y_smoothed = gaussian_filter(ac_y, expt)
    ac_y = gaussian_filter(ac_y, 0.01 * expt)

    return ac_x, ac_y, ac_y_smoothed


def estimate_kernel_width(ss, ac_x, ac_y, timestep):  # purely magical numbers. Don't even ask... xD
    errors = []
    max_width = np.min([ss.max_rotation_period, ac_x[-1]]) / 2 * 24

    m = 50
    id_min_interval = 1
    widths = np.linspace(max_width / m, max_width, m)
    found_min_interval = False

    for c, w in enumerate(widths):

        y = gaussian_filter(ac_y, 3600 * w / (timestep * ss.exptime))
        grad = np.gradient(y)

        # change from positive to negative gradient means a local maximum
        local_max_index = np.where(np.diff(np.sign(grad)) < 0)[0]

        maxs_x = ac_x[local_max_index]  # x-values of local minima

        if len(maxs_x) > 1:
            min_dist_between_peaks = 24 * np.min(np.abs(np.diff(maxs_x)))
        else:
            min_dist_between_peaks = np.inf

        error = np.sum((y - ac_y) ** 2) / len(y)
        errors.append(error)

        # fig, ax = plt.subplots()
        # ax.plot(ac_x, y)
        # ax.set_title('w: ' + str(w))

        if len(maxs_x) == 0:
            peak_num = 0.0001
        else:
            peak_num = len(maxs_x)

        # print("min peak dists" + str(min_dist_between_peaks))
        # print("avg peak dist" + str((ac_x[-1] * 24) / peak_num))
        # print("rel error" + str(error / errors[0]))

        if min_dist_between_peaks > 12 and (ac_x[
                                                -1] * 24) / peak_num > 24 and found_min_interval is False:  # found estimate when: A) no two peaks closer than 20 hours apart and B) the average peak distance longer than 1d and C) the error between smoothed and unsmoothed is at least twice the error of the first run (1h gaussian kernel)
            id_min_interval = np.max([1, c - 1])
            found_min_interval = True
            # print("minimum width due to min peak dist and avg peak dists (hours): " + str(widths[id_min_interval]))

    Y_frac = errors[-1] / errors[0]
    # err0_min = np.max([errors[0], 0.02*Y_frac])

    rel_err = errors[0] + 0.5 * (errors[-1] - errors[0]) / np.sqrt(Y_frac)
    rel_err = np.min([rel_err, errors[0] + 0.6 * (errors[-1] - errors[0])])
    min_error_index = np.argmin(np.abs(errors - rel_err))
    est = np.max([widths[min_error_index], widths[id_min_interval]])

    # fig, ax = plt.subplots()
    # ax.plot(widths, errors)
    # ax.scatter(widths[min_error_index], errors[min_error_index], c='r')
    # plt.show()
    # print('last rel error: ' + str(errors[-1]/errors[0]))
    # print('est 1 (relative error): ' + str(widths[min_error_index]))
    # print('est 2 (distance between peaks): ' + str(widths[id_min_interval]))
    print("Estimated kernel width = %1.1fh" % est)
    return est


def find_rotation_period(ss, lcc, x=None, y=None):
    if x is None:
        x = lcc.ac_x
    if y is None:
        y = lcc.ac_y_smoothed

    grad = np.gradient(y)

    # change from positive to negative gradient means a local maximum
    local_max_index = np.where(np.diff(np.sign(grad)) < 0)[0]
    local_min_index = np.where(np.diff(np.sign(grad)) > 0)[0]

    if len(local_max_index) == 0:
        return 0, 0, 0, 0, 0, 0, 0

    mins_x = x[local_min_index]  # x-values of local minima
    mins_y = y[local_min_index]  # y-values of local minima
    maxs_x = x[local_max_index]
    maxs_y = y[local_max_index]

    highest_peak = 0
    P = 0
    rotation_period_error = 0
    rotP_err = 0
    n_peak = 0
    buffer = 1.45

    # run through all maxima and find the one with highest relative peak height
    for n in range(len(maxs_x)):

        if ss.override_period is not None and maxs_x[n] > 1 / buffer * ss.override_period and maxs_x[
            n] < buffer * ss.override_period:
            relative_peak_height, rotP_err, x_HW, y_HW = h_p(maxs_x[n], maxs_y[n], mins_x, mins_y, x, y)

            P = maxs_x[n]
            n_peak = n
            highest_peak = relative_peak_height
            rotation_period_error = rotP_err
            lcc.x_HWHM, lcc.y_HWHM = x_HW, y_HW

            break

        # only check if period is physically possible given vsini and v
        if maxs_x[n] < 1.05*ss.max_rotation_period:
            relative_peak_height, rotP_err, x_HW, y_HW = h_p(maxs_x[n], maxs_y[n], mins_x, mins_y, x, y)

            if relative_peak_height > highest_peak:
                P = maxs_x[n]
                n_peak = n
                highest_peak = relative_peak_height
                rotation_period_error = rotP_err
                lcc.x_HWHM, lcc.y_HWHM = x_HW, y_HW

    if rotP_err == 0:
        rotation_period_error = 0.1 * P

    i = [0, 1]
    all_periods = [0, P]
    rotation_period = P
    newP = []

    if len(maxs_x) != 1:
        maxs_x = np.append(maxs_x,1000000)  # kind of hacky, but prevents a lot of if's, that would need to be there, to use the last peak

        for x in maxs_x[
                 n_peak + 1:]:  # then go through all maxima again, starting from the second, now finding all peaks that lie approximately an integer times the period

            lower_limit = 1 / buffer * rotation_period
            upper_limit = buffer * rotation_period

            dist_to_prev_peak = x - all_periods[-1]

            if dist_to_prev_peak < lower_limit:
                continue
            elif dist_to_prev_peak > lower_limit and dist_to_prev_peak < upper_limit and x != maxs_x[-1]:
                newP.append(x)

            elif dist_to_prev_peak > upper_limit or x == maxs_x[-1]:
                if len(newP) == 0:  # if not last peak and no new period were found in the interval, stop looking for more periods

                    break
                else:
                    i.append(i[-1] + 1)  # if a new period was found further ahead, add the previous period and integers to the list of all periods

                    if len(newP) > 1:
                        idx = np.abs(newP - len(all_periods) * rotation_period).argmin()
                        all_periods.append(newP[idx])
                    else:
                        all_periods.append(newP[0])

                if len(all_periods) == 2:
                    rotation_period = P

                elif len(all_periods) > 2:
                    z, cov = np.polyfit(i, all_periods, 1, cov=True)
                    rotation_period = z[0]
                    rotation_period_error = np.sqrt(np.diag(cov))[0]

                if x - all_periods[-1] < 1 / buffer * rotation_period:
                    newP = []
                elif x - all_periods[-1] > 1 / buffer * rotation_period and x - all_periods[
                    -1] < buffer * rotation_period:
                    newP = [x]
                else:

                    break

        maxs_x = maxs_x[:-1]

    return rotation_period, rotation_period_error, maxs_x, maxs_y, highest_peak, i, all_periods


def FWHM(X, Y, half_point):
    d = Y - half_point
    indexes = np.where(d > 0)[0]

    if len(indexes) == 0:
        res = 0
    else:
        res = abs(X[indexes[-1]] - X[indexes[0]])

    return res


def h_p(max_x, max_y, mins_x, mins_y, x, y):
    amount = 0
    left_min_y = 0
    right_min_y = 0
    mask_left = mins_x < max_x
    mask_right = mins_x > max_x

    if mask_left.sum() > 0:

        left_min_x = np.max(mins_x[mask_left])
        left_min_y = mins_y[list(mins_x).index(left_min_x)]
        amount += 1

        left_condition = x > left_min_x
    else:
        left_condition = x > max_x

    if mask_right.sum() > 0:

        right_min_x = np.min(mins_x[mask_right])
        right_min_y = mins_y[list(mins_x).index(right_min_x)]
        amount += 1

        right_condition = x < right_min_x
    else:
        right_condition = x < max_x

    if amount == 0:
        return 0, 0, 0, 0
    bottom = (left_min_y + right_min_y) / amount
    hp = max_y - bottom

    total_condition = np.logical_and(left_condition, right_condition)

    HW = 0.5 * FWHM(x[total_condition], y[total_condition], bottom + 0.5 * hp)

    return hp, HW, x[total_condition], y[total_condition]


def plot_LC_and_ACF(ss, lcc, maxs_x, maxs_y, lc_original, mask):
    fig = plt.figure()
    fig.set_size_inches(18, 25)

    gs = fig.add_gridspec(16, 7, hspace=4, wspace=0.35)
    # plt.subplots_adjust(hspace = 1)

    ###############################################
    # ax0:

    ax0 = fig.add_subplot(gs[0:3, -1:])

    if lc_original is not None and mask is not None:
        ax0.scatter(lc_original[mask].time.value, lc_original[mask].flux.value, s=1, color='r', alpha=0.2)

    ax0.scatter(lcc.lc.time.value, lcc.lc.flux.value, s=1, alpha=0.2)
    ax0.set_title('Sections overview')

    bracket = int(0.01 * len(lcc.lc.flux.value))
    maxf = np.max(np.sort(lcc.lc.flux.value)[-bracket])
    minf = np.min(np.sort(lcc.lc.flux.value)[::-1][-bracket])
    # ax0.set_ylim([minf - 0.3*(maxf - minf), maxf + 0.3*(maxf - minf)])
    # ax0.set_ylim([])
    ax0.set_xlim([lcc.lc.time.value[0], lcc.lc.time.value[-1]])
    ax0.set_yticks([])
    ax0.set_xlabel('Time [days]')

    ###############################################
    # ax1:

    ax1 = fig.add_subplot(gs[0:3, :-1])
    ax1.set_title(ss.name + ' lightcurve cutout, ' + ss.title)

    if lc_original is not None and mask is not None:
        ss.handles.append(ax1.scatter(lc_original[mask].time.value, lc_original[mask].flux.value, s=3, color='r'))
        ss.names.append('Removed data points')

    ss.handles.append(ax1.scatter(lcc.lc.time.value, lcc.lc.flux.value, s=4, alpha=0.4))
    ss.names.append('Data points (exptime = %1.0fs)' % ss.exptime)

    binned_lc = copy.copy(lcc.lc.bin)(time_bin_size=10 * ss.exptime / (24 * 3600))

    bt = binned_lc.time.value
    bf = binned_lc.flux.value
    t = lcc.lc.time.value

    tim = t[0]
    c = 1

    def get_col(i):

        if i == 1:
            return plt.cm.Paired(6)
        else:
            return plt.cm.Paired(7)

    if lcc.found_rotation_period > 0:
        while tim < t[-1] - lcc.found_rotation_period:
            c = c*-1
            start_index = (np.abs(bt - tim)).argmin()

            end_index = (np.abs(bt - tim - lcc.found_rotation_period)).argmin()

            ax1.scatter(bt[start_index:end_index], bf[start_index:end_index], s=15, alpha=1, color=get_col(c))
            tim += lcc.found_rotation_period
        c = c * -1

        ss.handles.append(ax1.scatter(bt[(np.abs(bt - tim)).argmin():], bf[(np.abs(bt - tim)).argmin():], s=15, alpha=1, color=get_col(c)))

    else:

        ss.handles.append(ax1.scatter(binned_lc.time.value, binned_lc.flux.value, s=14, color='k'))

    ss.names.append('Binned data points (binsize = %1.0fs) ' % (10 * ss.exptime))

    maxx1 = np.min([lcc.lc.time.value[0] + 40, lcc.lc.time.value[-1]])
    ax1.set_ylim([minf - 0.3 * (maxf - minf), maxf + 0.3 * (maxf - minf)])
    ax1.set_xlim([lcc.lc.time.value[0], maxx1])
    ax1.set_xlabel('Time [days]')
    ax1.set_ylabel('Flux, normalized')


    ###############################################
    # ax2:

    ax2 = fig.add_subplot(gs[3:7, :])

    h, = ax2.plot(lcc.ac_x, lcc.ac_y, color='g', linewidth=1, alpha=0.75)
    ss.handles.append(h)
    ss.names.append('ACF')

    h, = ax2.plot(lcc.ac_x, lcc.ac_y_smoothed, linewidth=4, color='k')
    ss.handles.append(h)
    ss.names.append('Smoothed ACF')

    draw_extra_rotation_lines(ss, ax2)

    A = 0.08 * (max(lcc.ac_y_smoothed) - min(lcc.ac_y_smoothed))

    if lcc.found_rotation_period != 0:

        for k, x in enumerate(maxs_x):

            if x in lcc.all_periods:
                c = 'orange'
                h, = ax2.plot([x, x], [maxs_y[k] - A,
                                       maxs_y[k] + A], color=c, linewidth=3)
                if 'Local maxima used to calculate $P_{rot}$' not in ss.names:
                    ss.handles.append(h)
                    ss.names.append('Local maxima used to calculate $P_{rot}$')

            else:
                c = 'blue'
                h, = ax2.plot([x, x], [maxs_y[k] - A,
                                       maxs_y[k] + A], color=c, linewidth=3)
                if 'Local maxima' not in ss.names:
                    ss.handles.append(h)
                    ss.names.append('Local maxima')

        h, = ax2.plot([lcc.found_rotation_period, lcc.found_rotation_period],
                      [max(lcc.y_HWHM) - A, max(lcc.y_HWHM) + A], color='r', linewidth=4)
        ss.handles.append(h)
        ss.names.append('Determined period')

        ax2.legend(ss.handles, ss.names, bbox_to_anchor=(1.02, 0.75))
        ax2.set_title('ACF')
        ax2.set_xlabel(r"$\tau_k$ [days]")
        ax2.set_ylabel('$r_k$')
        ax2.set_ylim([min(lcc.ac_y_smoothed) - A * 3, max(lcc.ac_y_smoothed) + A * 3])

        maxx2 = np.min([max(lcc.ac_x), 60, lcc.found_rotation_period * 15])
        ax2.set_xlim([min(lcc.ac_x), maxx2])


        ###############################################
        # ax3:

        ax3 = fig.add_subplot(gs[7:10, :-2])
        ax3.set_title('Period determination')

        if len(lcc.all_periods) > 2:

            ax3.scatter(lcc.all_periods, lcc.indeces, color='orange')

            x = np.linspace(0, np.max(lcc.indeces), 100)
            ax3.plot(lcc.found_rotation_period * x, x, color='purple')

            ax3.set_xlim([min(lcc.ac_x), maxx2])
            ax3.set_ylabel('index of local maximum')
            ax3.set_xlabel(r"$\tau_k$ [days] of local maximum")
            ax3.text(0.5*np.max(lcc.indeces), 1.2, "$P_{rot} = $%1.2f$\pm$%1.2f" % (
                lcc.found_rotation_period, lcc.found_rotation_period_err))

        else:
            ax3.plot(lcc.x_HWHM, lcc.y_HWHM, color='k', linewidth=4)
            midpoint = min(lcc.y_HWHM) + 0.5 * (max(lcc.y_HWHM) - min(lcc.y_HWHM))
            ax3.plot([lcc.found_rotation_period, lcc.found_rotation_period], [
                max(lcc.y_HWHM) - A, max(lcc.y_HWHM) + A], color='r', linewidth=3)
            ax3.plot([lcc.found_rotation_period, lcc.found_rotation_period +
                      lcc.found_rotation_period_err], [midpoint, midpoint], color='C0', linestyle='--')
            ax3.plot([lcc.found_rotation_period - lcc.found_rotation_period_err, lcc.found_rotation_period -
                      lcc.found_rotation_period_err], [midpoint, max(lcc.y_HWHM)], color='C0')
            ax3.plot([lcc.found_rotation_period + lcc.found_rotation_period_err, lcc.found_rotation_period +
                      lcc.found_rotation_period_err], [midpoint, max(lcc.y_HWHM)], color='C0')
            ax3.text(lcc.found_rotation_period + 0.5 * lcc.found_rotation_period_err, max(lcc.y_HWHM) + 0.5 * A,
                     "$P_{rot} = $%1.2f$\pm$%1.2f" % (lcc.found_rotation_period, lcc.found_rotation_period_err))

        ###############################################
        # ax4:

        ax4 = fig.add_subplot(gs[7:10, -2:])
        ax4.set_title('Folded lightcurve')

        folded = binned_lc.fold(period=lcc.found_rotation_period)
        ax4.scatter(folded.time.value, folded.flux.value, s=4, alpha=1)
        ax4.plot([min(folded.time.value), max(folded.time.value)], [1, 1], color='r', linestyle='--', linewidth=2)
        ax4.set_xlim([min(folded.time.value), max(folded.time.value)])
        ax4.set_yticks([])
        ax4.set_xlabel('Time [days]')
        plt.show()

        return fig


def Jackknives(ss, lcc):
    fig = plt.figure()

    fig.set_size_inches(18, 24)
    gs = fig.add_gridspec(4, 6, hspace=0.3, wspace=0.8)
    ax1 = fig.add_subplot(gs[:2, 2:])
    ax2 = fig.add_subplot(gs[:2, :2])
    ax3 = fig.add_subplot(gs[2:3, :3])
    ax4 = fig.add_subplot(gs[2:3, 3:])
    ax5 = fig.add_subplot(gs[3:, :])

    ACF_remove_increments(ss, lcc, ax1, ax2, ax3, ax4)
    ACF_small_increments(ss, lcc, ax5, fig)
    plt.show()


def ACF_small_increments(ss, lcc, ax, fig):
    def colorline(x, y, ax, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):

        def make_segments(x, y):

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            return segments

        # Default colors equally spaced on [0,1]:
        if z is None:
            z = np.linspace(0.0, 1.0, len(x))

        # Special case if a single number:
        if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
            z = np.array([z])

        z = np.asarray(z)

        segments = make_segments(x, y)
        currentvals = cmap

        N = len(segments)
        vals = np.ones((N, 4))
        vals[:, 0] = currentvals[0]
        vals[:, 1] = currentvals[1]
        vals[:, 2] = currentvals[2]
        vals[:, -1] = np.linspace(alpha, 0, N)

        newcmp = mpl.colors.ListedColormap(vals)
        lc = mpl.collections.LineCollection(segments, array=z, cmap=newcmp, norm=norm,
                                            linewidth=linewidth)

        ax.add_collection(lc)

        return lc

    resolution = 11

    total_time = lcc.lc.time.value[-1] - lcc.lc.time.value[0]

    cmap = plt.get_cmap('inferno', 10 * resolution)

    fractions = []
    periods = []

    def make_lines(fraction, amount, m):

        for k in range(0, amount + 1):
            end = (1 - fraction) * total_time * fraction

            progress = k / amount
            t_high = lcc.lc.time.value[0] + total_time * fraction + progress * end
            t_low = lcc.lc.time.value[0] + progress * end
            t = lcc.lc.time.value[lcc.lc.time.value < t_high]
            mask = t > t_low
            t = t[mask]

            f = lcc.lc.flux.value[lcc.lc.time.value < t_high]
            f = f[mask]
            [x, y, yb] = ACF(ss, lcc, lcc.lc.time.value, f, timestep=10)

            P, Perr, maxs_x, maxs_y, highest_peak, i, all_P = find_rotation_period(
                ss, lcc, x, yb)
            periods.append(P)
            fractions.append(fraction)

            start = len(yb) - 40
            ax.plot(x[:start], yb[:start], linewidth=3, alpha=0.4, c=cmap(m * resolution + k))

            path = mpl.path.Path(np.column_stack([x[start:], yb[start:]]))
            verts = path.interpolated(steps=3).vertices
            x, y = verts[:, 0], verts[:, 1]
            z = np.linspace(0, 1, len(x))
            colorline(x, y, ax, z, cmap=cmap(m * resolution + k), linewidth=3, alpha=0.3)

    for m in range(10):
        fraction = 0.3 + (1 - 0.4) * m / 9
        make_lines(fraction, resolution, m)

    l, = ax.plot(lcc.ac_x, lcc.ac_y, linewidth=1, c='green', alpha=0.65)

    l2, = ax.plot(lcc.ac_x, lcc.ac_y_smoothed, linewidth=4, c='black')

    ax.set_xlabel(r"$\tau_k$ [days]")
    ax.set_ylabel("ACF")

    maxx2 = np.min([max(lcc.ac_x), 60, lcc.found_rotation_period * 15])
    ax.set_xlim([0, maxx2])
    ax.grid(False)
    ax.legend([l, l2], ["ACF", "Smoothed ACF"])

    cmap = mpl.cm.get_cmap('inferno', 9)
    norm = mpl.cm.colors.Normalize(vmin=0.3, vmax=0.9)
    cbaxes = fig.add_axes([0.907, 0.12, 0.015, 0.17])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01, cax=cbaxes)
    cbar.set_label('Fraction of lightcurve used', labelpad=10)


def ACF_remove_increments(ss, lcc, ax, ax2, ax7, ax8):
    total_time = lcc.lc.time.value[-1] - lcc.lc.time.value[0]

    cmap = plt.get_cmap('viridis', 11)  # ,resolution+2

    def make_lines(fraction, amount, m):

        periods = []
        hps = []

        bracket = int(0.01 * len(lcc.lc.flux.value))
        maxf = np.max(np.sort(lcc.lc.flux.value)[-bracket])
        maxA = 0.6 * (np.max(lcc.ac_y_smoothed) - np.min(lcc.ac_y_smoothed))

        for k in range(0, amount + 1):

            progress1 = k / (amount) * (1 - fraction)
            progress2 = progress1 + fraction  # (k+1)/(1+amount)

            t_low = lcc.lc.time.value[0] + progress1 * total_time
            t_high = lcc.lc.time.value[0] + progress2 * total_time

            mask = np.logical_and(lcc.lc.time.value > t_low,
                                  lcc.lc.time.value < t_high)
            f = np.copy(lcc.lc.flux.value)
            f[mask] = 1

            # a bit less precise to save time
            [x, y, yb] = ACF(ss, lcc, lcc.lc.time.value, f, timestep=10)

            P, Perr, maxs_x, maxs_y, highest_peak, i, all_P = find_rotation_period(
                ss, lcc, x, yb)
            periods.append(P)
            hps.append(highest_peak)

            if m == 7:

                A = 0.07 * (max(lcc.ac_y_smoothed) - min(lcc.ac_y_smoothed))

                for n in range(len(maxs_x)):
                    b, = ax.plot([maxs_x[n], maxs_x[n]], [
                        maxs_y[n] - A + maxA * k, maxs_y[n] + A + maxA * k], color='orange', zorder=11)

                ax.plot([0, x[-1]], [maxA * k, maxA * k],
                        linestyle='--', color='k')
                ax.plot(x, yb + maxA * k, linewidth=6, c=cmap(k))
                ax.plot([P, P], [maxA * k - A * 2 + highest_peak / 2, maxA * k + A *
                                 2 + highest_peak / 2], linewidth=5, color='red', zorder=10)

                ax2.scatter(lcc.lc.time.value, (f - 1) + 3.3 * (maxf - 1)
                            * k, s=2, color='C0', alpha=0.3)

        return periods, hps

    ax.set_title("ACFs of lightcurve with differently placed cuts")
    ax.set_xlabel(r"$\tau_k$ [days]")
    ax.set_ylabel('r$_k$')

    maxx2 = np.min([max(lcc.ac_x), 60, lcc.found_rotation_period * 15])
    ax.set_xlim([0, maxx2])

    ax2.set_title(ss.name + " lightcurves with 24% cuts")
    ax2.set_xlabel("Time [BKJD days]")
    ax2.set_ylabel("Flux [normalized]")

    highestA = 0

    for m in range(10):
        fraction = 0.03 * (m + 1)
        Ps, As = make_lines(fraction, 10, m)

        if np.max(As) > highestA:
            highestA = np.max(As)

    cmap2 = plt.get_cmap("plasma")

    for m in range(10):
        fraction = 0.03 * (m + 1)
        Ps, As = make_lines(fraction, 10, m)

        for n in range(len(Ps)):
            A = np.asarray(As)[n]
            ax7.scatter(fraction, np.asarray(
                Ps)[n], edgecolor='k', linewidth=3, color=cmap2(A / highestA), alpha=0.6, s=160)
            ax8.scatter(fraction, A, edgecolor='k', linewidth=3,
                        color=cmap2(A / highestA), alpha=0.6, s=160)

    ax8.set_ylim([0, highestA * 1.05])
    ax7.set_xlim([0, 0.32])
    ax8.set_xlim([0, 0.32])

    ax7.set_xlabel('Fraction cut')
    ax8.set_xlabel('Fraction cut')
    ax7.set_ylabel('Found rotation period [days]')
    ax8.set_ylabel('Relative peak height h$_p$')


def make_lcs(results, gaussian_kernel_width_hours, split_lc=True):
    section_size = 0
    section = []
    lcs = []
    l_prev = []

    if split_lc:
        if len(results) > 1:
            for n, r in enumerate(results):
                l = r.download()
                l.remove_nans().normalize()

                if section_size == 0:
                    if n == len(results) - 1:
                        lcs.append(lc_container(l.remove_nans().normalize()))
                    else:
                        section_size = 1
                        section.append(l)
                elif l.meta["SECTOR"] - section[-1].meta[
                    "SECTOR"] < 5:  # if there are less than 5 gaps between sectors:
                    section.append(l)

                    if n == len(results) - 1:
                        lccol = lk.LightCurveCollection(section)
                        lcs.append(lc_container(lccol.stitch().remove_nans().normalize()))

                    else:

                        section_size = section_size + 1
                else:  # if there is too far between sectors

                    if section_size == 1:  # only of sector in this section, so simply lightcurve
                        lcs.append(lc_container(l_prev.remove_nans().normalize()))  # add the previous lc to all lcs.

                        if n == len(results) - 1:
                            lcs.append(lc_container(l.remove_nans().normalize()))

                        else:
                            section = [l]  # and start a new section with this new lc
                    else:  # if there were several lcs in this sector,
                        if n == len(results) - 1:

                            lccol = lk.LightCurveCollection(section)
                            lcs.append(lc_container(lccol.stitch().remove_nans().normalize()))

                            lcs.append(lc_container(l.remove_nans().normalize()))

                        else:
                            lccol = lk.LightCurveCollection(section)
                            lcs.append(lc_container(lccol.stitch().remove_nans().normalize()))

                            section = [l]
                            section_size = 1  # and make the new section size 0

                l_prev = l
        else:
            lcs.append(lc_container(results.download().remove_nans().normalize()))
    else:
        lcs.append(lc_container(results.download_all().stitch().remove_nans().normalize()))

    for lc in lcs:
        lc.gaussian_kernel_width_hours = gaussian_kernel_width_hours

    return lcs


def Rvar(f):
    sf = np.sort(f)
    N = len(f)

    topindex = int(N * 0.95)
    bottomindex = int(N * 0.05)

    return sf[topindex] - sf[bottomindex]


def draw_max_rotation_arrow(ax, ss):

    xmax = ax.get_xlim()[-1]
    if ss.max_rotation_period < xmax:
        trans2 = ax.get_xaxis_transform()
        txt2 = "$P_{rot,max}$= %1.1fd"%ss.max_rotation_period
        ax.annotate(txt2, size=18,
                    xy=(ss.max_rotation_period, 0), xycoords=trans2,
                    xytext=(0, -20), textcoords='offset points',
                    horizontalalignment='center', verticalalignment='top',
                    arrowprops=dict(arrowstyle="->", facecolor='red'))


def vertical_line(x, ax, kwargs=None):

    xmin, xmax = ax.get_xbound()
    ymin, ymax = ax.get_ybound()

    l = mpl.lines.Line2D([x, x], [ymin, ymax], **kwargs)
    ax.add_line(l)

    return l


def draw_extra_rotation_lines(ss, ax):

    xmax = ax.get_xlim()[-1]

    if ss.orbital_period is not None and ss.orbital_period < xmax:

        kwargs = {'color': 'pink', 'linewidth': 3, 'alpha': 1, 'linestyle' : 'dotted'}
        ss.handles.append(vertical_line(ss.orbital_period, ax, kwargs))
        ss.names.append("Orbital period = %1.1fd" % ss.orbital_period)

    if ss.max_rotation_period < xmax:

        ax.axvspan(ss.max_rotation_period,xmax,facecolor = '0.2',alpha = 0.4)

        kwargs = {'color': 'black','linewidth':2, 'linestyle': '--'}

        ss.handles.append(vertical_line(ss.max_rotation_period, ax, kwargs))
        ss.names.append("Max rotation period (given vsini and R) =  %1.1fd" % ss.max_rotation_period)

        draw_max_rotation_arrow(ax, ss)

    if ss.T_eff is not None and 5900 < ss.T_eff < 6600:

        def Louden_v(T):
            tau = (T - 6250) / 300

            c0 = 9.57
            c1 = 8.01
            c2 = 3.3

            v = c0 + c1 * tau + c2 * tau ** 2

            return v

        v_spocs = Louden_v(ss.T_eff)

        P_spocs = 2*np.pi*ss.R*695700/v_spocs * 1/(24*3600)

        kwargs3 = {'color': 'purple', 'linewidth': 3, 'alpha': 1, 'linestyle' : '-.'}
        ss.handles.append(vertical_line(P_spocs, ax, kwargs3))
        ss.names.append("Louden rotation period = %1.1fd" % P_spocs)


def get_Unconfirmed_TOIs(prioritylim=3):
    df = pd.read_csv('TOI_candidates.csv')
    df['sy_gaiamag'] = df.iloc[:]['st_tmag']

    dffop = pd.read_csv('exofop_tess_tois.csv')

    priorities = np.ones(len(df)) * 5
    mets = np.zeros(len(df))
    loggs = np.ones(len(df))
    names = []

    for x in range(len(df)):
        # print(df.iloc[x]['toi'])
        names.append("TOI-" + str(int(df.iloc[x]['toi'])))
        idx = np.where(np.array(dffop['TOI']) == df.iloc[x]['toi'])

        if (len(idx)) == 1:
            id_here = idx[0][0]
            priorities[x] = dffop.iloc[id_here]['Master priority']
            mets[x] = dffop.iloc[id_here]['Stellar Metallicity']
            loggs[x] = dffop.iloc[id_here]['Stellar log(g) (cm/s2)']

    df['Master priority'] = priorities
    df['st_met'] = mets
    df['st_logg'] = loggs
    df['hostname'] = names

    mask = df.iloc[:]['Master priority'] <= prioritylim
    df = df[mask]

    ids = np.array(df['tid'])

    multies = np.zeros(len(df))

    for d in range(len(df)):
        for e in range(len(df)):

            if d == e:
                continue
            if ids[d] == ids[e]:
                multies[d] = 1

    df['multi'] = multies
    df['tic_id'] = "TIC " + df['tid'].astype(str)

    return df


def get_confirmed():
    df = pd.read_csv('All_confirmed_planets_extended_default.csv')
    df2 = pd.read_csv('All_confirmed_planets_extended.csv')
    df2 = df2.groupby('pl_name', as_index=False).mean()
    df.fillna(df2, inplace=True)  # use default parameters, but if values are missing from the default parameters, use the mean of alle other values.
    return df


def get_system(name):

    df = get_confirmed()
    TOIs = get_Unconfirmed_TOIs(5)

    tics = list(df['tic_id'])
    for t in range(len(tics)):
        if str(tics[t]) == "nan":
            tics[t] = 0
        else:
            tics[t] = int(tics[t].replace('TIC ', ''))

    df['tid'] = np.array(tics)

    idcs = []
    for n in range(len(TOIs)):  # this is to make sure there are no overlapping systems between all planets and unconfirmed planets
        if TOIs.iloc[n]['tid'] in np.array(tics):
            idcs.append(TOIs.index[n])

    TOIs.drop(np.array(idcs), inplace=True)
    df = pd.concat([df, TOIs], ignore_index=True)

    mask = np.any([
        df['hostname'] == name,
        df['tic_id'] == name
    ], axis=0)

    rt = pd.read_csv('planets_review_table.csv')

    if mask.sum() > 1:
        sys = df.loc[mask].iloc[0]
    if mask.sum() == 1:
        sys = df.loc[mask]
    if mask.sum() == 0:
        print(name + " not recognized. Is this a planet host? If so, try a different name ")
        sys = None

    if sys is None:
        ss = System(name)
    else:
        ss = System(name, T_eff=float(sys['st_teff']),
                    orbital_period=float(sys['pl_orbper']), transit_duration=float(sys['pl_trandur'])/24,
                    R=float(sys['st_rad']), R_err=float(sys['st_raderr1']),
                    vsini=float(sys['st_vsin']), vsini_err=float(sys['st_vsinerr2']),
                    i_o=float(sys['pl_orbincl'])*np.pi/180, i_o_err=float(sys['pl_orbinclerr1'])*np.pi/180)

    for n in range(len(rt)):
        if rt.loc[n]['system'] == name:
            ss.lmbda = rt.loc[n]['lam_ori']*np.pi/180
            ss.lmbda_err1 = rt.loc[n]['lam_ori_down']*np.pi/180
            ss.lmbda_err2 = rt.loc[n]['lam_ori_up']*np.pi/180
            ss.vsini = rt.loc[n]['vsini']
            ss.vsini_err = rt.loc[n]['vsini_up']

    return ss


def transit_mask(ss, lcc):

    original_lc, mask = None, None

    period = np.linspace(ss.orbital_period * 0.9, ss.orbital_period * 1.1, 1000)
    bls = lcc.lc.to_periodogram(method='bls', period=period, frequency_factor=1500);
    ss.transit_first_time = bls.transit_time_at_max_power

    if ss.transit_duration is None or str(ss.transit_duration) == 'nan':
        ss.transit_duration = bls.duration_at_max_power * 1.5

    if None not in (ss.orbital_period, ss.transit_first_time, ss.transit_duration):

        original_lc = copy.copy(lcc.lc)
        mask = lcc.lc.create_transit_mask(ss.orbital_period, ss.transit_first_time, ss.transit_duration)
        lcc.lc = lcc.lc[~mask]

    return original_lc, mask


def Rotation_period(ss, lcc, jackknives=False):
    original_lc, mask = None, None
    ss.handles = []
    ss.names = []

    #lcc.lc = lcc.lc.remove_outliers(4)

    if ss.orbital_period is not None:
        original_lc, mask = transit_mask(ss, lcc)

    lcc.lc = fill_gaps(lcc.lc)
    lcc.ac_x, lcc.ac_y, lcc.ac_y_smoothed = ACF(ss, lcc, lcc.lc.time.value, lcc.lc.flux.value)
    lcc.found_rotation_period, lcc.found_rotation_period_err, maxs_x, maxs_y, highest_peak, lcc.indeces, lcc.all_periods = find_rotation_period(
        ss, lcc)

    fig = plot_LC_and_ACF(ss, lcc, maxs_x, maxs_y, original_lc, mask)

    print("Rvar = " + str(Rvar(lcc.lc.flux.value)))
    print("h_p = " + str(highest_peak))

    if ss.orbital_period is not None and lcc.found_rotation_period != 0 and np.abs(ss.orbital_period/lcc.found_rotation_period - 1) < 0.1:
        print('Careful, found period is close to orbital period')

    if jackknives:
        Jackknives(ss, lcc)

    return ss, fig


def get_lightcurves_and_rotation_period(n, obliquity=True, gaussian_kernel_width_hours=0, jackknives=False):  # First check for SPOC light curve, if not available, use Full Frame Images instead.
    ss = get_system(n)
    figs = []
    authors = ['SPOC', 'TESS-SPOC', 'QLP']
    exptimes = [120, None, None]
    titles = ['SPOC', 'TESS-SPOC (from Full-Frame Images)', 'QLP (MIT Quick-Look Pipeline from FFI)']

    for c, a in enumerate(authors):
        name = copy.copy(ss.name)

        if ss.name[0:3] == "TOI":
            name = ss.name + ".01"

        results = lk.search_lightcurve(name, mission="TESS", author=a, exptime=exptimes[c])

        if len(results) != 0:
            ss.exptime = results[0].exptime[0].value
            ss.title = titles[c]

            if ss.T_eff is not None:
                ss.title += ". T = %1.0fK" % ss.T_eff

            if ss.lmbda is not None:
                ss.title += ". $\lambda = %1.1f^{+%1.1f^\circ}_{-%1.1f^\circ}$" % (ss.lmbda*180/np.pi, ss.lmbda_err1*180/np.pi, ss.lmbda_err2*180/np.pi)

            ss.lccs = make_lcs(results, gaussian_kernel_width_hours, ss.split_lc)

            for lcc in ss.lccs:
                s, f = Rotation_period(ss, lcc, jackknives)
                figs.append(f)

                if obliquity:
                    figs.append(obl.obliquity(s, lcc))

            if c == 0:  # if SPOC lightcurve is available, don't look for FFI lightcurves
                break
    return ss, figs


ss, figs = get_lightcurves_and_rotation_period('WASP-12', obliquity=True, jackknives=True)

for c, f in enumerate(figs):
    if f is not None:
        f.savefig(ss.name + "v" + str(c+1) + '.png')
