# simulations are performed by creating a your_config.toml file including the parameters that you want (above the definition of the main() function is a list of possible config parameters)
# and then executing "python3 protein_simulation_prw_exp_config.py your_config.toml" in the command line (compatibility tested with python 3.9.7)

from numba import njit
import numpy as np
from scipy import interpolate
from motility import msd_fast
import sys
import tidynamics

# for loading config
import toml
# for parsing config
import pydantic
# extracting MFPTs from trajectories
from mfpts import *
# for MSD calculation and correlations
from correlation_mkl import correlation


def msd_fft(x):
    ''' compute MSD from trajectory x via fft (position correlation)'''
    mu = np.mean(x)
    N = len(x)
    D = np.square(x - mu)
    D = np.append(D, 0) 
    pos_corr = correlation(x - mu)
    Q = 2 * np.sum(D)
    running_av_sq = np.zeros(N)
    for m in range(N):
        Q = Q - D[m-1] - D[N-m]
        running_av_sq[m] = Q / (N - m)
    return running_av_sq - 2 * pos_corr

def msd_fast(x, y=[], z=[], less_memory=False):
    # choose which MSD function to use and check dimensionality
    if less_memory:
        msd_fun = msd_fft # a bit slower but uses less memory
    else:
        msd_fun = tidynamics.msd
    if len(z) > 0:
        # 3D
        msd = msd_fun(x) + msd_fun(y) + msd_fun(z)
        
    elif len(y) > 0:
        # 2D
        msd = msd_fun(x) + msd_fun(y)
    else:
        # 1D
        msd = msd_fun(x)
    return msd


def spline_fe_for_sim(
    pos,
    fe,
    dx=10,
    der=1,
    add_bounds_right=False,
    start_right=0,
    add_bounds_left=False,
    start_left=0,
):
    ''' find out at which position the trajectory is and which free energy exerts a force for GLE simulations '''
    dxf = (pos[1] - pos[0]) / dx
    fe_spline = interpolate.splrep(pos, fe, s=0, per=0)
    force_bins = np.arange(pos[0], pos[-1], dxf)
    force_matrix = interpolate.splev(force_bins, fe_spline, der=der)

    if add_bounds_right:
        # add quadratic boundary at ends

        # xfine_new = np.arange(pos[0],pos[-1]*1.5,dxf)
        xfine_new = np.arange(pos[0], pos[-1], dxf)
        xfine_right = xfine_new[len(force_bins) - start_right :]
        force_fine_right = (xfine_right - force_bins[-start_right]) ** 2 * (dxf) * 1e9
        force_fine_right += force_matrix[-start_right]
        force_matrix = np.append(force_matrix[:-start_right], force_fine_right)
        force_bins = xfine_new

    if add_bounds_left:

        xfine_new = np.arange(pos[0], pos[-1], dxf)
        xfine_left = xfine_new[:start_left]
        force_fine_left = -((xfine_left - force_bins[start_left]) ** 2) * (dxf) * 1e9
        force_fine_left += force_matrix[start_left]

        force_matrix = np.append(force_fine_left, force_matrix[start_left:])
        force_bins = xfine_new

    return fe_spline, force_bins, force_matrix


@njit
def integrate_gle_prw(nsteps, dt, m, gamma, x0, v0, kT, force_bins, force_matrix):
    ''' simulate Langevin equation with mass, ie.e PRW '''
    x = np.zeros(nsteps, dtype=np.float64)
    x[0] = x0 # initial position

    fac_rand = np.sqrt(2 * kT * gamma / dt)
    xx = x[0]
    vv = v0

    # runge kutta integration (4 step)
    for var in range(1, nsteps):
        xi = np.random.normal(0.0, 1.0)

        kx1 = dt * vv
        kv1 = dt * (-gamma * (vv) - dU2(xx, force_bins, force_matrix) + fac_rand * xi) / m
        x1 = xx + kx1 / 2
        v1 = vv + kv1 / 2

        kx2 = dt * v1
        kv2 = dt * (-gamma * (v1) - dU2(x1, force_bins, force_matrix) + fac_rand * xi) / m
        x2 = xx + kx2 / 2
        v2 = vv + kv2 / 2

        kx3 = dt * v2
        kv3 = dt * (-gamma * (v2) - dU2(x2, force_bins, force_matrix) + fac_rand * xi) / m
        x3 = xx + kx3
        v3 = vv + kv3

        kx4 = dt * v3
        kv4 = dt * (-gamma * (v3) - dU2(x3, force_bins, force_matrix) + fac_rand * xi) / m
        xx += (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6
        vv += (kv1 + 2 * kv2 + 2 * kv3 + kv4) / 6

        x[var] = xx
        # v[var]=vv

    return x, vv


@njit()
def integrate_gle_multi_exp(nsteps, dt, m, gammas, taus, x0, y0, v0, kT, force_bins, force_matrix):
    ''' integrate GLE for nsteps simulation steps with dt as time step with multi-exponential memory described by gammas and taus
    force_bins and force_matrix originate from spline function'''

    ks = gammas / taus # prefactors of exponential memory contributions

    n_exp = len(ks)
    x = np.zeros(nsteps, dtype=np.float64)
    x[0] = x0

    fac_randy = np.zeros(n_exp)
    for i in range(n_exp):

        fac_randy[i] = np.sqrt(2 * kT / gammas[i] / dt)

    xx = x0
    yy = y0
    vv = v0

    xi = np.zeros(n_exp)
    for var in range(1, int(nsteps)):
        for i in range(n_exp):

            xi[i] = np.random.normal(0.0, 1.0)

        fr = np.zeros(n_exp)
        for i in range(n_exp):

            fr[i] = fac_randy[i] * xi[i]

        kx1 = dt * vv
        ff = dU2(xx, force_bins, force_matrix)
        kv1 = 0
        kv1 -= dt * ff / m

        ky1 = np.zeros((n_exp,))
        for i in range(n_exp):
            kv1 += -dt * (ks[i] * (xx - yy[i])) / m
            ky1[i] = -dt * ((yy[i] - xx) / taus[i] + fr[i])

        x1 = xx + kx1 / 2
        v1 = vv + kv1 / 2
        y1 = yy + ky1 / 2

        kx2 = dt * v1
        ff = dU2(x1, force_bins, force_matrix)
        kv2 = 0
        kv2 -= dt * ff / m

        ky2 = np.zeros((n_exp,))
        for i in range(n_exp):
            kv2 += -dt * (ks[i] * (x1 - y1[i])) / m
            ky2[i] = -dt * ((y1[i] - x1) / taus[i] + fr[i])

        x2 = xx + kx2 / 2
        v2 = vv + kv2 / 2
        y2 = yy + ky2 / 2

        kx3 = dt * v2
        ff = dU2(x2, force_bins, force_matrix)
        kv3 = 0
        kv3 -= dt * ff / m

        ky3 = np.zeros((n_exp,))
        for i in range(n_exp):
            kv3 += -dt * (ks[i] * (x2 - y2[i])) / m
            ky3[i] = -dt * ((y2[i] - x2) / taus[i] + fr[i])

        x3 = xx + kx3
        v3 = vv + kv3
        y3 = yy + ky3

        kx4 = dt * v3
        ff = dU2(x3, force_bins, force_matrix)
        kv4 = 0
        kv4 -= dt * ff / m

        ky4 = np.zeros((n_exp,))
        for i in range(n_exp):
            kv4 += -dt * (ks[i] * (x3 - y3[i])) / m
            ky4[i] = -dt * ((y3[i] - x3) / taus[i] + fr[i])

        xx = xx + (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6
        vv = vv + (kv1 + 2 * kv2 + 2 * kv3 + kv4) / 6

        for i in range(n_exp):

            yy[i] += (ky1[i] + (2 * ky2[i]) + (2 * ky3[i]) + ky4[i]) / 6

        x[var] = xx

    return x, vv, yy


@njit
def dU2(x, force_bins, force_matrix):

    idx = bisection(force_bins, x)
    value = force_matrix[idx]

    return value


@njit()
def bisection(array, value):
    """Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively."""
    n = len(array)
    if value < array[0]:
        return 0  # -1
    elif value > array[n - 1]:
        return n - 1
    jl = 0  # Initialize lower
    ju = n - 1  # and upper limits.
    while ju - jl > 1:  # If we are not yet done,
        jm = (ju + jl) >> 1  # compute a midpoint with a bitshift
        if value >= array[jm]:
            jl = jm  # and replace either the lower limit
        else:
            ju = jm  # or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if value == array[0]:  # edge cases at bottom
        return 0
    elif value == array[n - 1]:  # and top
        return n - 1
    else:
        return jl


def simulation_prw(config, count_label):
    kT = 2.49 * (config.temp / 300)
    taum = config.mass / np.sum(config.gammas)
    dt = config.dt * taum

    if config.dw:
        if config.Ur == config.Ul:
            pos = np.linspace(-2*config.ll, 0, int(1000 * config.ll))
            pos = np.unique(np.append(pos, np.linspace(0, 2*config.rr, int(1000 * config.rr))))
            # print(np.unique(np.diff(pos)))
            Ull = config.Ul*kT
            fel = Ull * ((pos[np.where(pos<0)] / config.ll)**2 - 1)**2
            fer = Ull * ((pos[np.where(pos>=0)] / config.rr)**2 - 1)**2
            fe = np.append(fel, fer)
            # plt.plot(pos, fe)
            # plt.show()
        
        else:
            pos = np.linspace(-2*config.ll, 0, int(1000 * config.ll))
            pos = np.unique(np.append(pos, np.linspace(0, 2*config.rr, int(1000 * config.rr))))
            # print(np.unique(np.diff(pos)))
            Ull = config.Ul*kT
            Urr = config.Ur*kT
            fel = Ull * ((pos[np.where(pos<0)] / config.ll)**2 - 1)**2
            fer = Urr * ((pos[np.where(pos>=0)] / config.rr)**2 - 1)**2 + (Ull - Urr)
            fe = np.append(fel, fer)
        fe_spline, force_bins, force_matrix = spline_fe_for_sim(pos, fe)
        prob = np.exp(-fe/kT) / (np.sum(np.exp(-fe/kT))) # * (pos[1] - pos[0])

    else:
        fe = np.loadtxt(
            config.path, delimiter=config.delimiter
        )  # path, where free energy profile is stored in format (x, p(x), U(x)/kT)
        pos = fe[:, 0]
        fe_spline, force_bins, force_matrix = spline_fe_for_sim(pos, kT * fe[:, 2]) # energy in fe has to be in units of kT
        prob = fe[:, 1] / (np.sum(fe[:, 1]))

    x0 = np.random.choice(pos, p=prob) # initial condition from Boltzmann distribution
    v0 = np.random.normal(0.0, np.sqrt(kT / config.mass)) # initial condition from Boltzmann distribution

    x, vv = integrate_gle_prw(config.n_steps, dt, config.mass, np.sum(config.gammas), x0, v0, kT, force_bins, force_matrix)
    vv = 0

    if config.save_trj:
        np.save(config.path_save + '/trjprw_'
                + config.name
                + "_bins"
                + str(config.mfpt_bins)
                + "_r"
                + str(config.right_bound)
                + "_l"
                + str(config.left_bound)
                + "_tm"
                + f'{taum:e}'[:3] + f'{taum:e}'[-3:]
                + "_n1e"
                + str(int(np.log10(config.n_steps)))
                + "_dt"
                + str(config.dt), x)

    if config.mfpt == 'mfpt':
        end_points = np.linspace(config.left_bound, config.right_bound, config.mfpt_bins)
        start_points = [np.min(start_points), np.max(start_points)]
        fpt, counts = get_mfpt(x, start_points, end_points, dt, return_sum=True)


    elif config.mfpt == 'mffpt':
        end_points = np.linspace(config.left_bound, config.right_bound, config.mfpt_bins)
        start_points = [np.min(end_points), np.max(end_points)]
        fpt, counts = get_mffpt(x, start_points, end_points, dt, return_sum=True)


    if config.save_trj:
        msd_sim = 0
    else:
        msd_sim = msd_fast(x, less_memory=True)
    x = 0  # eject trj from memory

    log_ind = np.unique(
        np.int64(np.logspace(0, np.log10(len(msd_sim) - 1), 1000))
    )  # use 1000 logarithmically spaced points of msd
    t = np.arange(len(msd_sim)) * dt  # * stride # time step increases if stride > 1
    if config.aver > 1:
        if config.mfpt == 'mfpt' or config.mfpt == 'mffpt':
            return msd_sim , fpt, counts
        else:
            return msd_sim




def simulation(config, count_label):
    kT = 2.49 * (config.temp / 300)
    gammas = np.array(config.gammas)
    taus = np.array(config.taus)
    taum = config.mass / np.sum(gammas)
    dt = config.dt * taum

    if config.dw:
        if config.Ur == config.Ul:
            pos = np.linspace(-2*config.ll, 0, int(1000 * config.ll))
            pos = np.unique(np.append(pos, np.linspace(0, 2*config.rr, int(1000 * config.rr))))
            # print(np.unique(np.diff(pos)))
            Ull = config.Ul*kT
            fel = Ull * ((pos[np.where(pos<0)] / config.ll)**2 - 1)**2
            fer = Ull * ((pos[np.where(pos>=0)] / config.rr)**2 - 1)**2
            fe = np.append(fel, fer)
            # plt.plot(pos, fe)
            # plt.show()
        
        else:
            pos = np.linspace(-2*config.ll, 0, int(1000 * config.ll))
            pos = np.unique(np.append(pos, np.linspace(0, 2*config.rr, int(1000 * config.rr))))
            # print(np.unique(np.diff(pos)))
            Ull = config.Ul*kT
            Urr = config.Ur*kT
            fel = Ull * ((pos[np.where(pos<0)] / config.ll)**2 - 1)**2
            fer = Urr * ((pos[np.where(pos>=0)] / config.rr)**2 - 1)**2 + (Ull - Urr)
            fe = np.append(fel, fer)
        fe_spline, force_bins, force_matrix = spline_fe_for_sim(pos, fe)
        prob = np.exp(-fe/kT) / (np.sum(np.exp(-fe/kT)))

    else:
        fe = np.loadtxt(
            config.path, delimiter=config.delimiter
        )  # path, where free energy profile is stored in format (x, p(x), U(x)/kT)
        pos = fe[:, 0]
        fe_spline, force_bins, force_matrix = spline_fe_for_sim(pos, kT * fe[:, 2]) # energy in fe has to be in units of kT
        prob = fe[:, 1] / (np.sum(fe[:, 1]))

    n_exp = len(gammas)

    x0 = np.random.choice(pos, p=prob) # initial condition from Boltzmann distribution
    v0 = np.random.normal(0.0, np.sqrt(kT / config.mass)) # initial condition from Boltzmann distribution
    y0 = np.zeros(n_exp)

    for j in range(n_exp):
        y0[j] = np.random.normal(x0, np.sqrt(kT * np.absolute(taus[j] / gammas[j])))

    x, vv, yy = integrate_gle_multi_exp(
        config.n_steps, dt, config.mass, gammas, taus, x0, y0, v0, kT, force_bins, force_matrix
    )
    vv = 0 # not needed so eject from memory in lazy way
    yy = 0

    if config.save_trj:
        np.save(config.path_save + '/trj_'
                + config.name
                + "_bins"
                + str(config.mfpt_bins)
                + "_r"
                + str(config.right_bound)
                + "_l"
                + str(config.left_bound)
                + "_tm"
                + f'{taum:e}'[:3] + f'{taum:e}'[-3:]
                + "_n1e"
                + str(int(np.log10(config.n_steps)))
                + "_dt"
                + str(config.dt), x)

    end_points = np.linspace(config.left_bound, config.right_bound, config.mfpt_bins)
    start_points = [np.min(end_points), np.max(end_points)]
    if config.mfpt == 'mfpt':
        fpt, counts = get_mfpt(x, start_points, end_points, dt, return_sum=True)

    elif config.mfpt == 'mffpt':
        fpt, counts = get_mffpt(x, start_points, end_points, dt, return_sum=True) # obtained correct results for ala9 mfpt with this

    if config.save_trj:
        msd_sim = 0 # in order to have more ram for trajectory
    else:
        msd_sim = msd_fast(x, less_memory=True)
    x = 0  # eject trj from memory
    log_ind = np.unique(
        np.int64(np.logspace(0, np.log10(len(msd_sim) - 1), 1000))
    )  # use 1000 logarithmically spaced points of msd
    t = np.arange(len(msd_sim)) * dt  # * stride # time step increases if stride > 1
    if config.aver > 1:
        if config.mfpt == 'mfpt' or config.mfpt == 'mffpt':
            return msd_sim , fpt, counts
        else:
            return msd_sim



class Config(pydantic.BaseModel):
    dt: float
    mass: float
    n_steps: float
    temp: float
    name: str
    gammas: list
    taus: list = [None]
    delimiter: str = "\t"
    path: str
    path_save: str = "../msd"
    aver: int = 1
    mfpt: str = 'mffpt'
    left_bound: float
    right_bound: float
    mfpt_bins: int = 100
    save_trj: bool = False #save to path_save as MSD

    dw: bool = False
    ll: float = 1 # distance from left minimum to maximum
    rr: float = 1
    Ul: float = 2 # in kT
    Ur: float = 2 # in kT depth from right well perspective
    c: float = 1 # scaling of memory times
    d: float = 1 # scaling of friction amplitudes
    tau_scale: float = 1 # scaling of first memory time in units of taum
    n: int = 3 # number of memory times
    taum: float = 1 # inertial time


def main():
    # args = parser.parse_args()
    if len(sys.argv) < 2:
        raise ValueError("Please give path to config as input.")
    config_file = sys.argv[1]
    config = toml.load(config_file)
    config = Config(**config)
    print(f"{config=}")
    config.n_steps = int(config.n_steps)
    if config.left_bound == None or config.right_bound == None:
        if config.mfpt == 'mfpt' or config.mfpt == 'mffpt':
            print('Please give left and right positions to calculate MFPT in between.')
    if np.any(config.taus) == None:
        prw = True
    else:
        prw = False
        config.taus = np.array(config.taus) * 1e3  # units for simulation in ps from input in ns

    config.gammas = np.array(config.gammas) / 1e3 # units for simulation in ps from input in ns
    

    if config.dw:
        Ustr = str(int(config.Ur))
        if config.Ur < 1:
            Ustr = '0' + str(int(config.Ur * 10))
        if config.Ur == config.Ul:
            config.name = 'c' + str(int(config.c)) + '_d' + str(int(config.d)) + '_U' + Ustr + '_n' + str(int(config.n)) + '_sc' + str(int(config.tau_scale)) + '_dw'
        else:
            Ulstr = str(int(config.Ul))
            if config.Ul < 1:
                Ulstr = '0' + str(int(config.Ul * 10))
            config.name = 'c' + str(int(config.c)) + '_d' + str(int(config.d)) + '_Ul' + Ulstr + '_Ur' + Ustr + '_n' + str(int(config.n)) + '_sc' + str(int(config.tau_scale)) + '_dw'
        
        config.gammas = np.array([config.d**i for i in range(config.n)])
        config.gammas = config.gammas / np.sum(config.gammas) * 2.49 * (config.temp/300)
        config.taus = np.array([config.c**i for i in range(config.n)]) * config.tau_scale * config.taum

        taud = 1
        L = 1
        config.mass = config.taum * taud * 2.49 * (config.temp/300) / L**2

    taum = config.mass / np.sum(config.gammas)

    if prw:
        if config.aver > 1:
            for i in range(config.aver):
                if i == 0:
                    if config.mfpt == 'mfpt' or config.mfpt == 'mffpt':
                        add_init_msd, add_init_mfpt, counts_add = 0, 0, 0 #, 0, 0 # lazy version, when not knowing shape of msd etc. # , add_init_mfpt_qf, counts_add_qf
                        msd_sim_init, mfpt_init, mfpt_count = simulation_prw(config, config.name + str(i)) # , mfpt_init_qf, mfpt_count_qf

                    else:
                        add_init_msd = 0  # , 0 # lazy version, when not knowing shape of msd etc.
                        msd_sim_init = simulation_prw(config, config.name + str(i))

                else:
                    if config.mfpt == 'mfpt' or config.mfpt == 'mffpt':
                        add_init_msd, add_init_mfpt, counts_add = simulation_prw(config, config.name + str(i)) # , add_init_mfpt_qf, counts_add_qf
                    else:
                        add_init_msd = simulation_prw(config, config.name + str(i))

                msd_sim_init += add_init_msd

                if config.mfpt == 'mfpt' or config.mfpt == 'mffpt':
                    mfpt_init += add_init_mfpt
                    mfpt_count += counts_add


            msd_aver = msd_sim_init / config.aver
            if config.mfpt == 'mfpt' or config.mfpt == 'mffpt':
                mfpt_aver = np.divide(mfpt_init, mfpt_count, where=mfpt_count!=0) * config.dt * taum # mfpt count can be zero for some transitions
                end_points = np.linspace(config.left_bound, config.right_bound, config.mfpt_bins) # approximate end_points by linear interpolation with given number of points 
                mfpt_aver = np.c_[mfpt_aver.T, end_points] # add positions from which mfpt to start and end point are computed as last column
                np.save(
                    config.path_save + "/" + config.mfpt + "_prw_av"
                    + str(config.aver)
                    + config.name
                    + "_bins"
                    + str(config.mfpt_bins)
                    + "_r"
                    + str(config.right_bound)
                    + "_l"
                    + str(config.left_bound)
                    + "_tm"
                    + f'{taum:e}'[:3] + f'{taum:e}'[-3:]
                    + "_n1e"
                    + str(int(np.log10(config.n_steps)))
                    + "_dt"
                    + str(config.dt),
                    mfpt_aver
                )


            t = np.arange(len(msd_aver)) * config.dt * taum  # obtain time in physical units of ps
            log_ind = np.unique(
                np.int64(np.logspace(0, np.log10(len(msd_aver) - 1), 1000))
            )  # use 1000 logarithmically spaced points of msd
            t = t[log_ind]
            msd_aver = msd_aver[log_ind]
            alpha_aver = np.diff(np.log(msd_aver)) / np.diff(np.log(t))

            np.save(
                config.path_save + "/msd_prw_av"
                + str(config.aver)
                + config.name
                + "_bins"
                + str(config.mfpt_bins)
                + "_r"
                + str(config.right_bound)
                + "_l"
                + str(config.left_bound)
                + "_tm"
                + f'{taum:e}'[:3] + f'{taum:e}'[-3:] # changed on 02.09.24 to not show 0.0 from previous round(taum, 1)
                + "_n1e"
                + str(int(np.log10(config.n_steps)))
                + "_dt"
                + str(config.dt),
                np.array([msd_aver, t])
            )
            np.save(
                config.path_save + "/alpha_prw_av"
                + str(config.aver)
                + config.name
                + "_bins"
                + str(config.mfpt_bins)
                + "_r"
                + str(config.right_bound)
                + "_l"
                + str(config.left_bound)
                + "_tm"
                + f'{taum:e}'[:3] + f'{taum:e}'[-3:]
                + "_n1e"
                + str(int(np.log10(config.n_steps)))
                + "_dt"
                + str(config.dt),
                alpha_aver
            )

        else:
            simulation_prw(config, config.name + str(0))


    else:
        # mfpt_count = 0
        if config.aver > 1:
            for i in range(config.aver):
                if i == 0:
                    if config.mfpt == 'mfpt' or config.mfpt == 'mffpt':
                        add_init_msd, add_init_mfpt, counts_add = 0, 0, 0 #, 0, 0 # lazy version, when not knowing shape of msd etc. # , add_init_mfpt_qf, counts_add_qf
                        msd_sim_init, mfpt_init, mfpt_count= simulation(config, config.name + str(i)) #, mfpt_init_qf, mfpt_count_qf 

                    else:
                        add_init_msd = 0  # , 0 # lazy version, when not knowing shape of msd etc.
                        msd_sim_init = simulation(config, config.name + str(i))

                else:
                    if config.mfpt == 'mfpt' or config.mfpt == 'mffpt':
                        add_init_msd, add_init_mfpt, counts_add = simulation(config, config.name + str(i)) # , add_init_mfpt_qf, counts_add_qf
                    else:
                        add_init_msd = simulation(config, config.name + str(i))

                msd_sim_init += add_init_msd

                if config.mfpt == 'mfpt' or config.mfpt == 'mffpt':
                    mfpt_init += add_init_mfpt
                    mfpt_count += counts_add


            msd_aver = msd_sim_init / config.aver
            if config.mfpt == 'mfpt' or config.mfpt == 'mffpt':
                mfpt_aver = np.divide(mfpt_init, mfpt_count, where=mfpt_count!=0) * config.dt * taum # mfpt count can be zero for some transitions
                end_points = np.linspace(config.left_bound, config.right_bound, config.mfpt_bins) # approximate end_points by linear interpolation with given number of points 
                mfpt_aver = np.c_[mfpt_aver.T, end_points] # add positions from which mfpt to start and end point are computed as last column
                np.save(
                    config.path_save + "/" + config.mfpt + "_av"
                    + str(config.aver)
                    + config.name
                    + "_bins"
                    + str(config.mfpt_bins)
                    + "_r"
                    + str(config.right_bound)
                    + "_l"
                    + str(config.left_bound)
                    + "_tm"
                    + f'{taum:e}'[:3] + f'{taum:e}'[-3:]
                    + "_n1e"
                    + str(int(np.log10(config.n_steps)))
                    + "_dt"
                    + str(config.dt),
                    mfpt_aver
                )


            t = np.arange(len(msd_aver)) * config.dt * taum  # obtain time in physical units of ps
            log_ind = np.unique(
                np.int64(np.logspace(0, np.log10(len(msd_aver) - 1), 1000))
            )  # use 1000 logarithmically spaced points of msd
            t = t[log_ind]
            msd_aver = msd_aver[log_ind]
            alpha_aver = np.diff(np.log(msd_aver)) / np.diff(np.log(t))
            
            np.save(
                config.path_save + "/msd_av"
                + str(config.aver)
                + config.name
                + "_bins"
                + str(config.mfpt_bins)
                + "_r"
                + str(config.right_bound)
                + "_l"
                + str(config.left_bound)
                + "_tm"
                + f'{taum:e}'[:3] + f'{taum:e}'[-3:]
                + "_n1e"
                + str(int(np.log10(config.n_steps)))
                + "_dt"
                + str(config.dt),
                np.array([msd_aver, t])
            )
            np.save(
                config.path_save + "/alpha_av"
                + str(config.aver)
                + config.name
                + "_bins"
                + str(config.mfpt_bins)
                + "_r"
                + str(config.right_bound)
                + "_l"
                + str(config.left_bound)
                + "_tm"
                + f'{taum:e}'[:3] + f'{taum:e}'[-3:]
                + "_n1e"
                + str(int(np.log10(config.n_steps)))
                + "_dt"
                + str(config.dt),
                alpha_aver
            )

        else:
            simulation(
                config, config.name + str(0)
            )


if __name__ == "__main__":
    main()
