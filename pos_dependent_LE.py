# simulations are performed by creating a your_config.toml file including the parameters that you want (above the definition of the main() function is a list of possible config parameters)
# and then executing "python3 pos_dependent_LE.py your_config.toml" in the command line (compatibility tested with python 3.9.7)
# free energy and friction profiles have to be saved in the correct format for the simulation to work (explained below)

import sys
from numba import njit
import numpy as np
from scipy import interpolate
from motility import msd_fast
# for loading config
import toml
# for parsing config
import pydantic
from mfpts import *

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
    # to determine behavior of Langevin simulations at given position and determine force due to free energy
    dxf = (pos[1] - pos[0]) / dx
    fe_spline = interpolate.splrep(pos, fe, s=0, per=0)
    force_bins = np.arange(pos[0], pos[-1], dxf)
    force_matrix = interpolate.splev(force_bins, fe_spline, der=der) # der is derivative

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


@njit()
def integrate_pos_dep(nsteps, dt, m, x0, v0, kT, force_bins, force_matrix, friction_bins, friction_matrix, seed):
    # integration of position dependent Langevin equation with mass term
    x = np.zeros(nsteps, dtype=np.float64)

    x[0] = x0
    xx = x[0]
    vv = v0

    np.random.seed(seed)

    for var in range(1, nsteps):
        xi = np.random.normal(0.0, 1.0)

        gammax = dU2(xx, friction_bins, friction_matrix)
        fac_rand = np.sqrt(2 * kT * gammax / dt)
        kx1 = dt * vv
        kv1 = dt * (-gammax * vv - dU2(xx, force_bins, force_matrix) + fac_rand * xi) / m
        x1 = xx + kx1 / 2
        v1 = vv + kv1 / 2

        gammax = dU2(x1, friction_bins, friction_matrix) 
        fac_rand = np.sqrt(2 * kT * gammax/ dt)
        kx2 = dt * v1
        kv2 = dt * (-gammax* v1 - dU2(x1, force_bins, force_matrix) + fac_rand * xi) / m
        x2 = xx + kx2 / 2
        v2 = vv + kv2 / 2

        gammax = dU2(x2, friction_bins, friction_matrix)
        fac_rand = np.sqrt(2 * kT * gammax / dt)
        kx3 = dt * v2
        kv3 = dt * (-gammax * v2 - dU2(x2, force_bins, force_matrix) + fac_rand * xi) / m
        x3 = xx + kx3
        v3 = vv + kv3

        gammax = dU2(x3, friction_bins, friction_matrix)
        fac_rand = np.sqrt(2 * kT *  gammax/ dt)
        kx4 = dt * v3
        kv4 = dt * (-gammax * v3 - dU2(x3, force_bins, force_matrix) + fac_rand * xi) / m
        xx += (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6
        vv += (kv1 + 2 * kv2 + 2 * kv3 + kv4) / 6

        x[var] = xx
        # v[var]=vv

    return x



@njit()
def integrate_pos_dep_OD(nsteps, dt, x0, kT, force_bins, force_matrix, friction_bins, friction_matrix, friction_der_bins, friction_der_matrix, seed):
    # integration of position dependent Langevin equation in overdamped (OD) limit (which is the analytically solvable case, where position dependent friction follows from MFPT profiles)
    x = np.zeros(nsteps, dtype=np.float64)

    x[0] = x0
    xx = x[0]
    
    np.random.seed(seed)

    for var in range(1, nsteps):
        xi = np.random.normal(0.0, 1.0)

        gammax = dU2(xx, friction_bins, friction_matrix)
        gamma_derx = dU2(xx, friction_der_bins, friction_der_matrix)
        fac_rand = np.sqrt(2 * kT * gammax / dt)
        kx1 = dt * (-dU2(xx, force_bins, force_matrix) - 0.5*kT * gamma_derx / gammax + fac_rand * xi) / gammax
        x1 = xx + kx1 / 2

        gammax = dU2(x1, friction_bins, friction_matrix) 
        gamma_derx = dU2(x1, friction_der_bins, friction_der_matrix)
        fac_rand = np.sqrt(2 * kT * gammax/ dt)
        kx2 = dt * (-dU2(x1, force_bins, force_matrix) - 0.5*kT * gamma_derx / gammax + fac_rand * xi) / gammax
        x2 = xx + kx2 / 2

        gammax = dU2(x2, friction_bins, friction_matrix)
        gamma_derx = dU2(x2, friction_der_bins, friction_der_matrix)
        fac_rand = np.sqrt(2 * kT * gammax / dt)
        kx3 = dt * (-dU2(x2, force_bins, force_matrix) - 0.5*kT * gamma_derx / gammax + fac_rand * xi) / gammax
        x3 = xx + kx3

        gammax = dU2(x3, friction_bins, friction_matrix)
        gamma_derx = dU2(x3, friction_der_bins, friction_der_matrix)
        fac_rand = np.sqrt(2 * kT *  gammax/ dt)
        kx4 = dt * (-dU2(x3, force_bins, force_matrix) - 0.5*kT * gamma_derx / gammax + fac_rand * xi) / gammax
        xx += (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6

        x[var] = xx

    return x


def add_const_sides(pos, friction_profile, min_pos=0, max_pos=1):
    # prolonging the extracted friction profiles on the sides to set behavior of langevin simulation at positions that were not sampled in the MD simulation, of which the friction profile was extracted from
    dpos = pos[1] - pos[0]
    add_left = int((np.min(pos) - min_pos) / dpos) # steps to add to the left until arriving at min_pos
    add_right = int((max_pos - np.max(pos)) / dpos) # steps to add to the right until arriving at max_pos
    friction_profile_conc = np.concatenate((np.ones(add_left) * friction_profile[0], friction_profile, np.ones(add_right) * friction_profile[-1]))
    pos_conc = np.concatenate((np.arange(1, add_left + 1) * dpos , pos,
    np.arange(add_right) * dpos + dpos + pos[-1]))

    return pos_conc, friction_profile_conc



def simulation_pos_dep(config, count_label):
    kT = 2.49 * (config.temp / 300)
    taum = config.mass / np.sum(config.gammas)
    dt = config.dt * taum

    fe = np.loadtxt(
        config.path, delimiter=config.delimiter
    )  # path, where free energy profile is stored in format (x, p(x), U(x)/kT)

    friction_profile = np.load(config.path_fric) # path, where friction profile is stored in format (gamma_fold, gamma_unfold, x)

    pos = fe[:, 0]
    fe_spline, force_bins, force_matrix = spline_fe_for_sim(pos, kT * fe[:, 2])
    if config.const_side:
        pos_conc, friction_profile_conc = add_const_sides(friction_profile[:, -1], friction_profile[:, config.fold], pos_min=config.min_pos, pos_max=config.max_pos)
        friction_spline, friction_bins, friction_matrix = spline_fe_for_sim(pos_conc, friction_profile_conc, der=0)
        
    else:
        friction_spline, friction_bins, friction_matrix = spline_fe_for_sim(friction_profile[:, -1], friction_profile[:, config.fold], der=0)

    prob = fe[:, 1] / (np.sum(fe[:, 1]))
    np.random.seed(config.seed)
    x0 = np.random.choice(pos, p=prob)
    v0 = np.random.normal(0.0, np.sqrt(kT / config.mass))

    x = integrate_pos_dep(config.n_steps, dt, config.mass, x0, v0, kT, force_bins, force_matrix, friction_bins, friction_matrix, config.seed)
    np.save(
        "/net/storage/kanton/protein_gle_sim/seed/seed_posdep"
        + count_label
        + "_av"
        + str(config.aver)
        + "_tm"
        + str(round(taum, 1))
        + "_n1e"
        + str(int(np.log10(config.n_steps)))
        + "_dt"
        + str(config.dt),
        config.seed
    )

    if config.mfpt:
        end_points = np.linspace(config.left_bound, config.right_bound, len(friction_profile[:, -1]))
        start_points = [np.min(end_points), np.max(end_points)]
        mfpt = get_mffpt(x, start_points, end_points, dt, return_sum=False)
        np.save(
            "/net/storage/kanton/protein_gle_sim/mfpt/mfpt_posdep"
            + count_label
            + "_av"
            + str(config.aver)
            + "_tm"
            + str(round(taum, 1))
            + "_n1e"
            + str(int(np.log10(config.n_steps)))
            + "_dt"
            + str(config.dt),
            np.array(mfpt))
    
    msd_sim = msd_fast(x, less_memory=True)
    x = 0  # eject trj from memory

    log_ind = np.unique(
        np.int64(np.logspace(0, np.log10(len(msd_sim) - 1), 1000))
    )  # use 1000 logarithmically spaced points of msd
    t = np.arange(len(msd_sim)) * dt  # * stride # time step increases if stride > 1
    if config.aver > 1:
        if config.mfpt:
            return msd_sim , np.array(mfpt)
        else:
            return msd_sim
    t = t[log_ind]
    msd_sim = msd_sim[log_ind]
    alpha = np.diff(np.log(msd_sim)) / np.diff(np.log(t))
    np.save(
        "/net/storage/kanton/protein_gle_sim/msd/msd_posdep"
        + count_label
        + "_av"
        + str(config.aver)
        + "_tm"
        + str(round(taum, 1))
        + "_n1e"
        + str(int(np.log10(config.n_steps)))
        + "_dt"
        + str(config.dt),
        np.array([msd_sim, t])
    )
    np.save(
        "/net/storage/kanton/protein_gle_sim/msd/alpha_posdep"
        + count_label
        + "_av"
        + str(config.aver)
        + "_tm"
        + str(round(taum, 1))
        + "_n1e"
        + str(int(np.log10(config.n_steps)))
        + "_dt"
        + str(config.dt),
        alpha
    )




def simulation_pos_dep_OD(config, count_label):
    kT = 2.49 * (config.temp / 300)
    taum = config.mass / np.sum(config.gammas)
    dt = config.dt * taum

    fe = np.loadtxt(
        config.path, delimiter=config.delimiter
    )  # path, where free energy profile is stored in format (x, p(x), U(x)/kT)

    friction_profile = np.load(config.path_fric) # path, where friction profile is stored in format (gamma_fold, gamma_unfold, x)

    pos = fe[:, 0]
    fe_spline, force_bins, force_matrix = spline_fe_for_sim(pos, kT * fe[:, 2])
    if config.const_side:
        pos_conc, friction_profile_conc = add_const_sides(friction_profile[:, -1], friction_profile[:, config.fold], pos_min=config.min_pos, pos_max=config.max_pos)

        friction_spline, friction_bins, friction_matrix = spline_fe_for_sim(pos_conc, friction_profile_conc, der=0)
        friction_spline, friction_der_bins, friction_der_matrix = spline_fe_for_sim(pos_conc, friction_profile_conc, der=1)

    else:
        friction_spline, friction_bins, friction_matrix = spline_fe_for_sim(friction_profile[:, -1], friction_profile[:, config.fold], der=0)
        friction_spline, friction_der_bins, friction_der_matrix = spline_fe_for_sim(friction_profile[:, -1], friction_profile[:, config.fold], der=1)

    prob = fe[:, 1] / (np.sum(fe[:, 1]))
    np.random.seed(config.seed)
    x0 = np.random.choice(pos, p=prob)

    x = integrate_pos_dep_OD(config.n_steps, dt, x0, kT, force_bins, force_matrix, friction_bins, friction_matrix, friction_der_bins, friction_der_matrix, config.seed)
    np.save(
        "/net/storage/kanton/protein_gle_sim/seed/seed_posdep_OD"
        + count_label
        + "_av"
        + str(config.aver)
        + "_tm"
        + str(round(taum, 1))
        + "_n1e"
        + str(int(np.log10(config.n_steps)))
        + "_dt"
        + str(config.dt),
        config.seed
    )

    if config.mfpt:
        end_points = np.linspace(config.left_bound, config.right_bound, len(friction_profile[:, -1]))
        start_points = [np.min(end_points), np.max(end_points)]
        mfpt = get_mffpt(x, start_points, end_points, dt, return_sum=False)
        np.save(
            "/net/storage/kanton/protein_gle_sim/mfpt/mfpt_posdep_OD"
            + count_label
            + "_av"
            + str(config.aver)
            + "_tm"
            + str(round(taum, 1))
            + "_n1e"
            + str(int(np.log10(config.n_steps)))
            + "_dt"
            + str(config.dt),
            np.array(mfpt))

    msd_sim = msd_fast(x, less_memory=True)
    x = 0  # eject trj from memory

    log_ind = np.unique(
        np.int64(np.logspace(0, np.log10(len(msd_sim) - 1), 1000))
    )  # use 1000 logarithmically spaced points of msd
    t = np.arange(len(msd_sim)) * dt  # * stride # time step increases if stride > 1
    if config.aver > 1:
        if config.mfpt:
            return msd_sim , np.array(mfpt)
        else:
            return msd_sim
    t = t[log_ind]
    msd_sim = msd_sim[log_ind]
    alpha = np.diff(np.log(msd_sim)) / np.diff(np.log(t))
    np.save(
        "/net/storage/kanton/protein_gle_sim/msd/msd_posdep_OD"
        + count_label
        + "_av"
        + str(config.aver)
        + "_tm"
        + str(round(taum, 1))
        + "_n1e"
        + str(int(np.log10(config.n_steps)))
        + "_dt"
        + str(config.dt),
        np.array([msd_sim, t])
    )
    np.save(
        "/net/storage/kanton/protein_gle_sim/msd/alpha_posdep_OD"
        + count_label
        + "_av"
        + str(config.aver)
        + "_tm"
        + str(round(taum, 1))
        + "_n1e"
        + str(int(np.log10(config.n_steps)))
        + "_dt"
        + str(config.dt),
        alpha
    )


class Config(pydantic.BaseModel):
    dt: float
    mass: float
    n_steps: float
    temp: float #temperature in Kelvin
    name: str
    gammas: list #gammas in u/ns
    delimiter: str = "\t"
    path: str # path to free energy profile in format (x, p(x), U(x)/kT) in txt
    path_fric: str # path to friction profile in format (gamma_fold, gamma_unfold, x) in npy, units in u/ps
    seed: int = None
    aver: int = 1
    od: bool = False # overdamped or not
    const_side: bool = False # decide whether or not to apply constant sides to the friction profile automatically
    mfpt: bool = True # decide whether or not to calculate mffpt
    left_bound: float
    right_bound: float # bound where to calculate mffpt between
    min_pos: float = 0
    max_pos: float = 1 # apply constant sides to the friction profile
    fold: int = 0 # if 0 use fold, if 1 use unfold friction profile

def main():
    # args = parser.parse_args()
    if len(sys.argv) < 2:
        raise ValueError("Please give path to config as input.")
    config_file = sys.argv[1]
    config = toml.load(config_file)
    config = Config(**config)
    print(f"{config=}")
    config.n_steps = int(config.n_steps)
    if config.fold == 1:
        config.name += '_unfold'
    if config.left_bound == None or config.right_bound == None:
        if config.mfpt:
            print('Please give left and right positions to calculate MFPT in between.')
    config.gammas = np.array(config.gammas) / 1e3
    taum = config.mass / np.sum(config.gammas)
    config.seed = np.random.randint(low=1, high=2**31)
    if config.const_side:
        config.name = config.name + 'cs'
    mfpt_count = 0
    if config.od:
        if config.aver > 1:
            for i in range(config.aver):
                if i == 0:
                    if config.mfpt:
                        add_init_msd, add_init_mfpt = 0 , 0 # lazy version, when not knowing shape of msd etc.
                        msd_sim_init, mfpt_init = simulation_pos_dep_OD(config, config.name + str(i))
                        if np.any(mfpt_init != 0):
                            count_flag = np.ones(mfpt_init.shape)
                            count_flag[np.where(mfpt_init == 0)] = np.zeros(len(np.where(mfpt_init == 0)[0]))
                            mfpt_count += count_flag
                    else:
                        add_init_msd = 0  # lazy version, when not knowing shape of msd etc.
                        msd_sim_init = simulation_pos_dep_OD(config, config.name + str(i))
                else:
                    if config.mfpt:
                        add_init_msd, add_init_mfpt = simulation_pos_dep_OD(config, config.name + str(i))
                    else:
                        add_init_msd = simulation_pos_dep_OD(config, config.name + str(i))

                
                msd_sim_init += add_init_msd
                if config.mfpt:
                    if np.any(add_init_mfpt != 0):
                        mfpt_init += add_init_mfpt
                        count_flag = np.ones(mfpt_init.shape)
                        count_flag[np.where(add_init_mfpt == 0)] = np.zeros(len(np.where(add_init_mfpt == 0)[0])) # only average where contribution is non-zero
                        mfpt_count += count_flag

            msd_aver = msd_sim_init / config.aver
            if config.mfpt:
                mfpt_aver = np.divide(mfpt_init, mfpt_count, where=mfpt_count!=0) # mfpt count can be zero for some transitions
                start_points = np.linspace(config.left_bound, config.right_bound, len(mfpt_aver[:, 0])) # approximate start_points by linear interpolation with given number of points 
                mfpt_aver = np.c_[mfpt_aver, start_points] # add positions from which mfpt to start and end point are computed as last column
            
                np.save(
                    "/net/storage/kanton/protein_gle_sim/msd/mfpt_posdep_OD_av"
                    + str(config.aver)
                    + config.name
                    + "_tm"
                    + str(round(taum, 1))
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
                "/net/storage/kanton/protein_gle_sim/msd/msd_posdep_OD_av"
                + str(config.aver)
                + config.name
                + "_tm"
                + str(round(taum, 1))
                + "_n1e"
                + str(int(np.log10(config.n_steps)))
                + "_dt"
                + str(config.dt),
                np.array([msd_aver, t])
            )
            np.save(
                "/net/storage/kanton/protein_gle_sim/msd/alpha_posdep_OD_av"
                + str(config.aver)
                + config.name
                + "_tm"
                + str(round(taum, 1))
                + "_n1e"
                + str(int(np.log10(config.n_steps)))
                + "_dt"
                + str(config.dt),
                alpha_aver
            )
            # with open("path", "r") as fh:
            #     toml.dump(config.dict(), fh)

        else:
            simulation_pos_dep_OD(config, config.name + str(0))

    else:
        if config.aver > 1:
            for i in range(config.aver):
                if i == 0:
                    if config.mfpt:
                        add_init_msd, add_init_mfpt = 0 , 0 # lazy version, when not knowing shape of msd etc.
                        msd_sim_init, mfpt_init = simulation_pos_dep_OD(config, config.name + str(i))
                        if np.any(mfpt_init != 0):
                            count_flag = np.ones(mfpt_init.shape)
                            count_flag[np.where(mfpt_init == 0)] = np.zeros(len(np.where(mfpt_init == 0)[0]))
                            mfpt_count += count_flag
                    else:
                        add_init_msd = 0  # lazy version, when not knowing shape of msd etc.
                        msd_sim_init = simulation_pos_dep_OD(config, config.name + str(i))
                else:
                    if config.mfpt:
                        add_init_msd, add_init_mfpt = simulation_pos_dep_OD(config, config.name + str(i))
                    else:
                        add_init_msd = simulation_pos_dep_OD(config, config.name + str(i))
                
                msd_sim_init += add_init_msd
                if config.mfpt:
                    if np.any(add_init_mfpt != 0):
                        mfpt_init += add_init_mfpt
                        count_flag = np.ones(mfpt_init.shape)
                        count_flag[np.where(add_init_mfpt == 0)] = np.zeros(len(np.where(add_init_mfpt == 0)[0])) # only average where contribution is non-zero
                        mfpt_count += count_flag

            msd_aver = msd_sim_init / config.aver
            if config.mfpt:
                mfpt_aver = np.divide(mfpt_init, mfpt_count, where=mfpt_count!=0) # mfpt count can be zero for some transitions
                start_points = np.linspace(config.left_bound, config.right_bound, len(mfpt_aver[:, 0])) # approximate start_points by linear interpolation with given number of points 
                mfpt_aver = np.c_[mfpt_aver, start_points] # add positions from which mfpt to start and end point are computed as last column
                np.save(
                    "/net/storage/kanton/protein_gle_sim/msd/mfpt_posdep_av"
                    + str(config.aver)
                    + config.name
                    + "_tm"
                    + str(round(taum, 1))
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
                "/net/storage/kanton/protein_gle_sim/msd/msd_posdep_av"
                + str(config.aver)
                + config.name
                + "_tm"
                + str(round(taum, 1))
                + "_n1e"
                + str(int(np.log10(config.n_steps)))
                + "_dt"
                + str(config.dt),
                np.array([msd_aver, t])
            )
            np.save(
                "/net/storage/kanton/protein_gle_sim/msd/alpha_posdep_av"
                + str(config.aver)
                + config.name
                + "_tm"
                + str(round(taum, 1))
                + "_n1e"
                + str(int(np.log10(config.n_steps)))
                + "_dt"
                + str(config.dt),
                alpha_aver
            )

        else:
            simulation_pos_dep(config, config.name + str(0))


if __name__ == "__main__":
    main()
