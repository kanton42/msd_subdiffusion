# simulations are performed by creating a your_config.toml file including the parameters that you want (above the definition of the main() function is a list of possible config parameters)
# and then executing "python3 mfpt_gammaq_md.py your_config.toml" in the command line (compatibility tested with python 3.9.7)

import numpy as np
from mfpts import *
import sys
import matplotlib.pyplot as plt

# for loading config
import toml
# for parsing config
import pydantic
import matplotlib.pyplot as plt
import h5py # for loading of ala9 trajectory

def load_potential(config):
    if config.path_pot.endswith('.txt'):
        potential_data = np.loadtxt(config.path_pot, delimiter=config.delimiter)
    elif config.path_pot.endswith('.npy'):
        potential_data = np.load(config.path_pot)

    return potential_data[:, 0], potential_data[:, 2]


def compute_friction_profile(pos_mfpt, mfpt, pos_potential, potential, fold=True):
    # tested for ramp potential with const friction, free diffusion and without potential for parabolic friction profile on 2024-02-14
    kT = 2.49 # times have to be in units of ps, positions in nm, energy in kT
    dpos_pot = (pos_potential[1] - pos_potential[0])
    running_integral = np.zeros(len(potential))
    dpos_mfpt = (pos_mfpt[1] - pos_mfpt[0])

    if not fold: # different reaction coordinate than in cihans pnas, therefore swapped fold and unfold
        # mfpt from big q to small q
        closest_indices = np.argmin(np.abs(pos_potential[:, np.newaxis] - pos_mfpt[:-1]), axis=0)
        for run_ind in range(len(potential)): # loop for running integral
            for integral_ind in range(run_ind, len(potential)):
                # integral Z2 of eq11b cihans pnas from q to qmax, where qmax is position at which there is 0 flux
                # use trapezoidal rule
                if integral_ind == len(potential)-1 or integral_ind == run_ind: #for parabolic friction profile only integral_ind == run_ind gives better agreement, for free or ramp potential and const fric use bothe conditions
                    running_integral[run_ind] += np.exp(-potential[integral_ind]) * dpos_pot * 0.5
                else:
                    running_integral[run_ind] += np.exp(-potential[integral_ind]) * dpos_pot

        if np.array_equal(pos_potential, pos_mfpt):
            closest_indices = np.arange(len(pos_potential) - 1) # indices, where pos_potential is closest to pos_mfpt for derivative
                
        # friction = -kT * np.diff(mfpt) / dpos_mfpt * np.exp(-potential[closest_indices]) / running_integral[closest_indices] # [:-1]
        friction = -kT * np.gradient(mfpt)[:-1] / dpos_mfpt * np.exp(-potential[closest_indices]) / running_integral[closest_indices]

    else:
        # mfpt from small q to big q
        # closest_indices = np.argmin(np.abs(pos_potential[:, np.newaxis] - (pos_mfpt[1:] + pos_mfpt[:-1]) / 2), axis=0)
        closest_indices = np.argmin(np.abs(pos_potential[:, np.newaxis] - pos_mfpt[1:]), axis=0)
        for run_ind in range(len(potential)):
            # integral Z1 of eq11a cihans pnas from qmin to q, where qmin is position at which there is 0 flux (reflective boundary)
            for integral_ind in range(run_ind + 1):
                # use trapezoidal rule
                if integral_ind == 0 or integral_ind == run_ind:
                    running_integral[run_ind] += np.exp(-potential[integral_ind]) * dpos_pot * 0.5
                else:
                    running_integral[run_ind] += np.exp(-potential[integral_ind]) * dpos_pot
        if np.array_equal(pos_potential, pos_mfpt):
            closest_indices = np.arange(1, len(pos_potential)) # indices, where pos_potential is closest to pos_mfpt for derivative
                
        friction = kT * np.gradient(mfpt)[1:] / dpos_mfpt * np.exp(-potential[closest_indices]) / running_integral[closest_indices]

    # friction will have units of u/ps, if position has units of nm, or in u nm^2/ps if position has no units
    return friction


def compute_mfpt_from_gammaq(friction, pos, potential, pos_potential, fold=True):
    kT = 2.49 # times have to be in units of ps, positions in nm, energy in kT
    dpos = pos[1] - pos[0]
    dpos_pot = pos_potential[1] - pos_potential[0]
    inner_integral = np.zeros(len(potential))
    mfpt = np.zeros(len(pos))
    closest_indices = np.argmin(np.abs(pos_potential[:, np.newaxis] - pos[:len(friction)]), axis=0) # since bins of potential can be different from bins of mfpt
    if fold:
        for inner_ind in range(len(potential)):
            inner_integral[inner_ind] = np.sum(np.exp(-potential[:inner_ind + 1]) * dpos_pot)
        
        for outer_ind in range(len(pos)):
            mfpt[outer_ind] = np.sum(friction[:outer_ind + 1] * dpos * np.exp(potential[closest_indices][:outer_ind + 1]) * inner_integral[closest_indices][:outer_ind + 1]) / kT

    else:
        for inner_ind in range(len(potential)):
            inner_integral[inner_ind] = np.sum(np.exp(-potential[inner_ind:]) * dpos_pot)
        
        for outer_ind in range(len(pos)):
            mfpt[outer_ind] = np.sum(friction[outer_ind:] * dpos * np.exp(potential[closest_indices][outer_ind:]) * inner_integral[closest_indices][outer_ind:]) / kT

    return mfpt


def ramp_potential(x, xs=0, xf=10, U0=2):
    U = (x - xs) / (xf - xs) * U0
    return U

def tmfp_ramp(x, xs=0, xf=10, U0=2, D=1, xfvar=True):
    if xfvar:
        return -(xf - xs) * (x - xs) / (D * U0) - (xf - xs)**2 / U0**2 * (1 - np.exp(U0 * (x - xs) / (xf - xs))) / D
    else:
        U0 = -U0
        tmfp = -(xf - xs) * (x - xs) / (D * U0) - (xf - xs)**2 / U0**2 * (1 - np.exp(U0 * (x - xs) / (xf - xs))) / D
        return tmfp[::-1]


def test_friction_profile(arg='ramp', skip=1):
    if arg == 'ramp':
        pos_potential = np.arange(0, 10, 0.005)
        potential = ramp_potential(pos_potential, U0=2)
        tmfp_theo = tmfp_ramp(pos_potential, U0=2)
        tmfp_theo2 = tmfp_ramp(pos_potential, U0=-2)[::-1]
        

    elif arg == 'parab':
        pos_potential = np.arange(0, 10, 0.005)
        potential = np.zeros(len(pos_potential))
        tmfp_theo = pos_potential**4/4 # free diffusion with parabolic friction profile
        tmfp_theo2 =  - (pos_potential**3 * 10 / 3 - pos_potential**4 / 4) # free diffusion with parabolic friction profile  (10**3 * 10 / 3 - 10**4 / 4)
        

    elif arg == 'free':
        pos_potential = np.arange(0, 10, 0.005)
        potential = np.zeros(len(pos_potential))
        tmfp_theo = pos_potential**2/2
        tmfp_theo2 = tmfp_theo[::-1]

        
    friction = compute_friction_profile(pos_potential[::skip], tmfp_theo[::skip], pos_potential, potential)
    friction2 = compute_friction_profile(pos_potential[::skip], tmfp_theo2[::skip], pos_potential, potential, fold=False)
    plt.plot(pos_potential[::skip][:-1], friction/2.49)
    plt.plot(pos_potential[::skip][:-1], friction2/2.49)
    plt.xlabel('position')
    plt.ylabel('friction')
    plt.show()

def friction_aver(mfpt, start_points, pot_pos, pot, cut_sides=2):
    # average friction profiles for different start points of mfpt to get smoother profile (disregard cut_sides amount of points at side of friction profile because of artefacts)
    # last column of mfpt has to be positions of mfpt
    if mfpt.shape[0] < mfpt.shape[1]:
        mfpt = mfpt.T
        print('Warning: mfpt shape transposed, to match convention of fewer start_points than end_points.')
    fold_aver = np.zeros(len(mfpt[:, 0]) - 1) # - 2*cut_sides
    unfold_aver = np.zeros(len(mfpt[:, 0]) - 1) # - 2*cut_sides
    ind_counter_unfold = np.zeros(len(mfpt[:, 0]) - 1) # - 2*cut_sides
    ind_counter_fold = np.zeros(len(mfpt[:, 0]) - 1) # - 2*cut_sides
    for idx_start in range(len(start_points)):
        if idx_start == 0:
            friction_profile_fold = compute_friction_profile(mfpt[:, -1], mfpt[:, idx_start], pot_pos, pot) #case of first point only gives folding (small to big q)
            ind_counter_fold += 1
            fold_aver += friction_profile_fold
        elif idx_start == len(start_points) - 1:
            friction_profile_unfold = compute_friction_profile(mfpt[:, -1], mfpt[:, idx_start], pot_pos, pot, fold=False) # last point only gives unfolding (big to small q)
            ind_counter_unfold += 1
            unfold_aver += friction_profile_unfold

        else:
            ind_smaller = np.where(mfpt[:, -1] <= start_points[idx_start])[0] # first part of mfpt is unfolding (big to smaller q) until reaching 0 then folding 
            ind_larger = np.where(mfpt[:, -1] >= start_points[idx_start])[0]
            # print(len(ind_smaller), len(ind_larger))
            # print(mfpt[ind_larger, -1].shape, start_points[idx_start], idx_start, len(start_points)-1, start_points )
            friction_profile_unfold = compute_friction_profile(mfpt[ind_smaller, -1], mfpt[ind_smaller, idx_start], pot_pos[np.where(pot_pos <= start_points[idx_start])[0]], pot[np.where(pot_pos <= start_points[idx_start])[0]], fold=False)
            friction_profile_fold = compute_friction_profile(mfpt[ind_larger, -1], mfpt[ind_larger, idx_start], pot_pos[np.where(pot_pos >= start_points[idx_start])[0]], pot[np.where(pot_pos >= start_points[idx_start])[0]])

            ind_counter_unfold[cut_sides : len(friction_profile_unfold) - cut_sides] += 1 # add where friction profile can be calculated, average later 
            ind_counter_fold[len(friction_profile_unfold) + cut_sides : - cut_sides] += 1
            unfold_aver[cut_sides : len(friction_profile_unfold) - cut_sides] += friction_profile_unfold[cut_sides:-cut_sides]
            fold_aver[len(friction_profile_unfold) + cut_sides : - cut_sides] += friction_profile_fold[cut_sides:-cut_sides]
        
    unfold_aver = unfold_aver / ind_counter_unfold
    fold_aver = fold_aver / ind_counter_fold

    return fold_aver, unfold_aver

def test_fric_aver(points = 10, cut_sides=10):
    pos_potential = np.arange(0, 10, 0.005)
    potential = np.zeros(len(pos_potential))

    edges = np.append(pos_potential[::int(len(pos_potential) / (points))], pos_potential[-1])
    tmfp_tot = np.zeros((len(edges), len(pos_potential)))
    for i in range(len(edges)):
        tmfp_tot[i, :] = (pos_potential - edges[i])**2 / 2
    tmfp_tot = np.c_[tmfp_tot.T, pos_potential]

    fric_fold, fric_unfold = friction_aver(tmfp_tot, edges, pos_potential, potential, cut_sides=cut_sides)

    plt.plot(pos_potential[:-1], fric_fold / 2.49, ls='-', label='fold')
    plt.plot(pos_potential[:-1], fric_unfold / 2.49, ls='--', label='unfold')
    plt.legend(loc=0)
    plt.show()



def test_mfpt_from_friction(arg='ramp', skip=1):
    if arg == 'ramp':
        pos_potential = np.arange(0, 10, 0.005)
        potential = ramp_potential(pos_potential, U0=2)
        tmfp_theo = tmfp_ramp(pos_potential, U0=2)
        tmfp_theo2 = tmfp_ramp(pos_potential, U0=-2)[::-1]
        friction = compute_friction_profile(pos_potential[::skip], tmfp_theo[::skip], pos_potential, potential)
        friction2 = compute_friction_profile(pos_potential[::skip], tmfp_theo2[::skip], pos_potential, potential, fold=False)

    elif arg == 'parab':
        pos_potential = np.arange(0, 10, 0.005)
        potential = np.zeros(len(pos_potential))
        tmfp_theo = pos_potential**4/4 # free diffusion with parabolic friction profile
        tmfp_theo2 =  - (pos_potential**3 * 10 / 3 - pos_potential**4 / 4) # free diffusion with parabolic friction profile  (10**3 * 10 / 3 - 10**4 / 4)
        tmfp_theo2 -= np.min(tmfp_theo2)
        friction = compute_friction_profile(pos_potential[::skip], tmfp_theo[::skip], pos_potential, potential)
        friction2 = compute_friction_profile(pos_potential[::skip], tmfp_theo2[::skip], pos_potential, potential, fold=False)

    elif arg == 'free':
        pos_potential = np.arange(0, 10, 0.005)
        potential = np.zeros(len(pos_potential))
        tmfp_theo = pos_potential**2/2
        tmfp_theo2 = tmfp_theo[::-1]
        friction = compute_friction_profile(pos_potential[::skip], tmfp_theo[::skip], pos_potential, potential)
        friction2 = compute_friction_profile(pos_potential[::skip], tmfp_theo2[::skip], pos_potential, potential, fold=False)
    
    mfpt_back = compute_mfpt_from_gammaq(friction, pos_potential[::skip], potential, pos_potential)
    mfpt_back2 = compute_mfpt_from_gammaq(friction2, pos_potential[::skip], potential, pos_potential, fold=False)

    plt.plot(pos_potential[::skip], tmfp_theo[::skip])
    plt.plot(pos_potential[::skip], mfpt_back, ls='--')

    plt.plot(pos_potential[::skip], tmfp_theo2[::skip])
    plt.plot(pos_potential[::skip], mfpt_back2, ls='--')
    plt.xlabel('position')
    plt.ylabel('MFPT')
    plt.show()
    

def compute_mfpt(config):
    
    end_points = np.linspace(config.left_bound, config.right_bound, config.steps) # one needs passage events in order to obtain mfpt, thus one should not choose min and max - left side of potential less steep
    start_points = np.unique(np.append(end_points[::int(config.steps / config.n_startpoint)], end_points[-1]))

    fpt_sum = 0
    count_sum = 0

    i = 0
    for path in config.trj_list:
        if path.endswith('.txt'):
            trj = np.loadtxt(path)[::config.stride]
        elif path.endswith('.npy'):
            trj = np.load(path)[::config.stride]
        elif path.endswith('.h5'):
            with h5py.File(path, "r") as dset:
                trj = dset["hb4"][::config.stride]


        if config.mfpt == 'mfpt':
            fpt, count = get_mfpt(trj, start_points, end_points, config.dt * config.stride, return_sum=True)

        elif config.mfpt == 'mffpt':
            fpt, count = get_mffpt(trj, start_points, end_points, config.dt * config.stride, return_sum=True)

        
        i += 1
        fpt_sum += fpt
        count_sum += count


    mfpt = np.divide(fpt_sum, count_sum, where=count_sum!=0)  * config.dt * config.stride
    mfpt = np.c_[mfpt.T, end_points] #append last column as positions where mfpt is computed (to start col1 and end col2, respectively)
    np.save(config.path_save + config.mfpt + config.name + '_av' + '_rb' + str(config.right_bound) + '_lb' + str(config.left_bound)
            + '_bins' + str(config.steps) + '_nstart' + str(int(config.n_startpoint)) + '_stride' + str(config.stride), mfpt)

    return mfpt


class Config(pydantic.BaseModel):
    delimiter: str = "\t"
    #path_trj: str # trj saved as .npy or as .txt
    path_pot: str # potential saved as .npy or as .txt. If txt, then with "delimiter"
    path_save: str = "/net/storage/kanton/protein_gle_sim/mfpt_md/" #previously data04
    dt: float #time step has to be given in ps
    name: str
    steps: int = 100 # 100 steps across potential landscape, where mfpt is calculated
    stride: int = 1 # stride for loading trajectory

    left_bound: float
    right_bound: float # default is start = [left_bound, right_bound] and end points as linspace with steps in between bounds, see n_startpoint
    trj_list: list # list of paths to trj files
    mfpt: str = "mffpt" # "mfpt" or "mffpt"
    n_startpoint: int = 1 # number of start points - 1 (right bound is always added as last start point), default is start = [left_bound, right_bound] ie. 1 + 1 start points
    cut_ends: int = 1 # cut first and last cut_ends points of friction profile


def main():
    if len(sys.argv) < 2:
        raise ValueError("Please give path to config as input.")
    config_file = sys.argv[1]
    config = toml.load(config_file)
    config = Config(**config)
    print(f"{config=}")

    try:
        mfpt = np.load(config.path_save + config.mfpt + config.name + '_av' + '_rb' + str(config.right_bound) + '_lb' + str(config.left_bound)
            + '_bins' + str(config.steps) + '_nstart' + str(int(config.n_startpoint)) + '_stride' + str(config.stride) + '.npy')
    except:
        mfpt = compute_mfpt(config) # [:, 0] are mfpts from left_bound to positions, [:, -2] are mfpts from right_bound to positions, [:, -1] are positions
    pot_pos, pot = load_potential(config)

    end_points = np.linspace(config.left_bound, config.right_bound, config.steps)
    start_points = np.unique(np.append(end_points[::int(config.steps / config.n_startpoint)], end_points[-1]))
    # print(start_points)

    fric_f = compute_friction_profile(mfpt[:, -1], mfpt[:, 0], pot_pos, pot)
    fric_b = compute_friction_profile(mfpt[:, -1], mfpt[:, -2], pot_pos, pot, fold=False)
    plt.scatter(mfpt[:-1, -1], fric_f, label='forward')
    plt.scatter(mfpt[:-1, -1], fric_b, label='backward')

    fric_fold, fric_unfold = friction_aver(mfpt, start_points, pot_pos, pot, cut_sides=config.cut_ends)

    plt.plot(mfpt[:-1, -1], fric_fold, label='fold')
    plt.plot(mfpt[:-1, -1], fric_unfold, label='unfold')

    plt.xlabel('q')
    plt.ylabel('friction')
    plt.semilogy()
    plt.show()

    np.save(config.path_save + 'pos_dep_fric_' + config.name + '_rb' + str(config.right_bound) + '_lb' + str(config.left_bound)
            + '_bins' + str(config.steps) + '_nstart' + str(int(config.n_startpoint)), np.array([fric_fold, fric_unfold, mfpt[:-1, -1]]).T)


if __name__ == "__main__":
    main()
