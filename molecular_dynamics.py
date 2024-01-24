import hoomd
import hoomd.md as md
import random
import numpy as np
import gsd.hoomd
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import sympy as sp
import warnings
from datetime import date
import pandas as pd
import os
warnings.filterwarnings('ignore')
os.chdir(os.getcwd()+'/GSD')


def calc_distance(part1, part2):
    """
    Description: Calculate the distance between two particles.

    Inputs: 
        - particle 1
        - particle 2
    Output: 
        - distance between particle 1 and 2
    """
    return(math.sqrt((part1[0]-part2[0])**2 + 
                     (part1[1]-part2[1])**2 + 
                     (part1[2]-part2[2])**2))

def init_particle_positions(n_particles, cutoff_distance):
    """
    Description: Initialize particle positions randomly.

    Inputs: 
        - n_particles (int): number of particles
        - cutoff_distance (int): particles within this distance are invalid
    Output: 
        - list of particle positions
    """
    particle_positions = []

    while len(particle_positions) < n_particles:
        l = [random.randint(-L//3, L//3) for i in range(3)]
        distances = [calc_distance(l, particle) for particle in particle_positions]
        
        if all(distance >= cutoff_distance for distance in distances):
            particle_positions.append(l)

    print(f'Particle positions initialized: \n{particle_positions}')
    return particle_positions

def setup_frame(N, positions, L, type):
    """
    Description: Set up a frame object for maximum_forceturing simulation states.

    Inputs: 
        - N  (int): number of particles
        - positions (list): list of particle positions
        - L (int): box dimension
        - type (str): name of particle
    Output: 
        - Frame: Frame object to maximum_forceture frames in the simulation
    """
    Frame = gsd.hoomd.Frame()
    Frame.particles.N = N
    Frame.particles.position = positions
    Frame.particles.typeid = [0] * N
    Frame.configuration.box = [L, L, L, 0, 0, 0]
    Frame.particles.types = [type]
    return Frame

def create_DLVO_potential(rad, start_i):
    """
    Description: Create DLVO potential list without infinite potentials.

    Inputs:
        - rad (array): radius list
        - start_i (int): point to start potential calculations
    Outputs:
        - V (DataFrame): dataset with radius, potential, and force
    """
    V = pd.DataFrame()
    potentials = [-a/6*(2*r*r/(i**2-(r+r)**2)
                            + 2*r*r/(i**2-(r-r)**2)
                            + np.log((i**2-(r+r)**2)/(i**2-(r-r)**2)))
                            + (r*r/(r+r))*z*np.exp(-k*(i-(r+r)))
                            for i in rad]

    V['DLVO Potential'] = potentials
    V['Radius'] = rad[:len(V['DLVO Potential'])]
    V = V.dropna()
    init_potential = pd.Series([V['DLVO Potential'].values[0]] * start_i)
    init_radius = pd.Series(range(start_i))
    append = pd.DataFrame({'Radius': init_radius, 'DLVO Potential': init_potential})
    V = pd.concat([append, V], ignore_index=True)
    print('Potential created')

    f = list(np.diff(V['DLVO Potential'])/np.diff(V['Radius']))
    for i in range(start_i):
        f[i] = f[start_i]
    f.append(f[-1])
    f = [-i for i in f]
    f = [maximum_force if i >= maximum_force else i for i in f]
    V['Force'] = f
    print('Force calculations completed')
    return V

def plot_potential_force(V, end_point):
    """
    Description: Plot potential and forces to visualize interpolated values.
    
    Inputs: 
        - r_shift (list): shifted radius list
        - potential_list (list): list of potentials
        - force_list (list): list of forces
    Output: 
        - plots interpolated potential and original potential vs radius
        - plots force vs radius
    """
    _, axs = plt.subplots(2)
    axs[0].plot(V['Radius'], V['DLVO Potential'])
    axs[0].set_xlim([-10, end_point + 50])
    axs[0].set_title('Potential v Radius')
    axs[1].plot(V['Radius'], V['Force'])
    axs[1].set_xlim([-10, end_point + 50])
    axs[1].set_title('Force v Radius')
    plt.show()


# PARTICLE PARAMETERS
N_particles = 5                             # num of particles
r = 1000                                    # radius of particle
L = 20000                                   # dimension of sim cube
a = 7e-46                                   # Hamaker constant
k = 0.001                                   # screening parameter
z = 1                                       # surface potential
b = 100                                     # buffer length
t_step = 5*10**3                              # dt between steps
kt = 1.0                                    # kT constant
drag = 1000.0                               # drag coefficient
radius_min = 5
# SIMULATION PARAMETERS
initial_cut = 6000                          # cutoff distance for particle interactions
start = 2000                                # start radius
end = initial_cut                           # end radius
num_steps = 100000                          # num of data points
maximum_force = 30                          # maximum force values
total_time = 6*10**13                       # 60 s = 6*10**13 ps
sim_time = int(total_time/t_step)           # 6e9 simulation steps
num_frames = 10**6                          # number of frames to output
period = int(sim_time/num_frames)           # how often to take frames
visualize = False                           # True: plot Potential/Force vs Radius


# FILE HANDLING
initial_file = 'init.gsd'
date = date.today()
current_date = date.isoformat()
hoomd_file_name = current_date + '_'+str(t_step)+'_hoomd_data.gsd'


# INITIALIZE PARTICLE POSITIONS
part_positions = init_particle_positions(N_particles, initial_cut)
Frame = setup_frame(N_particles, part_positions, L, 'Silica')
with gsd.hoomd.open(name=initial_file, mode='wb') as f:
    f.append(Frame)


# DLVO POTENTIAL
radius = np.linspace(start, end, num_steps)
potential_and_force = create_DLVO_potential(radius, start)


# INTEGRATOR SET-UP
cell = hoomd.md.nlist.Cell(buffer = b)
tabular = hoomd.md.pair.Table(nlist = cell)
tabular.params[('Silica', 'Silica')] = dict(r_min = radius_min,
                                            U = potential_and_force['DLVO Potential'], 
                                            F = potential_and_force['Force'])
tabular.r_cut[('Silica', 'Silica')] = initial_cut
brownian = hoomd.md.methods.Brownian(filter = hoomd.filter.All(),
                                     kT = kt, 
                                     default_gamma = drag)


# SIMULATION SET-UP
integrator = hoomd.md.Integrator(dt = t_step)
integrator.methods.append(brownian)
integrator.forces.append(tabular)
cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device = cpu, seed = 1)
sim.operations.integrator = integrator
sim.create_state_from_gsd(filename = initial_file)
sim.state.thermalize_particle_momenta(filter = hoomd.filter.All(), kT = kt)


# RUN SIMULATION AND OUTPUT
print('Running simulation')
gsd_writer = hoomd.write.GSD(filename = hoomd_file_name,
                             trigger = hoomd.trigger.Periodic(period),
                             mode = 'wb')
sim.operations.writers.append(gsd_writer)

if visualize:
    plot_potential_force(potential_and_force, initial_cut)

for i in tqdm(range(sim_time)):
    sim.run(1)