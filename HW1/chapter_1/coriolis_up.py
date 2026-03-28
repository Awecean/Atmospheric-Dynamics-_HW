# coriolis.py
"""
Module Description:
    This module simulates constant angular momentum trajectories in spherical coordinates
    and compares cases with and without curvature terms.


    
Functions:
    - Computes trajectories based on given parameters (initial latitude, zonal wind u0,
      meridional wind v0, runtime in days).
    - Uses differential equation functions from xprim.py for numerical integration.
    - Plots trajectories for both cases (with and without curvature terms).

Usage:
    Call run_simulation() with appropriate parameters to execute the simulation.

Author: TDRC
Revisor: P-T, Lin B11501037
Revised Date: 2026/03/28
"""

import numpy as np
import scipy.integrate as inte
import matplotlib.pyplot as plt
from xprim import xprim1, xprim2  # Corrected import

def run_simulation(init_lat, u0, v0, runtime_days, a=6.37e6, omega=7.292e-5):
    """
    Run the Coriolis force trajectory simulation (both with and without curvature terms).
    
    Parameters:
        init_lat (float): Initial latitude (degrees).
        u0 (float): Initial zonal wind speed (m/s).
        v0 (float): Initial meridional wind speed (m/s).
        runtime_days (float): Simulation duration (days).
        a (float): Earth’s radius (default: 6.37e6 m).
        omega (float): Earth’s angular velocity (default: 7.292e-5 s^-1).
    
    Returns:
        None (Plots the trajectories).
    """
    # Convert initial latitude from degrees to radians
    lat0 = np.radians(init_lat)
    # Convert simulation duration from days to seconds
    time_total = runtime_days * 24 * 3600
    # Define time evaluation points
    t_eval = np.linspace(0, time_total, int(time_total) + 1)
    
    # Solve ODE with curvature terms
    sol_with = inte.solve_ivp(xprim1, [0, time_total], [u0, v0, 0, lat0],
                              vectorized=True, args=(a, omega), t_eval=t_eval)
    long_with = np.degrees(sol_with.y[2])
    lat_with = np.degrees(sol_with.y[3])
    print(np.max(lat_with),np.min(lat_with))
    mask = (lat_with[1:] >= np.rad2deg(lat0)) & (lat_with[:-1] < np.rad2deg(lat0))
    tag = np.where(mask)[0][0]  # Get the first index where the condition is met
    #print
    print(f'time:{t_eval[tag]/86400:.2f} days, longitude:{long_with[tag]} degrees, latitude:{lat_with[tag]} degrees')
    global t_period_with_curvature
    t_period_with_curvature = t_eval[tag+1] 

    # Plot trajectory with curvature terms
    plt.figure(figsize=[6, 4], dpi=200)
    plt.plot(long_with, lat_with, label='With curvature terms')
    plt.plot(long_with[0], lat_with[0], 'kd', label='Initial position')
    plt.title(f'Trajectory with curvature terms\n u0 = {u0}, v0 = {v0}')
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'Coriolis_with_curvature_{init_lat}_{u0}_{v0}.png')

    plt.show()

    
    # Solve ODE without curvature terms
    sol_without = inte.solve_ivp(xprim2, [0, time_total], [u0, v0, 0, lat0],
                                 vectorized=True, args=(a, omega), t_eval=t_eval)
    long_without = np.degrees(sol_without.y[2])
    lat_without = np.degrees(sol_without.y[3])
    print(np.max(lat_without),np.min(lat_without))


    # Plot trajectory without curvature terms
    plt.figure(figsize=[6, 4], dpi=200)
    plt.plot(long_without, lat_without, 'r', label='Without curvature terms')
    plt.plot(long_without[0], lat_without[0], 'kd', label='Initial position')
    plt.title(f'Trajectory without curvature terms\n u0 = {u0}, v0 = {v0}')
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    mask = (lat_without[1:] >= np.rad2deg(lat0)) & (lat_without[:-1] < np.rad2deg(lat0))
    tag = np.where(mask)[0][0]  # Get the first index where the condition is met
    #print
    print(f'time:{t_eval[tag]/86400:.2f} days, longitude:{long_without[tag]} degrees, latitude:{lat_without[tag]} degrees')
    global t_period_without_curvature
    t_period_without_curvature = t_eval[tag+1]
    plt.savefig(f'Coriolis_without_curvature_{init_lat}_{u0}_{v0}.png')

    plt.show()

def plot_1_circulation_in_1_window(init_lat, u0, v0, runtime_days_with_c, runtime_days_without_c, a=6.37e6, omega=7.292e-5):
    """
    Draw two trajectories (with and without curvature terms) for one full circulation in the same plot for better comparison.
    """
    
    lat0 = np.radians(init_lat)
    # Convert simulation duration from days to seconds
    time_total_c = runtime_days_with_c * 24 * 3600
    time_total_without_c = runtime_days_without_c * 24 * 3600
    # Define time evaluation points
    t_eval_c = np.linspace(0, time_total_c, int(time_total_c) + 1)
    t_eval_without_c = np.linspace(0, time_total_without_c, int(time_total_without_c) + 1)

    # Solve ODE with curvature terms
    sol_with_one = inte.solve_ivp(xprim1, [0, time_total_c], [u0, v0, 0, lat0],
                              vectorized=True, args=(a, omega), t_eval=t_eval_c)
    long_with_one = np.degrees(sol_with_one.y[2])
    lat_with_one = np.degrees(sol_with_one.y[3])

    sol_without_one = inte.solve_ivp(xprim2, [0, time_total_without_c], [u0, v0, 0, lat0],
                                 vectorized=True, args=(a, omega), t_eval=t_eval_without_c)
    long_without_one = np.degrees(sol_without_one.y[2])
    lat_without_one = np.degrees(sol_without_one.y[3])

    plt.plot(figsize=(16, 10), dpi=200)
    
    # Subplot for trajectory with curvature terms 
    min_lat = min(np.min(lat_with_one), np.min(lat_without_one))
    max_lat = max(np.max(lat_with_one), np.max(lat_without_one))
    min_long = min(np.min(long_with_one), np.min(long_without_one))
    max_long = max(np.max(long_with_one), np.max(long_without_one))
    plt.plot(long_with_one, lat_with_one, label='With curvature terms') #line color: rainbow
    plt.plot(long_without_one, lat_without_one, 'r', label='Without curvature terms')
    plt.plot(long_with_one[0], lat_with_one[0], 'kd', label='Initial position')
    plt.title(f'Trajectory comparison for one full circulation u0 = {u0}, v0 = {v0}:\n {runtime_days_with_c:.4f} days (with curvature)\n {runtime_days_without_c:.4f} days (without curvature)')
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.ylim(min_lat-2, max_lat+2)  # Set y-axis limits to focus on the region around the initial latitude
    plt.xlim(min_long-2, max_long+10)  # Set x-axis limits to show one       
    plt.legend(loc='best')  # Place legend outside the plot area

    plt.tight_layout()
    plt.savefig(f'Trajectory_comparison_1window_{init_lat}_{u0}_{v0}.png')
    plt.show()

def plot_1_circulation_in_2_windows(init_lat, u0, v0, runtime_days_with_c, runtime_days_without_c, a=6.37e6, omega=7.292e-5):
    """
    The period for one full circulation is determined from the previous simulation results.
    Plot the trajectories for one full circulation for both cases (with and without curvature terms).
    And the period are different for the two cases, so we need to specify the runtime for each case separately.
    """
    
    lat0 = np.radians(init_lat)
    # Convert simulation duration from days to seconds
    time_total_c = runtime_days_with_c * 24 * 3600
    time_total_without_c = runtime_days_without_c * 24 * 3600
    # Define time evaluation points
    t_eval_c = np.linspace(0, time_total_c, int(time_total_c) + 1)
    t_eval_without_c = np.linspace(0, time_total_without_c, int(time_total_without_c) + 1)

    # Solve ODE with curvature terms
    sol_with_one = inte.solve_ivp(xprim1, [0, time_total_c], [u0, v0, 0, lat0],
                              vectorized=True, args=(a, omega), t_eval=t_eval_c)
    long_with_one = np.degrees(sol_with_one.y[2])
    lat_with_one = np.degrees(sol_with_one.y[3])

    sol_without_one = inte.solve_ivp(xprim2, [0, time_total_without_c], [u0, v0, 0, lat0],
                                 vectorized=True, args=(a, omega), t_eval=t_eval_without_c)
    long_without_one = np.degrees(sol_without_one.y[2])
    lat_without_one = np.degrees(sol_without_one.y[3])

    fig, axes = plt.subplots(1, 2, figsize=(9, 5), dpi=200)
    
    # Subplot for trajectory with curvature terms 
    min_lat = min(np.min(lat_with_one), np.min(lat_without_one))
    max_lat = max(np.max(lat_with_one), np.max(lat_without_one))
    min_long = min(np.min(long_with_one), np.min(long_without_one))
    max_long = max(np.max(long_with_one), np.max(long_without_one))
    

    axes[0].plot(long_with_one, lat_with_one, label='With curvature terms') #line color: rainbow
    axes[0].plot(long_with_one[0], lat_with_one[0], 'kd', label='Initial position')
    axes[0].set_title(f'With curvature terms\nTime for one full circulation: {runtime_days_with_c:.4f} days')  
    axes[0].set_xlabel('Longitude (degrees)')
    axes[0].set_ylabel('Latitude (degrees)')
    axes[0].set_ylim(min_lat-2, max_lat+2)  # Set y-axis limits to focus on the region around the initial latitude
    axes[0].set_xlim(min_long-2, max_long+2)  # Set x-axis limits to show one full circulation

    axes[0].legend(loc='lower right')
    # Subplot for trajectory without curvature terms
    
    axes[1].plot(long_without_one, lat_without_one, 'r', label='Without curvature terms')
    axes[1].plot(long_without_one[0], lat_without_one[0], 'kd', label='Initial position')
    axes[1].set_title(f'Without curvature terms\nTime for one full circulation: {runtime_days_without_c:.4f} days')
    axes[1].set_xlabel('Longitude (degrees)')
    axes[1].set_ylim(min_lat-2, max_lat+2)  # Set y-axis limits to focus on the region around the initial latitude
    axes[1].set_xlim(min_long-2, max_long+2)  # Set x-axis limits to show one full circulation

    axes[1].set_ylabel('')  # Remove y-axis label for the second subplot
    axes[1].legend(loc='lower right')

    fig.suptitle(f'One Period_Trajectory Comparison u0 = {u0}, v0 = {v0}')  # Set the overall title for the figure
    plt.tight_layout()
    plt.savefig(f'Coriolis_comparison_2window_{init_lat}_{u0}_{v0}.png')
    plt.show()

# Example calls
def main():
    """
    Execute the simulation for the two scenarios from the problem statement:
    
    (a) Initial latitude 60°, u0 = 40 m/s, v0 = 40 m/s, runtime = 5 days.
    (b) Initial latitude 60°, u0 = 40 m/s, v0 = 80 m/s, runtime adjusted for full circuit observation.
    """
    ## Case in code note but not in presentation
    # Case (a)
    print('Running Case in code but not in presentation:')
    print("Running Case (a):")
    run_simulation(init_lat=60, u0=40, v0=40, runtime_days=5) # case (a) in code but not in presentation
    # Case (b)
    print("Running Case (b):")
    run_simulation(init_lat=60, u0=40, v0=80, runtime_days=5) # case (b) in code but not in presentation
    plot_1_circulation_in_1_window(init_lat=60, u0=40, v0=80, runtime_days_with_c=t_period_with_curvature/86400, runtime_days_without_c=t_period_without_curvature/86400)
    # One circulation for with/without curvature terms, plot in the same window for better comparison
    plot_1_circulation_in_2_windows(init_lat=60, u0=40, v0=80, runtime_days_with_c=t_period_with_curvature/86400, runtime_days_without_c=t_period_without_curvature/86400)

    ## Case in Presentation
    # Case (a)
    print('Running Case in Presentation')
    print("Running Case (a):")
    run_simulation(init_lat=60, u0=0, v0=40, runtime_days=5)
    print("Running Case (b):")
    run_simulation(init_lat=60, u0=0, v0=80, runtime_days=5)   
    # One circulation for with/without curvature terms, plot in the two windows separately
    plot_1_circulation_in_1_window(init_lat=60, u0=0, v0=80, runtime_days_with_c=t_period_with_curvature/86400, runtime_days_without_c=t_period_without_curvature/86400)
    # One circulation for with/without curvature terms, plot in the same window for better comparison
    plot_1_circulation_in_2_windows(init_lat=60, u0=0, v0=80, runtime_days_with_c=t_period_with_curvature/86400, runtime_days_without_c=t_period_without_curvature/86400)
    
    period = np.pi / (2 * np.pi / 86400) / np.sin(np.radians(60))  # Time for full circuit
    
    print(f'analytical period: {period/86400:.4f} days')
    print(f'experimental period (with curvature): {t_period_with_curvature/86400:.4f} days')
    print(f'experimental period (without curvature): {t_period_without_curvature/86400:.4f} days')

if __name__ == '__main__':
    main()