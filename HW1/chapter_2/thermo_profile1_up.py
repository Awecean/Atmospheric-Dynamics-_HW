# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:54:33 2024
Refactored on [current date]

This script reads a data file containing pressure, temperature, and 
number of soundings and plots:
    1. Temperature versus Pressure
    2. Temperature versus Altitude

The data file (tropical_temp.dat) is assumed to have 3 columns with M rows.
Altitude is computed assuming a 0.5 km interval starting at 0 km.

Revisor: P-T, Lin B11501037
Revised Date: 2026/03/28
"""

import numpy as np
import matplotlib.pyplot as plt

def compute_altitude(num_rows, interval=0.5):
    """
    Compute altitude array given the number of data rows.
    
    Parameters:
        num_rows (int): Number of rows in the data.
        interval (float): Altitude interval in km. Default is 0.5.
        
    Returns:
        numpy.ndarray: Array of altitudes in km.
    """
    # Creates an array from 0 to (num_rows*interval) with step size of interval.
    return np.arange(0.5, (num_rows+1) * interval, interval)


def plot_temperature_pressure(ax, data):
    """
    Plot temperature vs pressure on the given axes.
    
    Parameters:
        ax (matplotlib.axes.Axes): Axes object to plot on.
        data (numpy.ndarray): Data array where column 1 is temperature and column 0 is pressure.
    """
    ax.plot(data[:, 1], data[:, 0],'b',label='Temperature (K)') # modified, added label
    #ax.set_ylim(1000, -100)  # Reverse y-axis: high pressure at the bottom
    ax.invert_yaxis()  # Invert y-axis to have pressure decrease upwards
    ax.set_xlabel('Value (K)') # modified, changed label to 'Value (K)'
    ax.set_ylabel('Pressure (mb)')
    ax.set_title('Temperature vs Pressure: Tropical Sounding')

def plot_temperature_altitude(ax, data, altitude):
    """
    Plot temperature vs altitude on the given axes.
    
    Parameters:
        ax (matplotlib.axes.Axes): Axes object to plot on.
        data (numpy.ndarray): Data array where column 1 is temperature.
        altitude (numpy.ndarray): Array of altitude values.
    """
    ax.plot(data[:, 1], altitude, 'b', label = 'Temperature (K)') # modified, added label
    #ax.set_ylim(altitude[0], altitude[-1])
    ax.set_xlabel('Value (K)') # modified, changed label to 'Value (K)'
    ax.set_ylabel('Geopotential Height (km)') # modified, changed label to 'Geopotential Height (km)'
    ax.set_title('Temperature vs Altitude: Tropical Sounding')

def compute_potential_temperature(temperature, pressure):
    """
    Compute potential temperature given temperature and pressure.
    
    Parameters:
        temperature (numpy.ndarray): Array of temperatures in K.
        pressure (numpy.ndarray): Array of pressures in mb.
        reference_pressure (float): Reference pressure in mb. Default is 1000 mb.
        
    Returns:
        numpy.ndarray: Array of potential temperatures in K.
    """
    R_d = 287  # Gas constant for dry air in J/(kg*K)
    c_p = 1004 # Specific heat at constant pressure for dry air in J/(kg*K)
    return temperature * (pressure[0] / pressure) ** (R_d / c_p)

def main():
    # Define file path and load the data
    file_path = 'tropical_temp.dat'
    data = np.loadtxt(file_path)
    print(np.size(data))
    # Compute altitude array based on the number of rows in the data
    altitude = compute_altitude(data.shape[0], interval=0.5)
    print(altitude)
    
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(9, 5), dpi=200)
    plt.subplots_adjust(wspace=0.4)
    
    # Plot temperature vs pressure in the first subplot
    plot_temperature_pressure(axes[0], data)
    
    # Compute potential temperature $\theta$ by own made function and plot it on the same axes
    Potential_Temperature = compute_potential_temperature(data[:,1], data[:,0])
    
    # 1st subplot, plot potential temperature vs pressure
    axes[0].plot(Potential_Temperature[:-1],data[:-1,0],'r',label='Potential Temperature (K)')
    axes[0].grid(True) # grid on
    axes[0].set_title('T & $\\theta$ vs. Pressure')
    axes[0].legend(loc=4)
    print(Potential_Temperature)


    # Plot temperature vs altitude in the second subplot
    plot_temperature_altitude(axes[1], data, altitude*1e3) # Convert altitude to meters for plotting
    
    # 2nd subplot, plot potential temperature vs altitude
    axes[1].plot(Potential_Temperature, altitude*1e3,'r',label='Potential Temperature (K)')
    axes[1].grid(True) # grid on
    axes[1].set_ylabel('Geopotential Height (m)') # modified, changed label to 'Geopotential Height (km)'
    axes[1].set_title('T & $\\theta$ vs. Geopotential Height')
    axes[1].legend(loc=4)

    # I thought the presentation's graph is wrong, about the altitude.
    # The altitude should be in range [0.5, 35]km, not [0, 35.5]km
    
    # Display the figure
    plt.tight_layout()
    plt.savefig('thermo_profile1_up.png', dpi=200) # Save the figure as a PNG file with specified DPI
    plt.show()

if __name__ == '__main__':
    main()
    

