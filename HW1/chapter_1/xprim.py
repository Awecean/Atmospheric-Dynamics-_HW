# xprim.py
"""
Module for computing time derivatives used in the coriolis simulation.
Contains two functions:
    - xprim1: ODEs with curvature terms included.
    - xprim2: ODEs with curvature terms omitted.
    
Both functions return the derivatives of:
    [du/dt, dv/dt, d(longitude)/dt, d(latitude)/dt]
    
Parameters:
    - t: Time variable (not used explicitly in the formulas).
    - n: Array of state variables [u, v, x (longitude), y (latitude)].
    - a: Radius of the Earth.
    - omega: Angular velocity of the Earth.
    
Author: TDRC
"""

import numpy as np

def xprim1(t, n, a, omega):
    """
    Compute time derivatives with curvature terms included.
    
    Parameters:
        t (float): Time.
        n (array): [u, v, longitude, latitude] where latitude is in radians.
        a (float): Radius of the Earth.
        omega (float): Angular velocity of the Earth.
    
    Returns:
        prim (array): [du/dt, dv/dt, d(longitude)/dt, d(latitude)/dt]
    """
    prim = np.zeros(4)
    # Zonal momentum equation with curvature term
    prim[0] = (2 * omega + n[0] / (a * np.cos(n[3]))) * np.sin(n[3]) * n[1]
    # Meridional momentum equation with curvature term
    prim[1] = - (2 * omega + n[0] / (a * np.cos(n[3]))) * np.sin(n[3]) * n[0]
    # Zonal movement (longitude evolution)
    prim[2] = n[0] / (a * np.cos(n[3]))
    # Meridional movement (latitude evolution)
    prim[3] = n[1] / a
    return prim

def xprim2(t, n, a, omega):
    """
    Compute time derivatives with curvature terms omitted.
    
    Parameters:
        t (float): Time.
        n (array): [u, v, longitude, latitude] where latitude is in radians.
        a (float): Radius of the Earth.
        omega (float): Angular velocity of the Earth.
    
    Returns:
        prim (array): [du/dt, dv/dt, d(longitude)/dt, d(latitude)/dt]
    """
    prim = np.zeros(4)
    # Zonal momentum equation without curvature term
    prim[0] = 2 * omega * np.sin(n[3]) * n[1]
    # Meridional momentum equation without curvature term
    prim[1] = - 2 * omega * np.sin(n[3]) * n[0]
    # Zonal movement (longitude evolution)
    prim[2] = n[0] / (a * np.cos(n[3]))
    # Meridional movement (latitude evolution)
    prim[3] = n[1] / a
    return prim
