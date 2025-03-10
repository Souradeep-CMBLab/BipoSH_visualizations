import numpy as np
import scipy.integrate as spi
import scipy.interpolate as spint
from scipy.special import eval_legendre
import matplotlib.pyplot as plt
import healpy as hp
import camb
import os


def get_C_psi_legendre(Cl, psi_values):
    """
    Function to calculate the correlation from Angular power spectrum using Legendre Summation
    Parameters:
    Cl: Power spectrum
    psi_values: Values of psi
    Returns:
    C(psi_values): Correlation values
    """
    c_psi = np.zeros_like(psi_values)   # Planck C(psi)
    ell = np.arange(len(Cl))
    
    for i, t in enumerate(psi_values):
        legendre_vals = eval_legendre(ell, np.cos(t))  # P_l(cosθ)

    c_psi[i] = np.sum((2 * ell+ 1) / (4 * np.pi) * Cl* legendre_vals)
  
    return c_psi

def get_nSI_Map(o_map, A,p, f= "cos"):
    """
    Function to generate a non-Gaussian CMB map with nSI signal
    Parameters:
    o_map: Original CMB map
    A: Modulation factor
    p: Special Direction
    nside: HEALPix resolution parameter (default=512)
    f: Function to generate anisotropy
    Returns:
    nSI_map: CMB map with nSI signal
    """
    nside = hp.get_nside(o_map)
    npix = hp.nside2npix(nside)
    

    if f == "cos":
        vecs = np.array(hp.pix2vec(nside, np.arange(npix))) # Get pixel coordinates
        dot_prod = np.einsum('i,ij->j', p, vecs)/np.linalg.norm(p) # Compute dot product
        F = dot_prod
    elif f == "sin":
        vecs = np.array(hp.pix2vec(nside, np.arange(npix))) # Get pixel coordinates
        dot_prod = np.einsum('i,ij->j', p, vecs)/np.linalg.norm(p) # Compute dot product
        F = np.sqrt(1 - dot_prod**2)
    else:
        theta_all, phi_all = hp.pix2ang(nside, np.arange(hp.nside2npix(nside))) # Get pixel angles
        if p == [0,0,1]:
            F = f(theta_all, phi_all)
        else:
            F = f(theta_all, phi_all)
            "Define the Reverse rotation to get the F rotated such that F features are defined around Direction P. So Z-axis -> P"
            thetap, phip = hp.vec2ang(np.array(p))
            # print(p, thetap, phip)
            if phip <= np.pi:
                Phi = np.arctan(-1/np.tan(phip[0])) + np.pi 
            else:
                Phi = np.arctan(-1/np.tan(phip[0]))

            # Now calculate the Euler angles:
            alpha = -Phi + np.pi/2
            beta = thetap[0]
            gamma = 0 #-alpha
            # print(Phi, alpha, beta, gamma)

            Reverse_Rotation1 = hp.rotator.Rotator( rot = [0, beta, 0], deg =False)
            Reverse_Rotation2 = hp.rotator.Rotator( rot = [alpha, 0, 0], deg =False)

            F = Reverse_Rotation2.rotate_map_pixel(Reverse_Rotation1.rotate_map_pixel(F))

    nSI_map = o_map*(1 + A*F)    
    return nSI_map


def get_C001(rmap, psi_values, Delta_psi):
    nside = hp.get_nside(rmap)
    C_001 = np.zeros_like(psi_values)
    theta_all, phi_all = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))

    # Calculate C_001(psi) = Integral dvarphi DeltaT(0,0) * DeltaT(psi, varphi)
    for i, psi in enumerate(psi_values):
        # Create an array to store the temperature differences
        temp_prod = []

        # Get the pixel angles
        same_theta_pixels = np.where((theta_all >= (psi - Delta_psi/2)) & (theta_all <= (psi + Delta_psi/2)))[0]
        temp_prod.append(rmap[hp.ang2pix(nside, theta_all[same_theta_pixels], phi_all[same_theta_pixels])]*rmap[hp.ang2pix(nside, 0, 0)])


        if len(temp_prod) > 0:
            C_001[i] = np.mean(temp_prod)


    return C_001


def get_anchored_correlation(map, alpha, beta, gamma):

    """Compute the C_001(psi, phi) correlation function for a map, by averaging over n Rotated maps, rotations are choosed randomly."""
    Anchored_correlation = np.zeros_like(map)
    nside = hp.get_nside(map)
    n_anchors = len(gamma)
    
    for i in range(n_anchors):
        if i % 200 == 0:
            print(f"Rotated Sky Map {i}: alpha={alpha[i]}, beta={beta[i]}, gamma={gamma[i]}")
        rotation= hp.rotator.Rotator( rot = [alpha[i], beta[i], gamma[i]], deg =False)
        rotated_map = rotation.rotate_map_pixel(map)
        T_001 = rotated_map[hp.ang2pix(nside, 0, 0)]
        Anchored_correlation += T_001 * rotated_map

        if i>0 and i%10000 ==0:
            output_filename = f"correlation_map_{nside}_{i}.fits"
            hp.write_map(output_filename, Anchored_correlation/i, overwrite=True)
            print(f" Correlation Map saved as {output_filename}")
            previous_file = f"correlation_map_{nside}_{i-1000}.fits"
            if os.path.exists(previous_file):
                os.remove(previous_file)
                print(f" Removed {previous_file}")

    return Anchored_correlation / len(gamma)


def get_anchored_correlation_rndm(map, anchors):

    """Compute the C_001(psi, phi) correlation function for a map, by averaging over n anchor points choosen randomly from npix."""
    Anchored_correlation = np.zeros_like(map)
    nside = hp.get_nside(map)
    n_anchors = len(anchors)
    theta_anchors, phi_anchors = hp.pix2ang(nside, anchors, lonlat = False)  # Get anchor points

    # Calculate the angles for rotation of anchor point to z-axis
    omega = -theta_anchors
    Theta = np.pi / 2
    # Phi is given by the inverse tangent, but for spherical coordinates we already have it from phi1
    # if phi_anchors <= np.pi:    #     Phi = np.arctan(-1/np.tan(phi_anchors)) + np.pi # else:    #     Phi = np.arctan(-1/np.tan(phi_anchors))
    Phi = np.where(phi_anchors <= np.pi,   
                np.arctan(-1/np.tan(phi_anchors)) + np.pi, 
                np.arctan(-1/np.tan(phi_anchors)))

    # Now calculate the Euler angles:
    alpha = Phi - np.pi/2
    beta = -theta_anchors # omega
    gamma = np.zeros_like(alpha)  #-alpha

    for i in range(n_anchors):

        if i % 200 == 0:
            print(f"Rotated Sky Map {i}: alpha={alpha[i]}, beta={beta[i]}, gamma={gamma[i]}")
        rotation= hp.rotator.Rotator( rot = [alpha[i], beta[i], gamma[i]], deg =False)
        rotated_map = rotation.rotate_map_pixel(map)
        T_001 = rotated_map[hp.ang2pix(nside, 0, 0)]
        Anchored_correlation += T_001 * rotated_map

        if i == 0:
            hp.mollview(map, title="Original Sky Map", unit="K", norm="hist", cmap = "coolwarm")
            hp.graticule()
            hp.projscatter(theta_anchors[i], phi_anchors[i], marker='*', color='black', s=100, label= "Anchor Point original")
            plt.legend()
            plt.show()
            theta_rot, phi_rot = rotation((theta_anchors[i], phi_anchors[i]))
            hp.mollview(rotated_map, title="Rotated Sky Map", unit="K", norm="hist", cmap = "coolwarm")
            hp.graticule()
            hp.projscatter(0, 0, marker='*', color='red', s=100)
            hp.projscatter(theta_rot, phi_rot, marker='*', color='black', s=100, label= "Anchor Point after Rotation")
            plt.legend()
            plt.show()

        if i>0 and i%10000 ==0:
            output_filename = f"correlation_map_{nside}_{i}.fits"
            hp.write_map(output_filename, Anchored_correlation/i, overwrite=True)
            print(f" Correlation Map saved as {output_filename}")
            previous_file = f"correlation_map_{nside}_{i-1000}.fits"
            if os.path.exists(previous_file):
                os.remove(previous_file)
                print(f" Removed {previous_file}")


    return Anchored_correlation / n_anchors


def get_C001_psi(map):

    "Here the given map is the Averaged correlation map produced, C(theta) calculated by averaging over all phi values"
    nside = hp.get_nside(map)
    theta_all, phi_all = hp.pix2ang(nside, np.arange(hp.nside2npix(nside))) # Get pixel angles

    unique_theta = np.unique(theta_all)  # Get unique theta values
    num_unique_theta = len(unique_theta)  # Count unique values

    print(f"Number of distinct theta values: {num_unique_theta}")
    print(f"Unique theta values: {unique_theta}")


    C_0001_EA = np.zeros_like(unique_theta) # Array to store the C_001 values

    for i, theta in enumerate(unique_theta):
        same_theta_pixels = np.where(theta_all == theta)[0]
        C_0001_EA[i] = np.mean(map[same_theta_pixels])  

    return unique_theta, C_0001_EA


def get_C001_phi(map, n_phi = 1000):

    "Here the given map is the Averaged correlation map produced, C(phi) calculated by averaging over all theta values"
    
    nside = hp.get_nside(map)
    theta_all, phi_all = hp.pix2ang(nside, np.arange(hp.nside2npix(nside))) # Get pixel angles

    if n_phi == False:
        unique_phi = np.unique(phi_all)  # Get unique theta values
        unique_phi = unique_phi[::5]
        num_unique_phi = len(unique_phi)  # Count unique values
    else:
        Delta_phi = 2*np.pi / n_phi
        unique_phi = np.linspace(0, 2*np.pi, n_phi)  # Get unique theta values

    print(f"Number of distinct phi values: { len(unique_phi) }")
    # print(f"Unique phi values: {unique_phi}")


    C_0001_EA = np.zeros_like(unique_phi) # Array to store the C_001 values

    for i, phi in enumerate(unique_phi):
        if n_phi == False:
            same_phi_pixels = np.where(phi_all == phi)[0]
        else:
            same_phi_pixels = np.where((phi_all >= (phi - Delta_phi/2)) & (phi_all <= (phi + Delta_phi/2)))[0]
        C_0001_EA[i] = np.mean(map[same_phi_pixels])  

    return unique_phi, C_0001_EA









