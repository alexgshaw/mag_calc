from __future__ import division
from scipy.optimize import least_squares
import numpy as np

class MagCalc:
    """Object to calculate magnetic fields in an atomic structure.

    Attributes:
        atoms (ndarray): The positions of the atoms in the crystal structure.
            Should be a 2D numpy array with each row as a 1D numpy array with 3
            values designating an x,y,z location.
        spins (ndarray): The magnetic dipole moments of the atoms in the crystal
            structure. Should be a 2D numpy array with each row as a 1D numpy
            array with 3 values designating an i,j,k spin direction.
        locations (ndarray): The locations of where to calculate the magnetic
            field. Should be a 2D numpy array with each row as a 1D numpy array
            with 3 values designating an x,y,z location.
    """

    def __init__(self,
                 atoms,
                 spins,
                 locations=None,
                 g_factor=1,
                 spin=1/2,
                 magneton='mu_B'):
        """Initializes the spins and atomic positions of a crystal structure.
        Optionally intializes the locations for where to calculate the field.

        Parameters:
            atoms (ndarray): The positions of the atoms in the crystal
                structure. Should be a 2D numpy array with each row as a 1D
                numpy array with 3 values designating an x,y,z location.
            spins (ndarray): The magnetic dipole moments of the atoms in the
                crystal structure. Should be a 2D numpy array with each row as a
                1D numpy array with 3 values designating an i,j,k spin
                direction.
            locations (ndarray): Optional. The locations of where to calculate
                the magnetic field. Should be a 2D numpy array with each row as
                a 1D numpy array with 3 values designating an x,y,z location.
            g_factor (int or float): The dimensionless g-factor used to
                calculate the spin magnetic moment.
            spin (int or float): The spin quantum number.
            magneton (string): Either 'mu_B' or 'mu_N' depending on whether the
                magnetic moment depends on the nuclei or electrons.
        """

        CONST_DICT = {'mu_B':9.274009994e-24, 'mu_N':5.050783699e-27}

        try:
            const = CONST_DICT[magneton]
        except:
            raise Exception("magneton must be equal to 'mu_B' or 'mu_N'")


        eigenvalue = (spin * (spin+1))**(1/2)
        self.atoms = atoms
        self.spins = spins * g_factor * eigenvalue * const

        if locations is not None:
            self.locations = locations

    def calculate_field(self,
                        location,
                        return_vector=True,
                        mask_radius=None):
        """ Calculates the magnetic field at the specified location.

        Parameters:
            location (ndarray): A 1D numpy array of length 3 specifying where to
                calculate the magnetic field.
            return_vector (boolean): Optional, default is True. See below for
                details.
            mask_radius (int or float): Optional, default is None. If
                mask_radius is None, all atoms and spins will be taken into
                account for the calculation. If mask_radius is set to a number,
                only atoms and spins within a sphere of radius mask_radius
                centered about the location parameter will be used for
                calculations. (To speed up calculations, 8 is
                recommended.)

        Returns:
            (float or ndarray): The magnetic field at the given location. If
                return_vector is False it is a float of the magnitude. If
                return_vector is True, it is a 1D numpy array with 3 values.

        """
        if mask_radius is not None:
            mask = np.apply_along_axis(np.linalg.norm,
                                       1,
                                       location - self.atoms) <= mask_radius
            atoms = self.atoms[mask]
            spins = self.spins[mask]
        else:
            atoms = self.atoms
            spins = self.spins

        r = (location - atoms) * 1e-10
        m = 1.0 * spins

        m_dot_r = np.sum(r*m, axis=1).reshape(r.shape[0])[..., np.newaxis]
        r_mag = np.sqrt((r**2).sum(-1))[..., np.newaxis]
        r5 = r_mag**5
        r3 = r_mag**3
        Bvals = 3.0*m_dot_r / r5*r - m/r3
        Btot = Bvals.sum(0) * 1e-7 #(mu_0 / (4 * pi))

        #We return the net magnetic field at the location specified.
        if return_vector is True:
            return Btot
        else:
            return np.linalg.norm(Btot)

    def calculate_fields(self,
                         locations=None,
                         return_vector=True,
                         mask_radius=None):
        """ Calculates the magnetic field at the specified locations.

        Parameters:
            locations (ndarray): Optional. A 2-Dimensional numpy array
                specifying at which locations to calculate the magnetic field.
                Each row in the array should be a 1-Dimensional numpy array of
                length 3. If no value is passed in then it will use
                self.locations.
            return_vector (boolean): Optional, default is True. See below for
                details.
            mask_radius (int or float): Optional, default is None. If
                mask_radius is None, all atoms and spins will be taken into
                account for the calculations. If mask_radius is set to a number,
                only atoms and spins within a sphere of radius mask_radius
                centered about the each location in the locations parameter will
                be used for calculations. (To speed up calculations, 8 is
                recommended.)

        Returns:
            (ndarray): A numpy array of either floats or 1D numpy arrays for the
                magnetic field at each of the given locations. If return_vector
                is False it is a 1D array of floats representing the magnitude.
                If return_vector is True, it is a 2D numpy array of 1D numpy
                arrays with 3 values.

        """

        if locations is None:
            if self.locations is None:
                raise Exception('Please specify locations first')
            else:
                locations = self.locations

        return np.array([self.calculate_field(location,
                                              return_vector,
                                              mask_radius)
                                              for location in locations])

    def find_field(self,
                   field,
                   mask_radius=None):
        """ Finds the location of a magnetic field in the crystal structure
        using least squares minimization.

        Parameters:
            field (float or int): The value of the magnetic field the function
                searches for. (Tesla)
            mask_radius (int or float): Optional, default is None. If
                mask_radius is None, all atoms and spins will be taken into
                account for the calculations. If mask_radius is set to a number,
                only atoms and spins within a sphere of radius mask_radius
                centered about the each location in the locations parameter will
                be used for calculations. (To speed up calculations, 8 is
                recommended.)

        Returns:
            (ndarray): A 1D array containing the x,y,z location in the
                structure where the magnetic field is closest to the input
                field.

        """

        f = lambda x, y, z: (self.calculate_field(location=x,
                                                  return_vector=False,
                                                  mask_radius=y) - z)

        minimum = least_squares(f, np.random.rand(3), args=(mask_radius, field))

        return minimum.x

    def make_grid(self,
                  side_length,
                  resolution,
                  center_point,
                  norm_axis='z',
                  return_vector=False,
                  mask_radius=None):
        """ Calculates the magnetic field over a grid specified by the input
        parameters.

        Parameters:
            side_length (int or float): The side length of the grid in
                Angstroms.
            resolution (int): The number of measurments to take per
                Angstrom.
            center_point (ndarray): A 1D numpy array with 3 values specifying an
                x,y,z location that the grid will be centered on.
            return_vector (boolean): Optional, default is True. See below for
                details.
            mask_radius (int or float): Optional, default is None. If
                mask_radius is None, all atoms and spins will be taken into
                account for the calculations. If mask_radius is set to a number,
                only atoms and spins within a sphere of radius mask_radius
                centered about the each location in the locations parameter will
                be used for calculations. (To speed up calculations, 8 is
                recommended.)

        Returns:
            (ndarray): A numpy array containing the magnetic field at each point
                in the grid. The array will have side_length*resolution columns
                and rows. If return_vector is False, the magnetic fields at each
                location in the array will be magnitudes. If return_vector is
                True, the magnetic fields will be 1D numpy arrays with 3 values.

        """
        axis_dict = {k:v for v,k in enumerate(['x','y','z'])}
        try:
            del axis_dict[norm_axis]
        except:
            raise Exception("norm_axis can only be set to: 'x', 'y', or 'z'")

        a_index = min(axis_dict.values())
        b_index = max(axis_dict.values())

        a = np.linspace(center_point[a_index] - side_length/2,
                        center_point[a_index] + side_length/2,
                        resolution*side_length)
        b = np.linspace(center_point[b_index] - side_length/2,
                        center_point[b_index] + side_length/2,
                        resolution*side_length)

        A,B = np.meshgrid(a,b)

        locations = [self.make_location(n=n,
                                        m=m,
                                        norm_axis=norm_axis,
                                        center_point=center_point)
                                        for n,m in np.nditer([A,B])]

        fields = self.calculate_fields(locations=np.array(locations),
                                       return_vector=return_vector,
                                       mask_radius=mask_radius)

        return np.reshape(np.array(fields), (len(a),len(b)))

    #Helper functions
    def make_location(self,
                      n,
                      m,
                      norm_axis,
                      center_point):
        """ Helper function to make locations for make_grid. """

        if norm_axis == 'x':
            location = [center_point[0], n, m]
        elif norm_axis == 'y':
            location = [n, center_point[1], m]
        else:
            location = [n, m, center_point[2]]

        return location
