from __future__ import division
import numpy as np

class MagCalc:
    """Object to calculate magnetic fields in an atomic structure.

    Attributes:
        atoms (numpy array): The positions of the atoms in the crystal
            structure. Should be a 2D numpy array with each row as a 1D numpy
            array with 3 values designating an x,y,z location.
        spins (numpy array): The magnetic dipole moments of the atoms in the
            crystal structure. Should be a 2D numpy array with each row as a 1D
            numpy array with 3 values designating an i,j,k spin direction.
        locations (numpy array): The locations of where to calculate the
            magnetic field. Should be a 2D numpy array with each row as a 1D
            numpy array with 3 values designating an x,y,z location.
    """

    def __init__(self, atoms, spins, locations = None):
        """Initializes the spins and atomic positions of a crystal structure.
        Optionally intializes the locations for where to calculate the field.

        Parameters:
            atoms (numpy array): The positions of the atoms in the crystal
                structure. Should be a 2D numpy array with each row as a 1D
                numpy array with 3 values designating an x,y,z location.
            spins (numpy array): The magnetic dipole moments of the atoms in the
                crystal structure. Should be a 2D numpy array with each row as a
                1D numpy array with 3 values designating an i,j,k spin
                direction.
            locations (numpy array): Optional. The locations of where to
                calculate the magnetic field. Should be a 2D numpy array with
                each row as a 1D numpy array with 3 values designating an x,y,z
                location.
        """
        self.atoms = atoms
        self.spins = spins

        if locations is not None:
            self.locations = locations

    def set_locations(self, locations):
        """ Sets the locations for where to calculate the magnetic field.

        Parameters:
            location (numpy array): A 2D numpy array specifying at which
                locations to calculate the magnetic field. Each row in the array
                should be a 1D numpy array of length 3.

        """
        self.locations = locations

    def calculate_field(self, location, return_vector=True):
        """ Calculates the magnetic field at the specified location.

        Parameters:
            location (numpy array): A 1D numpy array of length 3 specifying
                where to calculate the magnetic field.
            return_vector (boolean): Optional, default is True. See below for
                details.

        Returns:
            (float or numpy array): The magnetic field at the given location. If
                return_vector is False it is a float of the magnitude. If
                return_vector is True, it is a 1D numpy array with 3 values.

        """


        r = (location - self.atoms) * 1e-10
        m = 1.0 * self.spins

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

    def calculate_fields(self, locations=None, return_vector=True):
        """ Calculates the magnetic field at the specified locations.

        Parameters:
            location (numpy array): Optional. A 2-Dimensional numpy array
                specifying at which locations to calculate the magnetic field.
                Each row in the array should be a 1-Dimensional numpy array of
                length 3. If no value is passed in then it will use
                self.locations.
            return_vector (boolean): Optional, default is True. See below for
                details.

        Returns:
            (list): A list of either floats or 1D numpy arrays for the magnetic
                field at each of the given locations. If return_vector is False
                it is a list of floats representing the magnitude. If
                return_vector is True, it is a list of 1D numpy arrays with 3
                values.

        """

        if locations is not None:
            self.locations = locations
        elif self.locations is None:
            print('Please specify locations first')
            return []

        return [self.calculate_field(location, return_vector)
                                                 for location in self.locations]
