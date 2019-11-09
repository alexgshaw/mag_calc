# mag_calc

mag_calc is a Python module used to calculate magnetic fields in crystal structures.

## Dependencies

The following python libraries are required:

```bash
pip install numpy
pip install scipy
```

## Usage

```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from mag_calc import MagCalc

>>> atoms = np.load('files/MnO_atoms.npy')
>>> spins = np.load('files/MnO_spins.npy')
>>> locations = np.random.rand(10,3)

>>> calc = MagCalc(atoms=atoms,
                   spins=spins,
                   locations=locations,
                   g_factor=2,
                   spin=1/2,
                   magneton='mu_B')

>>> print(calc)
atoms shape:     (10149, 3)
spins shape:     (10149, 3)
locations shape: (10, 3)
g_factor:        2
spin:            0.5
magneton:        mu_B

>>> B = calc.calculate_field(location=np.random.rand(3),
                             return_vector=True,
                             mask=None)
>>> B_list = calc.calculate_fields(return_vector=False,
                                   mask_radius=8)

>>> field_location = calc.find_field(field=0.1, mask=None)
>>> print(calc.calculate_field(location=field_location, return_vector=False))
0.1

>>> plane = calc.make_plane(side_length=4,
                            resolution=200,
                            center_point=np.array([.5,.5,.5]),
                            norm_vec=np.array([0,0,1]),
                            return_vector=False,
                            mask_radius=8)

>>> plt.axes().set_aspect('equal')
>>> plt.contourf(plane, 20, cmap='Spectral')
>>> plt.colorbar().outline.set_visible(False)
>>> plt.axis('off')
>>> plt.show()
```

![Field Plot](/files/readme_plot.png)

## Support

Feel free to reach out at
[ashaw8@byu.edu](mailto:ashaw8@byu.edu)
if you have any questions.
