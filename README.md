# mag_calc

mag_calc is a Python module used to calculate magnetic fields in crystal structures.

## Usage

```python
from mag_calc import MagCalc

atoms = np.genfromtxt('atoms_example.csv', delimiter=',')
spins = np.genfromtxt('spins_example.csv', delimiter=',')
locations = np.reshape(np.arange(30), (10,3))

calc = MagCalc(atoms=atoms,
               spins=spins,
               locations=locations,
               g_factor=2,
               spin=1/2,
               magneton='mu_B')

B = calc.calculate_field(location=np.arange(3), return_vector=True, mask_radius=None)
B_list = calc.calculate_fields(return_vector=False, mask_radius=8)

field_location = calc.find_field(field=0.1, mask_radius=None)
```

## Dependencies

The following python library is required:

```bash
pip install numpy
pip install scipy
```

## Support

Feel free to reach out at
```
ashaw8@byu.edu
```
if you have any questions.
