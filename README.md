# mag_calc

mag_calc is a Python module used to calculate magnetic fields in crystal structures.

## Usage

```python
from mag_calc import MagCalc

atoms = np.genfromtxt('atoms_example.csv', delimiter=',')
spins = np.genfromtxt('spins_example.csv', delimiter=',')
locations = np.reshape(np.arange(30), (10,3))

calc = MagCalc(atoms=atoms, spins=spins, locations=locations)

B = calc.calculate_field(location=np.arange(3), return_vector=True)
B_list = calc.calculate_fields(return_vector=True)
```

## Dependencies

The following python library is required:

```bash
pip install numpy
```

## Support

Feel free to reach out at
```
ashaw8@byu.edu
```
if you have any questions.
