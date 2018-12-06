import numpy as np
import dask.array as da
from dask_quantity import DaskQuantity
import astropy.units as u

array = da.from_array(np.ones((1024, 1024)) * 100, chunks=(256, 256))

print(type(array))

quantity1 = DaskQuantity(array, u.m / u.s)
quantity2 = DaskQuantity(array, u.cm / u.s)

print(quantity1.compute())
print(np.sqrt(quantity1).value[0,0].compute())
print(np.sqrt(quantity1)[0,0].compute())
print(np.add(quantity1, quantity2).compute())
