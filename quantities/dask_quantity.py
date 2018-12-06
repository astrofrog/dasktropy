import numpy as np
import astropy.units as u
from astropy.units.quantity_helper import converters_and_unit


class DaskQuantity:

    def __init__(self, value, unit):
        self.value = value
        self.unit = unit

    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        """
        Wrap numpy ufuncs, taking care of units.

        Parameters
        ----------
        function : callable
            ufunc to wrap.
        method : str
            ufunc method: ``__call__``, ``at``, ``reduce``, etc.
        inputs : tuple
            Input arrays.
        kwargs : keyword arguments
            As passed on, with ``out`` containing possible quantity output.

        Returns
        -------
        result : `~astropy.units.DaskQuantity`
            Results of the ufunc, with the unit set properly.
        """

        # Determine required conversion functions -- to bring the unit of the
        # input to that expected (e.g., radian for np.sin), or to get
        # consistent units between two inputs (e.g., in np.add) --
        # and the unit of the result (or tuple of units for nout > 1).
        converters, unit = converters_and_unit(function, method, *inputs)

        # Same for inputs, but here also convert if necessary.
        arrays = [(converter(input_.value) if converter else
                   getattr(input_, 'value', input_))
                  for input_, converter in zip(inputs, converters)]

        # Call our superclass's __array_ufunc__

        result = self.value.__array_ufunc__(function, method, *arrays, **kwargs)

        # If unit is None, a plain array is expected (e.g., comparisons), which
        # means we're done.
        # We're also done if the result was None (for method 'at') or
        # NotImplemented, which can happen if other inputs/outputs override
        # __array_ufunc__; hopefully, they can then deal with us.
        if unit is None or result is None or result is NotImplemented:
            return result

        return DaskQuantity(result, unit)

    def to(self, unit):
        return DaskQuantity(self.value * self.unit.to(unit), unit)

    def compute(self):
        return u.Quantity(self.value.compute(), self.unit)

    def __getitem__(self, item):
        return DaskQuantity(self.value[item], self.unit)

    def __neg__(self):
        return DaskQuantity(-self.value, self.unit)

    def __add__(self, other):
        return np.add(self, other)

    def __radd__(self, other):
        return np.add(other, self)

    def __sub__(self, other):
        return np.subtract(self, other)

    def __rsub__(self, other):
        return np.subtract(other, self)
