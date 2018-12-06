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

        print(converters, unit, type(self.value))

        # Same for inputs, but here also convert if necessary.
        arrays = [(converter(input_.value) if converter else
                   getattr(input_, 'value', input_))
                  for input_, converter in zip(inputs, converters)]

        # Call our superclass's __array_ufunc__

        print(type(arrays[0]))
        result = self.value.__array_ufunc__(function, method, *arrays, **kwargs)

        print(type(result))

        # If unit is None, a plain array is expected (e.g., comparisons), which
        # means we're done.
        # We're also done if the result was None (for method 'at') or
        # NotImplemented, which can happen if other inputs/outputs override
        # __array_ufunc__; hopefully, they can then deal with us.
        if unit is None or result is None or result is NotImplemented:
            return result

        return DaskQuantity(result, unit)

    def compute(self):
        return u.Quantity(self.value.compute(), self.unit)

    def __getitem__(self, item):
        return DaskQuantity(self.value[item], self.unit)
