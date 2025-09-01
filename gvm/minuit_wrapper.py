from iminuit import Minuit


def minimize(array_func, x0, names, errordef=0.5):
    """Run minimization using iminuit and return the Minuit object."""
    if hasattr(Minuit, "from_array_func"):
        m = Minuit.from_array_func(
            array_func,
            x0,
            name=names,
            errordef=errordef,
        )
    else:
        m = Minuit(
            array_func,
            x0,
            name=names,
        )
        m.errordef = errordef
    m.migrad()
    return m

