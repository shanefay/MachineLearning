from functools import reduce, update_wrapper
import functools

def compose(*fs):
    """Compose a list of functions
    e.g. compose(f, g, h)(x) = f(g(h(x)))
    """
    return reduce(compose2, fs)

def compose2(f, g):
    """Compose function f with g
    e.g. compose2(f, g)(x) = f(g(x))
    """
    return lambda *a, **kw: f(g(*a, **kw))

def partial(func, *args, **kwargs):
    """Wrapper for functools.partial to include missing attributes.
    e.g. __name__ attribute
    """
    partial_func = functools.partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func