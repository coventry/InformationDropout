"""Shim for mypy type checker to allow attr-class initialization with kw args.

Does nothing, except allows, e.g.

    >>> import attr
    >>> @attr.s
    ... class A:
    ...     b = attr.ib()
    >>> a = A(b=3)

...which would otherwise cause mypy to complain about an unspecified keyword `b`.

This is a half-assed solution... There is a real solution being developed here:
https://github.com/python/mypy/pull/4397

Once that's mature, this should be removed.

The main drawback to this approach is that there is no mypy checking of the
keyword arguments.

You can't just wrap `attr.s`, because mypy never executes that AFAICT.

"""


class MyPyShim:

    def __init__(self, *args, **kw):
        "Shim for mypy type checker to allow keyword args. Does nothing."
        raise RuntimeError('Should never be called')
