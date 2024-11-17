import torch

try:
    from . import gsrt_cpp_extension as cpp
except ImportError as err_:
    err = err_
    # TODO: Raise error
    print("\033[91;1mERROR: could not load the cpp extension. Build the project first.\033[0m")

    class LazyError:
        class LazyErrorObj:
            def __call__(self, *args, **kwds):
                raise RuntimeError("ERROR: could not load cpp extension. Please build the project first") from err

            def __getattribute__(self, __name: str):
                raise RuntimeError("ERROR: could not load cpp extension. Please build the project first") from err

        def __getattribute__(self, __name: str):
            return LazyError.LazyErrorObj()

    cpp = LazyError()

GaussiansTracer = cpp.GaussiansTracer
TracerCustom = cpp.TracerCustom