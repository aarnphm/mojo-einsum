from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder


@export
def PyInit__native() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_native")
        module.def_function[passthrough]("passthrough")
        _ = (
            module.add_type[Greeter]("Greeter")
            .def_py_init[Greeter.py_init]()
            .def_method[Greeter.greet]("greet")
        )
        return module.finalize()
    except e:
        abort(String("failed to create Python module: ", e))


def passthrough(value: PythonObject) raises -> PythonObject:
    return value + " world from Mojo"


@fieldwise_init
struct Greeter(Movable, Writable):
    var suffix: String

    @staticmethod
    def _get_self_ptr(py_self: PythonObject) -> UnsafePointer[Self, MutAnyOrigin]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(String("Greeter method receiver had the wrong type: ", e))

    @staticmethod
    def py_init(out self: Greeter, args: PythonObject, kwargs: PythonObject) raises:
        if len(args) != 1:
            raise Error("Greeter() takes exactly one suffix argument")
        self = Self(String(py=args[0]))

    @staticmethod
    def greet(py_self: PythonObject, value: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return value + " " + self_ptr[].suffix
