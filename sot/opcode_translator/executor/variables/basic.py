from __future__ import annotations

import operator
import types
from functools import reduce
from typing import TYPE_CHECKING, Any

import paddle

from ....infer_meta import MetaInfo
from ....symbolic.statement_ir import Symbol
from ....utils import (
    BreakGraphError,
    NameGenerator,
    log_do,
    paddle_tensor_methods,
)
from ....utils.exceptions import InnerError
from ..guard import StringifyExpression, union_free_vars
from ..pycode_generator import PyCodeGen
from ..tracker import ConstTracker, DummyTracker, GetAttrTracker, Tracker
from .base import ConstTypes, VariableBase, VariableFactory

if TYPE_CHECKING:
    from ..function_graph import FunctionGraph


class ConstantVariable(VariableBase):
    def __init__(
        self,
        value: Any,
        tracker: Tracker,
    ):
        super().__init__(tracker)
        self.value = value

    def get_value(self):
        return self.value

    @property
    def debug_name(self) -> str:
        return f"{self.value}"

    @debug_name.setter
    def debug_name(self, name):
        pass

    def _reconstruct(self, codegen: PyCodeGen):
        codegen.gen_load_const(self.value)

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}

    def __bool__(self) -> bool:
        return bool(self.value)

    def apply_unary_operator(self, magic_name):
        operator = getattr(self.value, magic_name)
        var = VariableFactory.from_value(
            operator(),
            None,
            tracker=DummyTracker(
                [
                    self,
                ]
            ),
        )
        return var

    def apply_binary_operator(self, other, magic_name):
        if not isinstance(other, ConstantVariable):
            return NotImplemented
        operator = getattr(self.value, magic_name)
        var = VariableFactory.from_value(
            operator(other.value), None, tracker=DummyTracker([self, other])
        )
        return var

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, ConstTypes):
            return ConstantVariable(value, tracker)
        return None

    @staticmethod
    def wrap_literal(value: Any) -> ConstantVariable:
        if isinstance(value, ConstantVariable):
            return value
        assert isinstance(
            value, ConstTypes
        ), f"value: {value},type: {type(value)}"
        return ConstantVariable(value, ConstTracker(value))


class TensorVariable(VariableBase):
    var_name_generator = NameGenerator("var_")

    def __init__(
        self,
        tensor: paddle.Tensor | MetaInfo,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(tracker)
        if isinstance(tensor, paddle.Tensor):
            self.value = tensor
            self.meta = MetaInfo.from_tensor(tensor)
        elif isinstance(tensor, MetaInfo):
            self.value = None
            self.meta = tensor
        else:
            raise InnerError(
                "Required type(tensor) is paddle.Tensor or ProxyTensor, but received {}.".format(
                    type(tensor).__name__
                )
            )
        self.var_name = TensorVariable.var_name_generator.next()
        self.graph = graph

    def get_value(self):
        if self.value is None:
            raise InnerError("Can not get value from a inner tensor variable.")
        return self.value

    def get_type(self):
        return paddle.Tensor

    def get_symbol(self) -> Symbol:
        return Symbol(self.var_name)

    @property
    def out_var_name(self):
        return f"{self.graph.out_var_prefix}{self.var_name}"

    def _reconstruct(self, codegen: PyCodeGen):
        codegen.gen_load_fast(self.out_var_name)

    def make_stringify_guard(self) -> StringifyExpression:
        assert not isinstance(
            self.tracker, DummyTracker
        ), "Can not make guard from dummy tracker"

        frame_value_tracer = self.tracker.trace_value_from_frame()
        log_do(
            4,
            lambda: print(
                f"[Guard]: guard_fn for {self}, tracker={self.tracker.__class__.__name__}, value={frame_value_tracer.expr}"
            ),
        )
        return StringifyExpression(
            f"str(MetaInfo.from_tensor({frame_value_tracer.expr})) == '{self.meta}'",
            union_free_vars(
                {"MetaInfo": MetaInfo},
                frame_value_tracer.free_vars,
            ),
        )

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "shape": self.meta.shape,
            "dtype": self.meta.dtype,
            "stop_gradient": self.meta.stop_gradient,
        }

    def __getitem__(self, key):
        return self.graph.call_tensor_method(
            '__getitem__',
            self,
            VariableFactory.from_value(
                key, self.graph, tracker=ConstTracker(key)
            ),
        )

    def __setitem__(self, key, value):
        return self.graph.call_tensor_method(
            '__setitem__',
            self,
            VariableFactory.from_value(
                key, self.graph, tracker=ConstTracker(key)
            ),
            value,
        )

    @property
    def T(self):
        perm = list(range(len(self.meta.shape) - 1, -1, -1))
        perm_var = VariableFactory.from_value(
            perm, self.graph, tracker=ConstTracker(perm)
        )
        out = self.graph.call_paddle_api(paddle.transpose, self, perm_var)
        return out

    @property
    def ndim(self):
        return ConstantVariable.wrap_literal(len(self.meta.shape))

    @property
    def shape(self):
        if self.meta.is_dynamic_shape():
            raise BreakGraphError(
                f"Getting size for a dynamic shape tensor causes graph break. shape = {self.meta.shape}"
            )
        self.graph.add_global_guarded_variable(self)
        return VariableFactory.from_value(
            self.meta.shape, self.graph, tracker=ConstTracker(self.meta.shape)
        )

    @property
    def size(self):
        # TODO: maybe break graph.
        if self.meta.is_dynamic_shape():
            raise BreakGraphError(
                f"Getting size for a dynamic shape tensor causes graph break. shape = {self.meta.shape}"
            )
        elements = reduce(operator.mul, self.meta.shape, 1)
        return ConstantVariable.wrap_literal(elements)

    def __getattr__(self, name: str):
        if name in ["shape", "dtype", "stop_gradient"]:
            return VariableFactory.from_value(
                getattr(self.meta, name),
                self.graph,
                tracker=GetAttrTracker(self, name),
            )
        elif name in paddle_tensor_methods:
            from .callable import TensorFunctionVariable

            fn_var = TensorFunctionVariable(
                name, graph=self.graph, tracker=DummyTracker([])
            )
            return fn_var.bind(self, name)
        elif name in ["T", "ndim", "size"]:
            return getattr(self, name)
        else:
            raise InnerError(f"Unknown Tensor attribute: {name}")

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, (paddle.Tensor, MetaInfo)):
            assert graph is not None
            return TensorVariable(value, graph, tracker)
        return None


class ObjectVariable(VariableBase):
    def __init__(self, obj, graph, tracker):
        super().__init__(tracker)
        self.value = obj
        self.graph = graph

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}

    def get_value(self) -> Any:
        return self.value


class SliceVariable(VariableBase):
    def __init__(self, slice_: slice, graph, tracker):
        super().__init__(tracker)
        self.value = slice_
        self.graph = graph

    @property
    def debug_name(self) -> str:
        return ":".join(
            [
                str(self.value.start) if self.value.start is not None else "",
                str(self.value.stop) if self.value.stop is not None else "",
                str(self.value.step) if self.value.step is not None else "",
            ]
        )

    @debug_name.setter
    def debug_name(self, name):
        pass

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}

    def get_value(self):
        return self.value

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, slice):
            return SliceVariable(value, graph, tracker)
        return None


class ModuleVariable(VariableBase):
    def __init__(self, func, graph, tracker):
        super().__init__(tracker)
        self.value = func
        self.graph = graph

    def get_value(self):
        return self.value

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, types.ModuleType):
            return ModuleVariable(value, graph, tracker)
        return None


class DygraphTracerVariable(VariableBase):
    # TODO(SigureMo): Remove this trick after we add CompareTracker
    def __init__(self, value, graph, tracker):
        super().__init__(tracker)
        self.value = value
        self.graph = graph

    def get_value(self):
        return self.value

    def make_stringify_guard(self) -> StringifyExpression:
        assert not isinstance(
            self.tracker, DummyTracker
        ), "Can not make guard from dummy tracker"

        frame_value_tracer = self.tracker.trace_value_from_frame()
        log_do(
            4,
            lambda: print(
                f"[Guard]: guard_fn for {self}, tracker={self.tracker.__class__.__name__}, value={frame_value_tracer.expr}"
            ),
        )
        return StringifyExpression("True", {})

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "is_none": self.value is None,
        }

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph | None, tracker: Tracker):
        if isinstance(value, paddle.fluid.dygraph.tracer.Tracer):
            return DygraphTracerVariable(value, graph, tracker)
        return None


class DummyVariable(VariableBase):
    def __init__(self):
        super().__init__(DummyTracker([]))

    def reconstruct(self, codegen: PyCodeGen):
        codegen.gen_push_null()


class ClosureVariable(VariableBase):
    def __init__(self, name):
        super().__init__(DummyTracker([]))
        self.name = name

    def get_name(self):
        return self.name