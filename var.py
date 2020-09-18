from __future__ import annotations

from decimal import Decimal
from functools import total_ordering
from typing import Sequence, SupportsFloat, Dict, List, Union, overload, Callable, Optional, Tuple
from warnings import catch_warnings, simplefilter

from numpy import sqrt, array, diag, isnan
from sympy.core.symbol import Symbol, Expr
from sympy.utilities import lambdify

DictSymVar: Dict[Symbol, Var] = {}


@total_ordering
class Var:
    @overload
    def __init__(self, _story: Expr, exp=0):
        """todo: this is used only by library, not by user"""
        ...

    @overload
    def __init__(self, value: SupportsFloat, error: SupportsFloat, exp=0):
        ...

    def __init__(self, *args, exp=0):
        if len(args) == 2:
            self._value: float = float(args[0]) * 10 ** exp
            self._error: float = float(args[1]) * 10 ** exp
            self._story: Symbol = Symbol('s' + str(id(self)))
            DictSymVar.update({self._story: self})
        else:
            self._story: Expr = args[0]

    def val_err(self) -> Tuple[float, float]:
        args = tuple(self._story.free_symbols)
        func = lambdify(args, self._story, modules='numpy')
        base_vars = tuple(DictSymVar[sym] for sym in args)
        values = array(tuple(var._value for var in base_vars), dtype=float)
        diag_err = diag(array(tuple(var._error for var in base_vars)))
        val = func(*values)
        if isnan(val):
            raise TypeError('The argument does not belong to the definition scope')
        # we check that values of function on both sides exist
        err = sqrt(sum((estimated_error(func, values, diag_err[i], val)) ** 2 for i in range(len(values))))
        return val, err

    def val(self) -> float:
        args = tuple(self._story.free_symbols)
        return lambdify(args, self._story, 'numpy')(*(DictSymVar[sym]._value for sym in args))

    def err(self) -> float:
        return self.val_err()[1]

    def __repr__(self) -> str:
        return f'~{self.val()}'

    def __str__(self) -> str:
        return normalize(self)

    def __le__(self, other: Union[SupportsFloat, Var]) -> bool:
        if isinstance(other, Var):
            return self.val() <= other.val()
        else:
            return self.val() <= float(other)

    def __eq__(self, other: Union[SupportsFloat, Var]) -> bool:
        if isinstance(other, Var):
            return self.val() == other.val()
        else:
            return self.val() == float(other)

    def _binary_operation(self, other: TypicalArgument, func: Callable) -> Union[Var, GroupVar]:
        if isinstance(other, Var):
            return _Var(func(self._story, other._story))
        if isinstance(other, SupportsFloat):
            return _Var(func(self._story, other))
        if isinstance(other, GroupVar):
            return GroupVar([...])

    def __add__(self, other: TypicalArgument) -> Var:
        return self._binary_operation(other, lambda x, y: x + y)

    def __radd__(self, other: TypicalArgument) -> Var:
        return self._binary_operation(other, lambda x, y: y + x)

    def __sub__(self, other: TypicalArgument) -> Var:
        return self._binary_operation(other, lambda x, y: x - y)

    def __rsub__(self, other: TypicalArgument) -> Var:
        return self._binary_operation(other, lambda x, y: y - x)

    def __mul__(self, other: TypicalArgument) -> Var:
        return self._binary_operation(other, lambda x, y: x * y)

    def __rmul__(self, other: TypicalArgument) -> Var:
        return self._binary_operation(other, lambda x, y: y * x)

    def __truediv__(self, other: TypicalArgument) -> Var:
        return self._binary_operation(other, lambda x, y: x / y)

    def __rtruediv__(self, other: TypicalArgument) -> Var:
        return self._binary_operation(other, lambda x, y: y / x)

    def __pow__(self, power: TypicalArgument) -> Var:
        # TODO: smth
        return self._binary_operation(power, lambda x, y: x ** y)

    def __pos__(self) -> Var:
        return self

    def __neg__(self) -> Var:
        return Var(-self._story)


class GroupVar:
    @overload
    def __init__(self, variables: Sequence[Var], exp=0):
        """todo: this is used only by library, not by user"""
        ...

    @overload
    def __init__(self, values: Sequence[SupportsFloat], errors: Sequence[SupportsFloat], exp=0):
        ...

    @overload
    def __init__(self, values: Sequence[SupportsFloat], error: SupportsFloat, exp=0):
        ...

    def __init__(self, *args, exp=0):
        # look group and ungroup
        if len(args) == 2:
            values, errors = args[0], {True: args[1], False: [args[1]] * len(args[0])}[hasattr(args[1], '__iter__')]
            if len(values) != len(errors):
                raise TypeError('Arguments must be the same length')
            self.variables: List[Var] = [Var(val, err) * 10 ** exp for val, err in zip(values, errors)]
        else:
            self.variables: List[Var] = [var * 10 ** exp for var in args[0]]

    def val_err(self) -> Tuple[Tuple[float, ...], ...]:
        return tuple(zip(*(var.val_err() for var in self)))

    def val(self) -> List[float]:
        return [var.val() for var in self.variables]

    def err(self) -> List[float]:
        return [var.err() for var in self.variables]

    def __getitem__(self, item):
        return self.variables[item]

    def __iter__(self):
        return self.variables.__iter__()

    def __len__(self) -> int:
        return len(self.variables)

    def __repr__(self):
        return str([var.__repr__() for var in self.variables])

    def __str__(self):
        return str([var.__str__() for var in self.variables])

    def _binary_operation(self, other: TypicalArgument, func: Callable) -> Union[Var, GroupVar]:
        if isinstance(other, GroupVar):
            if len(self) != len(other):
                raise TypeError('Arguments must be the same length')
            return GroupVar(tuple(func(self[i], other[i]) for i in range(len(self))))
        else:
            return GroupVar(tuple(func(var, other) for var in self.variables))

    def __add__(self, other: TypicalArgument) -> GroupVar:
        return self._binary_operation(other, lambda x, y: x + y)

    def __radd__(self, other: TypicalArgument) -> GroupVar:
        return self._binary_operation(other, lambda x, y: y + x)

    def __sub__(self, other: TypicalArgument) -> GroupVar:
        return self._binary_operation(other, lambda x, y: x - y)

    def __rsub__(self, other: TypicalArgument) -> GroupVar:
        return self._binary_operation(other, lambda x, y: y - x)

    def __mul__(self, other: TypicalArgument) -> GroupVar:
        return self._binary_operation(other, lambda x, y: x * y)

    def __rmul__(self, other: TypicalArgument) -> GroupVar:
        return self._binary_operation(other, lambda x, y: y * x)

    def __truediv__(self, other: TypicalArgument) -> GroupVar:
        return self._binary_operation(other, lambda x, y: x / y)

    def __rtruediv__(self, other: TypicalArgument) -> GroupVar:
        return self._binary_operation(other, lambda x, y: y / x)

    def __pow__(self, power: Union[TypicalArgument]) -> GroupVar:
        return self._binary_operation(power, lambda x, y: x ** y)

    def __pos__(self) -> GroupVar:
        return self

    def __neg__(self) -> GroupVar:
        return _GroupVar([-var for var in self.variables])


def group(variables: Sequence[Var]) -> GroupVar:
    return GroupVar(variables)


def ungroup(var: GroupVar) -> List[Var]:
    return var.variables


TypicalArgument = Union[SupportsFloat, Var, GroupVar]


def _Var(_story: Expr) -> Var:
    return Var(_story)


def _GroupVar(variables: Sequence[Var]):
    return GroupVar(variables)


error_accuracy: float = 0.4
value_accuracy: float = 0.05


def set_error_accuracy(accuracy: float):
    global error_accuracy
    error_accuracy = accuracy


def set_value_accuracy(accuracy: float):
    global value_accuracy
    value_accuracy = accuracy


def suitable_accuracy(val: float, err: float) -> int:
    if err == 0:
        return Decimal.from_float(val * value_accuracy).adjusted()
    return Decimal.from_float(err * error_accuracy).adjusted()


def normalize(var: Var, accuracy: Optional[int] = None) -> str:
    val, err = var.val_err()
    if accuracy is None:
        accuracy = suitable_accuracy(val, err)
    return '{0} \\pm {1}'.format(
        *({True: float, False: int}[accuracy < 0](round(num, -accuracy)) for num in (val, err)))


BIG_NUMBER = 50


def set_big_number(n: int):
    global BIG_NUMBER
    BIG_NUMBER = n


def estimated_error(func: Callable[[...], float], values: array, err_vector: array, val: float):
    err_vector /= BIG_NUMBER
    with catch_warnings():
        simplefilter("error")
        try:
            v_plus = func(*(values + err_vector))
        except RuntimeWarning:
            try:
                return (val - func(*(values - err_vector))) * BIG_NUMBER
            except RuntimeWarning:
                raise TypeError('Your errors are too big')
        try:
            v_minus = func(*(values - err_vector))
        except RuntimeWarning:
            return (val - v_plus) * BIG_NUMBER
        return (v_plus - v_minus) / 2 * BIG_NUMBER
