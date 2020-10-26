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
    """
    Class Var is one of the main classes in lab_cheat.
    It is used to store the pair of value and error of any directly measured variable OR
    the function of how to calculate indirect variable from direct variables.
    There is no public slots in this class.
    For series of similar variables it's more convenient to use GroupVar.
    """

    @overload
    def __init__(self, _story: Expr, exp: int = 0):
        """Never use Var(...) in this way. This is used only by library"""
        ...

    @overload
    def __init__(self, value: SupportsFloat, error: SupportsFloat, exp: int = 0):
        """
        Generates object of Var class.
        :param value: the approximate value of directly measured variable
        :param error: it's standard deviation
        :param exp: if exp!=0 then value and error will be changed:
            value -> value * 10 ** exp
            error -> error * 10 ** exp
        """
        ...

    def __init__(self, *args, exp: int = 0):
        if len(args) == 2:
            self._value: float = float(args[0]) * 10 ** exp
            self._error: float = float(args[1]) * 10 ** exp
            self._story: Symbol = Symbol('s' + str(id(self)))
            DictSymVar.update({self._story: self})
        else:
            self._story: Expr = args[0]

    def val_err(self) -> Tuple[float, float]:
        """
        :return: tuple containing value and error
        (This method is more efficient than calling val() and err() separately.)
        """
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
        """
        :return: value of your variable
        """
        args = tuple(self._story.free_symbols)
        return lambdify(args, self._story, 'numpy')(*(DictSymVar[sym]._value for sym in args))

    def err(self) -> float:
        """
        :return: error of your variable
        The following is to explain how does library finds it:
            In most cases we have random independent errors of variables:

            delta(f(x1, x2, ...)) ~= sqrt( (df/dx1 * delta(x1))**2 + (df/dx2 * delta(x2))**2 + ... ),

            where the error by x1 for example in this method is calculated by:

            df/dx1 * delta(x1) = (f(x1+dx1, x2, ...) - f(x1 - dx1, x2, ...)) * BIG_NUMBER / 2
            dx1 = delta(x1) / BIG_NUMBER

            As you see, it's very similar to finding  partial derivative numerically.
            You may change BIG_NUMBER in function 'set_big_number' in this module. (default is 50)
        """
        return self.val_err()[1]

    def __repr__(self) -> str:
        """
        :return: short rough representation of variable
        """
        return f'~{self.val()}'

    def __str__(self) -> str:
        """
        :return: string looking like "value \\pm error", where value end error are rounded.
        Digits that have the same order as the 40% error will not be shown.
        If error is zero, there won't be shown digits having the same order as 5% value.
        (40% and 5% may be changed in 'set_error_accuracy' and 'value_accuracy' respectively.)
        """
        return normalize(self)

    def __le__(self, other: Union[SupportsFloat, Var]) -> bool:
        """
        As other order operations compares self.value with other.value or number
        """
        if isinstance(other, Var):
            return self.val() <= other.val()
        else:
            return self.val() <= float(other)

    def __eq__(self, other: Union[SupportsFloat, Var]) -> bool:
        """
        As other order operations compares self.value with other.value or number
        """
        if isinstance(other, Var):
            return self.val() == other.val()
        else:
            return self.val() == float(other)

    def _binary_operation(self, other: TypicalArgument, func: Callable) -> Union[Var, GroupVar]:
        if isinstance(other, Var):
            return Var(func(self._story, other._story))
        if isinstance(other, SupportsFloat):
            return Var(func(self._story, float(other)))
        if isinstance(other, GroupVar):
            return GroupVar(tuple(func(self, other[i]) for i in range(len(other))))

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
        return self._binary_operation(power, lambda x, y: x ** y)

    def __pos__(self) -> Var:
        return self

    def __neg__(self) -> Var:
        return Var(-self._story)


class GroupVar:
    """
    This class implements the idea of a series of similar measurements. The behavior is the same as in numpy arrays.
    So, adding, multiplying of two GroupVar is corresponding operation between Vars inside GroupVars.
    The only slot in this class is 'variables'. It's a list containing Vars.
    (To read documentation to this class, it's necessary to understand the same methods in Var class.)
    """
    @overload
    def __init__(self, variables: Sequence[Var], exp=0):
        """
        Generates GroupVar where 'variables' will be in 'self.variables'
        :param variables: something containing Vars
        :param exp: if exp!=0 then all variables will be changed:
            var -> var * 10 ** exp
        """
        ...

    @overload
    def __init__(self, values: Sequence[SupportsFloat], errors: Sequence[SupportsFloat], exp=0):
        """
        Generates lots of 'Var(value, error)' and puts them to 'self.variables'
        :param values: something containing numbers
        :param errors: something containing numbers
        :param exp: exp: if exp!=0 then all variables will be changed:
            var -> var * 10 ** exp
        """
        ...

    @overload
    def __init__(self, values: Sequence[SupportsFloat], error: SupportsFloat, exp=0):
        """
        Very similar as the previous overloaded __init__ method, but all errors in variables will be the same.
        The same as calling
        GroupVar(values, [error]*len(values), exp)
        """
        ...

    def __init__(self, *args, exp=0):
        if len(args) == 2:
            values, errors = args[0], {True: args[1], False: [args[1]] * len(args[0])}[hasattr(args[1], '__iter__')]
            if len(values) != len(errors):
                raise TypeError('Arguments must be the same length')
            self.variables: List[Var] = [Var(val, err) * 10 ** exp for val, err in zip(values, errors)]
        else:
            self.variables: List[Var] = [var * 10 ** exp for var in args[0]]

    def val_err(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """
        :return: Tuple consisting of (tuple of values) and (tuple of errors)
        """
        return tuple(zip(*(var.val_err() for var in self)))

    def val(self) -> List[float]:
        """
        :return: list of values
        """
        return [var.val() for var in self.variables]

    def err(self) -> List[float]:
        """
        :return: list of errors
        """
        return [var.err() for var in self.variables]

    def __getitem__(self, item: Union[int, slice]):
        """
        If type of item is 'int', then returns self.variables[item].
        If type of item is 'slice', then returns GroupVar(self.variables[item])
        """
        if isinstance(item, slice):
            return GroupVar(self.variables[item])
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
        return GroupVar([-var for var in self.variables])

TypicalArgument = Union[SupportsFloat, Var, GroupVar]

error_accuracy: float = 0.4
value_accuracy: float = 0.05


def set_error_accuracy(accuracy: float):
    """Look in Var.__str__ documentation"""
    global error_accuracy
    error_accuracy = accuracy


def set_value_accuracy(accuracy: float):
    """Look in Var.__str__ documentation"""
    global value_accuracy
    value_accuracy = accuracy


def suitable_accuracy(val: float, err: float) -> int:
    """Finds needed number of digits as it was shown Var.__str__ documentation"""
    if err == 0:
        return Decimal.from_float(val * value_accuracy).adjusted()
    return Decimal.from_float(err * error_accuracy).adjusted()


def normalize(var: Var, accuracy: Optional[int] = None) -> str:
    """The same as str(var), but you can set the number of shown digits in parameter 'accuracy'"""
    val, err = var.val_err()
    if accuracy is None:
        accuracy = suitable_accuracy(val, err)
    return r'{0} \pm {1}'.format(
        *({True: float, False: int}[accuracy < 0](round(num, -accuracy)) for num in (val, err)))


BIG_NUMBER = 50


def set_big_number(n: int):
    """look in Var.err documentation"""
    global BIG_NUMBER
    BIG_NUMBER = n


def estimated_error(func: Callable[[...], float], values: array, err_vector: array, val: float):
    err_vector /= BIG_NUMBER
    # we catch warnings turning them to errors
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
