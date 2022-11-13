import sys as _sys
from typing import SupportsFloat, Tuple, Union, List, Callable, Optional, Sequence

import numpy as np
from numpy import argmin as _argmin
from numpy import argsort as _argsort
from numpy import var as _var
from scipy.interpolate import UnivariateSpline as _UnivariateSpline
from scipy.optimize import fmin as _fmin
from scipy.optimize import curve_fit as _curve_fit

from lab_cheat import *
from .table import rus_tex_formula
from .var import set_value_accuracy, set_error_accuracy, set_big_number, normalize, GroupVar

_rare_used_funcs = [set_value_accuracy, set_error_accuracy, set_big_number, rus_tex_formula, normalize]


def slicer(var: GroupVar, left_val: Optional[SupportsFloat] = None,
           right_val: Optional[SupportsFloat] = None,
           excluding: Optional[List[SupportsFloat]] = None) -> Union[List[int]]:
    """
    Makes list of indexes of variables which value is between given borders except the nearest vars to numbers in
    exluding list
    :param var: GroupVar that contains variables
    :param left_val: left border. If None then -infinity.
    :param right_val: right border. If None then +infinity.
    :param excluding: If None then []
    Ex:
    >>> v = GroupVar(range(10), 0)
    >>> s = slicer(v, left_val=1.1, right_val=8, excluding=[3.1, 6.3, 0])
    >>> v[s].val()
    [2.0, 4.0, 5.0, 7.0, 8.0]
    """
    vals = var.val()
    if left_val is None:
        left_val = float('-inf')
    if right_val is None:
        right_val = float('inf')
    # if excluding is None:
    #     needed_part = filter(lambda x: left_val <= x < right_val, var)
    #     return slice(var.variables.index(min(needed_part), var.variables.index(max(needed_part))))
    excluding_elements = {_argmin(tuple(map(lambda x: abs(x - ex), vals))) for ex in excluding} \
        if excluding is not None else set()
    return sorted(
        list(filter(lambda i: left_val <= vals[i] <= right_val and i not in excluding_elements, range(len(vals)))))


# def parabola_coefficients(x1, x2, x3, y1, y2, y3):
#     """
#     Finds parabola coefficients drown through 3 dots.
#     (x1, y1), (x2, y2), (x3, y3) - are these dots
#     y_exp = ax^2+bx+colour
#     :return: a, b, colour
#     """
#     a = (y3 - (x3 * (y2 - y1) + x2 * y1 - x1 * y2) / (x2 - x1)) / (x3 * (x3 - x1 - x2) + x1 * x2)
#     b = (y2 - y1) / (x2 - x1) - a * (x1 + x2)
#     c = (x2 * y1 - x1 * y2) / (x2 - x1) + a * x1 * x2
#     return a, b, c
#
#
# def parabola_func(a, b, c):
#     return lambda x: a * x ** 2 + b * x + c
#
#
# def draw_parabola(x_min, x_max, N, a, b, c):
#     """needed in my_func in Figure
#     N - number of dots, (a, b, colour) - parabola coefficients"""
#
#     def _draw_parabola(ax):
#         x = _linspace(x_min, x_max, N)
#         ax.scatter(x=x, y=a * x ** 2 + b * x + c, s=1)
#
#     return _draw_parabola


def prototype(XL: str, zero_in_corner=True) -> None:
    x_val, y_val = to_table(XL)
    x, y = GroupVar(x_val, 0), GroupVar(y_val, 0)
    Figure(zero_in_corner=zero_in_corner).plot(x, y).show()


def sorting(x: Union[array, List, GroupVar], y: Union[array, List, GroupVar]) -> Tuple[
    Union[array, List, GroupVar], Union[array, List, GroupVar]]:
    """
    !!! x and y must be the same type !!!
    Sorts pairs of data to make xt raising
    :return: the same types as x and y.
    """
    if isinstance(x, list) and isinstance(y, list):
        indexes = _argsort(array(x))
        return [x[i] for i in indexes], [y[i] for i in indexes]
    elif isinstance(x, GroupVar) and isinstance(y, GroupVar):
        indexes = _argsort(array(x.val()))
        return GroupVar([x[i] for i in indexes]), GroupVar([y[i] for i in indexes])
    else:
        try:
            indexes = _argsort(x)
            x: array
            y: array
            return x[indexes], y[indexes]
        except Exception:
            raise TypeError('x and y must be the same type')


def smoothing(x: Union[array, List, GroupVar], y: Union[array, List, GroupVar], smooth_factor) -> Callable:
    # todo: сделать покопаться в _UnivariateSpline и учитывать каждую точку с весом ошибки
    x, y = sorting(x, y)
    if isinstance(x, GroupVar):
        x = x.val()
    if isinstance(y, GroupVar):
        y = y.val()
    x, y = map(array, [x, y])
    f = _UnivariateSpline(x, y)
    f.set_smoothing_factor(smooth_factor)
    # todo: лучше, конечно, было бы если возвращаемая функция была приведена к sympy.Function и её можно было бы
    #  хранить, как sin это позволило бы передавать ей на вход Var и GroupVar!
    return f


def fmin(f: Callable, x0: Union[Var, SupportsFloat], x: Optional[GroupVar] = None, y: Optional[GroupVar] = None) \
        -> Union[Var, SupportsFloat]:
    """
    fmin is for very accurate search of minimum of known graph
    :param f: function to minimize
    :param x0: dot near minimum
    :params x and y to find the error accurately
    :return: x_min
    """
    # TODO: сделать так, чтоб искало минимум на заднанном промежутке. (Потому что, при данной реализации минимум
    #  функции-колокола будет ниже, чем любая переданная точка. Что не есть хорошо)
    v, _sys.stdout = _sys.stdout, None
    if isinstance(x0, Var):
        x0 = x0.val()
    x_min = _fmin(f, x0)[0]
    if isinstance(y, GroupVar) and isinstance(x, GroupVar):
        x, y = sorting(x, y)
        i_min = sorted(list(x.val()) + [x_min]).index(x_min)
        if i_min == len(x):
            f_min, y_err, err = f(x_min), y[i_min - 1].err(), x[i_min - 1].err()
        else:
            f_min, y_err, err = f(x_min), y[i_min].err(), x[i_min].err()
        x_err_approx = sqrt(2 * y_err / (f(x_min + err) + f(x_min - err) - 2 * f_min)) * err
        _sys.stdout = v
        return Var(x_min, sqrt(x_err_approx ** 2 + err ** 2))
    else:
        _sys.stdout = v
        return x_min


def fmax(f: Callable, x0: Union[Var, SupportsFloat], x: Optional[GroupVar] = None, y: Optional[GroupVar] = None) \
        -> Union[Var, SupportsFloat]:
    return fmin(lambda t: -f(t), x0, x=x, y=y)


def curve_fit(f: Callable, x: GroupVar, y: GroupVar, p0: Optional[Sequence[SupportsFloat]] = None):
    """unfortunately, ignores x.err()"""
    initial_p = None if p0 is None else np.array(p0)
    too_small_err = np.any(np.asarray(y.err()) < 1/np.finfo(np.float_).max)
    sigma = None if too_small_err else y.err()
    p_opt, p_cov = _curve_fit(f, xdata=x.val(), ydata=y.val(), p0=initial_p, sigma=sigma)
    return GroupVar(p_opt, np.diag(p_cov))


def sigma(variable: Union[GroupVar, Sequence]):
    if isinstance(variable, GroupVar):
        variable = variable.val()
    return sqrt(_var(variable))
