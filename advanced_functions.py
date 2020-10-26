from typing import SupportsFloat

from numpy import linspace as _linspace

from .table import rus_tex_formula
from .var import set_value_accuracy, set_error_accuracy, set_big_number, normalize, GroupVar

_rare_used_funcs = [set_value_accuracy, set_error_accuracy, set_big_number, rus_tex_formula, normalize]


def slicer(var: GroupVar, left_val: SupportsFloat, right_val: SupportsFloat) -> slice:
    """
    Makes python slice containing indexes of variables which value is between given borders.
    :param var: GroupVar that contains variables
    :param left_val: left border
    :param right_val: right border
    :return: slice
    """
    needed_part = filter(lambda x: left_val <= x < right_val, var)
    return slice(var.variables.index(min(needed_part), var.variables.index(max(needed_part))))


def parabola_coefficients(x1, x2, x3, y1, y2, y3):
    """
    Finds parabola coefficients drown through 3 dots.
    (x1, y1), (x2, y2), (x3, y3) - are these dots
    y = ax^2+bx+c
    :return: a, b, c
    """
    a = (y3 - (x3 * (y2 - y1) + x2 * y1 - x1 * y2) / (x2 - x1)) / (x3 * (x3 - x1 - x2) + x1 * x2)
    b = (y2 - y1) / (x2 - x1) - a * (x1 + x2)
    c = (x2 * y1 - x1 * y2) / (x2 - x1) + a * x1 * x2
    return a, b, c


def parabola_func(a, b, c):
    return lambda x: a * x ** 2 + b * x + c


def draw_parabola(x_min, x_max, N, a, b, c):
    """needed in my_func in Figure
    N - number of dots, (a, b, c) - parabola coefficients"""
    def _draw_parabola(ax):
        x = _linspace(x_min, x_max, N)
        ax.scatter(x=x, y=a * x ** 2 + b * x + c, s=1)

    return _draw_parabola
