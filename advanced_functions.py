from .var import set_value_accuracy, set_error_accuracy, set_big_number, normalize
from .table import rus_tex_formula
from numpy import linspace as _linspace

_very_useful_funcs = [set_value_accuracy, set_error_accuracy, set_big_number, rus_tex_formula, normalize]

def parabola_coefficients(x1, x2, x3, y1, y2, y3):
    # ax^2+bx+c
    a = (y3 - (x3 * (y2 - y1) + x2 * y1 - x1 * y2) / (x2 - x1)) / (x3 * (x3 - x1 - x2) + x1 * x2)
    b = (y2 - y1) / (x2 - x1) - a * (x1 + x2)
    c = (x2 * y1 - x1 * y2) / (x2 - x1) + a * x1 * x2
    return a, b, c


def draw_parabola_func(x_min, x_max, N, a, b, c):
    def draw_parabola(ax):
        x = _linspace(x_min, x_max, N)
        ax.scatter(x=x, y=a * x ** 2 + b * x + c, s=1)

    return draw_parabola
