from functools import reduce
from typing import Optional, Tuple, Union, Callable

import matplotlib.patches as _mp
import matplotlib.pyplot as plt

from .var import Var, GroupVar


class Figure:
    def __init__(self, x_label: str = '', y_label: str = '', bold_axes: bool = True, zero_in_corner: bool = True,
                 label_near_arrow: bool = True, my_func: Optional[Callable] = None):
        """
        :param x_label: label near x axis
        :param y_label: label near y axis
        :param bold_axes: need of y=0 and x=0 lines
        :param zero_in_corner: need of showing (0, 0) wherever the other dots
        :param label_near_arrow: if True: labels will be shown in the corners near arrows of axis.
                                 if False: labels will be near centers of appropriate axis.
        :param my_func: function that takes an object of the matplotlib.axes._subplots.AxesSubplot class as an argument.
        It will be called before fixing axes and adding lines. You may use it to add anything you want without using Var
        and GroupVar.
        """
        self.x_label, self.y_label = x_label, y_label
        self.bold_axes = bold_axes
        self.zero_in_corner = zero_in_corner
        self.label_near_arrow = label_near_arrow
        self.my_func = my_func
        # params for 'plot' method
        self._scatters_kwargs = []
        self._errorbars_kwargs = []
        # params for 'line' method
        self._lines_params = []

    def line(self, k: Union[float, int], b: Union[float, int], colour: Optional[str] = None,
             line_style: Optional[str] = None, label: Optional[str] = None):
        self._lines_params.append((k, b, colour, line_style, label))

    def plot(self, x: GroupVar, y: GroupVar, capsize=3, s=1, c=None, marker=None, label=None) -> None:
        x_val, x_err = x.val_err()
        y_val, y_err = y.val_err()
        self._scatters_kwargs.append(
            dict(x=x_val, y=y_val, s=s, c=c, marker=marker, label=label))
        self._errorbars_kwargs.append(
            dict(x=x_val, y=y_val, xerr=x_err, yerr=y_err, capsize=capsize, capthick=1, fmt='none', c=c))

    def show(self):
        """
        Generates matplotlib.Figure and shows it.
        :return: None
        """
        axes = plt.figure().add_subplot()
        self._grid_lines(axes)
        self._show_plots(axes)
        # maybe user wants to do something by himself
        if self.my_func is not None:
            self.my_func(axes)
        xy_limits = self._fix_axes(axes)
        self._set_label(axes)
        self._arrows(axes)
        if self.bold_axes is True:
            self._bold_axes(axes, *xy_limits)
        self._show_lines(axes, *xy_limits)
        plt.show()

    @staticmethod
    def _grid_lines(axes):
        axes.grid(axis='both', which='major', linestyle='--', linewidth=1)
        axes.grid(axis='both', which='minor', linestyle='--', linewidth=0.5)
        axes.minorticks_on()

    def _show_plots(self, axes):
        for scatter_kwargs, errorbar_kwargs in zip(self._scatters_kwargs, self._errorbars_kwargs):
            axes.scatter(**scatter_kwargs)
            axes.errorbar(**errorbar_kwargs)

    def _fix_axes(self, axes):
        x_min, x_max = axes.get_xlim()
        y_min, y_max = axes.get_ylim()
        if self.zero_in_corner is True:
            x_min = min(0, x_min)
            y_min = min(0, y_min)
        axes.set_xlim(x_min, x_max)
        axes.set_ylim(y_min, y_max)
        return x_min, x_max, y_min, y_max

    def _set_label(self, axes):
        if self.label_near_arrow is True:
            y_label_prop = dict(rotation=0, y=1)
            x_label_prop = dict(rotation=0, x=1)
        else:
            x_label_prop, y_label_prop = {}, {}
        if self.x_label.rstrip() != '':
            axes.set_xlabel('$' + self.x_label + '$', x_label_prop)
        if self.y_label.rstrip() != '':
            axes.set_ylabel('$' + self.y_label + '$', y_label_prop)

    def _arrows(self, axes):
        axes.annotate('', xy=(1.05, 0), xycoords='axes fraction', xytext=(-0.03, 0),
                      arrowprops=dict(arrowstyle=_mp.ArrowStyle.CurveB(head_length=1), color='black'))
        axes.annotate('', xy=(0, 1.06), xycoords='axes fraction', xytext=(0, -0.03),
                      arrowprops=dict(arrowstyle=_mp.ArrowStyle.CurveB(head_length=1), color='black'))

    def _bold_axes(self, axes, x_min, x_max, y_min, y_max):
        axes.hlines(0, x_min, x_max, linewidth=1, colors='black')
        axes.vlines(0, y_min, y_max, linewidth=1, colors='black')

    def _show_lines(self, axes, x_min, x_max, y_min, y_max):
        for k, b, c, ls, label in self._lines_params:
            points = []
            if y_min <= k * x_min + b <= y_max:
                points.append((x_min, k * x_min + b))
            if y_min <= k * x_max + b <= y_max:
                points.append((x_max, k * x_max + b))
            if len(points) < 2 and x_min < (y_max - b) / k < x_max:
                points.append(((y_max - b) / k, y_max))
            if len(points) < 2 and x_min < (y_min - b) / k < x_max:
                points.append(((y_min - b) / k, y_min))
            axes.plot((points[0][0], points[1][0]), (points[0][1], points[1][1]), c=c, ls=ls, label=label)


def mnk(x: GroupVar, y: GroupVar, figure: Optional[Figure] = None) -> Tuple[Var, Var]:
    if len(x) != len(y): raise TypeError('"x" and "y" must be the same length')
    x_sum: Var = reduce(lambda res, x_var: res + x_var, x)
    y_sum: Var = reduce(lambda res, y_var: res + y_var, y)
    k: Var = (len(x) * reduce(lambda res, i: res + x[i] * y[i], range(len(x)), 0) - x_sum * y_sum) / \
             (len(x) * reduce(lambda res, x_var: res + x_var * x_var, x, 0) - x_sum * x_sum)
    b: Var = (y_sum - k * x_sum) / len(x)
    if figure is not None:
        figure.line(k.val(), b.val())
    return k, b


def mnk_through0(x: GroupVar, y: GroupVar, figure: Optional[Figure] = None) -> Var:
    if len(x) != len(y): raise TypeError('"x" and "y" must be the same length')
    k: Var = reduce(lambda res, i: res + x[i] * y[i], range(len(x)), 0) / \
             reduce(lambda res, x_var: res + x_var * x_var, x[1:], x[0] * x[0])
    if figure is not None:
        figure.line(k.val(), 0)
    return k
