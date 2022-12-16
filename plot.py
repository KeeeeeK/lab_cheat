from __future__ import annotations

from functools import reduce
from tkinter.messagebox import showwarning, showerror
from typing import Optional, Tuple, Union, Callable, SupportsFloat, Sequence

import matplotlib.patches as _mp
import matplotlib.pyplot as plt
from numpy import array, linspace, sqrt

from .var import Var, GroupVar


class Figure:
    """
    Данный класс используется для работы с графиками
    Его объекты соответствуют figure matplotlib, т. е. один объект - полноценное окно с графиком
    """

    def __init__(self, x_label: str = '', y_label: str = '', bold_axes: bool = True, zero_in_corner: bool = True,
                 label_near_arrow: bool = True, my_func: Optional[Callable] = None,
                 x_label_coords: Sequence[SupportsFloat] = None,
                 y_label_coords: Sequence[SupportsFloat] = None,
                 legend_props: Optional[dict] = None):
        """
        :param x_label: Подпись около оси X
        :param y_label: Подпись около оси Y
        :param bold_axes: Нужны ли жирные оси X и Y
        :param zero_in_corner: Нужно ли чтобы точка (0, 0) отображалась на графике, и в ней пересекались оси X и Y
        :param label_near_arrow: Если True: Названия будут отображаться в углу рядом с концами стрелок осей
                                 Если False: Названия будут отображаться посередине осей
        :param my_func: функция, которая принимает объект класса matplotlib.axes._subplots.AxesSubplot как аргумент
        Будет вызвана до нормировки осей и рисования линий. Может быть использована для использования любых
        возможностей matplotlib.
        :param x_label_coords: Координаты размещения названия оси X.
        Легко двигать название (x_label) таким образом: figure.x_label_coords+=array([0.03, -0.04])
        :param y_label_coords: То же самое, что и x_label_coords
        :param legend_props: Словарь объектов того, что должно быть в легенде, нужен только для контроля легенды,
        если не указан, в легенде будут все элементы
        """
        self.x_label, self.y_label = x_label, y_label
        if x_label != '' and x_label.count(',') == 0:
            showwarning("Странное название оси X", "В названии оси X не обнаружена размерность, "
                                                   "вряд ли график с такими подписями осей кому-то нужен")
        if y_label != '' and y_label.count(',') == 0:
            showwarning("Странное название оси Y", "В названии оси Y не обнаружена размерность, "
                                                   "вряд ли график с такими подписями осей кому-то нужен")
        self.bold_axes = bold_axes
        self.zero_in_corner = zero_in_corner
        self.label_near_arrow = label_near_arrow
        self.my_func = my_func
        self.legend_props: dict = legend_props if legend_props is not None else {}
        self.alpha = 0.25  # the transparency of v_lines and h_lines
        # params for 'plot' method
        self._scatters_kwargs = []
        self._errorbars_kwargs = []
        # params for 'line' method
        self._lines_params = []
        self._v_lines_params = []
        self._h_lines_params = []
        self._func_graphs_before_fixing_axes = []
        self._func_graphs_after_fixing_axes = []

        # Выбираем место расположения названий осей
        t_x, t_y = False, False
        if x_label_coords is None:
            if len(x_label) <= 6:
                self.x_label_coords = [1.01 + 0.01 * len(x_label) * 0.8, 0.05]
            else:
                self.x_label_coords = [1.01, - 0.08]
                t_x = True
        else:
            self.x_label_coords = x_label_coords
        if y_label_coords is None:
            if len(y_label) <= 7:
                self.y_label_coords = [-0.02 - 0.01 * (len(y_label) * 0.7), 1.03]
            else:
                self.y_label_coords = [0, 1.05]
                t_y = True
        else:
            self.y_label_coords = y_label_coords
        if t_x and t_y:
            showwarning("Названия обеих осей слишком длинное",
                        "Так как названия обеих осей очень длинное, оно не помещается в обычное место, поэтому "
                        "рекомендуется установить параметру label_near_axes значение False")
        elif t_x:
            showwarning("Название оси X слишком длинное",
                        "Так как название оси X очень длинное, оно не помещается в обычное место для него, поэтому "
                        "рекомендуется установить параметру label_near_axes значение False")
        elif t_y:
            showwarning("Название оси Y слишком длинное",
                        "Так как название оси Y очень длинное, оно не помещается в обычное место для него, поэтому "
                        "рекомендуется установить параметру label_near_axes значение False")

    def line(self, k: Union[float, int, Var], b: Union[float, int, Var], colour: Optional[str] = None,
             line_style: Optional[str] = None, label: Optional[str] = None) -> Figure:
        """
        Строит наклонную прямую с заданными коэффициентами вида y = kx + b. Является вспомогательным объектом, т. е. не
        будет подгонять под себя масштаб осей.
        :param k: Коэффициент наклона прямой.
        :param b: Свободный член прямой.
        :param colour: цвет линии 'b' голубой, 'g' зелёный, 'r' красный, 'c' бирюзовый,
        'm' розовый, 'y' жёлтый, 'k' чёрный, 'w' белый.
        :param line_style: Стиль прямой 'solid', 'dotted', 'dashed', 'dashdot'.
        :param label: Название прямой для легенды.
        :return: Себя же, для однострочной записи.
        """
        if isinstance(k, Var):
            k = k.val()
        if isinstance(b, Var):
            b = b.val()
        self._lines_params.append((k, b, colour, line_style, label))
        return self

    def v_line(self, x: Union[float, int, Var], colour: Optional[str] = None, line_style: Optional[str] = None,
               label: Optional[str] = None) -> Figure:
        """
        Строит вертикальную прямую через весь график. вляется вспомогательным объектом, т. е. не будет подгонять
        под себя масштаб осей.
        :param x: Абсцисса прямой.
        :param colour: цвет линии 'b' голубой, 'g' зелёный, 'r' красный, 'c' бирюзовый,
        'm' розовый, 'y' жёлтый, 'k' чёрный, 'w' белый.
        :param line_style: Стиль прямой.
        :param label: Название прямой для создания легенды.
        :return: Себя же, для однострочной записи.
        """
        self._v_lines_params.append((x, colour, line_style, label))
        return self

    def h_line(self, y: Union[float, int, Var], colour: Optional[str] = None, line_style: Optional[str] = None,
               label: Optional[str] = None) -> Figure:
        """
        Строит горизонтальную прямую через весь график. Является вспомогательным объектом, т. е. не будет подгонять
        под себя масштаб осей.
        :param y: Ордината прямой.
        :param colour: цвет линии 'b' голубой, 'g' зелёный, 'r' красный, 'c' бирюзовый,
        'm' розовый, 'y' жёлтый, 'k' чёрный, 'w' белый.
        :param line_style: Стиль прямой 'solid', 'dotted', 'dashed', 'dashdot'.
        :param label: Название прямой для создания легенды.
        :return: Себя же, для однострочной записи.
        """
        self._h_lines_params.append((y, colour, line_style, label))
        return self

    def func_graph(self, func: Callable[[array], array], x_min: Union[int, float, Var], x_max: Union[int, float, Var],
                   n: int = 1000, line_style: Optional[str] = None, colour: Optional[str] = None,
                   label: Optional[str] = None, add_before_fixing_axes: bool = True) -> Figure:
        """
        todo: сделать возможность проводить линию до края графика по оси x влево или вправо, если указано None;
        Отвечает за создание графика данной функции на экране. Может являться как основным,
        так и второстепенным объектом, в зависимости от значения параметра add_before_fixing_axes,
        если он False - то основной объект и наоборот.
        :param func: Принимает функцию, получающую массив, выводящую массив.
        :param x_min: Наименьшее значение по оси X.
        :param x_max: Наибольшее значение по оси X.
        :param n: Количество точек в данном промежутке.
        :param line_style: Стиль линии 'solid', 'dotted', 'dashed', 'dashdot'.
        :param colour: цвет линии 'b' голубой, 'g' зелёный, 'r' красный, 'c' бирюзовый,
        'm' розовый, 'y' жёлтый, 'k' чёрный, 'w' белый.
        :param label: Название данной кривой для создания легенды.
        :param add_before_fixing_axes: Нужно ли подогнать размеры по осям.
        (True - без учёта данной кривой, False - с учётом)
        :return: Возвращает объект класса Figure, который представляет собой нарисованную кривую - график вашей функции
        Вывод этого объекта необходим для возможности записи кода в виде obj.func_graph(arg...).func().
        """
        if isinstance(x_min, Var):
            x_min = x_min.val()
        if isinstance(x_max, Var):
            x_max = x_max.val()
        x = linspace(x_min, x_max, n)
        y = func(x)
        needed_func_graph_list = self._func_graphs_after_fixing_axes if add_before_fixing_axes \
            else self._func_graphs_before_fixing_axes
        needed_func_graph_list.append((x, y, line_style, colour, label))
        return self

    def plot(self, x: Union[GroupVar, Sequence], y: Union[GroupVar, Sequence],
             capsize=3, s=1, colour=None, marker=None, label=None) -> Figure:
        """
        Отвечает за нанесение точек с погрешностями (крестов) на график.
        :param x: Итерируемый объект с абсциссами точке.
        :param y: Итерируемый объект с ординатами точек.
        :param capsize: Размер кончиков крестов погрешностей точек.
        :param s: Радиус точек.
        :param colour: цвет точек 'b' голубой, 'g' зелёный, 'r' красный, 'c' бирюзовый,
        'm' розовый, 'y' жёлтый, 'k' чёрный, 'w' белый.
        :param marker: Тип маркера точек
        Самые часто используемые маркеры: '.', '+', 'x'.
        :param label: Название точек для легенды.
        :return: Себя, для записи в одну строку.
        """

        def val_err(t):
            if isinstance(t, GroupVar):
                return t.val_err()
            nonlocal capsize
            capsize = 0
            return array(t), array([0] * len(t))

        x_val, x_err = val_err(x)
        y_val, y_err = val_err(y)
        self._scatters_kwargs.append(
            dict(x=x_val, y=y_val, s=s, c=colour, marker=marker, label=label))
        self._errorbars_kwargs.append(
            dict(x=x_val, y=y_val, xerr=x_err, yerr=y_err, capsize=capsize, capthick=1, fmt='none', c=colour))
        return self

    def show(self):
        """
        Создаёт окно matplotlib и рисует в нём всё, что было в объекте.
        :return: Ничего.
        """
        axes = plt.figure().add_subplot()
        self._grid_lines(axes)
        self._show_plots(axes)
        self._show_func_graphs_before_fixing_axes(axes)
        # maybe user wants to do something by himself
        if self.my_func is not None:
            self.my_func(axes)
        xy_limits = self._fix_axes(axes)
        self._v_lines(axes, *xy_limits)
        self._h_lines(axes, *xy_limits)
        self._set_label(axes)
        self._arrows(axes)
        if self.bold_axes is True:
            self._bold_axes(axes, *xy_limits)
        self._show_lines(axes, self.legend_props, *xy_limits)
        self._show_func_graphs_after_fixing_axes(axes)
        plt.show()

    @staticmethod
    def _grid_lines(axes):
        """
        Рисует сетку (второстепенные пунктирные линии).
        :param axes: Область, на которой отражаются все графики, оси и т.п.
        :return: Ничего.
        """
        axes.grid(axis='both', which='major', linestyle='--', linewidth=1)
        axes.grid(axis='both', which='minor', linestyle='--', linewidth=0.5)
        axes.minorticks_on()

    def _show_plots(self, axes):
        """
        Наносит на точки и погрешности на график.
        :param axes: Область, на которой отражаются все графики, оси и т.п.
        :return: Ничего.
        """
        for scatter_kwargs, errorbar_kwargs in zip(self._scatters_kwargs, self._errorbars_kwargs):
            axes.scatter(**scatter_kwargs)
            axes.errorbar(**errorbar_kwargs)

    def _show_func_graphs_before_fixing_axes(self, axes):
        """
        Отвечает за отрисовку функции до подгона масштаба осей.
        :param axes: Область, на которой отражаются все графики, оси и т.п.
        :return: Ничего.
        """
        for x, y, line_style, colour, label in self._func_graphs_before_fixing_axes:
            axes.plot(x, y, color=colour, linestyle=line_style, label=label)

    def _fix_axes(self, axes):
        """
        Настраивает объект axes.
        :param axes: Область, на которой отражаются все графики, оси и т.п.
        :return: Максимальные и минимальные значения по осям.
        """
        x_min, x_max = axes.get_xlim()
        y_min, y_max = axes.get_ylim()
        if self.zero_in_corner is True:
            x_min = 0
            y_min = 0
        axes.set_xlim(x_min, x_max)
        axes.set_ylim(y_min, y_max)
        return x_min, x_max, y_min, y_max

    def _v_lines(self, axes, x_min, x_max, y_min, y_max):
        """
        Функция отвечающая за построение вертикальных прямых.
        :param axes: Область, на которой отражаются все графики, оси и т.п.
        :param x_min: -
        :param x_max: -
        :param y_min: Координата начала прямой.
        :param y_max: Координата конца прямой.
        :return: Ничего.
        """
        for x, colour, line_style, label in self._v_lines_params:
            if isinstance(x, Var):
                x_val, x_err = x.val_err()
                plot, = axes.plot((x_val, x_val), (y_min, y_max), color=colour, linestyle=line_style, label=label)
                axes.plot((x_val - x_err, x_val - x_err), (y_min, y_max), color=plot.get_color(), linestyle=':')
                axes.plot((x_val + x_err, x_val + x_err), (y_min, y_max), color=plot.get_color(), linestyle=':')
                axes.fill_between((x_val - x_err, x_val + x_err), (y_min, y_min), (y_max, y_max),
                                  color=plot.get_color(), alpha=self.alpha)
            else:
                axes.plot((x, x), (y_min, y_max), color=colour, linestyle=line_style, label=label)

    def _h_lines(self, axes, x_min, x_max, y_min, y_max):
        """
        Функция отвечающая за построение вертикальных прямых.
        :param axes: Область, на которой отражаются все графики, оси и т.п.
        :param x_min: Координата начала прямой.
        :param x_max: Координата конца прямой.
        :param y_min: -
        :param y_max: -
        :return: Ничего.
        """
        for y, colour, line_style, label in self._h_lines_params:
            if isinstance(y, Var):
                y_val, y_err = y.val_err()
                plot, = axes.plot((x_min, x_max), (y_val, y_val), color=colour, linestyle=line_style, label=label)
                axes.plot((x_min, x_max), (y_val - y_err, y_val - y_err), color=plot.get_color(), linestyle=':')
                axes.plot((x_min, x_max), (y_val + y_err, y_val + y_err), color=plot.get_color(), linestyle=':')
                axes.fill_between((x_min, x_max), (y_val - y_err, y_val - y_err), (y_val + y_err, y_val + y_err),
                                  color=plot.get_color(), alpha=self.alpha)
            else:
                axes.plot((x_min, x_max), (y, y), color=colour, linestyle=line_style, label=label)

    def _set_label(self, axes):
        """
        Устанавливает названия осей по координатам.
        :param axes: Область, на которой отражаются все графики, оси и т.п.
        :return: Ничего.
        """
        for set_label, axis, label, label_coords in ((axes.set_xlabel, axes.xaxis, self.x_label, self.x_label_coords),
                                                     (axes.set_ylabel, axes.yaxis, self.y_label, self.y_label_coords)):
            if label.rstrip() != '':
                label_prop = {True: dict(rotation=0), False: {}}[self.label_near_arrow]
                set_label('$' + label + '$', label_prop)
                if self.label_near_arrow is True:
                    axis.set_label_coords(*label_coords)
    @staticmethod
    def _arrows(axes):
        """
        Отвечает за правильное позиционирование и отрисовку стрелок на графике.
        :param axes: Область, на которой отражаются все графики, оси и т.п.
        :return: Ничего.
        """
        arrowprops = dict(arrowstyle=_mp.ArrowStyle.CurveB(head_length=1), color='black')
        axes.annotate('', xy=(1.05, 0), xycoords='axes fraction', xytext=(-0.03, 0), arrowprops=arrowprops)
        axes.annotate('', xy=(0, 1.06), xycoords='axes fraction', xytext=(0, -0.03), arrowprops=arrowprops)

    @staticmethod
    def _bold_axes(axes, x_min, x_max, y_min, y_max):
        """
        Отвечает за отрисовку жирных осей.
        :param axes: Область, на которой отражаются все графики, оси и т.п.
        :param x_min: Начало оси абсцисс.
        :param x_max: Конец оси абсцисс.
        :param y_min: Начало оси ординат.
        :param y_max: Конец оси ординат.
        :return: Ничего.
        """
        axes.hlines(0, x_min, x_max, linewidth=1, colors='black')
        axes.vlines(0, y_min, y_max, linewidth=1, colors='black')

    def _show_lines(self, axes, legend_props, x_min, x_max, y_min, y_max):
        """
        Отвечает за отрисовку ВСПОМОГАТЕЛЬНЫХ линий.
        :param axes: Область, на которой отражаются все графики, оси и т.п.
        :param legend_props: Словарь объектов того, что должно быть в легенде, нужен только для контроля легенды.
        :param x_min: Абсцисса начала прямой.
        :param x_max: Абсцисса конца прямой.
        :param y_min: Ордината начала прямой.
        :param y_max: Ордината конца прямой.
        :return: Ничего.
        """
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
            if len(points) < 2:
                showwarning("Ваша прямая не помещается на график!", "Прямая с параметрами k =" + k + " , b = " + b +
                            "не помещается на график и не будет отрисована. Связано это с тем, что функция line() " +
                            "предназначена для построения вспомогательных линий, а не основных, так что она не " +
                            "подгоняет под себя масштаб осей, для этого используйте func_graph с параметром " +
                            "add_before_fixing_axes=False")
                return
            axes.plot((points[0][0], points[1][0]), (points[0][1], points[1][1]), c=c, ls=ls, label=label)
        if len(axes.get_legend_handles_labels()[1]) != 0:
            plt.legend(**legend_props)

    def _show_func_graphs_after_fixing_axes(self, axes):
        """
        Рисует точки на графике, для случая, когда мы подгоняем оси до отрисовки точек.
        :param axes: Область, на которой отражаются все графики, оси и т.п.
        :return: Ничего.
        """
        for x, y, line_style, colour, label in self._func_graphs_after_fixing_axes:
            axes.plot(x, y, color=colour, linestyle=line_style, label=label)


def mnk(x: Union[GroupVar, Sequence], y: Union[GroupVar, Sequence], figure: Optional[Figure] = None,
        colour: Optional[str] = None,
        line_style: Optional[str] = None, label: Optional[str] = None) -> Tuple[Var, Var]:
    """
    Данный метод считает два вида ошибок: вызываемый погрешностями и вызываемый статистикой.
    Если точки хорошо ложатся на прямую, то преобладать будет ошибка из-за погрешностей.
    Если точки измерены крайне точно, но на прямую они ложатся так себе, то преобладает статистическая ошибка.
    Результирующей ошибкой выдаётся корень из суммы квадратов двух видов этих ошибок.
    :param x: Итерируемый объект с абсциссами точек.
    :param y: Итерируемый объект с ординатами точек.
    :param figure: Объект класса Figure, передаётся если мы хотим, чтобы эта прямая была построена.
    :param colour: цвет прямой 'b' голубой, 'g' зелёный, 'r' красный, 'c' бирюзовый,
    'm' розовый, 'y' жёлтый, 'k' чёрный, 'w' белый.
    :param line_style: Стиль линии 'solid', 'dotted', 'dashed', 'dashdot'.
    :param label: Название прямой для легенды.
    :return: Коэффициент наклона аппроксимированной прямой и её свободный коэффициент.
    """
    if len(x) != len(y):
        raise TypeError('Количество абсцисс не совпадает с количеством ординат!')
    if len(x) == 0:
        raise ValueError('What should I do with no dots? Genius blyat!')
    if len(x) == 1:
        raise ValueError('One dot!?!? Are you serious, Sam?')
    if len(x) == 2:
        raise ValueError('There is no need in mnk if you have only 2 dots')

    if not isinstance(x, GroupVar):
        x = GroupVar(x, 0)
    if not isinstance(y, GroupVar):
        y = GroupVar(y, 0)
    x_sum: Var = reduce(lambda res, x_var: res + x_var, x)
    y_sum: Var = reduce(lambda res, y_var: res + y_var, y)
    k_ex: Var = (len(x) * reduce(lambda res, i: res + x[i] * y[i], range(len(x)), 0) - x_sum * y_sum) / \
                (len(x) * reduce(lambda res, x_var: res + x_var * x_var, x, 0) - x_sum * x_sum)
    b_ex: Var = (y_sum - k_ex * x_sum) / len(x)
    k_stat_err, b_stat_err = _find_stat_errors(array(x.val()), array(y.val()), k_ex.val(), b_ex.val())
    k, b = [Var(val, sqrt(stat_err ** 2 + exact_err ** 2)) for val, exact_err, stat_err in
            [(*k_ex.val_err(), k_stat_err),
             (*b_ex.val_err(), b_stat_err)]]
    if figure is not None:
        figure.line(k.val(), b.val(), colour=colour, line_style=line_style, label=label)
    return k, b


def mnk_through0(x: GroupVar, y: GroupVar, figure: Optional[Figure] = None, colour: Optional[str] = None,
                 line_style: Optional[str] = None, label: Optional[str] = None) -> Var:
    """
    Тот же самый мнк, но проводит линию через начало координат.
    :param x: Итерируемый объект с абсциссами точек.
    :param y: Итерируемый объект с ординатами точек.
    :param figure: Объект класса Figure, передаётся если мы хотим, чтобы эта прямая была построена.
    :param colour: цвет прямой 'b' голубой, 'g' зелёный, 'r' красный, 'c' бирюзовый,
    'm' розовый, 'y' жёлтый, 'k' чёрный, 'w' белый.
    :param line_style: Стиль линии 'solid', 'dotted', 'dashed', 'dashdot'.
    :param label: Название прямой для легенды.
    :return: Коэффициент наклона полученной прямой.
    """
    if len(x) != len(y):
        raise TypeError('Количество абсцисс не совпадает с количеством ординат!')
    k: Var = reduce(lambda res, i: res + x[i] * y[i], range(len(x)), 0) / \
             reduce(lambda res, x_var: res + x_var * x_var, x[1:], x[0] * x[0])
    if figure is not None:
        figure.line(k.val(), 0, colour=colour, line_style=line_style, label=label)
    return k


def _find_stat_errors(x: array, y: array, k, b):
    """
    Считает и выводит статистическую ошибку ТОЛЬКО ДЛЯ ЛИНЕЙНОЙ ФУНКЦИИ.
    :param x: Массив с значениями по оси X.
    :param y: Массив с значениями по оси Y.
    :param k: Коэффициент наклона прямой, которую мы нааппроксимировали ранее.
    :param b: Свободный коэффициент прямой.
    :return: Ошибку для k и b соответственно.
    """
    if len(x) != len(y):
        raise TypeError('Количество абсцисс не совпадает с количеством ординат!')
    sy = sum((y - b - k * x) ** 2) / (len(x) - 2)
    d = len(x) * sum(x ** 2) - (sum(x)) ** 2
    return sqrt(sy * len(x) / d), sqrt(sy * sum(x ** 2) / d)
