from .var import Var, GroupVar
import pandas as pd
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror


def read_data():
    """
    Отвечает за открытие окошка с выбором файла и импортом его в программу
    :return: DataFrame с данными из файла, если файл не выбран, возвращает None
    """
    file_path = askopenfilename(title="Выберите файл")
    file_type = file_path.split(".")[-1]
    if file_path != '':
        if file_type in ["xls", "xlsx", "xlsm", "xlsb", "odf", "ods", "odt"]:
            out_data = pd.read_excel(file_path)
        elif file_type == "csv":
            out_data = pd.read_csv(file_path)
        else:
            showerror("Ошибка формата данных", "Неподдерживаемый формат данных")
            out_data = read_data()
        return out_data
    return None


def shredder(dataf, constants_needed=False):
    """
    Забирает константы из таблицы DataFrame и режет её на маленькие таблички по столбцам pd.na
    :param dataf: Правильно отфарматированная таблица
    :param constants_needed: Есть ли в ней константы, которые нужно употребить
    :return: словарь с константами в формате Var и список с каждой табличкой отдельно
    """
    const = {}
    size_dataf = dataf.shape
    columns = dataf.columns
    if constants_needed:
        if not ('Unnamed: 0' in columns and 'Unnamed: 1' in columns):
            raise TypeError("Ошибка форматирования данных: Первая строка констант должна быть пустой")
        try:
            if pd.isna(dataf.iloc[0, 2]):
                raise TypeError("Ошибка форматирования данных: Константы должны быть с погрешностью")
        except:
            raise TypeError("Ошибка форматирования данных: Константа должна быть хотя бы одна")

        i = 0
        while i < size_dataf[0] and not (pd.isna(dataf.iloc[i, 0])):
            const[dataf.iloc[i, 0]] = Var(dataf.iloc[i, 1], dataf.iloc[i, 2])
            i += 1
        dataf = dataf.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3'], axis=1)

    columns = dataf.columns
    datat = []
    i_beg = 0
    for i in range(len(columns)):
        if 'Unnamed:' in columns[i]:
            local_df = pd.DataFrame()
            for j in range(i_beg, i):
                local_df[columns[j]] = dataf[columns[j]]
            i_beg = i + 1
            local_df = local_df.dropna()
            datat.append(local_df)

        if i == len(columns) - 1:
            local_df = pd.DataFrame()
            for j in range(i_beg, i + 1):
                local_df[columns[j]] = dataf[columns[j]]
            local_df = local_df.dropna()
            datat.append(local_df)
    return const, datat


def get_into_groupvar_col_to_col(data_frame):
    """
    Превращает табличку с погрешностями в формате столбец величины - столбец погрешности в словарь с GroupVar
    :param data_frame: Табличка в правильном виде (Получается из shredder)
    :return: Словарь, где ключи - названия столбцов с данными, а значения - GroupVar с данными
    """
    table = {}
    if len(data_frame.columns) == 0:
        raise TypeError("Столбцы не обнаружены, нужен хотя бы один")
    if len(data_frame.columns) % 2 == 1:
        raise TypeError("Похоже, один из столбцов без напарника с погрешностью, т. к. количество столбцов нечётно")
    for index in range(0, len(data_frame.columns), 2):
        table[data_frame.columns[index]] = GroupVar(data_frame[data_frame.columns[index]],
                                                    data_frame[data_frame.columns[index + 1]])
    return table


def get_into_groupvar_col_named(data_frame):
    """
    Превращает табличку с погрешностями в формате столбец величины и если название следующего начинается с delta,
    то с погрешностью в словарик из GroupVar
    :param data_frame: Табличка в правильном виде (Получается из shredder)
    :return: Словарь, где ключи - названия столбцов с данными, а значения - GroupVar с данными
    """
    if len(data_frame.columns) == 0:
        raise TypeError("Столбцы не обнаружены, нужен хотя бы один")
    table = {}
    for index in range(len(data_frame.columns) - 1):
        if not ("delta" in data_frame.columns[index]):
            if "delta" in data_frame.columns[index + 1]:
                table[data_frame.columns[index]] = GroupVar(data_frame[data_frame.columns[index]],
                                                            data_frame[data_frame.columns[index + 1]])
            else:
                table[data_frame.columns[index]] = GroupVar(data_frame[data_frame.columns[index]], 0)
    return table


def quick_use_form(dictionary):
    """
    Преобразует словарь с данными для более быстрого использования в коде
    :param dictionary: словарь с ключами вида "<название величины>, <её размерность>"
    :return: Словарь, дополненный упрощёнными ключами
    """
    keys_arr = dictionary.keys()
    for key in list(keys_arr):
        if key.count(',') != 0:
            key_up = key.split(',')[0]
            dictionary[key_up] = dictionary[key]
    return dictionary

df = read_data()
print(df)
c, tables = shredder(df, True)
print(c)
for i in tables:
    print(i)
    print('<<>>')
dv = get_into_groupvar_col_named(tables[1])
print(dv)
dv = quick_use_form(dv)
print(dv['u, м/с2'])
print(dv['u'])
