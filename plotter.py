from matplotlib import pyplot
import numpy
from scipy.optimize import curve_fit
import parser


# constants which define the variable names for input file parameters
FILE_IN_FIT_EXPRESSION = 'FIT_EXPRESSION'
FILE_IN_FIT_PARAS = 'FIT_PARAS'
FILE_IN_FIT_PARAS_INIT = 'FIT_PARAS_INIT'
FILE_IN_FIT_X_PARA = 'FIT_X'
FILE_IN_FIT_OPT = 'FIT_OPT'
FILE_IN_FIT_OPT_PARA = 'FIT_OPT_PARA'
FILE_IN_YES = 'yes'
FILE_IN_NO = 'no'
FILE_IN_FIT_FIXED = 'FIT_FIXED'
FILE_IN_FIT_BOUNDS = 'FIT_BOUNDS'
FILE_IN_FIT_PRINT = 'FIT_PRINT'
FILE_IN_SEP = ';'
FILE_IN_FIG_HEIGHT = 'FIG_HEIGHT'
FILE_IN_FIG_WIDTH = 'FIG_WIDTH'
FILE_IN_X_LABEL = 'X_LABEL'
FILE_IN_Y_LABEL = 'Y_LABEL'
FILE_IN_MARKER = 'MARKER'
FILE_IN_MARKER_COLOR = 'MARKER_COLOR'
FILE_IN_MARKER_SIZE = 'MARKER_SIZE'
FILE_IN_LINE_COLOR = 'LINE_COLOR'
FILE_IN_LINE_STYLE = 'LINE_STYLE'
FILE_IN_LINE_WIDTH = 'LINE_WIDTH'
FILE_IN_X_LIM = 'X_LIM'
FILE_IN_Y_LIM = 'Y_LIM'
FILE_IN_Y_EXP = 'Y_EXPONENT'
FILE_IN_OUTPUT_DIR = 'OUTPUT_DIR'
FILE_IN_TEXT_X = 'TEXT_X'
FILE_IN_TEXT_Y = 'TEXT_Y'
FILE_IN_REMOVE_LAST = 'REMOVE_LAST_N_POINTS'


def fit(callable, x_data, y_data, init_paras, sigma=None, method='trf', maxfev=10000, bounds=None):
    """
    Fits x and y data through a given function.
    :param callable: callable, function
    :param x_data: array containing the x data
    :param y_data: array containing the y data
    :param init_paras: initiale guess on the parameter of the callable function
    :param sigma: optional, standard deviation error of the measurement
    :param method: Algorithm to perform minimization.
    ‘trf’ : Trust Region Reflective algorithm, particularly suitable for large sparse problems with bounds. Generally robust method.
    ‘dogbox’ : dogleg algorithm with rectangular trust regions, typical use case is small problems with bounds. Not recommended for problems with rank-deficient Jacobian.
    ‘lm’ : Levenberg-Marquardt algorithm as implemented in MINPACK. Doesn’t handle bounds and sparse Jacobians. Usually the most efficient method for small unconstrained problems.
    Default is ‘trf’.
    :return:
    """

    # calculates curve fit
    paras, cov = curve_fit(callable, x_data, y_data, init_paras, sigma=None, method=method, maxfev=maxfev, bounds=bounds)
    # calculates standard deviation from covariances
    paras_std = numpy.sqrt(numpy.diag(cov))

    return paras, paras_std


def data_points(callable, x_min, x_max, paras=[], n_points=1000):
    """
    Calculates 2D points of a function/callable from x_min to x_max.
    :param callable: function/callable
    :param x_min: start x value
    :param x_max: end x value
    :param paras: parameters of the function/callable
    :param n_points: number of points, default: n_points = 256
    :return:
    """

    x_data = numpy.logspace(numpy.log10(x_min), numpy.log10(x_max), n_points)
    y_data = numpy.array([callable(x, *paras) for x in x_data])

    return x_data, y_data


class Plotter:

    def __init__(self, plate, input_file_names):
        self._plate = plate
        self._input_file_names = input_file_names
        self._fit_dict = dict()

        self.fit()

    def fit(self):

        print('Fitting...')

        # read user input file
        paras_dict = parser.read_in(self._input_file_names)

        fit_expression = paras_dict[FILE_IN_FIT_EXPRESSION][0]
        fit_paras = paras_dict[FILE_IN_FIT_PARAS]
        fit_x_para = paras_dict[FILE_IN_FIT_X_PARA][0]
        fit_paras_init = paras_dict[FILE_IN_FIT_PARAS_INIT]
        fit_fixed = paras_dict[FILE_IN_FIT_FIXED]
        fit_bounds = paras_dict[FILE_IN_FIT_BOUNDS]
        fit_print = paras_dict[FILE_IN_FIT_PRINT]
        fit_opt = paras_dict[FILE_IN_FIT_OPT][0] == FILE_IN_YES
        fit_opt_para = paras_dict[FILE_IN_FIT_OPT_PARA][0]
        remove_last_n_points = int(paras_dict[FILE_IN_REMOVE_LAST][0])

        # extract variable parameters

        fit_var_paras = [v for v, fixed in zip(fit_paras, fit_fixed) if fixed == FILE_IN_NO]
        fit_var_print = [p for p, fixed in zip(fit_print, fit_fixed) if fixed == FILE_IN_NO]

        try:
            fit_var_values = [float(v) for v, fixed in zip(fit_paras_init, fit_fixed) if fixed == FILE_IN_NO]
        except ValueError:
            print('Error: Initial fit parameters must have a numerical value!')
            exit(1)

        try:
            tmp_lower = [float(i.split(FILE_IN_SEP)[0]) for i, fixed in zip(fit_bounds, fit_fixed) if
                         fixed == FILE_IN_NO]
            tmp_upper = [float(i.split(FILE_IN_SEP)[1]) for i, fixed in zip(fit_bounds, fit_fixed) if
                         fixed == FILE_IN_NO]
            fit_var_bounds = (tmp_lower, tmp_upper)
        except ValueError:
            print('Error: Bound values must have a numerical value!')
            exit(1)

        # extract constant parameters
        fit_const_paras = [c for c, fixed in zip(fit_paras, fit_fixed) if fixed == FILE_IN_YES]
        try:
            fit_const_values = [float(c) for c, fixed in zip(fit_paras_init, fit_fixed) if fixed == FILE_IN_YES]
        except ValueError:
            print('Error: Initial fit parameters must have a numerical value!')
            exit(1)

        # get data sets from plate
        labels = self._plate.labels()

        # set up function that represents the fit function
        def fit_func(x, *args):

            p = dict()

            for k, var in zip(fit_var_paras, args):
                try:
                    p[k] = float(var)
                except ValueError:
                    print('Error: fit parameters must be numerical values!')
                    exit(1)

            for k, c in zip(fit_const_paras, fit_const_values):
                try:
                    p[k] = float(c)
                except ValueError:
                    print('Error: initial fit parameters must be numerical values!')
                    exit(1)
            p[fit_x_para] = x
            return eval(fit_expression, p)

        # iterate through every data set and perform fit
        fit_result, fit_std_result = None, None
        for label in labels:

            if fit_opt:
                # try to find optimal fit parameters by iteratively removing the last x values

                tmp_fit_dict = dict()

                for i in range(remove_last_n_points):
                    x_data = self._plate.fit_x(label)
                    y_data = self._plate.fit_y(label)
                    y_std_data = self._plate.fit_y_std(label)

                    x_fit_data = x_data[:len(x_data) - i]
                    y_fit_data = y_data[:len(y_data) - i]
                    y_std_fit_data = y_std_data[:len(y_std_data) - i]

                    fit_result, fit_std_result = fit(fit_func, x_fit_data, y_fit_data, fit_var_values,
                                                      sigma=y_std_fit_data,
                                                      bounds=fit_var_bounds)

                    para_index = fit_var_paras.index(fit_opt_para)
                    para_std = fit_std_result[para_index]

                    tmp_fit_dict[para_std] = (fit_result, fit_std_result, x_fit_data, y_fit_data, y_std_fit_data)

                min_std = min(list(tmp_fit_dict.keys()))

                best_fit_result = tmp_fit_dict[min_std][0]
                best_fit_std_result = tmp_fit_dict[min_std][1]
                best_x_data = tmp_fit_dict[min_std][2]
                best_y_data = tmp_fit_dict[min_std][3]
                best_y_std_data = tmp_fit_dict[min_std][4]

                self._fit_dict[label] = [fit_func, fit_var_paras, best_fit_result, best_fit_std_result,
                                                          best_x_data, best_y_data, best_y_std_data, fit_var_print]

            else:
                x_data = self._plate.fit_x(label)
                y_data = self._plate.fit_y(label)
                y_std_data = self._plate.fit_y_std(label)
                fit_result, fit_std_result = fit(fit_func, x_data, y_data, fit_var_values, sigma=y_std_data,
                                                  bounds=fit_var_bounds)

                self._fit_dict[label] = [fit_func, fit_var_paras, fit_result, fit_std_result, x_data, y_data,
                                         y_std_data, fit_var_print]

            print('{}:'.format(label))
            print('\tPara:\t{}'.format(fit_var_paras))
            print('\tFit:\t{}'.format(fit_result))
            print('\tFit SD:\t{}'.format(fit_std_result))

        print('Fitting finished!')
        print('-----------------\n')

    def plot(self):

        print('Plotting...')

        file_name = self._plate.file_name
        plot_paras = parser.read_in(self._input_file_names)
        xlabel = plot_paras[FILE_IN_X_LABEL][0]
        ylabel = plot_paras[FILE_IN_Y_LABEL][0]
        marker = plot_paras[FILE_IN_MARKER][0]
        marker_size = plot_paras[FILE_IN_MARKER_SIZE][0]
        marker_color = plot_paras[FILE_IN_MARKER_COLOR][0]
        line_color = plot_paras[FILE_IN_LINE_COLOR][0]
        line_style = plot_paras[FILE_IN_LINE_STYLE][0]

        try:
            fig_height = int(plot_paras[FILE_IN_FIG_HEIGHT][0])
            fig_width = int(plot_paras[FILE_IN_FIG_WIDTH][0])
            line_width = float(plot_paras[FILE_IN_LINE_WIDTH][0])
            ylim_low = float(plot_paras[FILE_IN_Y_LIM][0].split(FILE_IN_SEP)[0])
            ylim_up = float(plot_paras[FILE_IN_Y_LIM][0].split(FILE_IN_SEP)[1])
            xlim_low = float(plot_paras[FILE_IN_X_LIM][0].split(FILE_IN_SEP)[0])
            xlim_up = float(plot_paras[FILE_IN_X_LIM][0].split(FILE_IN_SEP)[1])
            yexp = int(numpy.floor(float(plot_paras[FILE_IN_Y_EXP][0])))
            text_x = float(plot_paras[FILE_IN_TEXT_X][0])
            text_y = float(plot_paras[FILE_IN_TEXT_Y][0])
        except ValueError:
            print('Error: A parameter from the input file could not be parsed to a numerical value! I am too lazy to'
                  'find out which parameter it is :(')
            exit(1)

        output_dir = plot_paras[FILE_IN_OUTPUT_DIR][0]

        # create pyplot figure
        fig, axs = pyplot.subplots(len(self._plate.labels()), 1, figsize=(fig_width, fig_height))

        for label, ax in zip(self._plate.labels(), axs):
            x = self._plate.fit_x(label)
            x_all = self._plate.x(label)
            y = self._plate.fit_y(label)
            y_all = self._plate.y(label)
            y_std = self._plate.fit_y_std(label)
            y_std_all = self._plate.y_std(label)

            fit_x = self._fit_dict[label][4]
            fit_y = self._fit_dict[label][5]
            fit_y_std = self._fit_dict[label][6]

            f_x, f_y = data_points(self._fit_dict[label][0], 0.001, 10000, self._fit_dict[label][2], n_points=100)

            # draw fit parameters onto graph
            fit_label = ''
            for para, value, err, out in zip(self._fit_dict[label][1], self._fit_dict[label][2], self._fit_dict[label][3], self._fit_dict[label][7]):
                v = round(value, 2)
                e = round(err, 2)
                if out == FILE_IN_YES:
                    fit_label += '{}'.format(para) + r'$=$' + '{}'.format(v) + r'$\pm$' + '{}'.format(e) + '\n'

            ax.text(text_x, text_y, fit_label, fontsize=12)
            # ax.set_title(file_name + '\n' + label + '\n' + fit_label, fontsize=8)

            ax.set_title(file_name + '\n' + label, fontsize=8)

            ax.set_xlabel(xlabel)
            if yexp == 0.0:
                ax.set_ylabel(ylabel)
            else:
                i = '{}'.format(yexp)
                ax.set_ylabel(ylabel + r'$\ (10^{})$'.format(i))
            ax.set_ylim(ylim_low, ylim_up)
            ax.set_xlim(xlim_low, xlim_up)
            ax.set_xscale('log')

            y_factor = 10.0 ** yexp
            ax.errorbar(x_all, numpy.array(y_all) / y_factor, numpy.array(y_std_all) / y_factor, linestyle='', capsize=3, color=marker_color, markersize=marker_size, marker=marker, alpha=0.25)
            ax.errorbar(fit_x, numpy.array(fit_y) / y_factor, numpy.array(fit_y_std) / y_factor, linestyle='', capsize=3,
                        color=marker_color, markersize=marker_size, marker=marker)

            ax.plot(f_x, f_y / y_factor, linestyle=line_style, linewidth=line_width, color=line_color)

            # v = round(self._fit_dict[label][2][2], 2)
            # e = round(self._fit_dict[label][3][2], 2)

            print('Graph created...')

        pyplot.tight_layout()
        print('Saving image...')
        pyplot.savefig('{}/{}.png'.format(output_dir, file_name.split('/')[-1]), dpi=400)
        print('Image saved!')
        pyplot.show()

