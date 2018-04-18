import argparse
import numpy as np
import matplotlib
# matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import keras
import os


def parse_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="Run dCNN")

    parser.add_argument('--with-mas', required=True, type=str)
    parser.add_argument('--mode', required=True, type=str)
    parser.add_argument('--direction', required=True, type=str)
    parser.add_argument('--start', required=True, type=int)
    parser.add_argument('--instances', required=True, type=int)

    parser.add_argument('--target-const', required=False, type=float, default=2.0)
    parser.add_argument('--l0-values', required=False, type=int, default=3)
    parser.add_argument('--l0-pixels', required=False, type=int, default=10000)
    parser.add_argument('--li-multi-eps', required=False, type=float, default=0.01)
    parser.add_argument('--li-multi-steps', required=False, type=int, default=1)
    parser.add_argument('--multi-sep', required=False, type=int, default=10)

    parser.add_argument('--work-dir', required=False, help='Where to put all the data', type=str, default='./')
    parser.add_argument('--exp-name', required=False, help='Name of the experiment being run saved under work_dir ',
                        type=str, default='results')
    parser.add_argument('--layer', required=False, help='Number of layer of cDNN model', type=int, default=10)
    parser.add_argument('--device', required=False, type=str, default=None)
    parser.add_argument('--save-data', required=False, type=bool, default=False)

    args = parser.parse_args()
    args.with_mas = str2bool(args.with_mas)

    return args


def show_slices(x_list, show_bar, save_name=None):
    slices = []
    image_shape = (172, 220, 156)
    for x in x_list:
        x = np.reshape(x, image_shape)
        slices.append([x[86, :, :], x[:, 110, :], x[:, :, 78]])

    scale = 10
    interval = 50
    edge = 300
    axis1 = 156 * scale
    axis2 = [0, 220 * scale, 392 * scale, int(513.963636 * scale)]

    plt.figure(figsize=((axis2[-1] + (len(slices[0]) - 1) * interval + edge) / 600,
                        (len(slices) * axis1 + (len(slices) - 1) * interval) / 600))
    gs1 = gridspec.GridSpec(len(slices) * axis1 + (len(slices) - 1) * interval,
                            axis2[-1] + (len(slices[0]) - 1) * interval + edge)

    for j in range(len(slices)):
        for i, one_slice in enumerate(slices[j]):
            if show_bar[j] and i == len(slices[j]) - 1:
                ax = plt.subplot(gs1[j * axis1 + j * interval:(j + 1) * axis1 + j * interval,
                                 axis2[i] + i * interval:axis2[i + 1] + i * interval + edge])
            else:
                ax = plt.subplot(gs1[j * axis1 + j * interval:(j + 1) * axis1 + j * interval,
                                 axis2[i] + i * interval:axis2[i + 1] + i * interval])
            ax.set_axis_off()
            im = ax.imshow(one_slice.T, cmap="gray", origin="lower")
            if show_bar[j] and i == len(slices[j]) - 1:
                cb = plt.colorbar(im, use_gridspec=True)
                tick_locator = ticker.MaxNLocator(nbins=5)
                cb.locator = tick_locator
                cb.update_ticks()

    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name, bbox_inches='tight', pad_inches=0)


def get_file_name(tag1, tag2, args):
    mode = args.mode
    direction = args.direction
    instances = args.instances
    start = args.start

    target_const = args.target_const
    values = args.l0_values
    pixels = args.l0_pixels
    eps = args.li_multi_eps
    steps = args.li_multi_steps
    sep = args.multi_sep

    work_dir = args.work_dir
    exp = args.exp_name

    output_dir = os.path.join(work_dir, exp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_name = ''
    if mode == 'li_non_target' or mode == 'gsm_non_target':
        output_name = '%s_%s_gsm_non_target_eps%05d_steps%d' % (tag1, direction, eps * 10000, steps)
    elif mode == 'l0_non_target' or mode == 'pixels_non_target':
        output_name = '%s_%s_pixels_non_target_pixels%d_values%d' % (tag1, direction, pixels, values)
    elif mode == 'li_target' or mode == 'gsm_target':
        output_name = '%s_%s_gsm_target%2f_steps%d' % (tag1, direction, target_const, steps)
    elif mode == 'l0_target' or mode == 'pixels_target':
        output_name = '%s_%s_pixels_target%2f_values%d' % (tag1, direction, target_const, values)
    elif mode == 'li_exp' or mode == 'gsm_exp':
        output_name = '%s_%s_gsm_steps%d' % (tag1, direction, steps)
    elif mode == 'l0_exp' or mode == 'pixels_exp':
        output_name = '%s_%s_l0' % (tag1, direction)
    elif mode == 'l2_exp':
        output_name = '%s_%s_l2' % (tag1, direction)
    elif mode == 'multi':
        output_name = '%s_%s_multi_eps%05d_steps%d_sep%d' % (tag1, direction, eps * 10000, steps, sep)

    if tag2 == 'csv':
        output_name += '_start%d_instances%d.csv' % (start, instances)
    elif tag2.endswith('png'):
        output_name += tag2

    if not (direction == 'max' or direction == 'min'):
        raise Exception('Invalid direction')

    output_name = os.path.join(output_dir, output_name)

    return output_name


def combine_model():
    work_dir = ''
    print("Loading Trained Convolutional Model")
    model1 = keras.models.load_model(os.path.join(work_dir, 'models', 'model1.h5'))
    print("Loading Trained MAS Model")
    model2 = keras.models.load_model(os.path.join(work_dir, 'models', 'model2p.h5'))
    first_stage_input = model1.layers[0].input
    first_stage_output = model1.layers[10].output
    model2_input = keras.layers.Input((134,))
    second_stage_input = keras.layers.concatenate([first_stage_output, model2_input], axis=-1)
    second_stage_output = model2(second_stage_input)
    model3 = keras.models.Model(inputs=[first_stage_input, model2_input], outputs=second_stage_output)

    model3.save(os.path.join(work_dir, 'models', 'model2.h5'))


def show_slices_from_saved_data(file_name):
    f = open(file_name, 'rb')
    data = pickle.load(f)
    f.close()
    print(data[2:])
    x = (data[0].raw_image - data[0].min_value) / (data[0].max_value - data[0].min_value)
    x_adv = (data[1].raw_image - data[1].min_value) / (data[1].max_value - data[1].min_value)
    show_slices(x, x_adv)


def show_comparison_between_samples(file_name_adv, file_name_random, file_name_cmp):
    f_adv = open(file_name_adv, 'rb')
    f_random = open(file_name_random, 'rb')
    f_cmp = open(file_name_cmp, 'rb')
    data_adv = pickle.load(f_adv)
    data_random = pickle.load(f_random)
    data_cmp = pickle.load(f_cmp)

    x = (data_adv[0].raw_image - data_adv[0].min_value) / (data_adv[0].max_value - data_adv[0].min_value)
    x_adv = (data_adv[1].raw_image - data_adv[1].min_value) / (data_adv[1].max_value - data_adv[1].min_value)
    x_rand = (data_random[1].raw_image - data_random[1].min_value) / (
        data_random[1].max_value - data_random[1].min_value)
    x_cmp = (data_cmp[0].raw_image - data_cmp[0].min_value) / (data_cmp[0].max_value - data_cmp[0].min_value)

    print(data_adv[2:], data_random[2:], data_cmp[2:])
    show_slices([x], [False], save_name="./results/figures/org_sample_age_%d.png" % data_adv[2])
    show_slices([x_adv], [False], save_name="./results/figures/adv_sample_age_%d.png" % data_adv[3])
    show_slices([x_rand], [False], save_name="./results/figures/rand_sample_age_%d.png" % data_random[3])
    show_slices([x_cmp], [False], save_name="./results/figures/cmp_sample_age_%d.png" % data_cmp[2])


def show_three_attacks_samples(file_name_gsm, file_name_l2, file_name_l0):
    f_gsm = open(file_name_gsm, 'rb')
    f_l2 = open(file_name_l2, 'rb')
    f_l0 = open(file_name_l0, 'rb')

    data_gsm = pickle.load(f_gsm)
    data_l2 = pickle.load(f_l2)
    data_l0 = pickle.load(f_l0)

    x = (data_l0[0].raw_image - data_l0[0].min_value) / (data_l0[0].max_value - data_l0[0].min_value)
    x_gsm = (data_gsm[1].raw_image - data_gsm[1].min_value) / (data_gsm[1].max_value - data_gsm[1].min_value)
    x_l2 = (data_l2[1].raw_image - data_l2[1].min_value) / (data_l2[1].max_value - data_l2[1].min_value)
    x_l0 = (data_l0[1].raw_image - data_l0[1].min_value) / (data_l0[1].max_value - data_l0[1].min_value)

    show_slices([x_gsm], [False], save_name="./results/figures/gsm_sample_age_%d.png" % data_gsm[3])
    show_slices([x_l2], [False], save_name="./results/figures/l2_sample_age_%d.png" % data_l2[3])
    show_slices([x_l0], [False], save_name="./results/figures/l0_sample_age_%d.png" % data_l0[3])
    show_slices([x_gsm - x], [True], save_name="./results/figures/gsm_sample_noise.png")
    show_slices([x_l2 - x], [True], save_name="./results/figures/l2_sample_noise.png")
    show_slices([x_l0 - x], [True], save_name="./results/figures/l0_sample_noise.png")


def show_three_attacks_figures(figure_type):
    use_ratio = False
    directions = ['max', 'min']
    modes = ['gsm', 'l2', 'l0']
    mode_labels = ['gsm_steps4', 'l2', 'l0']
    appendix = 'start0_instances100.csv'

    models = ['model1']
    for model in models:
        for direction in directions:
            for i, mode in enumerate(modes):
                file_name = '%s_%s_%s_%s' % (model, direction, mode_labels[i], appendix)
                file_name = os.path.join(os.getcwd(), 'results', model, file_name)
                print(file_name)
                draw_figure_from_csv(file_name, mode, direction,use_ratio=use_ratio, figure_type=figure_type)

    for direction in directions:
        for i, mode in enumerate(modes):
            file_name = '%s_%s_%s_%s' % ('model1', direction, mode_labels[i], appendix)
            file_name = os.path.join(os.getcwd(), 'results', 'model1', file_name)
            file_name_cmp = '%s_%s_%s_%s' % ('model2', direction, mode_labels[i], appendix)
            file_name_cmp = os.path.join(os.getcwd(), 'results', 'model2', file_name_cmp)
            print(file_name, file_name_cmp)
            draw_figure_from_csv(file_name, mode, direction, use_ratio=use_ratio, figure_type=figure_type, file_name_cmp=file_name_cmp)


def draw_figure_from_csv(file_name, mode, direction, use_ratio=True, figure_type='plot', file_name_cmp=None):
    def get_index(value, ranges):
        index = 0
        for i in ranges:
            if value > i:
                index += 1
        return index

    def fmt(x, pos=0):
        # a, b = '{:.1e}'.format(x).split('e')
        # b = int(b)
        x = x*pow(10, pos)
        a = '{:.1f}'.format(x).split('e')[0]
        b = -pos
        return r'${} \times 10^{{{}}}$'.format(a, b)

    AGE_RANGES = [15, 25, 50, 65]
    adj = 2 if mode == 'l2' else 1

    data = np.genfromtxt(file_name, delimiter=',', dtype=float)[:, :-1]
    deviations = [[] for _ in range(len(AGE_RANGES) + 1)]
    total_deviations = []
    average_deviations = [[] for _ in range(len(AGE_RANGES) + 2)]
    for i in range(data.shape[0] // adj):
        index = get_index(data[adj * i, 0], AGE_RANGES)
        deviation = data[adj * i] / data[adj * i, 0] if use_ratio else data[adj * i] - data[adj * i, 0]
        deviations[index].append(deviation)
        total_deviations.append(deviation)
    for i in range(len(AGE_RANGES) + 2):
        tmp = np.array(total_deviations) if i == len(AGE_RANGES) + 1 else np.array(deviations[i])
        if not use_ratio:
            average_deviations[i] = np.abs(np.mean(tmp, axis=0))
            # average_deviations[i] = np.abs(average_deviations[i] - average_deviations[i][0])
        elif direction == 'max':
            average_deviations[i] = np.mean(tmp, axis=0) - 1
        else:
            average_deviations[i] = 1 - np.mean(tmp, axis=0)

    if file_name_cmp is not None:
        data_cmp = np.genfromtxt(file_name_cmp, delimiter=',', dtype=float)[:, :-1]
        total_deviations_cmp = []
        for i in range(data_cmp.shape[0] // adj):
            deviation = data_cmp[adj * i] / data_cmp[adj * i, 0] if use_ratio else data_cmp[adj * i] - data_cmp[adj * i, 0]
            total_deviations_cmp.append(deviation)
        if not use_ratio:
            average_deviations_cmp = np.abs(np.mean(total_deviations_cmp, axis=0))
            # average_deviations_cmp = np.abs(average_deviations_cmp - average_deviations_cmp[0])
        elif direction == 'max':
            average_deviations_cmp = np.mean(total_deviations_cmp, axis=0) - 1
        else:
            average_deviations_cmp = 1 - np.mean(total_deviations_cmp, axis=0)

    plt.figure()
    if mode == 'l2':
        distortions = np.sqrt(np.array([0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]) / (172 * 220 * 156))
        x_range = (0, 0.0006)
        x_values = [1, 2, 3]
        x_ticks = [fmt(distortions[i], pos=4) for i in x_values]
        x_label = 'difference, $\| \cdot \|_2$'
    elif mode == 'l0':
        distortions = np.array([30 * i / (172 * 220 * 156) for i in range(40 + 1)])
        x_range = (0, 0.0002)
        x_values = [12, 24, 36]
        x_ticks = [fmt(distortions[i], pos=4) for i in x_values]
        x_label = 'difference, $\| \cdot \|_0$'
    elif mode == 'gsm':
        distortions = np.array([0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
        x_range = (0, 0.002)
        x_values = [3, 4, 5]
        x_ticks = [fmt(distortions[i], pos=3) for i in x_values]
        x_label = 'difference, $\| \cdot \|_\infty$'
    else:
        raise Exception('Invalid mode')

    elements = [None for _ in range(len(AGE_RANGES) + 2)]
    ax = plt.subplot(111)
    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))

    plt.xlabel(x_label)
    plt.ylabel('deviation')
    if not use_ratio:
        plt.ylim((0, 100))
    elif direction == 'min':
        plt.ylim((0, 1.2))
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0%', '20%', '40%', '60%', '80%', '100%'])
    else:
        plt.ylim((0, 7))
        plt.yticks([0, 1, 2, 3, 4, 5, 6], ['0%', '100%', '200%', '300%', '400%', '500%', '600%'])

    if figure_type == 'plot':
        plt.xlim(x_range)
        for i in range(len(AGE_RANGES) + 2):
            elements[i], = plt.plot(distortions, average_deviations[i])
        plt.legend(handles=[elements[i] for i in range(len(AGE_RANGES) + 2)],
                   labels=['<15', '15-25', '25-50', '50-65', '>65',
                           'average'], loc='upper left')
    elif figure_type == 'bar' and file_name_cmp is None:
        ind = np.arange(len(x_values))
        width = 0.1
        for i in range(len(AGE_RANGES) + 1):
            y_values = average_deviations[i][x_values]
            elements[i] = ax.bar(ind+width*i, y_values, width)
        y_values = average_deviations[len(AGE_RANGES) + 1][x_values]
        for i in range(len(y_values)):
            elements[len(AGE_RANGES) + 1] = ax.hlines(y_values[i], ind[i]-width/2, ind[i]+9*width/2, colors="c", linestyles="dashed")
        plt.xticks(ind+len(AGE_RANGES)*width/2)
        ax.set_xticklabels(x_ticks)
        plt.legend(handles=[elements[i] for i in range(len(AGE_RANGES) + 2)],
                   labels=['<15', '15-25', '25-50', '50-65', '>65',
                           'average'], loc='upper left', fontsize='medium')
        print(y_values)
    elif figure_type == 'bar' and file_name_cmp is not None:
        ind = np.arange(len(x_values))
        width = 0.25
        y_values = average_deviations[len(AGE_RANGES) + 1][x_values]
        y_values_cmp = average_deviations_cmp[x_values]
        elements[0] = ax.bar(ind, y_values, width)
        elements[1] = ax.bar(ind+width, y_values_cmp, width)
        plt.xticks(ind + width)
        ax.set_xticklabels(x_ticks)
        plt.legend(handles=[elements[0], elements[1]], labels=['conventional DNN', 'context-aware model'], loc='upper left', fontsize='medium')
        print(y_values)
    else:
        raise Exception('Invalid type of figure')

    fig_name = file_name if file_name_cmp is None else file_name_cmp
    fig_name = os.path.join(os.getcwd(), 'results', 'figures', os.path.split(fig_name)[1].split('.')[0] + '_bar.pdf')
    plt.savefig(fig_name, bbox_inches='tight')
    # plt.gcf().canvas.set_window_title(fig_name)
    # plt.show()


if __name__ == '__main__':
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    matplotlib.rcParams.update({'font.size': 14})
    # show_slices_from_saved_data('./results/samples/model1_max_gsm_exp_sample0.txt')
    # show_three_attacks_samples('./results/samples/model1_max_gsm_steps4_sample0.txt',
    #                            './results/samples/model1_max_l2_sample0.txt',
    #                            './results/samples/model1_max_l0_sample0.txt')
    # show_comparison_between_samples('./results/samples/model1_max_gsm_steps4_sample0.txt',
    #                                 './results/samples/model1_max_rand_steps4_sample0.txt',
    #                                 './results/samples/model1_max_gsm_steps4_sample639.txt')
    # show_three_attacks_figures('bar')
