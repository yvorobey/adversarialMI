import numpy as np
import time

import utils
from l2_attack import CarliniL2
from brainage import BrainAgeModel1, BrainAgeModel2


class Experiment(object):
    def __init__(self, data, model, args):
        self.data = data
        self.model = model
        self.args = args
        if isinstance(model, BrainAgeModel1):
            self.file_name = utils.get_file_name('model1', 'csv', args)
        elif isinstance(model, BrainAgeModel2):
            self.file_name = utils.get_file_name('model2', 'csv', args)
        else:
            raise Exception('Invalid model')

    def write_results(self, results):
        f = open(self.file_name, 'a')
        for res in results:
            f.write('%f' % res)
            f.write(",")
        f.write("\n")

    def write_results2(self, file_name, results):
        f = open(file_name, 'a')
        for res in results:
            f.write('%.4f' % res)
            f.write(",")
        f.write("\n")

    def save_variables(self, variable):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        file_name_variable = '_'.join(self.file_name.split('_')[0:-2]) + '_sample%d.txt' % self.args.start
        f = open(file_name_variable, 'wb')
        pickle.dump(variable, f)
        f.close()


class GsmExp(Experiment):
    def __init__(self, data, model, args):
        super(GsmExp, self).__init__(data, model, args)
        self.EPS_LIST = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

    def attack(self):
        steps = self.args.li_multi_steps

        time_start = time.time()
        for i in range(self.args.start, self.args.start + self.args.instances):
            image = self.data.get_image([i])[0]
            org_prediction = self.model.predict_image(image)[0][0]

            predictions = [org_prediction]
            for eps in self.EPS_LIST:
                if self.args.direction == "max":
                    x_adv = self.gsm_attack(image, eps, steps)
                else:
                    x_adv = self.gsm_attack(image, -eps, steps)
                new_prediction = self.model.predict_image(x_adv)[0][0]
                predictions.append(new_prediction)

                if self.args.save_data and i >= self.args.start + self.args.instances - 1 and eps == 0.002:
                    # utils.show_slices(image.raw_image, x_adv.raw_image, self.file_name[:-4]+'.png')
                    self.save_variables([image, x_adv, predictions[0], new_prediction])

            print('Sample %d' % i, "Time used %.2f" % (time.time() - time_start), predictions)
            self.write_results(predictions)

    def gsm_attack(self, x, step_length, steps, normalize=np.sign):
        """from instances return adversarial examples"""
        x_copy = x.copy()
        for step in range(steps):
            # used to get random noise
            # grad = x.copy()
            # grad.raw_image = step_length/steps * (x_copy.max_value - x_copy.min_value) * \
            #                  np.sign(np.random.rand(*grad.raw_image.shape)-0.5)
            grad = self.model.get_grad(x_copy, step_length * (x_copy.max_value - x_copy.min_value) / steps, normalize)
            x_copy += grad
            x_copy.range_check()
        return x_copy


class MultiGsm(Experiment):
    def __init__(self, data, model, args):
        super(MultiGsm, self).__init__(data, model, args)
        # self.EPS_LIST = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
        self.EPS_LIST = [0.005]
        self.group_sz = [100, 300, 500, 1000, 3000]

    def attack(self):
        steps = self.args.li_multi_steps
        file_name_template = '{model}_{sl}_{group}.csv'
        time_start = time.time()
        for eps in self.EPS_LIST:
            for size in self.group_sz:
                groups = int(self.args.instances / size)
                for group in range(groups):
                    g_time = time.time()
                    instances_start = self.args.start + group * size
                    instances_end = self.args.start + (1 + group) * size
                    original_predictions = []
                    for idx in range(instances_start, instances_end):
                        instance = self.data.get_image([idx])[0]
                        prediction = self.model.predict_image(instance)[0][0]
                        original_predictions.append(prediction)
                    attacked_predictions = self.gsm_attack((instances_start, instances_end), eps, 8)
                    print(len(original_predictions), len(attacked_predictions))
                    assert len(original_predictions) == len(attacked_predictions)
                    for idx in range(len(attacked_predictions)):
                        predictions = [original_predictions[idx], attacked_predictions[idx]]
                        self.write_results2(file_name_template.format(model=self.model.name, sl=eps, group=size),
                                            predictions)
                    print('group' + str(group) + 'time' + str(time.time() - g_time))

    def gsm_attack(self, instances_range, step_length, steps, normalize=np.sign):
        """from instances return adversarial examples"""
        start = instances_range[0]
        end = instances_range[1]
        offset = None
        for step in range(steps):
            common_grad = None
            for idx in range(start, end):
                if offset == None:
                    instance = self.data.get_image([idx])[0]
                else:
                    instance = self.data.get_image([idx])[0]
                    offset_cp = offset.copy()
                    offset_cp.normalize((instance.max_value - instance.min_value) * step_length / steps * step)
                    instance += offset_cp

                grad = self.model.get_grad(instance, 1, normalize)

                if common_grad == None:
                    common_grad = grad
                else:
                    common_grad += grad
            common_grad.normalize(step_length / steps)
            if offset == None:
                offset = common_grad
            else:
                offset += common_grad
        res = []
        for idx in range(start, end):
            instance = self.data.get_image([idx])[0]
            offset_cp = offset.copy()
            offset_cp.normalize((instance.max_value - instance.min_value) * step_length)
            instance += offset_cp
            prediction = self.model.predict_image(instance)[0][0]
            res.append(prediction)
        return res


class L2Exp(Experiment):
    def __init__(self, data, model, args):
        super(L2Exp, self).__init__(data, model, args)
        self.CONST_LIST = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        # self.CONST_LIST = [1.90]
        self.MAX_ITERATION = 400

    def attack(self):
        attack = CarliniL2(self.model.sess, self.model, batch_size=1, max_iterations=self.MAX_ITERATION,
                           confidence=0,
                           direction=self.args.direction)
        for i in range(self.args.start, self.args.start + self.args.instances):
            image = self.data.get_image([i])[0]

            org_prediction = self.model.predict_image(image)[0][0]

            distortions = [0]
            predictions = [org_prediction]
            for j in self.CONST_LIST:
                time_start = time.time()
                inputs = self.data.get_image([i])

                adv = attack.attack(inputs, [j])
                new_prediction = self.model.predict_image(adv[0])[0][0]
                x_diff = adv[0] - inputs[0]

                distortion = np.sum(np.square(x_diff.raw_image/ (image.max_value - image.min_value)))
                distortions.append(distortion)
                predictions.append(new_prediction)

                print('Sample %d' % i, "Const %f" % j, "Time used %.2f" % (time.time() - time_start), new_prediction,
                      distortion)

                if self.args.save_data and i >= self.args.start + self.args.instances - 1 and j == self.CONST_LIST[-1]:
                    self.save_variables([inputs[0], adv[0], predictions[0], new_prediction, distortion])

            self.write_results(predictions)
            self.write_results(distortions)


class L0Exp(Experiment):
    def __init__(self, data, model, args):
        super(L0Exp, self).__init__(data, model, args)
        self.MAX_PIXELS_TO_CHANGE = int(0.1 * 172 * 220 * 156)
        self.INTERVAL = 100  # 100
        self.NUMBERS_INTERVAL = 50  # 40

    def attack(self):
        for i in range(self.args.start, self.args.start + self.args.instances):
            x = self.data.get_image([i])[0]
            max_value = x.max_value
            min_value = x.min_value

            x_adv = x.copy()
            best_prediction = self.model.predict_image(x)
            gradient_mat = self.model.get_grad_only(x)
            abs_gradient = np.abs(gradient_mat)

            cnt = 0
            cnt_failure = 0
            predictions = [best_prediction[0][0]]
            time_start = time.time()
            while True:
                argmax_num = np.nanargmax(abs_gradient)
                argmax_indices = np.unravel_index(argmax_num, abs_gradient.shape)
                abs_gradient[argmax_indices] = 0  # to find the next largest gradient pixel
                if self.args.direction == "max":
                    best_value = max_value if gradient_mat[argmax_indices] > 0 else min_value
                else:
                    best_value = min_value if gradient_mat[argmax_indices] > 0 else max_value
                saved_value = x_adv.raw_image[argmax_indices]
                flag_better = False
                for k in np.linspace(min_value, max_value, num=self.args.l0_values):
                    x_adv.raw_image[argmax_indices] = k
                    y_pred_tmp = self.model.predict_image(x_adv)
                    if (y_pred_tmp[0][0] > best_prediction[0][0] and self.args.direction == "max") or (
                            y_pred_tmp[0][0] < best_prediction[0][0] and self.args.direction == "min"):
                        best_value = k
                        best_prediction = y_pred_tmp
                        flag_better = True

                print("%.2f" % (time.time() - time_start), cnt, gradient_mat[argmax_indices], argmax_indices, best_value,
                      best_prediction, flag_better)
                x_adv.raw_image[argmax_indices] = best_value if flag_better else saved_value
                cnt = cnt + 1 if flag_better else cnt
                cnt_failure = cnt_failure + 1 if not flag_better else 0
                if cnt % self.INTERVAL == 0 and flag_better:
                    predictions.append(best_prediction[0][0])
                if cnt_failure >= 100:
                    predictions = predictions + [best_prediction[0][0]]*(self.NUMBERS_INTERVAL-len(predictions)+1)
                    break
                if cnt // self.INTERVAL >= self.NUMBERS_INTERVAL:
                    # save one example before exit
                    if self.args.save_data and i >= self.args.start + self.args.instances - 1:
                        y_pred_adv = self.model.predict_image(x_adv)
                        self.save_variables([x, x_adv, predictions[0], y_pred_adv[0][0]])
                    break

            self.write_results(predictions)
