import numpy as np
import os
import scipy.io
import nibabel as nib
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import load_model
from numpy import ndarray, array
from abc import abstractmethod
from sklearn.externals import joblib

from feature import RawImage, ImageWithMas


class BrainAgeData(object):
    def __init__(self, work_dir='./'):
        self.work_dir = work_dir

        random_path_list = os.path.join(work_dir, 'data', 'random_path_list.csv')
        path_list = np.genfromtxt(random_path_list, delimiter=',', dtype=str)
        X = list(path_list)
        n = (range(len(X)))
        X = [X[x] for x in n]

        self.data_paths = X
        self.cor_num = self.get_cor_num()
        self.mas_feats = self.load_MAS_feats()
        self.with_mas = None

    @abstractmethod
    def get_image(self, instance_list):
        pass

    def get_cor_num(self):
        cor_nums_file = os.path.join(self.work_dir, 'data', 'random_path_list_nums.csv')
        nums_old = np.genfromtxt(cor_nums_file, delimiter=',', dtype=int) - 1
        return nums_old

    def load_MAS_feats(self):
        # Load MAS and demographic features. These are already demeaned from testing set.
        test_features_file = os.path.join(self.work_dir, 'data', 'MAS_data.mat')
        mat = scipy.io.loadmat(test_features_file)
        test = mat['data']
        test = array(test)

        MM = joblib.load(os.path.join(self.work_dir, 'data', 'MAS_feats132_normer.save'))
        test = MM.transform(test)

        SC = joblib.load(os.path.join(self.work_dir, 'data', 'MAS_feats132_scaler.save'))
        test = SC.transform(test)

        return test

    @staticmethod
    def load_samples(sample_list):
        samples = []
        if not isinstance(sample_list, list):
            sample_list = [sample_list]

        for sample in range(0, len(sample_list)):
            x = nib.load(sample_list[sample])
            samples.append(np.expand_dims(np.squeeze(x.get_data()), axis=0))
        samples = np.concatenate(samples, axis=0)
        samples = np.expand_dims(samples, axis=4)
        return samples


class BrainAgeData1(BrainAgeData):
    def __init__(self, work_dir='./'):
        super(BrainAgeData1, self).__init__(work_dir)
        self.with_mas = True

    def get_image(self, instance_list):
        # get_RawImage
        res = []
        for idx in instance_list:
            raw = self.load_samples(self.data_paths[idx])[0]
            res.append(RawImage(raw))
        return res


class BrainAgeData2(BrainAgeData):
    def __init__(self, work_dir='./'):
        super(BrainAgeData2, self).__init__(work_dir)
        self.with_mas = False

    def get_image(self, instance_list):
        # get_ImageWithMas
        res = []
        for idx in instance_list:
            raw_image = self.load_samples(self.data_paths[idx])[0]
            mas = self.mas_feats[self.cor_num[idx]]
            res.append(ImageWithMas(raw_image, mas))
        return res


class BrainAgeModel(object):
    def __init__(self, session=None, work_dir='./'):
        self.shape = [172, 220, 156, 1]
        self.sess = session

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def predict_image(self, image):
        pass

    @abstractmethod
    def get_grad(self, var, step_length, normalize=np.sign):
        pass


class BrainAgeModel1(BrainAgeModel):
    def __init__(self, model_name, session=None, work_dir='./'):
        super(BrainAgeModel1, self).__init__(session, work_dir)
        model_path = os.path.join(work_dir, 'models', model_name)
        model = load_model(model_path)
        self.model = model
        self.name = 'model1'
        self.grad = tf.gradients(self.model.output, self.model.input)

    def predict(self, x):
        # use tensorflow model to predict tensor
        return self.model(x)

    def predict_image(self, raw_image):
        # use keras model to predict image
        return self.model.predict(raw_image.raw_image.reshape((1, 172, 220, 156, 1)))

    def get_grad(self, var, step_length, normalize=np.sign):
        grad = self.sess.run(self.grad, feed_dict={self.model.input: var.raw_image.reshape((1, 172, 220, 156, 1))})
        return RawImage(normalize(np.reshape(grad, var.raw_image.shape)) * step_length)

    def get_grad_only(self, var):
        grad = self.sess.run(self.grad, feed_dict={self.model.input: var.raw_image.reshape((1, 172, 220, 156, 1))})
        return np.reshape(grad, var.raw_image.shape)


class BrainAgeModel2(BrainAgeModel):
    def __init__(self, model_name1, model_name2, session=None, work_dir='./'):
        def transform(X):
            MM = joblib.load(os.path.join(work_dir, 'data', 'conv_feats_normer.save'))
            SC = joblib.load(os.path.join(work_dir, 'data', 'conv_feats_scaler.save'))
            X = K.tf.multiply(X, MM.scale_)
            X = K.tf.add(X, MM.min_)
            if SC.with_mean:
                X = K.tf.subtract(X, SC.mean_)
            if SC.with_std:
                X = K.tf.div(X, SC.scale_)
            return X

        super(BrainAgeModel2, self).__init__(session, work_dir)
        model1 = load_model(os.path.join(work_dir, 'models', model_name1))
        model2 = load_model(os.path.join(work_dir, 'models', model_name2))
        first_stage_input = model1.layers[0].input
        first_stage_output = model1.layers[10].output
        first_stage_output_after_transforming = keras.layers.Lambda(transform)(first_stage_output)
        model2_input = keras.layers.Input((134,))
        second_stage_output = model2([first_stage_output_after_transforming, model2_input])
        model3 = keras.models.Model(inputs=[first_stage_input, model2_input], outputs=second_stage_output)
        self.model = model3
        self.name = 'model2'
        self.grad = tf.gradients(self.model.output, self.model.input)

    def predict(self, x):
        # use tensorflow model to predict tensor
        return self.model(x)

    def predict_image(self, image_with_mas):
        _mas = image_with_mas.mas.reshape((1, 134))
        raw_image = image_with_mas.raw_image.reshape((1, 172, 220, 156, 1))
        x = [raw_image, _mas]
        return self.model.predict(x)

    def get_grad(self, var, step_length, normalize=np.sign):
        raw_image = var.raw_image.reshape((1, 172, 220, 156, 1))
        mas = var.mas
        grad = self.sess.run(self.grad, feed_dict={self.model.input[0]: raw_image, self.model.input[1]: np.reshape(mas,(1,134))})
        raw_image_grad = normalize(np.reshape(grad[0], var.raw_image.shape)) * step_length
        return ImageWithMas(raw_image_grad, mas)

    def get_grad_only(self, var):
        raw_image = var.raw_image.reshape((1, 172, 220, 156, 1))
        mas = var.mas
        grad = self.sess.run(self.grad,
                             feed_dict={self.model.input[0]: raw_image, self.model.input[1]: np.reshape(mas, (1, 134))})
        return np.reshape(grad[0], var.raw_image.shape)
