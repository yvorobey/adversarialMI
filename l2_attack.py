# modified by Huahong Zhang <huahong.zhang@vanderbilt.edu> to fit the L2 algorithm mentioned in the paper
# original copyright license follows.

# Copyright (c) 2016 Nicholas Carlini
#
# LICENSE
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import tensorflow as tf
import numpy as np
from feature import RawImage, ImageWithMas
from brainage import BrainAgeModel1, BrainAgeModel2

BINARY_SEARCH_STEPS = 10  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 1000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1e1     # the initial constant c to pick as a first guess
THRESHOLD = 1


class CarliniL2:
    def __init__(self, sess, model, batch_size=1, confidence = CONFIDENCE,
                 learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS,
                 abort_early = ABORT_EARLY, 
                 initial_const = INITIAL_CONST,
                 direction='max'):
        """
        The L_2 optimized attack. 

        This attack is the most efficient and should be used as the primary 
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence. 
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        boxmin: Minimum pixel value (default -0.5).
        boxmax: Maximum pixel value (default 0.5).
        """

        # image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.repeat = binary_search_steps >= 10
        self.direction = direction

        shape = (batch_size, model.shape[0], model.shape[1], model.shape[2], model.shape[3])
        
        # the variable we're going to optimize over
        modifier = tf.Variable(np.zeros(shape,dtype=np.float32))

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)
        self.mas = tf.Variable(np.zeros([batch_size, 134]), dtype=tf.float32)

        self.boxmul = tf.Variable(np.zeros(batch_size), dtype=tf.float32)
        self.boxplus = tf.Variable(np.zeros(batch_size), dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, [batch_size])
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        self.assign_mas = tf.placeholder(tf.float32, [batch_size, 134])

        self.assign_boxmul = tf.placeholder(tf.float32, [batch_size])
        self.assign_boxplus = tf.placeholder(tf.float32, [batch_size])
        
        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        self.newimg = tf.tanh(modifier + self.timg) * self.boxmul + self.boxplus

        if isinstance(model, BrainAgeModel2):
            self.output = model.predict([self.newimg, self.mas])
            self.org_output = model.predict([tf.tanh(self.timg)* self.boxmul + self.boxplus, self.mas])
        else:
            self.output = model.predict(self.newimg)
            self.org_output = model.predict(tf.tanh(self.timg) * self.boxmul + self.boxplus)
        
        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg-(tf.tanh(self.timg) * self.boxmul + self.boxplus)),[1,2,3,4])

        # modified here
        if self.direction == 'max':
            loss1 = -self.output + self.org_output
        else:
            loss1 = self.output - self.org_output

        # sum up the losses
        self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss1 = tf.reduce_sum(self.const*loss1)
        self.loss = self.loss1+self.loss2
        
        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.const.assign(self.assign_const))
        self.setup.append(self.boxmul.assign(self.assign_boxmul))
        self.setup.append(self.boxplus.assign(self.assign_boxplus))

        if isinstance(model, BrainAgeModel2):
            self.setup.append(self.mas.assign(self.assign_mas))
        
        self.init = tf.variables_initializer(var_list=[modifier]+new_vars)

    def attack(self, imgs, eps, mas=None):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to',len(imgs))
        for i in range(0,len(imgs),self.batch_size):
            print('tick', i)
            raw_images = np.array([x.raw_image for x in imgs[i:i + self.batch_size]])
            max_values = np.array([x.max_value for x in imgs[i:i + self.batch_size]])
            min_values = np.array([x.min_value for x in imgs[i:i + self.batch_size]])
            if isinstance(imgs[0], RawImage):
                res = self.attack_batch(raw_images, eps, max_values, min_values)
                r.extend([RawImage(res[i]) for i in range(self.batch_size)])
            elif isinstance(imgs[0], ImageWithMas):
                mas_feats = [x.mas for x in imgs[i:i+self.batch_size]]
                res = self.attack_batch(raw_images, eps, max_values, min_values, mas_feats)
                r.extend([ImageWithMas(res[i], mas_feats[i]) for i in range(self.batch_size)])
            else:
                raise Exception('Invalid data')

            # r.extend(self.attack_batch(imgs[i:i+self.batch_size], targets[i:i+self.batch_size]))

        return r

    def attack_batch(self, imgs, eps, max_values, min_values, mas_feats=None):
        """
        Run the attack on a batch of images and labels.
        """
        def compare(x, y):
            return (x>y and self.direction=="max") or (x<y and self.direction=="min")

        eps = [eps[0]*(max_values[0]-min_values[0])*(max_values[0]-min_values[0])]

        batch_size = self.batch_size

        # convert to tanh-space
        boxmul = (max_values - min_values)/2
        boxplus = (max_values + min_values)/2
        imgs = np.arctanh((imgs - boxplus) / boxmul * 0.999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # the best l2, score, and image attack
        # o_bestl2 = [1e10]*batch_size
        o_bestscore = [-1]*batch_size if self.direction == "max" else [1e10]*batch_size
        o_bestattack = [np.zeros(imgs[0].shape)]*batch_size

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print('STEP: %d, EPS: %f, CONST: %f, o_bestscore: %f' % (outer_step, eps[0], CONST[0], o_bestscore[0]))
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            if not mas_feats:
                self.sess.run(self.setup, {self.assign_timg: batch,
                                           self.assign_const: CONST,
                                           self.assign_boxmul: boxmul,
                                           self.assign_boxplus: boxplus})
            else:
                batchmas = mas_feats[:batch_size]
                self.sess.run(self.setup, {self.assign_timg: batch,
                                           self.assign_mas: batchmas,
                                           self.assign_const: CONST,
                                           self.assign_boxmul: boxmul,
                                           self.assign_boxplus: boxplus})

            # bestl2 = [1e10] * batch_size
            org_score = self.sess.run(self.org_output)[0][0]
            print("org_score", org_score)
            bestscore = [org_score+0.01] * batch_size if self.direction == "max" else [org_score-0.01] * batch_size

            prev = 1e6
            found = False
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                _, l, l2s, scores, nimg = self.sess.run([self.train, self.loss,
                                                                    self.l2dist, self.output,
                                                                    self.newimg])

                # print out the losses every 10%
                if iteration % (self.MAX_ITERATIONS // 10) == 0:
                    print(iteration, "loss: %f" % l, 'l1: %f' % (l - l2s), 'l2: %f' % l2s, 'score %f' % scores)
                    # print(iteration, self.sess.run((self.loss,self.loss1,self.loss2)), )

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration % (self.MAX_ITERATIONS // 10) == 0:
                    if l > prev * .9999:
                        break
                    prev = l

                # adjust the best result found so far
                for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
                    if l2 < eps and compare(sc[0], bestscore[e]):
                        bestscore[e] = sc[0]
                        found = True
                    if l2 < eps and compare(sc, o_bestscore[e]):
                        print("Found better result", sc[0], o_bestscore[e])
                        o_bestscore[e] = sc[0]
                        o_bestattack[e] = ii

            # adjust the constant as needed
            # the updates are different from the original version
            for e in range(batch_size):
                if found:
                    lower_bound[e] = max(lower_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        CONST[e] *= 2
                else:
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2

        # return the best solution found
        return o_bestattack
