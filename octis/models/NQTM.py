from sklearn.feature_extraction.text import CountVectorizer
from octis.models.model import AbstractModel
from octis.models.NQTM_model.NQTM_v2 import NQTM as NQTM_model

import os
import numpy as np
import tensorflow as tf

class NQTM(AbstractModel):

    def __init__(self, topic_num=10, layer1=100, layer2=100, batch_size=200,
                 learning_rate=0.002, keep_prob=1.0, epoch=200, word_sample_size=20,
                 word_sample_epoch=150, omega=1.0, commitment_cost=0.1, test_index=1,
                 use_partitions=True, use_validation=False):
        """
        initialization of NQTM
        """
        assert not(use_validation and use_partitions), "Validation data is not needed for BAT. Please set 'use_validation=False'."

        super().__init__()

        self.hyperparameters['topic_num'] = topic_num
        self.hyperparameters['layer1'] = layer1
        self.hyperparameters['layer2'] = layer2
        self.hyperparameters['batch_size'] = batch_size
        self.hyperparameters['learning_rate'] = learning_rate
        self.hyperparameters['keep_prob'] = keep_prob
        self.hyperparameters['epoch'] = epoch
        self.hyperparameters['word_sample_size'] = word_sample_size
        self.hyperparameters['word_sample_epoch'] = word_sample_epoch
        self.hyperparameters['omega'] = omega
        self.hyperparameters['commitment_cost'] = commitment_cost
        self.hyperparameters['test_index'] = test_index
        self.hyperparameters['active_fct'] = tf.nn.softplus

        self.use_partitions = use_partitions
        self.use_validation = use_validation

        self.model = None
        self.vocab = None

    def train_model(self, dataset, hyperparameters=None):
        """
        trains NQTM model
        """
        if hyperparameters is None:
            hyperparameters = {}

        self.set_params(hyperparameters)
        self.vocab = dataset.get_vocabulary()

        if self.use_partitions and not self.use_validation:
            train, test = dataset.get_partitioned_corpus(use_validation=False)
            data_corpus_train = [' '.join(i) for i in train]
            data_corpus_test = [' '.join(i) for i in test]

            x_train, x_test, input_size = self.preprocess(vocab=self.vocab,
                                                                   train=data_corpus_train,
                                                                   test=data_corpus_test)
            
            self.hyperparameters['vocab_size'] = input_size

            self.model = NQTM_model(self.hyperparameters)
            self.train(x_train, x_test)
            result = self.get_info()
            return result

        else:
            data_corpus = [' '.join(i) for i in dataset.get_corpus()]
            x_train, input_size = self.preprocess(self.vocab, train=data_corpus)
            self.hyperparameters['vocab_size'] = input_size

            self.model = NQTM_model(self.hyperparameters)
            self.train(x_train)
            result = self.get_info()
            return result

    def set_params(self, hyperparameters):
        for k in hyperparameters.keys():
            if k in self.hyperparameters.keys():
                self.hyperparameters[k] = hyperparameters.get(k, self.hyperparameters[k])
    
    def train(self, train_data, test_data=None, verbose=False):
        total_batch = int(train_data.shape[0] / self.hyperparameters['batch_size'])
        minibatches = self.create_minibatch(train_data)
        op = [self.model.train_op, self.model.loss]

        for epoch in range(self.hyperparameters['epoch']):
            omega = 0 if epoch < self.hyperparameters['word_sample_epoch'] else 1.0
            train_loss = list()
            for i in range(total_batch):
                batch_data = minibatches.__next__()
                feed_dict = {self.model.x: batch_data, self.model.w_omega: omega}
                _, batch_loss = self.model.sess.run(op, feed_dict=feed_dict)
                train_loss.append(batch_loss)
            if verbose==True:
                print('Epoch: ', '{:03d} loss: {:.3f}'.format(epoch + 1, np.mean(train_loss)))

        self.beta = self.model.sess.run((self.model.beta))
        self.train_theta = self.get_theta(train_data)
        if test_data is not None:
            self.test_theta = self.get_theta(test_data)

    
    def get_info(self, n_top_words=10):
        info = {}
        top_words = list()
        for i in range(len(self.beta)):
            top_words.append([self.vocab[j] for j in self.beta[i].argsort()[-n_top_words:][::-1]])
        info['topic-word-matrix'] = self.beta
        info['topic-document-matrix'] = self.train_theta.T
        if self.use_partitions:
            info['test-topic-document-matrix'] = self.test_theta.T
        info['topics'] = top_words
        return info
    
    def create_minibatch(self, data):
        rng = np.random.RandomState(10)
        while True:
            ixs = rng.randint(data.shape[0], size=self.hyperparameters['batch_size'])
            yield data[ixs]

    def get_theta(self, x):
        data_size = x.shape[0]
        batch_size = self.hyperparameters['batch_size']
        train_theta = np.zeros((data_size, self.hyperparameters['topic_num']))
        for i in range(int(data_size / batch_size)):
            start = i * batch_size
            end = (i + 1) * batch_size
            data_batch = x[start:end]
            train_theta[start:end] = self.model.sess.run(self.model.theta_e, feed_dict={self.model.x: data_batch})
        train_theta[-batch_size:] = self.model.sess.run(self.model.theta_e, feed_dict={self.model.x: x[-batch_size:]})
        return train_theta

    @staticmethod
    def preprocess(vocab, train, test=None, validation=None):
        vocab2id = {w: i for i, w in enumerate(vocab)}
        vec = CountVectorizer(
            vocabulary=vocab2id, token_pattern=r'(?u)\b[\w+|\-]+\b')
        entire_dataset = train.copy()
        if test is not None:
            entire_dataset.extend(test)
        if validation is not None:
            entire_dataset.extend(validation)

        vec.fit(entire_dataset)
        idx2token = {v: k for (k, v) in vec.vocabulary_.items()}
        x_train = vec.transform(train).todense()
        input_size = len(idx2token.keys())

        if test is not None and validation is not None:
            x_test = vec.transform(test).todense()
            x_valid = vec.transform(validation).todense()
            return x_train, x_test, x_valid, input_size
        
        if test is None and validation is not None:
            x_valid = vec.transform(validation).todense()
            return x_train, x_valid, input_size
        
        if test is not None and validation is None:
            x_test = vec.transform(test).todense()
            return x_train, x_test, input_size
        
        if test is None and validation is None:
            return x_train, input_size

