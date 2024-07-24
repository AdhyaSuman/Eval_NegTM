from sklearn.feature_extraction.text import CountVectorizer
import gensim
import torch
import sys
import pandas as pd

from octis.models.model import AbstractModel
from octis.models.clntm.scholar import Scholar

import os
import numpy as np

class SCHOLARN(AbstractModel):

    def __init__(self, num_topics=10, lr=1e-3, momentum=0.99, batch_size=64,
                 num_epochs=100, interactions=False, covars_predict=False,
                 min_prior_covar_count=None, min_topic_covar_count=None,
                 regularization=False, l1_topics=0.0, l1_topic_covars=0.0,
                 l1_interactions=0.0, l2_prior_covars=0.0, emb_dim=300, alpha=1.0,
                 no_bg=False, dev_folds=0, dev_fold=0, dist=None, seed=2021,
                 w2v=None, topk=1, top_word=10, model_type='DecoderNegScholar',
                 use_partitions=True, use_validation=False,
                 topic_perturb=1, tloss_weight=1.):

        super().__init__()

        self.hyperparameters['num_topics'] = num_topics
        self.hyperparameters['lr'] = lr
        self.hyperparameters['momentum'] = momentum
        self.hyperparameters['batch_size'] = batch_size
        self.hyperparameters['num_epochs'] = num_epochs
        self.hyperparameters['interactions'] = interactions
        self.hyperparameters["covars_predict"] = covars_predict
        self.hyperparameters["min_prior_covar_count"] = min_prior_covar_count
        self.hyperparameters["min_topic_covar_count"] = min_topic_covar_count
        self.hyperparameters["regularization"] = regularization

        self.hyperparameters["l1_topics"] = l1_topics
        self.hyperparameters["l1_topic_covars"] = l1_topic_covars
        self.hyperparameters["l1_interactions"] = l1_interactions

        self.hyperparameters["l2_prior_covars"] = l2_prior_covars
        self.hyperparameters["emb_dim"] = emb_dim
        self.hyperparameters["alpha"] = alpha
        self.hyperparameters["no_bg"] = no_bg
        self.hyperparameters["dev_folds"] = dev_folds
        self.hyperparameters["dev_fold"] = dev_fold
        self.hyperparameters["dist"] = dist
        self.hyperparameters["l1_interactions"] = l1_interactions
        self.hyperparameters["topk"] = topk

        self.hyperparameters['topic_perturb'] = topic_perturb
        self.hyperparameters['tloss_weight'] = tloss_weight

        self.w2v = w2v
        self.use_partitions = use_partitions
        self.use_validation = use_validation
        self.top_word = top_word
        self.model_type = model_type

        self.model = None
        self.vocab = None

        if self.hyperparameters["regularization"]:
            self.hyperparameters["l1_topics"] = 1.0
            self.hyperparameters["l1_topic_covars"] = 1.0
            self.hyperparameters["regularization"] = 1.0
        
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            self.hyperparameters["seed"] = seed
        else:
            rng = np.random.RandomState(np.random.randint(0, 100000))
            self.hyperparameters["seed"] = None

    def train_model(self, dataset, hyperparameters=None, top_words=10,):
        if hyperparameters is None:
            hyperparameters = {}
        
        self.set_params(hyperparameters)

        self.vocab = dataset.get_vocabulary()

        if self.use_partitions and not self.use_validation:
            train, test = dataset.get_partitioned_corpus(use_validation=False)

            data_corpus_train = [' '.join(i) for i in train]
            data_corpus_test = [' '.join(i) for i in test]

            train_X, test_X = self.preprocess(self.vocab, data_corpus_train, test=data_corpus_test, validation=None)
            
            train_prior_covars, prior_covar_selector, \
                prior_covar_names, n_prior_covars = self.load_covariates(input_dir=None,
                                                                         input_prefix=None,
                                                                         row_selector=None,
                                                                         covars_to_load=None,
                                                                         min_count=None)
            
            train_topic_covars, topic_covar_selector, \
                topic_covar_names, n_topic_covars = self.load_covariates(input_dir=None,
                                                                         input_prefix=None,
                                                                         row_selector=None,
                                                                         covars_to_load=None,
                                                                         min_count=None)
            
            test_prior_covars, _, _, _ = self.load_covariates(input_dir=None,
                                                              input_prefix=None,
                                                              row_selector=None,
                                                              covars_to_load=None,
                                                              covariate_selector=prior_covar_selector)
            
            test_topic_covars, _, _, _ = self.load_covariates(input_dir=None,
                                                              input_prefix=None,
                                                              row_selector=None,
                                                              covars_to_load=None,
                                                              covariate_selector=topic_covar_selector)
            
            init_bg = self.get_init_bg(train_X)
            if self.hyperparameters['no_bg']:
                init_bg = np.zeros_like(init_bg)
            
            # combine the network configuration parameters into a dictionary
            network_architecture = self.make_network(n_prior_covars, n_topic_covars)

            # print("Network architecture:-")
            # for key, val in network_architecture.items():
            #     print('\t' + key + ':', val)

            print('topic_perturb={}, tloss_weight={}'.format(network_architecture['topic_perturb'], network_architecture['tloss_weight']))
            
            # load word vectors
            embeddings, update_embeddings = self.load_word_vectors()
            
            self.model = Scholar(network_architecture,
                                 alpha=self.hyperparameters["alpha"],
                                 learning_rate=self.hyperparameters["lr"],
                                 init_embeddings=embeddings,
                                 update_embeddings=update_embeddings,
                                 init_bg=init_bg,
                                 adam_beta1=self.hyperparameters["momentum"],
                                 device=0 if torch.cuda.is_available() else None,
                                 seed=self.hyperparameters["seed"],
                                 classify_from_covars=self.hyperparameters["covars_predict"],
                                 model=self.model_type,
                                 topk=self.hyperparameters["topk"])

            # train the model
            # print("Optimizing full model")
            self.model = self._train(network_architecture,
                                     X=train_X,
                                     Y=None,
                                     PC=train_prior_covars,
                                     TC=train_topic_covars,
                                     training_epochs=self.hyperparameters['num_epochs'],
                                     batch_size=self.hyperparameters['batch_size'],
                                     rng=self.rng,
                                     X_dev=None,
                                     Y_dev=None,
                                     PC_dev=None,
                                     TC_dev=None)
            
            result = self.get_info()
            result['topic-document-matrix'] = self.get_doc_topic(train_X,
                                                                 Y=None,
                                                                 PC=train_prior_covars,
                                                                 TC=train_topic_covars,
                                                                 ids=None,
                                                                 output_dir=None,
                                                                 partition='train',
                                                                 batch_size=self.hyperparameters['batch_size']).T
            
            result['test-topic-document-matrix'] = self.get_doc_topic(test_X,
                                                                      Y=None,
                                                                      PC=test_prior_covars,
                                                                      TC=test_topic_covars,
                                                                      ids=None,
                                                                      output_dir=None,
                                                                      partition='test',
                                                                      batch_size=self.hyperparameters['batch_size']).T
        else:
            data_corpus = [' '.join(i) for i in dataset.get_corpus()]
            train_X = self.preprocess(self.vocab, train=data_corpus)

            train_prior_covars, prior_covar_selector, \
                prior_covar_names, n_prior_covars = self.load_covariates(input_dir=None,
                                                                         input_prefix=None,
                                                                         row_selector=None,
                                                                         covars_to_load=None,
                                                                         min_count=None)
            
            train_topic_covars, topic_covar_selector, \
                topic_covar_names, n_topic_covars = self.load_covariates(input_dir=None,
                                                                         input_prefix=None,
                                                                         row_selector=None,
                                                                         covars_to_load=None,
                                                                         min_count=None)


            init_bg = self.get_init_bg(train_X)
            if self.hyperparameters['no_bg']:
                init_bg = np.zeros_like(init_bg)
            
            # combine the network configuration parameters into a dictionary
            network_architecture = self.make_network(n_prior_covars, n_topic_covars)

            # print("Network architecture:")
            # for key, val in network_architecture.items():
            #     print(key + ':', val)
            # print('topk=', self.hyperparameters["topk"])
            print('topic_perturb={}, tloss_weight={}'.format(network_architecture['topic_perturb'], network_architecture['tloss_weight']))
            # load word vectors
            embeddings, update_embeddings = self.load_word_vectors()
            
            self.model = Scholar(network_architecture,
                                 alpha=self.hyperparameters["alpha"],
                                 learning_rate=self.hyperparameters["lr"],
                                 init_embeddings=embeddings,
                                 update_embeddings=update_embeddings,
                                 init_bg=init_bg,
                                 adam_beta1=self.hyperparameters["momentum"],
                                 device=0 if torch.cuda.is_available() else None,
                                 seed=self.hyperparameters["seed"],
                                 classify_from_covars=self.hyperparameters["covars_predict"],
                                 model=self.model_type,
                                 topk=self.hyperparameters["topk"])

            # train the model
            print("Optimizing full model")
            self.model = self._train(network_architecture,
                                     X=train_X,
                                     Y=None,
                                     PC=train_prior_covars,
                                     TC=train_topic_covars,
                                     training_epochs=self.hyperparameters['num_epochs'],
                                     batch_size=self.hyperparameters['batch_size'],
                                     rng=self.rng,
                                     X_dev=None,
                                     Y_dev=None,
                                     PC_dev=None,
                                     TC_dev=None)
            
            result = self.get_info()
            result['topic-document-matrix'] = self.get_doc_topic(train_X,
                                                                 Y=None,
                                                                 PC=train_prior_covars,
                                                                 TC=train_topic_covars,
                                                                 ids=None,
                                                                 output_dir=None,
                                                                 partition='train',
                                                                 batch_size=self.hyperparameters['batch_size']).T
            result = self.get_info()
        return result

    def set_params(self, hyperparameters):
        for k in hyperparameters.keys():
            if k in self.hyperparameters.keys():
                self.hyperparameters[k] = hyperparameters.get(k, self.hyperparameters[k])

    def get_info(self):
        info = {}
        beta = self.model.get_weights()
        info['topic-word-matrix'] = beta
        info['topics'] = SCHOLARN.show_topic_words(beta, self.vocab, topK=self.top_word)
        return info
    
    @staticmethod
    def show_topic_words(beta, vocab, topK):
        topic_w = []
        for k in range(len(beta)):
            if np.isnan(beta[k]).any():
                # to deal with nan matrices
                topic_w = None
                break
            else:
                top_words = list(beta[k].argsort()[-topK:][::-1])
            topic_words = [vocab[a] for a in top_words]
            topic_w.append(topic_words)

        return topic_w

    def set_params(self, hyperparameters):
        for k in hyperparameters.keys():
            if k in self.hyperparameters.keys():
                self.hyperparameters[k] = hyperparameters.get(k, self.hyperparameters[k])

    def inference(self, x_test):
        assert isinstance(self.use_partitions, bool) and self.use_partitions
        results = self.model.predict(x_test)
        return results

    def partitioning(self, use_partitions=False):
        self.use_partitions = use_partitions
    
    def get_init_bg(self, data):
        #Compute the log background frequency of all words
        #sums = np.sum(data, axis=0)+1
        n_items, vocab_size = data.shape
        sums = np.array(data.sum(axis=0)).reshape((vocab_size,))+1.
        print("Computing background frequencies")
        print("Min/max word counts in training data: %d %d" % (int(np.min(sums)), int(np.max(sums))))
        bg = np.array(np.log(sums) - np.log(float(np.sum(sums))), dtype=np.float32)
        return bg
    
    def make_network(self, n_prior_covars=0, n_topic_covars=0):
    # Assemble the network configuration parameters into a dictionary
        network_architecture = \
            dict(embedding_dim=self.hyperparameters["emb_dim"],
                n_topics=self.hyperparameters['num_topics'],
                vocab_size=len(self.vocab),
                label_type=None,
                n_labels=0,
                n_prior_covars=n_prior_covars,
                n_topic_covars=n_topic_covars,
                l1_beta_reg=self.hyperparameters["l1_topics"],
                l1_beta_c_reg=self.hyperparameters["l1_topic_covars"],
                l1_beta_ci_reg=self.hyperparameters["l1_interactions"],
                l2_prior_reg=self.hyperparameters["l2_prior_covars"],
                classifier_layers=1,
                use_interactions=self.hyperparameters["interactions"],
                dist=self.hyperparameters["dist"],
                model=self.model_type,
                topic_perturb=self.hyperparameters['topic_perturb'],
                tloss_weight=self.hyperparameters['tloss_weight']
                )
        return network_architecture
       
    def load_covariates(self, input_dir, input_prefix, row_selector, covars_to_load, min_count=None, covariate_selector=None):
        covariates = None
        covariate_names = None
        n_covariates = 0
        if covars_to_load is not None:
            covariate_list = []
            covariate_names_list = []
            covar_file_names = covars_to_load.split(',')
            # split the given covariate names by commas, and load each one
            for covar_file_name in covar_file_names:
                covariates_file = os.path.join(input_dir, input_prefix + '.' + covar_file_name + '.csv')
                if os.path.exists(covariates_file):
                    print("Loading covariates from", covariates_file)
                    temp = pd.read_csv(covariates_file, header=0, index_col=0)
                    covariate_names = temp.columns
                    covariates = np.array(temp.values, dtype=np.float32)
                    # select the rows that match the non-empty documents (from load_word_counts)
                    covariates = covariates[row_selector, :]
                    covariate_list.append(covariates)
                    covariate_names_list.extend(covariate_names)
                else:
                    raise(FileNotFoundError("Covariates file {:s} not found".format(covariates_file)))

            # combine the separate covariates into a single matrix
            covariates = np.hstack(covariate_list)
            covariate_names = covariate_names_list

            _, n_covariates = covariates.shape

            # if a covariate_selector has been given (from a previous call of load_covariates), drop columns
            if covariate_selector is not None:
                covariates = covariates[:, covariate_selector]
                covariate_names = [name for i, name in enumerate(covariate_names) if covariate_selector[i]]
                n_covariates = len(covariate_names)
            # otherwise, choose which columns to drop based on how common they are (for binary covariates)
            elif min_count is not None and int(min_count) > 0:
                print("Removing rare covariates")
                covar_sums = covariates.sum(axis=0).reshape((n_covariates, ))
                covariate_selector = covar_sums > int(min_count)
                covariates = covariates[:, covariate_selector]
                covariate_names = [name for i, name in enumerate(covariate_names) if covariate_selector[i]]
                n_covariates = len(covariate_names)

        return covariates, covariate_selector, covariate_names, n_covariates
    
    def load_word_vectors(self, word2vec_file=None):
        # load word2vec vectors if given
        if word2vec_file is not None:
            vocab_size = len(self.vocab)
            vocab_dict = dict(zip(self.vocab, range(vocab_size)))
            # randomly initialize word vectors for each term in the vocabualry
            embeddings = np.array(self.rng.rand(self.hyperparameters["emb_dim"], vocab_size) * 0.25 - 0.5, dtype=np.float32)
            count = 0
            print("Loading word vectors")
            # load the word2vec vectors
            pretrained = gensim.models.KeyedVectors.load_word2vec_format(self.w2v, binary=True)

            # replace the randomly initialized vectors with the word2vec ones for any that are available
            for word, index in vocab_dict.items():
                if word in pretrained:
                    count += 1
                    embeddings[:, index] = pretrained[word]

            print("Found embeddings for %d words" % count)
            update_embeddings = False
        else:
            embeddings = None
            update_embeddings = True

        return embeddings, update_embeddings
    
    def _train(self, network_architecture, X, Y, PC, TC, batch_size=200,
              training_epochs=100, display_step=10, X_dev=None, Y_dev=None,
              PC_dev=None, TC_dev=None, bn_anneal=True, init_eta_bn_prop=1.0,
              rng=None, min_weights_sq=1e-7):
        # Train the model
        n_train, vocab_size = X.shape
        mb_gen = SCHOLARN.create_minibatch(X, Y, PC, TC, batch_size=batch_size, rng=rng)
        total_batch = int(n_train / batch_size)
        batches = 0
        eta_bn_prop = init_eta_bn_prop  # interpolation between batch norm and no batch norm in final layer of recon

        self.model.train()

        n_topics = network_architecture['n_topics']
        n_topic_covars = network_architecture['n_topic_covars']
        vocab_size = network_architecture['vocab_size']

        # create matrices to track the current estimates of the priors on the individual weights
        if network_architecture['vocab_size'] > 0:
            l1_beta = 0.5 * np.ones([vocab_size, n_topics], dtype=np.float32) / float(n_train)
        else:
            l1_beta = None

        if network_architecture['l1_beta_c_reg'] > 0 and network_architecture['n_topic_covars'] > 0:
            l1_beta_c = 0.5 * np.ones([vocab_size, n_topic_covars], dtype=np.float32) / float(n_train)
        else:
            l1_beta_c = None

        if network_architecture['l1_beta_ci_reg'] > 0 and network_architecture['n_topic_covars'] > 0 and network_architecture['use_interactions']:
            l1_beta_ci = 0.5 * np.ones([vocab_size, n_topics * n_topic_covars], dtype=np.float32) / float(n_train)
        else:
            l1_beta_ci = None

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            accuracy = 0.
            avg_nl = 0.
            avg_kld = 0.
            avg_tl = 0.
            # Loop over all batches
            for i in range(total_batch):
                # get a minibatch
                batch_xs, batch_ys, batch_pcs, batch_tcs = next(mb_gen)
                # do one minibatch update
                cost, recon_y, thetas, nl, kld, tl = self.model.fit(batch_xs, batch_ys, batch_pcs, batch_tcs,
                                                                eta_bn_prop=eta_bn_prop, l1_beta=l1_beta,
                                                                l1_beta_c=l1_beta_c, l1_beta_ci=l1_beta_ci)

                # compute accuracy on minibatch
                if network_architecture['n_labels'] > 0:
                    accuracy += np.sum(np.argmax(recon_y, axis=1) == np.argmax(batch_ys, axis=1)) / float(n_train)

                # Compute average loss
                avg_cost += float(cost) / n_train * batch_size
                avg_nl += float(nl) / n_train * batch_size
                avg_kld += float(kld) / n_train * batch_size
                avg_tl += float(tl) / n_train * batch_size
                batches += 1
                if np.isnan(avg_cost):
                    print(epoch, i, np.sum(batch_xs, 1).astype(np.int), batch_xs.shape)
                    print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                    sys.exit()

            # if we're using regularization, update the priors on the individual weights
            if network_architecture['l1_beta_reg'] > 0:
                weights = self.model.get_weights().T
                weights_sq = weights ** 2
                # avoid infinite regularization
                weights_sq[weights_sq < min_weights_sq] = min_weights_sq
                l1_beta = 0.5 / weights_sq / float(n_train)

            if network_architecture['l1_beta_c_reg'] > 0 and network_architecture['n_topic_covars'] > 0:
                weights = self.model.get_covar_weights().T
                weights_sq = weights ** 2
                weights_sq[weights_sq < min_weights_sq] = min_weights_sq
                l1_beta_c = 0.5 / weights_sq / float(n_train)

            if network_architecture['l1_beta_ci_reg'] > 0 and network_architecture['n_topic_covars'] > 0 and network_architecture['use_interactions']:
                weights = self.model.get_covar_interaction_weights().T
                weights_sq = weights ** 2
                weights_sq[weights_sq < min_weights_sq] = min_weights_sq
                l1_beta_ci = 0.5 / weights_sq / float(n_train)

            # Display logs per epoch step
            if epoch % display_step == 0 and epoch > 0:
                # if network_architecture['n_labels'] > 0:
                #     print("Epoch:", '%d' % epoch, "; cost =", "{:.9f}".format(avg_cost), "; training accuracy (noisy) =", "{:.9f}".format(accuracy))
                # else:
                #     print("Epoch:", '%d' % epoch, "cost=", "{:.9f}".format(avg_cost))

                if X_dev is not None:
                    # switch to eval mode for intermediate evaluation
                    self.model.eval()
                    dev_perplexity = self.evaluate_perplexity(X_dev, Y_dev, PC_dev, TC_dev, batch_size, eta_bn_prop=eta_bn_prop)
                    n_dev, _ = X_dev.shape
                    if network_architecture['n_labels'] > 0:
                        dev_pred_probs = self.predict_label_probs(X_dev, PC_dev, TC_dev, eta_bn_prop=eta_bn_prop)
                        dev_predictions = np.argmax(dev_pred_probs, axis=1)
                        dev_accuracy = float(np.sum(dev_predictions == np.argmax(Y_dev, axis=1))) / float(n_dev)
                        # print("Epoch: %d; Dev perplexity = %0.4f; Dev accuracy = %0.4f" % (epoch, dev_perplexity, dev_accuracy))
                    # else:
                        # print("Epoch: %d; Dev perplexity = %0.4f" % (epoch, dev_perplexity))
                    # switch back to training mode
                    self.model.train()

            # anneal eta_bn_prop from 1.0 to 0.0 over training
            if bn_anneal:
                if eta_bn_prop > 0:
                    eta_bn_prop -= 1.0 / float(0.75 * training_epochs)
                    if eta_bn_prop < 0:
                        eta_bn_prop = 0.0

        # finish training
        self.model.eval()
        return self.model

    @staticmethod
    def create_minibatch(X, Y, PC, TC, batch_size=200, rng=None):
        # Yield a random minibatch
        while True:
            # Return random data samples of a size 'minibatch_size' at each iteration
            if rng is not None:
                ixs = rng.randint(X.shape[0], size=batch_size)
            else:
                ixs = np.random.randint(X.shape[0], size=batch_size)

            X_mb = np.array(X[ixs, :]).astype('float32')
            if Y is not None:
                Y_mb = Y[ixs, :].astype('float32')
            else:
                Y_mb = None

            if PC is not None:
                PC_mb = PC[ixs, :].astype('float32')
            else:
                PC_mb = None

            if TC is not None:
                TC_mb = TC[ixs, :].astype('float32')
            else:
                TC_mb = None

            yield X_mb, Y_mb, PC_mb, TC_mb


    def evaluate_perplexity(self, X, Y, PC, TC, batch_size, eta_bn_prop=0.0):
        # Evaluate the approximate perplexity on a subset of the data (using words, labels, and covariates)
        n_items, _ = X.shape
        doc_sums = np.array(X.sum(axis=1), dtype=float).reshape((n_items,))
        X = X.astype('float32')
        if Y is not None:
            Y = Y.astype('float32')
        if PC is not None:
            PC = PC.astype('float32')
        if TC is not None:
            TC = TC.astype('float32')
        losses = []

        n_items, _ = X.shape
        n_batches = int(np.ceil(n_items / batch_size))
        for i in range(n_batches):
            batch_xs, batch_ys, batch_pcs, batch_tcs = SCHOLARN.get_minibatch(X, Y, PC, TC, i, batch_size)
            batch_losses = self.model.get_losses(batch_xs, batch_ys, batch_pcs, batch_tcs, eta_bn_prop=eta_bn_prop)
            losses.append(batch_losses)
        losses = np.hstack(losses)
        perplexity = np.exp(np.mean(losses / doc_sums))
        return perplexity
    
    
    def predict_label_probs(self, X, PC, TC, batch_size=200, eta_bn_prop=0.0):
        # Predict a probability distribution over labels for each instance using the classifier part of the network

        n_items, _ = X.shape
        n_batches = int(np.ceil(n_items / batch_size))
        pred_probs_all = []

        # make predictions on minibatches and then combine
        for i in range(n_batches):
            batch_xs, batch_ys, batch_pcs, batch_tcs = SCHOLARN.get_minibatch(X, None, PC, TC, i, batch_size)
            Z, pred_probs = self.model.predict(batch_xs, batch_pcs, batch_tcs, eta_bn_prop=eta_bn_prop)
            pred_probs_all.append(pred_probs)

        pred_probs = np.vstack(pred_probs_all)

        return pred_probs
    
    @staticmethod
    def get_minibatch(X, Y, PC, TC, batch, batch_size=200):
        # Get a particular non-random segment of the data
        n_items, _ = X.shape
        n_batches = int(np.ceil(n_items / float(batch_size)))
        if batch < n_batches - 1:
            ixs = np.arange(batch * batch_size, (batch + 1) * batch_size)
        else:
            ixs = np.arange(batch * batch_size, n_items)

        X_mb = np.array(X[ixs, :]).astype('float32')
        if Y is not None:
            Y_mb = Y[ixs, :].astype('float32')
        else:
            Y_mb = None

        if PC is not None:
            PC_mb = PC[ixs, :].astype('float32')
        else:
            PC_mb = None

        if TC is not None:
            TC_mb = TC[ixs, :].astype('float32')
        else:
            TC_mb = None

        return X_mb, Y_mb, PC_mb, TC_mb


    def get_doc_topic(self, X, Y, PC, TC, ids, output_dir, partition, batch_size=200):
        # compute the mean of the posterior of the latent representation for each documetn and save it
        if Y is not None:
            Y = np.zeros_like(Y)

        n_items, _ = X.shape
        n_batches = int(np.ceil(n_items / batch_size))
        thetas = []

        for i in range(n_batches):
            batch_xs, batch_ys, batch_pcs, batch_tcs = SCHOLARN.get_minibatch(X, Y, PC, TC, i, batch_size)
            thetas.append(self.model.compute_theta(batch_xs, batch_ys, batch_pcs, batch_tcs))
        
        return np.concatenate(thetas, axis=0)

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
        
        x_train = vec.transform(train).toarray()

        if test is not None and validation is not None:
            x_test = vec.transform(test).toarray()
            x_valid = vec.transform(validation).toarray()
            return x_train, x_test, x_valid
        
        if test is None and validation is not None:
            x_valid = vec.transform(validation).toarray()
            return x_train, x_valid
        
        if test is not None and validation is None:
            x_test = vec.transform(test).toarray()
            return x_train, x_test
        if test is None and validation is None:
            return x_train
