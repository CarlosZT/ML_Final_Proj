from typing import Literal, Mapping, Sequence
import numpy as np
from numpy.random import RandomState
from data_tools import *
from scipy import stats
from scipy.special import logsumexp

class MLP(PythonModel):
    def __init__(self, topology, n_features, bias = 0.5) -> None:
        self.model = []
        self.b = bias
        #Esto se queda con dos capas nada mas, porque no me salió implementar mas :c
        for neurons in topology:
            self.model.append(Layer(n_features, neurons))
            n_features = neurons

        self.loss = (lambda y, y_hat: ((y_hat - y)**2)/2,
                     lambda y, y_hat: (y_hat - y))
        
        self.activation = (lambda x: 1/(1 + np.exp(-x)),
                           lambda x: self.activation[0](x) * (1 - self.activation[0](x)))
    
    def predict(self, X, trainning = False):
        y_hat = X
        for l in self.model:
           l.forward(y_hat)
           y_hat = l.output
        
        if trainning:
            return y_hat

        
        y_hat[y_hat > self.b] = 1
        y_hat[y_hat <= self.b] = 0
        
        return y_hat[...,0]

    
    def update_weights(self, y_true, lr):
        y_true = y_true[..., np.newaxis]
        alpha =  self.loss[1](y_true, self.model[-1].output) * self.activation[1](self.model[-1].z)
        i = 0
        for l in reversed(self.model):
            grad = l.input.T @ alpha
            if i == 0:
                beta = (self.activation[1](l.input) * l.W.T)[...,1:]
                alpha = alpha * beta
                i+=1
            l.W -= (grad) * lr

        


class Layer():
    def __init__(self, n_features, n_neurons) -> None:
        self.neurons = n_neurons
        self.W = np.random.normal(0, 1, (n_features + 1, self.neurons))
        self.activation = (lambda x: 1/(1 + np.exp(-x)),
                           lambda x: self.activation[0](x) * (1 - self.activation[0](x)))
        self.z = None
        self.output = None
        self.input = None
        
    def forward(self, X):
        self.input = np.concatenate([np.ones(X.shape[0])[...,np.newaxis], X], axis=1)
        self.z = self.input @ self.W
        self.output = self.activation[0](self.z)

class DecisionTree(DecisionTreeClassifier, PythonModel):
    pass
class LogisticClassifier(PythonModel):
    def __init__(self, dims = 1, bias = 0.5) -> None:
        self.W = np.random.normal(0, 1, dims + 1)
        self.b = bias
        self.loss = (lambda y, y_hat: ((y_hat - y)**2)/2,
                     lambda y, y_hat: (y_hat - y))
        self.activation = (lambda x: 1/(1 + np.exp(-x)),
                           lambda x: self.activation[0](x) * (1 - self.activation[0](x)))
    

    def predict(self, X, trainning = False):
        X_ = np.concatenate([np.ones(X.shape[0])[...,np.newaxis], X], axis=1)
        z = X_ @ self.W
        y_hat = self.activation[0](z)

        if trainning:
            return y_hat, z
        
        y_hat[y_hat > self.b] = 1
        y_hat[y_hat <= self.b] = 0
        return y_hat


    def update_weights(self, y_true, y_hat, z, X, lr = 0.01):
        X_ = np.concatenate([np.ones(X.shape[0])[...,np.newaxis], X], axis=1)
        grad = X_.T @ (self.loss[1](y_true, y_hat) * self.activation[1](z))
        self.W -= grad * lr

class LinearClassifier(PythonModel):
    def __init__(self, dims = 1, bias = 0.0) -> None:
        self.W = np.random.normal(0, 1, dims + 1)
        self.b = bias
        self.loss = (lambda y_true, y_hat: ((y_hat - y_true)**2)/2,
                     lambda y_true, y_hat: (y_hat - y_true))

    def predict(self, input):
        X_ = np.concatenate([np.ones(input.shape[0])[...,np.newaxis], input], axis=1)
        y_hat = X_ @ self.W
        y_hat[y_hat > self.b] = 1
        y_hat[y_hat <= self.b] = 0
        return y_hat

    def update_weights(self, y_true, y_hat, input, lr = 0.01):
        X_ = np.concatenate([np.ones(input.shape[0])[...,np.newaxis], input], axis=1)
        grad = X_.T @ self.loss[1](y_true, y_hat) #Cálculo del gradiente (Derivada de función de costo respecto de los pesos)
        self.W -= grad * lr    #Actualización de pesos (peso actual - gradiente * learning rate)
    

class MixtureModels(PythonModel):
    def __init__(self) -> None:
        self.params = None
    
    

    # def get_random_psd(self, n):
    #     x = np.random.normal(0, 1, size=(n, n))
    #     return np.dot(x, x.transpose())

    # def initialize_random_params(self, ):
    #     params = {'phi': np.random.uniform(0, 1),
    #             'mu0': np.random.normal(0, 1, size=(2,)),
    #             'mu1': np.random.normal(0, 1, size=(2,)),
    #             'sigma0': self.get_random_psd(2),
    #             'sigma1': self.get_random_psd(2)}
    #     return params


    def learn_params(self, x_labeled, y_labeled):
        n = x_labeled.shape[0]
        phi = x_labeled[y_labeled == 1].shape[0] / n
        mu0 = np.sum(x_labeled[y_labeled == 0], axis=0) / x_labeled[y_labeled == 0].shape[0]
        mu1 = np.sum(x_labeled[y_labeled == 1], axis=0) / x_labeled[y_labeled == 1].shape[0]
        sigma0 = np.cov(x_labeled[y_labeled == 0].T, bias= True)
        sigma1 = np.cov(x_labeled[y_labeled == 1].T, bias=True)
        self.params = {'phi': phi, 'mu0': mu0, 'mu1': mu1, 'sigma0': sigma0, 'sigma1': sigma1}
        return self.params

    def e_step(self, x):
        np.log([stats.multivariate_normal(self.params["mu0"], self.params["sigma0"]).pdf(x),
                stats.multivariate_normal(self.params["mu1"], self.params["sigma1"]).pdf(x)])
        log_p_y_x = np.log([1-self.params["phi"], self.params["phi"]])[np.newaxis, ...] + \
                    np.log([stats.multivariate_normal(self.params["mu0"], self.params["sigma0"]).pdf(x),
                stats.multivariate_normal(self.params["mu1"], self.params["sigma1"]).pdf(x)]).T
        log_p_y_x_norm = logsumexp(log_p_y_x, axis=1)
        return log_p_y_x_norm, np.exp(log_p_y_x - log_p_y_x_norm[..., np.newaxis])


    def m_step(self, x):
        total_count = x.shape[0]
        _, heuristics = self.e_step(x)
        heuristic0 = heuristics[:, 0]
        heuristic1 = heuristics[:, 1]
        sum_heuristic1 = np.sum(heuristic1)
        sum_heuristic0 = np.sum(heuristic0)
        phi = (sum_heuristic1/total_count)
        mu0 = (heuristic0[..., np.newaxis].T.dot(x)/sum_heuristic0).flatten()
        mu1 = (heuristic1[..., np.newaxis].T.dot(x)/sum_heuristic1).flatten()
        diff0 = x - mu0
        sigma0 = diff0.T.dot(diff0 * heuristic0[..., np.newaxis]) / sum_heuristic0
        diff1 = x - mu1
        sigma1 = diff1.T.dot(diff1 * heuristic1[..., np.newaxis]) / sum_heuristic1
        self.params = {'phi': phi, 'mu0': mu0, 'mu1': mu1, 'sigma0': sigma0, 'sigma1': sigma1}
        


    def get_avg_log_likelihood(self, x):
        loglikelihood, _ = self.e_step(x)
        return np.mean(loglikelihood)


    def run_em(self, x):
        avg_loglikelihoods = []
        while True:
            avg_loglikelihood = self.get_avg_log_likelihood(x)
            avg_loglikelihoods.append(avg_loglikelihood)
            if len(avg_loglikelihoods) > 2 and abs(avg_loglikelihoods[-1] - avg_loglikelihoods[-2]) < 0.0001:
                break
            self.m_step(x)
        # print("\tphi: %s\n\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s"
        #         % (self.params['phi'], self.params['mu0'], self.params['mu1'], self.params['sigma0'], self.params['sigma1']))
        # return forecasts, posterior, avg_loglikelihoods
    
    def predict(self, X):
        _, post = self.e_step(X)
        forecast = np.argmax(post, axis=1)
        return forecast



