import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.tree import DecisionTreeClassifier
import mlflow
from mlflow.pyfunc import PythonModel





def metrics(y_true, y_hat):
        y_t = y_hat[y_hat==y_true] #Trues
        y_n = y_hat[y_hat!=y_true] #Negatives

        false_negatives = len(y_n[y_n==0])
        false_positives = len(y_n[y_n==1])
        true_positives = len(y_t[y_t==1])

        recall = true_positives/(true_positives + false_negatives)
        precision = true_positives/(true_positives + false_positives)
        f1 = recall/(precision + recall)


        return {'recall':recall, 'precision': precision, 'f1': f1}

def get_feature_distribution(x, y):
    u0 = []
    u1 = []

    for i, e in enumerate(y):
        if e == 0:
            u0.append(x[i])
        else:
            u1.append(x[i])

    mean0 = np.mean(u0)
    std0 = np.std(u0)

    mean1 = np.mean(u1)
    std1 = np.std(u1)

    params = {
        'mu0':mean0,
        'std0':std0,
        'mu1':mean1,
        'std1':std1
    }
    
    return params

g = lambda x, mu, sigma: (1/(sigma * np.sqrt(2 * np.pi))) * np.exp((-1/2) * ((x - mu)/sigma)**2)

def overlapping(mu0, mu1, v0, v1):
    mid = ((mu0 - mu1)/2) + mu1
    alpha = 0
    beta = 0
    
    if mu0 < mu1:
        for e0, e1 in zip(v0, v1):
            if e0 >= mid:
                beta += e0
            if e1 <= mid:
                alpha += e1
    else:
        for e0, e1 in zip(v0, v1):
            if e0 <= mid:
                alpha += e0
            if e1 >= mid:
                beta += e1

    return alpha + beta

def feature_performance(x, y, mu0, sigma0, mu1, sigma1, linspace_lim = 1.5, plot=False):
    
    a = np.linspace(-linspace_lim, linspace_lim, 200)
    b = g(a, mu0, sigma0)
    b_ = g(a, mu1, sigma1)
    error = overlapping(mu0, mu1, b, b_)
    mean_diff = np.abs(mu0 - mu1)

    if plot:
        fig, ax = plt.subplots()
        
        ax.grid(True)
        ax.scatter(x[y==0], y[y==0], label='Class-0 Samples', alpha=0.3)
        ax.scatter(x[y==1], y[y==1], label='Class-1 Samples', alpha=0.3)
        # plt.scatter(c_x0, c_y0, label='Class-0 Samples', alpha=0.3)
        # plt.scatter(c_x1, c_y1, label='Class-1 Samples', alpha=0.3)
        ax.plot(a, b, color='skyblue', linewidth=2, label='Class-0 Distribution')
        ax.plot(a, b_, color='salmon', linewidth=2, label='Class-1 Distribution')
        
        ax.set_xlabel('x')
        ax.set_ylabel('P(x)')
        ax.set_title('Feature-Class Distribution')
        ax.legend(loc='best')
        #plt.savefig(f'features/feature_{time.time()}.png', dpi=300)
        return error, mean_diff, fig
 


    
    return error, mean_diff

def data_augmentation(x, y):
    class_difference = np.abs(y[y==1].shape[0] - y[y==0].shape[0])
    alt_x = []
    for i in range(x.shape[1]):
        params = get_feature_distribution(x[:,i], y)
        x_aug = np.concatenate([x[:,i], np.random.normal(params['mu0'], params['std0'], size=class_difference)])
        alt_x.append(x_aug)

    y_aug = np.concatenate([y, np.zeros(class_difference)])
    
    return np.array(alt_x).swapaxes(0,1), y_aug
    