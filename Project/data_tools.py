import numpy as np
import matplotlib.pyplot as plt
import time
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
    
    # if outlier_limit == 0:
    #     print('>No outlier limit')
    #     outlier_limit = 10
    # else:
    #     print(f'>Limiting the outliers til\' {outlier_limit}σ')
    a = np.linspace(-linspace_lim, linspace_lim, 200)
    b = g(a, mu0, sigma0)
    b_ = g(a, mu1, sigma1)
    error = overlapping(mu0, mu1, b, b_)
    mean_diff = np.abs(mu0 - mu1)
    
    # a_x0 = x[y==0]
    # b_x0 = a_x0[a_x0>=(mu0 - outlier_limit * sigma0)]
    # c_x0 = b_x0[b_x0<=(mu0 + outlier_limit * sigma0)]

    # a_y0 = y[y==0]
    # b_y0 = a_y0[a_x0>=(mu0 - outlier_limit * sigma0)]
    # c_y0 = b_y0[b_x0<=(mu0 + outlier_limit * sigma0)]

    # a_x1 = x[y==1]
    # b_x1 = a_x1[a_x1>=(mu1 - outlier_limit* sigma1)]
    # c_x1 = b_x1[b_x1<=(mu1 + outlier_limit * sigma1)]

    # a_y1 = y[y==1]
    # b_y1 = a_y1[a_x1>=(mu1 - outlier_limit * sigma1)]
    # c_y1 = b_y1[b_x1<=(mu1 + outlier_limit * sigma1)]


    if plot:
        plt.grid(True)
        plt.scatter(x[y==0], y[y==0], label='Class-0 Samples', alpha=0.3)
        plt.scatter(x[y==1], y[y==1], label='Class-1 Samples', alpha=0.3)
        # plt.scatter(c_x0, c_y0, label='Class-0 Samples', alpha=0.3)
        # plt.scatter(c_x1, c_y1, label='Class-1 Samples', alpha=0.3)
        plt.plot(a, b, color='skyblue', linewidth=2, label='Class-0 Distribution')
        plt.plot(a, b_, color='salmon', linewidth=2, label='Class-1 Distribution')
        plt.xlabel('x')
        plt.ylabel('P(x)')
        plt.title('Feature-Class Distribution')
        plt.legend(loc='best')
        #plt.savefig(f'features/feature_{time.time()}.png', dpi=300)
        plt.show()

    
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
    