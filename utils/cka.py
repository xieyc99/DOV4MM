import math
import numpy as np
import torch

def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH

def centering_torch(K):
    n = K.size(0)
    unit = torch.ones([n, n]).cuda()
    I = torch.eye(n).cuda()
    H = I - unit / n

    return torch.mm(torch.mm(H, K), H)

def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX

def rbf_torch(X, sigma=None):
    GX = torch.mm(X, X.T)
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = torch.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX

def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))

def kernel_HSIC_torch(X, Y, sigma):
    return torch.sum(centering_torch(rbf_torch(X, sigma)) * centering_torch(rbf_torch(Y, sigma)))

def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))

def linear_HSIC_torch(X, Y):
    L_X = torch.mm(X, X.T)
    L_Y = torch.mm(Y, Y.T)
    return torch.sum(centering_torch(L_X) * centering_torch(L_Y))

def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)

def linear_CKA_torch(X, Y):
    hsic = linear_HSIC_torch(X, Y)
    var1 = torch.sqrt(linear_HSIC_torch(X, X))
    var2 = torch.sqrt(linear_HSIC_torch(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)

def kernel_CKA_torch(X, Y, sigma=None):
    hsic = kernel_HSIC_torch(X, Y, sigma)
    var1 = torch.sqrt(kernel_HSIC_torch(X, X, sigma))
    var2 = torch.sqrt(kernel_HSIC_torch(Y, Y, sigma))

    return hsic / (var1 * var2)