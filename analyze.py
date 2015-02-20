#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from collections import Counter
import sys
import pickle

# Parameters
params = { 'N_sub': 300,
	   'B_tags': 5,
	   'reg_C': 10.0 }

# Set random seed for reproducible output
np.random.seed(137)

# Load pickled data
source_dir = sys.argv[1]
input = open('%s/parse.pkl' % source_dir, 'rb')
data = pickle.load(input)
tags, scp_tags, links = Counter(data['tags']), data['scp_tags'], data['links']

# Restrict to subset of most common tags
tags = tags.most_common(params['B_tags'])
print 'Tags used: ' + ' '.join([t for (t,c) in tags])

# Define mapping from covariates and parameters alpha, beta, kappa to
# edge probabilties
def edge_probabilities(alpha, beta, kappa, x):
    N = x.shape[0]
    logit_P = np.zeros((N,N))
    for i in range(N):
        logit_P[i,:] += alpha[0,i]
    for j in range(N):
        logit_P[:,j] += alpha[0,j]
    logit_P += np.dot(x, beta)
    logit_P += kappa

    return 1.0 / (np.exp(-logit_P) + 1.0)

# Define negative log-likelihood
def neg_log_likelihood(alpha, beta, kappa, A, x):
    P = edge_probabilities(alpha, beta, kappa, x)
    return -(np.sum(np.log(P ** A) + np.log((1.0 - P) ** (1.0 - A))) -
	     np.sum(np.log(np.diag(P) ** np.diag(A)) +
		    np.log((1.0 - np.diag(P)) ** (1.0 - np.diag(A)))))

# Procedure to find MLE via logistic regression
def infer(A, x, fit_alpha = False):
    N = A.shape[0]
    B = x.shape[2]
    offdiagonal = -np.diag([True] * N).reshape((N*N,))

    lr = LogisticRegression(fit_intercept = True,
                            C = params['reg_C'], penalty = 'l2')
    y = A.reshape((N*N,))[offdiagonal]
    if fit_alpha:
        Phi = np.zeros((N*N,(B + 2*N)))
    else:
        Phi = np.zeros((N*N,B))
    for b in range(B):
        Phi[:,b] = x[:,:,b].reshape((N*N,))
    if fit_alpha:
        for i in range(N):
            phi_row = np.zeros((N,N))
            phi_row[i,:] = 1.0
            Phi[:,B + i] = phi_row.reshape((N*N,))
	for j in range(N):
            phi_col = np.zeros((N,N))
            phi_col[:,j] = 1.0
            Phi[:,B + N + j] = phi_col.reshape((N*N,))
    Phi = Phi[offdiagonal]
    lr.fit(Phi, y)
    coefs = lr.coef_[0]
    intercept = lr.intercept_[0]

    alpha = np.zeros((2,N))
    out = {'alpha': alpha, 'beta': coefs[0:B], 'kappa': intercept}
    if fit_alpha:
        out['alpha'][0] = coefs[B:(B + N)]
        out['alpha'][1] = coefs[(B + N):(B + 2*N)]
    return out

# Procedure for extracting random subnetwork
def subnetwork(n):
    inds = np.arange(N)
    np.random.shuffle(inds)
    sub = inds[0:n]

    A_sub = A[sub][:,sub]
    x_sub = x[sub][:,sub]
    return A_sub, x_sub

# Construct connectivity matrix and covariate design matrix
nodes = scp_tags.keys()
N = len(nodes)
A = np.zeros((N,N), dtype = np.bool)
for i in range(N):
    i_links = links[nodes[i]]
    for j in range(N):
        if nodes[j] in i_links:
            A[i,j] = True
B = 2 * params['B_tags']
x = np.zeros((N,N,B))
for b in range(params['B_tags']):
    tag, c = tags[b]
    for i in range(N):
        if tag in scp_tags[nodes[i]]:
            x[i,:,b] = True
for b in range(params['B_tags']):
    tag, c = tags[b]
    for j in range(N):
        if tag in scp_tags[nodes[j]]:
            x[:,j,b+params['B_tags']] = True

# Fit model to random subnetwork
A_sub, x_sub = subnetwork(params['N_sub'])
fit = infer(A_sub, x_sub, fit_alpha = True)

print fit['beta'][0:params['B_tags']]
print fit['beta'][params['B_tags']:B]
plt.figure()
plt.hist(fit['alpha'][0], bins = 50)
plt.show()
