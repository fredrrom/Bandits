import numpy as np
from exp3_slate import update_importance_weighted

# Policy
def exp4_next_action(Q, E):
    P = np.matmul(Q,E)
    return np.random.choice(len(P),5,p=P,replace=False), P

# Expert weight update
def update_importance_weighted_with_experts(Q,E,arm,reward,P,c):
    X_hat = update_importance_weighted(np.zeros(E.shape[1]),arm,reward,P)
    X_tilde = np.dot(E,X_hat)
    Q = np.multiply(np.exp(c*X_tilde),Q)/np.sum(np.multiply(np.exp(c*X_tilde),Q))
    return Q

# Bandit interaction
def Exp4(M,reward,horizon,c,advice):
    all_rewards = []
    Q = np.full(M,1/M)
    for t in range(horizon):
        E = advice(t)
        As, P = exp4_next_action(Q,E)
        X, A = reward(t,As)
        if X is not None:
            Q = update_importance_weighted_with_experts(Q,E,A,X,P,c)
            all_rewards.append(X)
    return all_rewards

