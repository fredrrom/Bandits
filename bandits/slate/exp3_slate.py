import numpy as np

# Policy
def exp3_next_action(S, c):
    P = np.exp(c*S)/np.sum(np.exp(c*S))
    return np.random.choice(len(P),5,p=P,replace=False), P

# Action weight update
def update_importance_weighted(S,arm,reward,P):
    S[arm] = S[arm] + reward/P[arm]
    return S

# Bandit interaction
def Exp3(k,reward,horizon,eta):
    all_rewards = []
    S = np.zeros(k)
    for t in range(horizon):
        As, P = exp3_next_action(S, eta)
        X, A = reward(t,As)
        if X is not None:
            S = update_importance_weighted(S,A,X,P)
            all_rewards.append(X)
    return all_rewards