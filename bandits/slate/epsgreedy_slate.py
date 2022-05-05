import numpy as np

# Policy
def Greedy_next_action(Q, eps=0.1):
    if np.random.uniform(0, 1) < eps:
        return np.random.choice(len(Q), 5, replace=False)
    return np.argpartition(Q, -5)[-5:]

# Value updates and step sizes
def update_weighted_average(t, Q, arm, pulls, reward, alpha):
    Q[arm] = Q[arm] + alpha(t,pulls)*(reward - Q[arm])
    return Q

def const_step(*_):
    return 0.1

def sample_average_step(_,pulls):
    return (1/pulls)

def time_decrease_step(t,_):
    return (1/(t+1))

# Bandit interaction
def epsilon_greedy(k,reward,horizon,c=2,step_size=sample_average_step):
    all_rewards = []
    Q = np.zeros(k)
    N = np.zeros(k)
    t_hit = 0
    for t in range(horizon):
        As = Greedy_next_action(Q)
        X, A = reward(t,As)
        if X is not None:
            N[A] += 1
            Q = update_weighted_average(t_hit,Q,A,N[A],X,step_size)
            all_rewards.append(X)
            t_hit += 1
    return all_rewards