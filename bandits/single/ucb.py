import numpy as np

# Policy
def UCB_next_action(t, Q, N, c):
    min_action = np.argmin(N)
    if N[min_action] == 0:
        next_action = min_action
    else:
        next_action = np.argmax(Q+c*np.sqrt(np.log(t)/N))
    return next_action

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
def UCB(k,reward,horizon,c=2,step_size=sample_average_step):
    all_rewards = []
    Q = np.zeros(k)
    N = np.zeros(k)
    t_hit = 0
    for t in range(horizon):
        A = UCB_next_action(t_hit,Q,N,c)
        X = reward(t,A)
        if X is not None:
            N[A] += 1
            Q = update_weighted_average(t_hit,Q,A,N[A],X,step_size)
            all_rewards.append(X)
            t_hit += 1
    return all_rewards