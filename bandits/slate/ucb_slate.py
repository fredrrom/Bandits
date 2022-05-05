import numpy as np

# Not complete

# Policy
def UCB_next_action(t, Q, N, c):
    next_actions = np.argwhere(N == 0).flatten()
    #print(next_actions)
    #if next_actions.size > 5:
        #return np.random.choice(next_actions,5)
    if next_actions.size == 0:
        print(np.argpartition(Q+c*np.sqrt(np.log(t)/(N)), -5)[-5:])
        print(Q+c*np.sqrt(np.log(t)/(N)))
        return np.argpartition(Q+c*np.sqrt(np.log(t)/(N)), -5)[-5:]
    return next_actions

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
        As = UCB_next_action(t_hit,Q,N,c)
        X, A = reward(t,As)
        if X is not None:
            N[A] += 1
            Q = update_weighted_average(t_hit,Q,A,N[A],X,step_size)
            all_rewards.append(X)
            t_hit += 1
    return all_rewards