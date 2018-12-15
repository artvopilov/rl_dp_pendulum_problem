import numpy as np
import matplotlib.pyplot as plt


def select_indexes_for_pend(data):
    indx1 = np.where((data[:, 0] > -np.pi / 3) & (data[:, 0] < np.pi / 3))
    indx2 = np.where((data[:, 0] > np.pi / 3) | (data[:, 0] < -np.pi / 3))
    return indx1, indx2


def collect_training_data(env, iterations_n=10000, render=False, select_data=False, select_coef=1, from_top=5000,
                          from_bottom=5000):
    # gaussian_actions = np.random.normal(0, 1, iterations_n * (1 + (select_coef - 1) * select_data))
    gaussian_actions = np.random.uniform(env.action_space.low, env.action_space.high,
                                         iterations_n * (1 + (select_coef - 1) * select_data))

    training_data_x = []
    training_data_y = []
    full_collection = []
    observation = env.reset()
    done = False
    for action in gaussian_actions:
        if done:
            observation = env.reset()
        if render:
            env.render()
        observation_next, reward, done, info = env.step([action])

        training_data_x.append(np.append(observation, action))
        training_data_y.append(np.append(observation_next, reward))

        full_collection.append(np.array([observation_next, observation, action, reward]))
        observation = observation_next
    env.close()

    training_data_x, training_data_y = np.array(training_data_x), np.array(training_data_y)

    if select_data:
        indexes_1, indexes_2 = select_indexes_for_pend(training_data_x[:, :-1])
        training_data_x = np.concatenate((training_data_x[indexes_1][:from_top],
                                          training_data_x[indexes_2][:from_bottom]), axis=0)
        training_data_y = np.concatenate((training_data_y[indexes_1][:from_top],
                                          training_data_y[indexes_2][:from_bottom]))

    return training_data_x, training_data_y, full_collection


def create_lookup_table(model, states, actions, observation_space):
    obs_l = observation_space.low
    obs_h = observation_space.high

    # number of all possible states
    disc_states_num = len(states)
    # repeat each raw in states matrix, thus to combine each possible (discrete) state with each possible (discrete)
    # action
    states = np.repeat(states, len(actions), axis=0)
    # add to actions array as many same arrays as many discrete states exist, then to combine states with actions
    actions = np.tile(actions, (disc_states_num, 1))
    # reshape actions array to append then to each raw in states matrix one element to make predictions
    if actions.shape[1] != 1:
        actions = actions.reshape(actions.size, 1)

    lookup_table_states, lookup_table_rewards = model(states, actions)
    # make sure that predicted states are within observation space
    lookup_table_states = np.minimum(np.maximum(lookup_table_states, obs_l), obs_h)

    return lookup_table_rewards, lookup_table_states


def discretize_states(observation_space, n_states):
    obs_l = observation_space.low
    obs_h = observation_space.high

    if len(obs_l) == 2:
        # discretize each observation of a state separately
        obs0 = np.around(np.linspace(obs_l[0], obs_h[0], n_states), decimals=4)
        obs1 = np.around(np.linspace(obs_l[1], obs_h[1], n_states), decimals=4)

        obs0_for_conc = np.repeat(obs0, n_states)
        obs1_for_conc = np.tile(obs1, n_states)

        # all possible states with respective degree of discretization
        states = np.concatenate((obs0_for_conc.reshape(obs0_for_conc.size, 1),
                                 obs1_for_conc.reshape(obs1_for_conc.size, 1)),
                                axis=1)

    elif len(obs_l) == 3:
        # discretize each observation of a state separately
        obs0 = np.around(np.linspace(obs_l[0], obs_h[0], n_states), decimals=4)
        obs1 = np.around(np.linspace(obs_l[1], obs_h[1], n_states), decimals=4)
        obs2 = np.around(np.linspace(obs_l[2], obs_h[2], n_states), decimals=4)

        obs0_for_conc = np.repeat(obs0, np.power(n_states, len(obs_l) - 1))
        obs1_for_conc = np.tile(np.repeat(obs1, np.power(n_states, len(obs_l) - 2)), n_states)
        obs2_for_conc = np.tile(obs2, np.power(n_states, len(obs_l) - 1))

        # all possible states with respective degree of discretization
        states = np.concatenate((obs0_for_conc.reshape(obs0_for_conc.size, 1),
                                 obs1_for_conc.reshape(obs1_for_conc.size, 1),
                                 obs2_for_conc.reshape(obs2_for_conc.size, 1)),
                                axis=1)
    else:
        raise NotImplementedError(f'Number of observations for a state {len(obs_l)}: not supported')

    return states


def obs_to_discrete_state_index(obs, observation_space, n_states):
    obs_l = observation_space.low
    obs_h = observation_space.high

    if len(obs_l) == 2:
        # calculate steps for each observation in a state
        obs0_s = np.linspace(obs_l[0], obs_h[0], n_states, retstep=True)[1]
        obs1_s = np.linspace(obs_l[1], obs_h[1], n_states, retstep=True)[1]
        obs = np.maximum(np.minimum(obs, obs_h), obs_l)

        d_space = np.array([obs0_s, obs1_s])
    elif len(obs_l) == 3:
        # calculate steps for each observation in a state
        obs0_s = np.linspace(obs_l[0], obs_h[0], n_states, retstep=True)[1]
        obs1_s = np.linspace(obs_l[1], obs_h[1], n_states, retstep=True)[1]
        obs2_s = np.linspace(obs_l[2], obs_h[2], n_states, retstep=True)[1]
        obs = np.maximum(np.minimum(obs, obs_h), obs_l)

        d_space = np.array([obs0_s, obs1_s, obs2_s])
    else:
        raise NotImplementedError(f'Number of observations for a state {len(obs_l)}: not supported')

    discrete_states_numbers = np.array(np.round((obs - obs_l) / d_space), dtype=int)
    factors = np.array([n_states ** i for i in range(len(obs_l) - 1, -1, -1)])
    if len(discrete_states_numbers.shape) == 1:
        return np.sum(discrete_states_numbers * factors)
    return np.sum(discrete_states_numbers * factors, axis=1)


def discritize_actions(action_space, n_actions):
    act_l = action_space.low
    act_h = action_space.high

    actions = np.around(np.linspace(act_l[0], act_h[0], n_actions), decimals=3)

    return actions


def run_episode(env, policy, render=False):
    obs = env.reset()
    total_reward = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(policy(obs))
        total_reward += reward
        if done:
            break
    env.close()
    return total_reward


def evaluate_policy(env, policy, n=100, plot=True, alg_name='Algorithm'):
    scores = [run_episode(env, policy, False) for _ in range(n)]
    if plot:
        plt.figure(alg_name)
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.plot(range(n), scores)
        plt.show()
    return np.mean(scores)
