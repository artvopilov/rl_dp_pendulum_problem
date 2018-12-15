import numpy as np
from useful_functions_pendulum import obs_to_discrete_state_index


def action_to_discrete_action_index(act, action_space, n_actions):
    act_l = action_space.low
    act_h = action_space.high
    act = np.maximum(act, act_l)
    act = np.minimum(act, act_h)

    d_actions = np.linspace(act_l[0], act_h[0], n_actions, retstep=True)[1]
    discrete_actions_indx = np.array(np.round((act - act_l) / d_actions), dtype=int)
    return discrete_actions_indx.flatten()


def value_policy(policy, values, observation_space, action_space, lookup_t_s, lookup_t_r, states, n_states, n_actions,
                 gamma=.98, eps=1e-4):
    count = 0
    while True:
        count += 1
        # copy values of each step to compare later
        prev_values = np.copy(values)
        # find indexes of actions of corresponding policy to find rewards and states in lookup tables
        act_indexes = action_to_discrete_action_index(policy, action_space, n_actions)
        # lookup tables indexes of corresponding combinations of states and action
        lookup_t_indexes = np.arange(len(states)) * n_actions + act_indexes
        # predicted rewards and states by models in lookup tables
        p_rewards = lookup_t_r[lookup_t_indexes]
        p_states = lookup_t_s[lookup_t_indexes]

        p_state_indexes = obs_to_discrete_state_index(p_states, observation_space, n_states)

        values = p_rewards + prev_values[p_state_indexes] * gamma

        if (np.fabs(values - prev_values) < eps).all():
            print('Values count: {}'.format(count))
            return values


def improve_policy(values, states, actions, observation_space, lookup_t_s, lookup_t_r, n_states, gamma=.98):

    # all the rewards are needed as we have to know rewards from each state applying each action
    p_rewards = lookup_t_r
    # the same, all the predicted states are needed (from each state applying each action)
    p_states_indexes = obs_to_discrete_state_index(lookup_t_s, observation_space, n_states)
    # values from each state with each action
    q_sa = p_rewards + values[p_states_indexes] * gamma
    # matrix of values where each raw is values for each action of a state, num of raw is index of state
    q_sa = q_sa.reshape(len(states), len(actions))

    policy = actions[np.argmax(q_sa, axis=1)]
    return policy


def policy_iteration_run(observation_space, action_space, states, actions, lookup_t_s, lookup_t_r, n_states=60,
                         n_actions=20, max_iter=100):
    # initialize values with zeros and policy with random actions
    values = np.zeros(len(states))
    policy = np.random.normal(0, 1, len(states))
    for i in range(max_iter):
        print('Iter: {}'.format(i))
        new_values = value_policy(policy, values, observation_space, action_space, lookup_t_s, lookup_t_r, states,
                                  n_states, n_actions)
        new_policy = improve_policy(new_values, states, actions, observation_space, lookup_t_s, lookup_t_r, n_states)

        print('Diff policy actions: {}'.format(len(np.where(policy != new_policy)[0])))
        print(np.where(new_policy != policy)[0])
        if np.all(policy == new_policy):
            return policy
        policy = new_policy
        values = new_values
    return policy


def value_policy_with_loops(policy, values, observation_space, model, states, n_states, gamma=.95, eps=1e-4):
    count = 0
    while True:
        count += 1
        # copy values of each step to compare later
        prev_values = np.copy(values)

        for i, state in enumerate(states):
            pred_states, pred_rwds = model(state, policy[i])
            p_state_indexes = obs_to_discrete_state_index(pred_states, observation_space, n_states)
            values[i] = pred_rwds + prev_values[p_state_indexes] * gamma

        if (np.fabs(values - prev_values) < eps).all():
            print('Values count: {}'.format(count))
            return values


def improve_policy_with_loops(values, states, actions, observation_space, model, n_states, n_actions, gamma=.98):
    q_sa = np.zeros((len(states), len(actions)))
    for i, state in enumerate(states):
        state = np.repeat(np.expand_dims(state, axis=0), n_actions, axis=0)
        pred_states, pred_rwds = model(state, actions)
        p_state_indexes = obs_to_discrete_state_index(pred_states, observation_space, n_states)
        q_sa[i] = pred_rwds + values[p_state_indexes] * gamma

    policy = actions[np.argmax(q_sa, axis=1)]
    return policy


def policy_iteration_run_with_loops(observation_space, states, actions, model, n_states=60, n_actions=20, max_iter=100):
    # initialize values with zeros and policy with random actions
    values = np.zeros(len(states))
    policy = np.random.normal(0, 1, len(states))
    for i in range(max_iter):
        print('Iter: {}'.format(i))
        new_values = value_policy_with_loops(policy, values, observation_space, model, states, n_states)
        new_policy = improve_policy_with_loops(new_values, states, actions, observation_space, model, n_states,
                                               n_actions)

        print('Diff policy actions: {}'.format(len(np.where(policy != new_policy)[0])))
        print(np.where(new_policy != policy)[0])
        if np.all(policy == new_policy):
            return policy
        policy = new_policy
        values = new_values
    return policy
