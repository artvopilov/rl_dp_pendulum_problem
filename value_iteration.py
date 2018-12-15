import numpy as np
from useful_functions_pendulum import obs_to_discrete_state_index as obs_to_disc_st_inx_pend


def value_iteration_run(observation_space, states, actions, lookup_t_s, lookup_t_r, n_states=60, gamma=0.95, eps=1e-2):

    values = np.zeros(len(states))
    while True:
        # copy values of each step to compare later
        prev_values = np.copy(values)

        # predicted rewards and states by models in lookup tables for each combination of states and actions
        p_rewards = lookup_t_r
        p_states = lookup_t_s
        # calculate indexes of predicted states in the array of discrete states
        p_state_indexes = obs_to_disc_st_inx_pend(p_states, observation_space, n_states)
        # values from each state with each action
        q_sa = p_rewards + prev_values[p_state_indexes] * gamma
        # matrix of values where each raw is values for each action of a state, num of raw is index of state
        q_sa = q_sa.reshape(len(states), len(actions))

        values = np.max(q_sa, axis=1)

        if (np.fabs(values - prev_values) < eps).all():
            # extract policy from final matrix of values of states with given actions
            policy = actions[np.argmax(q_sa, axis=1)]
            return policy
        print('Diff in values:', len(np.where(np.fabs(values - prev_values) >= eps)[0]))


def value_iteration_run_with_loops(observation_space, states, actions, model, n_states=60, n_actions=30, gamma=0.95,
                                   eps=1e-2):
    values = np.zeros(len(states))
    q_sa = np.zeros((len(states), len(actions)))
    while True:
        # copy values of each step to compare later
        prev_values = np.copy(values)

        for i, state in enumerate(states):
            state = np.repeat(np.expand_dims(state, axis=0), n_actions, axis=0)
            pred_states, pred_rwds = model(state, actions)
            p_state_indexes = obs_to_disc_st_inx_pend(pred_states, observation_space, n_states)

            q_sa[i] = pred_rwds + prev_values[p_state_indexes] * gamma
            values[i] = np.max(q_sa[i])

        if (np.fabs(values - prev_values) < eps).all():
            # extract policy from final matrix of values of states with given actions
            policy = actions[np.argmax(q_sa, axis=1)]
            return policy
        print('Diff in values:', len(np.where(np.fabs(values - prev_values) >= eps)[0]))
