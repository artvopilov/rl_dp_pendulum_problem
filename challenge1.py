"""
Submission template for Programming Challenge 1: Dynamic Programming.
"""

from useful_functions_pendulum import collect_training_data, discretize_states, discritize_actions, \
    create_lookup_table, obs_to_discrete_state_index
from nn_models import train_models as train_nn_models
from rf_models import train_models as train_rf_models
from value_iteration import value_iteration_run, value_iteration_run_with_loops
from policy_iteration import policy_iteration_run, policy_iteration_run_with_loops
import numpy as np
import torch

info = dict(
    group_number=None,
    authors="Artem Vopilov; Tomash Pinto, Pascal",
    description="CONFIG can be used to adjust the model and algorithms. By default Neural Network model and Policy "
                "Iteration algorithm are used as they turned out to be the most efficient. It is also possible to use "
                "Random Forest model and Value Iteration algorithm. Speed states for using algorithm either with "
                "python loops or without python loops which is faster. As for number of discretized states and number "
                "of discretized actions, you can increase them if the size of your RAM is sufficient.")

CONFIG = {
    'model_type': 'nn',
    'n_states': 83,
    'n_actions': 71,
    'algorithm': {
        'type': 'pi',
        'speed': 'fast'
    }
}


def get_model(env, max_num_samples):
    """
    Sample up to max_num_samples transitions (s, a, s', r) from env
    and fit a parametric model s', r = f(s, a).

    :param env: gym.Env
    :param max_num_samples: maximum number of calls to env.step(a)
    :return: function f: s, a -> s', r
    """
    model_type = CONFIG['model_type']
    # choose whether to train Neural Network or Random forest according to CONFIG
    if model_type == 'nn':
        train_model = train_nn_models
    else:
        train_model = train_rf_models

    data_x, data_y, _ = collect_training_data(env, iterations_n=max_num_samples)
    model, _ = train_model(data_x, data_y, env.spec.id, samples_n_start=max_num_samples - 1)

    # this function is an interface to the original model
    def get_predictions(obs, act):
        if model_type == 'nn':
            with torch.no_grad():
                # predict for one sample
                if len(obs.shape) == 1:
                    pred = model(torch.from_numpy(np.append(obs, act)).float()).numpy()
                    # state and reward
                    return pred[:-1], pred[-1]
                # predict for several samples
                else:
                    if len(act.shape) == 1:
                        # actions should be in a column, so that it is possible to concanate them with states
                        act = act.reshape(act.size, 1)
                    pred = model(torch.from_numpy(np.append(obs, act, axis=1)).float()).numpy()
                    # states and rewards
                    return pred[:, :-1], pred[:, -1]
        if model_type == 'rf':
            # predict for one sample
            if len(obs.shape) == 1:
                pred = model.predict(np.expand_dims(np.append(obs, act), axis=0))
                # state and reward
                return pred[0][:-1], pred[0][-1]
            # predict for several samples
            else:
                if len(act.shape) == 1:
                    # actions should be in a column, so that it is possible to concanate them with states
                    act = act.reshape(act.size, 1)
                pred = model.predict(np.append(obs, act, axis=1))
                # states and rewards
                return pred[:, :-1], pred[:, -1]

    return get_predictions


def get_policy(model, observation_space, action_space):
    """
    Perform dynamic programming and return the optimal policy.

    :param model: function f: s, a -> s', r
    :param observation_space: gym.Space
    :param action_space: gym.Space
    :return: function pi: s -> a
    """
    n_states = CONFIG['n_states']
    n_actions = CONFIG['n_actions']
    algorithm_type = CONFIG['algorithm']['type']
    algorithm_speed = CONFIG['algorithm']['speed']

    disc_states = discretize_states(observation_space, n_states)
    disc_actions = discritize_actions(action_space, n_actions)

    # whether to use algorithm with loops or with vectors according to CONFIG
    if algorithm_speed == 'fast':
        lookup_t_r, lookup_t_s = create_lookup_table(model, disc_states, disc_actions, observation_space)
        if algorithm_type == 'pi':
            optimal_policy = policy_iteration_run(observation_space, action_space, disc_states, disc_actions,
                                                  lookup_t_s, lookup_t_r, n_states=n_states, n_actions=n_actions)

        elif algorithm_type == 'vi':
            optimal_policy = value_iteration_run(observation_space, disc_states, disc_actions, lookup_t_s, lookup_t_r,
                                                 n_states=n_states)
        else:
                raise NotImplementedError(f'Algorithm: {algorithm_type}: Try vi or pi')

    else:
        if algorithm_type == 'vi':
            optimal_policy = value_iteration_run_with_loops(observation_space, disc_states, disc_actions, model,
                                                            n_states, n_actions)
        elif algorithm_type == 'pi':
            optimal_policy = policy_iteration_run_with_loops(observation_space, disc_states, disc_actions, model,
                                                             n_states, n_actions)
        else:
            raise NotImplementedError(f'Algorithm: {algorithm_type}: Try vi or pi')

    # return interface to policy
    # obs_to_discrete_state_index converts observation to index of discretized state, so that a proper action can be
    # chosen from policy
    return lambda obs: [optimal_policy[obs_to_discrete_state_index(obs, observation_space, n_states)]]
