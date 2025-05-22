import numpy as np


def sgd_momentum(variables, gradients, config, state):
    """
    Implements stochastic gradient descent with momentum.

    Each parameter is updated using:
        v = momentum * v + learning_rate * grad
        param = param - v

    Parameters:
    - variables: list of lists of numpy arrays (parameters to optimize)
    - gradients: list of lists of numpy arrays (gradients for each parameter)
    - config: dict with keys:
        - 'learning_rate': float, step size
        - 'momentum': float, momentum factor (typically between 0 and 1)
    - state: dict used to store momentum buffers between steps
    """
    state.setdefault('accumulated_grads', {})

    var_index = 0
    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            old_grad = state['accumulated_grads'].setdefault(var_index, np.zeros_like(current_grad))

            np.add(config['momentum'] * old_grad, config['learning_rate'] * current_grad, out=old_grad)

            current_var -= old_grad
            var_index += 1


def adam_optimizer(variables, gradients, config, state):
    """
    Implements the Adam optimizer.

    Each parameter is updated using:
        m_t = beta1 * m + (1 - beta1) * grad
        v_t = beta2 * v + (1 - beta2) * grad^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        param = param - learning_rate * m_hat / (sqrt(v_hat) + epsilon)

    Parameters:
    - variables: list of lists of numpy arrays (parameters to optimize)
    - gradients: list of lists of numpy arrays (gradients for each parameter)
    - config: dict with keys:
        - 'learning_rate': float, base learning rate
        - 'beta1': float, exponential decay rate for the first moment
        - 'beta2': float, exponential decay rate for the second moment
        - 'epsilon': float, small value to prevent division by zero
    - state: dict used to store optimizer state (m, v, t)
    """
    state.setdefault('m', {})
    state.setdefault('v', {})
    state.setdefault('t', 0)
    state['t'] += 1
    for k in ['learning_rate', 'beta1', 'beta2', 'epsilon']:
        assert k in config, f"Missing key '{k}' in config. Got keys: {list(config.keys())}"

    var_index = 0
    lr_t = config['learning_rate'] * np.sqrt(1 - config['beta2'] ** state['t']) / (1 - config['beta1'] ** state['t'])
    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            var_first_moment = state['m'].setdefault(var_index, np.zeros_like(current_grad))
            var_second_moment = state['v'].setdefault(var_index, np.zeros_like(current_grad))

            np.add(
                config['beta1'] * var_first_moment, (1 - config['beta1']) * current_grad,
                out=var_first_moment
            )

            np.add(
                config['beta2'] * var_second_moment, (1 - config['beta2']) * (current_grad ** 2),
                out=var_second_moment
            )

            current_var -= lr_t * (var_first_moment / (np.sqrt(var_second_moment) + config['epsilon']))

            assert var_first_moment is state['m'].get(var_index)
            assert var_second_moment is state['v'].get(var_index)
            var_index += 1
