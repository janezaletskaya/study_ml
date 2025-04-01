from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1
    s0: float = 1e-3
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class LossFunctionCalculator:
    """
    Класс для вычисления значений функций потерь и их градиентов
    """

    @staticmethod
    def calc_loss(x: np.ndarray, y: np.ndarray, w: np.ndarray,
                  loss_function: LossFunction, delta: float = 1.0) -> float:
        """
        Вычисляет значение функции потерь

        :param x: features array
        :param y: targets array
        :param w: weights array
        :param loss_function: loss function
        :param delta: parameter of Huber Loss
        :return: function value
        """
        predictions = x.dot(w)
        diff = predictions - y

        if loss_function == LossFunction.MSE:
            return np.mean(diff ** 2)

        elif loss_function == LossFunction.MAE:
            return np.mean(np.abs(diff))

        elif loss_function == LossFunction.LogCosh:
            return np.mean(np.log(np.cosh(diff)))

        elif loss_function == LossFunction.Huber:
            mask = np.abs(diff) <= delta
            squared_loss = 0.5 * (diff ** 2)
            linear_loss = delta * (np.abs(diff) - 0.5 * delta)
            return np.mean(mask * squared_loss + (~mask) * linear_loss)

        else:
            raise ValueError(f"Неизвестная функция потерь: {loss_function}")

    @staticmethod
    def calc_gradient(x: np.ndarray, y: np.ndarray, w: np.ndarray,
                      loss_function: LossFunction, delta: float = 1.0) -> np.ndarray:
        """
        Вычисляет градиент функции потерь по весам

        :param x: features array
        :param y: targets array
        :param w: weights array
        :param loss_function: loss function
        :param delta: parameter of Huber Loss
        :return: gradient
        """
        n = x.shape[0]
        predictions = x.dot(w)
        diff = predictions - y

        if loss_function == LossFunction.MSE:
            return (1 / n) * x.T.dot(diff)

        elif loss_function == LossFunction.MAE:
            return (1 / n) * x.T.dot(np.sign(diff))

        elif loss_function == LossFunction.LogCosh:
            return (1 / n) * x.T.dot(np.tanh(diff))

        elif loss_function == LossFunction.Huber:
            mask = np.abs(diff) <= delta
            gradient_values = mask * diff + (~mask) * (delta * np.sign(diff))
            return (1 / n) * x.T.dot(gradient_values)

        else:
            raise ValueError(f"Неизвестная функция потерь: {loss_function}")


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, dimension: int, lambda_: float = 1e-1, loss_function: LossFunction = LossFunction.MSE,
                 huber_delta: float = 1.0):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        self.w: np.ndarray = np.random.rand(dimension)
        # self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.lr = lambda: lambda_
        self.loss_function: LossFunction = loss_function
        self.huber_delta: float = huber_delta
        self.loss_calculator = LossFunctionCalculator()

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Template for calc_gradient function
        Calculate gradient of loss function with respect to weights
        :param x: features array
        :param y: targets array
        :return: gradient: np.ndarray
        """
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for x and y with our weights
        :param x: features array
        :param y: targets array
        :return: loss: float
        """
        return self.loss_calculator.calc_loss(
            x, y, self.w, self.loss_function, self.huber_delta
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for x
        :param x: features array
        :return: prediction: np.ndarray
        """
        return x.dot(self.w)


class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights with respect to gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        grad_step = self.lr() * gradient
        self.w -= grad_step

        return -grad_step

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.loss_calculator.calc_gradient(
            x, y, self.w, self.loss_function, self.huber_delta
        )


class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):
        """
        :param batch_size: batch size (int)
        """
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not isinstance(y, np.ndarray):
            y = y.values

        n = x.shape[0]
        batch_indices = np.random.randint(0, n, size=self.batch_size)
        x_batch = x[batch_indices]
        y_batch = y[batch_indices]

        return self.loss_calculator.calc_gradient(
            x_batch, y_batch, self.w, self.loss_function, self.huber_delta
        )


class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights with respect to gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        new_h = self.alpha * self.h + self.lr() * gradient
        self.h = new_h
        self.w -= self.h

        return -self.h


class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights & params
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        new_m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        new_v = self.beta_2 * self.v + (1 - self.beta_2) * gradient ** 2

        self.iteration += 1

        m_hat = new_m / (1 - self.beta_1 ** self.iteration)
        v_hat = new_v / (1 - self.beta_2 ** self.iteration)

        weight_diff = -self.lr() * m_hat / (np.sqrt(v_hat) + self.eps)

        self.m = new_m
        self.v = new_v
        self.w += weight_diff

        return weight_diff


class NAdam(VanillaGradientDescent):

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights & params using Nesterov-accelerated Adam
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        new_m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        new_v = self.beta_2 * self.v + (1 - self.beta_2) * gradient ** 2

        self.iteration += 1

        m_hat = new_m / (1 - self.beta_1 ** self.iteration)
        v_hat = new_v / (1 - self.beta_2 ** self.iteration)

        m_nesterov = self.beta_1 * m_hat + (1 - self.beta_1) * gradient / (1 - self.beta_1 ** self.iteration)
        weight_diff = -self.lr() * m_nesterov / (np.sqrt(v_hat) + self.eps)

        self.m = new_m
        self.v = new_v
        self.w += weight_diff

        return weight_diff


class AMSGrad(VanillaGradientDescent):
    """
    AMSGrad optimization algorithm
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE,
                 huber_delta: float = 1.0):
        super().__init__(dimension, lambda_, loss_function, huber_delta)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)
        self.v_max: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights using AMSGrad
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """

        new_m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        new_v = self.beta_2 * self.v + (1 - self.beta_2) * gradient ** 2

        self.iteration += 1

        m_hat = new_m / (1 - self.beta_1 ** self.iteration)
        new_v_max = np.maximum(self.v_max, new_v)

        weight_diff = -self.lr() * m_hat / (np.sqrt(new_v_max) + self.eps)

        self.m = new_m
        self.v = new_v
        self.v_max = new_v_max
        self.w += weight_diff

        return weight_diff


class AdaMax(VanillaGradientDescent):
    """
    AdaMax optimization algorithm
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE,
                 huber_delta: float = 1.0):
        super().__init__(dimension, lambda_, loss_function, huber_delta)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.u: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights using AdaMax
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """

        new_m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        new_u = np.maximum(self.beta_2 * self.u, np.abs(gradient))

        self.iteration += 1

        m_hat = new_m / (1 - self.beta_1 ** self.iteration)

        weight_diff = -self.lr() * m_hat / (new_u + self.eps)

        self.m = new_m
        self.u = new_u
        self.w += weight_diff

        return weight_diff


class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """
        base_gradient = self.loss_calculator.calc_gradient(
            x, y, self.w, self.loss_function, self.huber_delta
        )
        l2_gradient = self.w
        l2_gradient[0] = 0

        return base_gradient + l2_gradient * self.mu


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """


class NAdamReg(BaseDescentReg, NAdam):
    """
    Adaptive gradient algorithm with Nesterov moments with regularization class
    """


class AMSGradReg(BaseDescentReg, AMSGrad):
    """
    AMSGrad optimization algorithm with regularization class
    """


class AdaMaxReg(BaseDescentReg, AdaMax):
    """
    AdaMax optimization algorithm with regularization class
    """


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)
    loss_function = descent_config.get('loss_function', LossFunction.MSE)
    huber_delta = descent_config.get('huber_delta', 1.0)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg,
        'nadam': NAdam if not regularized else NAdamReg,
        'adamax': AdaMax if not regularized else AdaMaxReg,
        'ams': AMSGrad if not regularized else AMSGradReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]
    kwargs = descent_config.get('kwargs', {}).copy()

    if 'loss_function' not in kwargs:
        kwargs['loss_function'] = loss_function

    if 'huber_delta' not in kwargs and loss_function == LossFunction.Huber:
        kwargs['huber_delta'] = huber_delta

    return descent_class(**kwargs)
