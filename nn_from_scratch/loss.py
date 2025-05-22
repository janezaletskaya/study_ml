import numpy as np


class Criterion(object):
    def __init__(self):
        self.output = None
        self.gradInput = None

    def forward(self, input, target):
        """
            Given an input and a target, compute the loss function
            associated to the criterion and return the result.

            For consistency this function should not be overrided,
            all the code goes in `updateOutput`.
        """
        return self.updateOutput(input, target)

    def backward(self, input, target):
        """
            Given an input and a target, compute the gradients of the loss function
            associated to the criterion and return the result.

            For consistency this function should not be overrided,
            all the code goes in `updateGradInput`.
        """
        return self.updateGradInput(input, target)

    def updateOutput(self, input, target):
        """
        Function to override.
        """
        return self.output

    def updateGradInput(self, input, target):
        """
        Function to override.
        """
        return self.gradInput

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want
        to have readable description.
        """
        return "Criterion"


class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()

    def updateOutput(self, input, target):
        self.output = np.sum(np.power(input - target, 2)) / input.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        self.gradInput = (input - target) * 2 / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "MSECriterion"


class ClassNLLCriterionUnstable(Criterion):
    EPS = 1e-15

    def __init__(self):
        super(ClassNLLCriterionUnstable, self).__init__()

    def updateOutput(self, input, target):
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
        batch_size = input.shape[0]
        target_idx = np.argmax(target, axis=1)
        loss_log = -1 * np.log(input_clamp[np.arange(batch_size), target_idx])
        self.output = np.mean(loss_log)

        return self.output

    def updateGradInput(self, input, target):
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
        batch_size = input.shape[0]
        target_idx = np.argmax(target, axis=1)
        self.gradInput = np.zeros_like(input)
        self.gradInput[np.arange(batch_size), target_idx] = (-1 / input_clamp[
            np.arange(batch_size), target_idx]) / batch_size

        return self.gradInput

    def __repr__(self):
        return "ClassNLLCriterionUnstable"


class ClassNLLCriterion(Criterion):
    def __init__(self):
        super(ClassNLLCriterion, self).__init__()

    def updateOutput(self, input, target):
        batch_size = input.shape[0]
        target_idx = np.argmax(target, axis=1)
        loss = -1 * input[np.arange(batch_size), target_idx]
        self.output = np.mean(loss)
        return self.output

    def updateGradInput(self, input, target):
        batch_size = input.shape[0]
        target_idx = np.argmax(target, axis=1)
        self.gradInput = np.zeros_like(input)
        self.gradInput[np.arange(batch_size), target_idx] = (-1) / batch_size

        return self.gradInput

    def __repr__(self):
        return "ClassNLLCriterion"
