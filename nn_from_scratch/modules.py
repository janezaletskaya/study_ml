import numpy as np
import scipy


class Module(object):
    """
    Basically, you can think of a module as of a something (black box)
    which can process `input` data and produce `ouput` data.
    This is like applying a function which is called `forward`:

        output = module.forward(input)

    The module should be able to perform a backward pass: to differentiate the `forward` function.
    More, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule.

        gradInput = module.backward(input, gradOutput)
    """

    def __init__(self):
        self.output = None
        self.gradInput = None
        self.training = True

    def forward(self, input):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        return self.updateOutput(input)

    def backward(self, input, gradOutput):
        """
        Performs a backpropagation step through the module, with respect to the given input.

        This includes
         - computing a gradient w.r.t. `input` (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput)
        return self.gradInput

    def updateOutput(self, input):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which is stored in the `output` field.

        Make sure to both store the data in `output` field and return it.
        """
        pass

    def updateGradInput(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own input.
        This is returned in `gradInput`. Also, the `gradInput` state variable is updated accordingly.

        The shape of `gradInput` is always the same as the shape of `input`.

        Make sure to both store the gradients in `gradInput` field and return it.
        """
        pass

    def accGradParameters(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass

    def zeroGradParameters(self):
        """
        Zeroes `gradParams` variable if the module has params.
        """
        pass

    def getParameters(self):
        """
        Returns a list with its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def getGradParameters(self):
        """
        Returns a list with gradients with respect to its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True

    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want
        to have readable description.
        """
        return "Module"


class Sequential(Module):
    """
         This class implements a container, which processes `input` data sequentially.
         `input` is processed by each module (layer) in self.modules consecutively.
         The resulting array is called `output`.
    """

    def __init__(self):
        super(Sequential, self).__init__()
        self.modules = []
        self.outputs = []

    def add(self, module):
        """
        Adds a module to the container.
        """
        self.modules.append(module)

    def updateOutput(self, input):
        """
        Basic workflow of FORWARD PASS:

            y_0    = module[0].forward(input)
            y_1    = module[1].forward(y_0)
            ...
            output = module[n-1].forward(y_{n-2})


        Just write a little loop.
        """
        self.outputs = []

        self.outputs.append(input)
        cur_output = input
        for module in self.modules:
            cur_output = module.forward(cur_output)
            self.outputs.append(cur_output)

        self.output = cur_output

        return self.output

    def backward(self, input, gradOutput):
        """
        Workflow of BACKWARD PASS:

            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)
            gradInput = module[0].backward(input, g_1)


        !!!

        To ech module you need to provide the input, module saw while forward pass,
        it is used while computing gradients.
        Make sure that the input for `i-th` layer the output of `module[i]` (just the same input as in forward pass)
        and NOT `input` to this Sequential module.

        !!!

        """
        cur_grad = gradOutput
        for i in range(len(self.modules) - 1, -1, -1):
            cur_module, cur_input = self.modules[i], self.outputs[i]
            cur_grad = cur_module.backward(cur_input, cur_grad)

        self.gradInput = cur_grad

        return self.gradInput

    def zeroGradParameters(self):
        for module in self.modules:
            module.zeroGradParameters()

    def getParameters(self):
        """
        Should gather all parameters in a list.
        """
        return [x.getParameters() for x in self.modules]

    def getGradParameters(self):
        """
        Should gather all gradients w.r.t parameters in a list.
        """
        return [x.getGradParameters() for x in self.modules]

    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string

    def __getitem__(self, x):
        return self.modules.__getitem__(x)

    def train(self):
        """
        Propagates training parameter through all modules
        """
        self.training = True
        for module in self.modules:
            module.train()

    def evaluate(self):
        """
        Propagates training parameter through all modules
        """
        self.training = False
        for module in self.modules:
            module.evaluate()


class Linear(Module):
    """
    A module which applies a linear transformation
    A common name is fully-connected layer, InnerProductLayer in caffe.

    The module should work with 2D input of shape (n_samples, n_feature).
    """

    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()

        stdv = 1. / np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size=(n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, input):
        self.input = input
        self.output = np.dot(input, self.W.T) + self.b

        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.dot(gradOutput, self.W)
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        self.gradW += np.dot(gradOutput.T, input)
        self.gradb += np.sum(gradOutput, axis=0)

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' % (s[1], s[0])
        return q


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2d, self).__init__()
        assert kernel_size % 2 == 1, kernel_size

        stdv = 1. / np.sqrt(in_channels)
        self.W = np.random.uniform(-stdv, stdv, size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.random.uniform(-stdv, stdv, size=(out_channels,))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, input):
        pad_size = self.kernel_size // 2
        batch_size, in_channels, h, w = input.shape
        self.output = np.zeros((batch_size, self.out_channels, h, w))

        padded_input = np.pad(input, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode='constant')

        for batch_num in range(batch_size):
            for out_channel_num in range(self.out_channels):
                for in_channel_num in range(self.in_channels):
                    self.output[batch_num, out_channel_num] += scipy.signal.correlate(
                        padded_input[batch_num, in_channel_num], self.W[out_channel_num, in_channel_num], mode='valid'
                    )
                self.output[batch_num, out_channel_num] += self.b[out_channel_num]

        return self.output

    def updateGradInput(self, input, gradOutput):
        batch_size, in_channels, h, w = input.shape
        pad_size = self.kernel_size // 2

        padded = np.zeros((batch_size, in_channels, h + 2 * pad_size, w + 2 * pad_size))

        for batch_num in range(batch_size):
            for in_channel_num in range(in_channels):
                for out_channel_num in range(self.out_channels):
                    rotated_kernel = np.flip(self.W[out_channel_num, in_channel_num], (0, 1))

                    padded[batch_num, in_channel_num] += scipy.signal.correlate(
                        gradOutput[batch_num, out_channel_num],
                        rotated_kernel,
                        mode='full'
                    )

        self.gradInput = padded[:, :, pad_size:h + pad_size, pad_size:w + pad_size]
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        batch_size, in_channels, h, w = input.shape
        pad_size = self.kernel_size // 2

        for batch_num in range(batch_size):
            for out_channel_num in range(self.out_channels):
                for in_channel_num in range(in_channels):
                    padded_input = np.pad(
                        input[batch_num, in_channel_num],
                        ((pad_size, pad_size), (pad_size, pad_size)),
                        mode='constant'
                    )

                    self.gradW[out_channel_num, in_channel_num] += scipy.signal.correlate(
                        padded_input,
                        gradOutput[batch_num, out_channel_num],
                        mode='valid'
                    )

                self.gradb[out_channel_num] += np.sum(gradOutput[batch_num, out_channel_num])

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        q = 'Conv2d %d -> %d' % (s[1], s[0])
        return q


class MaxPool2d(Module):
    def __init__(self, kernel_size):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size

    def updateOutput(self, input):
        batch_size, in_channels, height, width = input.shape
        out_h, out_w = height // self.kernel_size, width // self.kernel_size

        assert height % self.kernel_size == 0
        assert width % self.kernel_size == 0

        self.output = np.zeros((batch_size, in_channels, out_h, out_w))
        self.max_indices = np.zeros((batch_size, in_channels, out_h, out_w), dtype=np.int64)

        for batch_size_num in range(batch_size):
            for in_channel_num in range(in_channels):
                for h_idx in range(out_h):
                    for w_idx in range(out_w):
                        h_start, w_start = h_idx * self.kernel_size, w_idx * self.kernel_size
                        h_end, w_end = h_start + self.kernel_size, w_start + self.kernel_size
                        window = input[batch_size_num, in_channel_num, h_start:h_end, w_start:w_end]

                        max_val = np.max(window)
                        max_idx = np.argmax(window.flatten())

                        self.output[batch_size_num, in_channel_num, h_idx, w_idx] = max_val
                        self.max_indices[batch_size_num, in_channel_num, h_idx, w_idx] = max_idx

        return self.output

    def updateGradInput(self, input, gradOutput):
        batch_size, in_channels, height, width = input.shape
        out_h, out_w = height // self.kernel_size, width // self.kernel_size

        self.gradInput = np.zeros_like(input)

        for batch_size_num in range(batch_size):
            for in_channel_num in range(in_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start, w_start = oh * self.kernel_size, ow * self.kernel_size
                        max_idx = self.max_indices[batch_size_num, in_channel_num, oh, ow]
                        local_h, local_w = np.unravel_index(max_idx, (self.kernel_size, self.kernel_size))
                        self.gradInput[batch_size_num, in_channel_num, h_start + local_h, w_start + local_w] = \
                        gradOutput[batch_size_num, in_channel_num, oh, ow]

        return self.gradInput

    def __repr__(self):
        q = 'MaxPool2d, kern %d, stride %d' % (self.kernel_size, self.kernel_size)
        return q


class SoftMax(Module):
    def __init__(self):
        super(SoftMax, self).__init__()

    def updateOutput(self, input):
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        self.output = np.exp(self.output)
        self.output = self.output / np.sum(self.output, axis=1, keepdims=True)

        return self.output

    def updateGradInput(self, input, gradOutput):
        batch_size = input.shape[0]
        self.gradInput = np.empty_like(input)

        for i in range(batch_size):
            col_vec = self.output[i].reshape(-1, 1)
            jacoby_matrix = np.diagflat(col_vec) - np.dot(col_vec, col_vec.T)
            self.gradInput[i] = np.dot(jacoby_matrix, gradOutput[i])

        return self.gradInput

    def __repr__(self):
        return "SoftMax"


class LogSoftMax(Module):
    def __init__(self):
        super(LogSoftMax, self).__init__()

    def updateOutput(self, input):
        stable_value = np.subtract(input, input.max(axis=1, keepdims=True))
        log_sum_exp = np.log(np.sum(np.exp(stable_value), axis=1, keepdims=True))
        self.output = stable_value - log_sum_exp

        return self.output

    def updateGradInput(self, input, gradOutput):
        batch_size, _ = input.shape
        self.gradInput = np.zeros_like(input)

        for i in range(batch_size):
            col_vec = np.exp(self.output[i])
            sum_grad = np.sum(gradOutput[i])
            self.gradInput[i] = gradOutput[i] - col_vec * sum_grad

        return self.gradInput

    def __repr__(self):
        return "LogSoftMax"


class BatchNormalization(Module):
    EPS = 1e-3

    def __init__(self, alpha=0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = None
        self.moving_variance = None

    def updateOutput(self, input):
        n_feats = input.shape[1]
        if self.moving_mean is None:
            self.moving_mean = np.zeros(n_feats)
            self.moving_variance = np.ones(n_feats)

        if self.training is True:  # флаг в Module train() и eval()
            self.batch_mean = input.mean(axis=0)
            self.batch_variance = input.var(axis=0)

            self.output = (input - self.batch_mean) / (np.sqrt(self.batch_variance + self.EPS))
            self.moving_mean = self.moving_mean * self.alpha + self.batch_mean * (1 - self.alpha)
            self.moving_variance = self.moving_variance * self.alpha + self.batch_variance * (1 - self.alpha)

            return self.output

        if self.training is False:
            self.output = (input - self.moving_mean) / (np.sqrt(self.moving_variance + self.EPS))

            return self.output

    def updateGradInput(self, input, gradOutput):
        mean_deviation = input - self.batch_mean
        std_denominator = np.sqrt(self.batch_variance + self.EPS)
        grad_prod = np.mean(gradOutput * mean_deviation, axis=0)

        self.gradInput = (
                                 gradOutput - np.mean(gradOutput, axis=0) - mean_deviation * grad_prod * (
                                     1 / std_denominator) ** 2) * (1 / std_denominator)

        return self.gradInput

    def __repr__(self):
        return "BatchNormalization"


class ChannelwiseScaling(Module):
    r"""
    Implements linear transform of input y = \gamma * x + \beta
    where \gamma, \beta - learnable vectors of length x.shape[-1]
    """

    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1. / np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def updateOutput(self, input):
        self.output = input * self.gamma + self.beta
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.gamma
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        self.gradBeta += np.sum(gradOutput, axis=0)
        self.gradGamma += np.sum(gradOutput * input, axis=0)

    def zeroGradParameters(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)

    def getParameters(self):
        return [self.gamma, self.beta]

    def getGradParameters(self):
        return [self.gradGamma, self.gradBeta]

    def __repr__(self):
        return "ChannelwiseScaling"


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()

        self.p = p
        self.mask = None

    def updateOutput(self, input):
        if self.training is True:
            batch_size, n_feats = input.shape
            self.mask = np.random.rand(batch_size, n_feats)
            self.mask = np.where(self.mask >= self.p, 1.0, 0.0)
            self.output = input * self.mask / (1 - self.p)
            return self.output

        if self.training is False:
            self.output = input
            return self.output

    def updateGradInput(self, input, gradOutput):
        if self.training is True:
            self.gradInput = gradOutput * self.mask / (1 - self.p)
            return self.gradInput

        if self.training is False:
            self.gradInput = gradOutput
            return self.gradInput

    def __repr__(self):
        return "Dropout"


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput, input > 0)
        return self.gradInput

    def __repr__(self):
        return "ReLU"


class LeakyReLU(Module):
    def __init__(self, slope=0.07):
        super(LeakyReLU, self).__init__()

        self.slope = slope

    def updateOutput(self, input):
        self.output = (1 + self.slope) / 2 * input + (1 - self.slope) / 2 * np.abs(input)
        return self.output

    def updateGradInput(self, input, gradOutput):
        grad = np.ones_like(input)
        grad = np.where(input <= 0, self.slope, 1.0)
        self.gradInput = gradOutput * grad

        return self.gradInput

    def __repr__(self):
        return "LeakyReLU"


class ELU(Module):
    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()

        self.alpha = alpha

    def updateOutput(self, input):
        self.output = np.where(input <= 0, self.alpha * (np.exp(input) - 1), input)
        return self.output

    def updateGradInput(self, input, gradOutput):
        grad = np.ones_like(input)
        grad = np.where(input <= 0, self.alpha * np.exp(input), 1.0)
        self.gradInput = gradOutput * grad

        return self.gradInput

    def __repr__(self):
        return "ELU"


class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()

    def updateOutput(self, input):
        self.output = np.log1p(np.exp(input))
        return self.output

    def updateGradInput(self, input, gradOutput):
        grad = 1 / (1 + np.exp(-input))
        self.gradInput = gradOutput * grad

        return self.gradInput

    def __repr__(self):
        return "SoftPlus"
