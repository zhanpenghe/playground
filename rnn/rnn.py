import torch
import torch.nn as nn
import torch.nn.functional as F


class VanilaRNN(nn.Module):

    def __init__(self, 
                 input_size,
                 output_size,
                 hidden_sizes=[32, 32],
                 hidden_nonlinearity=F.relu,
                 output_nonlinearity=F.softmax,
                 ):
        self._input_size = input_size
        self._hidden_sizes = hidden_sizes
        self._output_size = output_size

        self.params = dict()
        self._init_params()

    def _init_params(self):
        for i, h in enumerate(self._hidden_sizes):
            size = self._input_size + h
            self.params["fc_{}".format(i)] = nn.Linear(size)
            if i == len(hidden_sizes) - 1:
                self.params["nonlinearity_{}".format(i)] = output_nonlinearity()
            else:
                self.params["nonlinearity_{}".format(i)] = hidden_nonlinearity()

    def forward(self, inputs):
        length = inputs.size()[-1]
        if length % self._input_size != 0:
            raise ValueError("Wrong shape of input for RNN!")

        # TODO: This is not recurrent now.
        context = torch.zeros(size=(0, 0))  # TODO: fix size
        for i in range(len(self._hidden_sizes)):
            input_window = self.slice_input(inputs, i)
            input_context = torch.cat(input_window, context)
            o = self.params["fc_{}".format(i)](input_context)
            context = self.params["nonlinearity_{}".format(i)](o)
        return context

    def _slice_input(self, inputs, i):
        pass
