import math
import torch
import warnings
import itertools
import numbers

from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence
from torch.nn import init
from torch.nn import functional as F
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class my_LSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, if_bias=True, batch_first=False):
		super(my_LSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.if_bias = if_bias
		self.batch_first = batch_first

		self.w_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.w_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
		if if_bias:
        	self.b_ih = Parameter(torch.Tensor(4 * hidden_size))
        	self.b_hh = Parameter(torch.Tensor(4 * hidden_size))
		else:
			self.register_parameter('b_ih', None)
			self.register_parameter('b_hh', None)

		'''
			TO-DO: define each matric multiplication here
		'''

		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()

		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1.0 / math.sqrt(self.hidden_size)
		for weight in self.parameters():
			init.uniform_(weight, -stdv, stdv)

	def forward(self, input_seq, hx=None):
		'''
			TO-DO: check if input_seq is a packed sequence. If yes, unpack it.
		'''
		is_packed = isinstance(input_seq, PackedSequence)
		if is_packed:
			input_seq, batch_sizes = input_seq
			max_batch_size = int(batch_sizes[0])
		else:
			batch_sizes = torch.tensor([1])
			max_batch_size = input_seq.size(1) if self.batch_first else input_seq.size(0)
			input_seq = input_seq[0]


		# outputs
		hidden_states = []
		cell_states = []

		'''
			TO-DO: if hx is None, initialize it.
		'''
		if hx is None:
			hx = input_seq.new_zeros(max_batch_size, self.hidden_size, requires_grad=False).to(device)
			hx = (hx, hx)

		'''
			TO-DO: implement LSTM here
		'''
		hx, cx = hx
		idx = 0
		for batch_size in batch_sizes:
			length = batch_size.item()
			x = input_seq[idx:idx+length]
			idx += length
			hx = hx[0:length]
			cx = cx[0:length]
			'''
			linear: (1,input_feature_size)*(4*hidden_size, input_feature_size)=>(1,4*hidden_size)
			'''
			gates = F.linear(input_seq[i], self.w_ih, self.b_ih) + F.linear(hx, self.w_hh, self.b_hh)
			ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
			ingate = F.sigmoid(ingate)
			forgetgate = F.sigmoid(forgetgate)
			cellgate = F.tanh(cellgate)
			outgate = F.sigmoid(outgate)
			cx = (forgetgate * cx) + (ingate * cellgate)
			hx = outgate * F.tanh(cx)
			hidden_states.append(hx)
			cell_states.append(cx)

		hidden_state_list = torch.cat(hidden_states, 0)
		cell_state_list = torch.cat(cell_states, 0)
		return hidden_state_list, (hidden_states, cell_states)
