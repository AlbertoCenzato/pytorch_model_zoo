import unittest
import random
import time

import torch
from ..package import convlstm, convlstm_cpp, convlstm_cuda

class TestConvLSTMCPP(unittest.TestCase):

    def setUp(self):
        input_size  = (30,30)
        input_dim   = 1
        hidden_dim  = 1
        kernel_size = (5,5)
        bias = True

        self.convlstm_cell     = convlstm.ConvLSTMCell       (input_size, input_dim, hidden_dim, kernel_size, bias)
        self.convlstmcpp_cell  = convlstm_cpp.ConvLSTMCPPCell(input_size, input_dim, hidden_dim, kernel_size, bias)
        
        # The two models must have the same initial conditions
        self.convlstmcpp_cell.weights.data = self.convlstm_cell.conv.weight.data
        self.convlstmcpp_cell.bias.data    = self.convlstm_cell.conv.bias.data
        
        batch_size = 16
        self.input  = torch.rand((batch_size, input_dim) + input_size)
        self.state = (torch.rand((batch_size, hidden_dim) + input_size), torch.rand((batch_size, hidden_dim) + input_size))


    def tearDown(self):
        self.convlstm_cell    = None
        self.convlstmcpp_cell = None
        self.input = None
        self.state = None

    def test_forward(self):
        output     = self.convlstm_cell(self.input, self.state)
        output_cpp = self.convlstmcpp_cell(self.input, self.state)
        self.assertTrue(torch.equal(output[0], output_cpp[0]), 'The two output tensors are not equal')
        
    def test_backpropagation(self):
        output     = self.convlstm_cell(self.input, self.state)
        output_cpp = self.convlstmcpp_cell(self.input, self.state)

        ground_truth = torch.full_like(output[0], 20)
        criterion = torch.nn.MSELoss()
        loss     = criterion(output[0],     ground_truth)
        loss_cpp = criterion(output_cpp[0], ground_truth)

        loss.backward()
        loss_cpp.backward()

        weight_grad     = self.convlstm_cell.conv.weight._grad
        weight_grad_cpp = self.convlstmcpp_cell.weights._grad

        self.assertTrue(torch.allclose(weight_grad, weight_grad_cpp), 'The two weights gradients are not equal')
        

    def test_bptt(self):
        time_steps = 100 #random.randint(1,20)
        criterion = torch.nn.MSELoss()

        # python model
        start = time.perf_counter()
        output = self.convlstm_cell(self.input, self.state)        
        for _ in range(time_steps):
            output = self.convlstm_cell(output[0], self.state)

        ground_truth = torch.full_like(output[0], 20)
        
        loss = criterion(output[0], ground_truth)
        loss.backward()
        grad = self.convlstm_cell.conv.weight._grad
        print('Python time: {}'.format(time.perf_counter() - start))

        # c++ model
        start = time.perf_counter()
        output_cpp = self.convlstmcpp_cell(self.input, self.state)
        for _ in range(time_steps):
            output_cpp = self.convlstmcpp_cell(output_cpp[0], self.state)

        loss_cpp = criterion(output_cpp[0], ground_truth)
        loss_cpp.backward()
        grad_cpp = self.convlstmcpp_cell.weights._grad
        print('C++ time: {}'.format(time.perf_counter() - start))

        self.assertTrue(torch.allclose(grad, grad_cpp), 'The two gradients are not equal')


class TestConvLSTMCuda(unittest.TestCase):

    def setUp(self):
        input_size  = (30,30)
        input_dim   = 1
        hidden_dim  = 1
        kernel_size = (5,5)
        bias = True

        self.convlstm_cell     = convlstm.ConvLSTMCell         (input_size, input_dim, hidden_dim, kernel_size, bias)
        self.convlstmcuda_cell = convlstm_cuda.ConvLSTMCudaCell(input_size, input_dim, hidden_dim, kernel_size, bias)
        
        # The two models must have the same initial conditions
        self.convlstmcuda_cell.weights.data = self.convlstm_cell.conv.weight.data
        self.convlstmcuda_cell.bias.data    = self.convlstm_cell.conv.bias.data
        
        batch_size = 16
        self.input  = torch.rand((batch_size, input_dim) + input_size)
        self.state = (torch.rand((batch_size, hidden_dim) + input_size), torch.rand((batch_size, hidden_dim) + input_size))


    def tearDown(self):
        self.convlstm_cell    = None
        self.convlstmcuda_cell = None
        self.input = None
        self.state = None

    def test_forward(self):
        output     = self.convlstm_cell(self.input, self.state)
        output_cpp = self.convlstmcuda_cell(self.input, self.state)
        self.assertTrue(torch.equal(output[0], output_cpp[0]), 'The two output tensors are not equal')
        
    def test_backpropagation(self):
        output     = self.convlstm_cell(self.input, self.state)
        output_cpp = self.convlstmcuda_cell(self.input, self.state)

        ground_truth = torch.full_like(output[0], 20)
        criterion = torch.nn.MSELoss()
        loss     = criterion(output[0],     ground_truth)
        loss_cpp = criterion(output_cpp[0], ground_truth)

        loss.backward()
        loss_cpp.backward()

        weight_grad     = self.convlstm_cell.conv.weight._grad
        weight_grad_cpp = self.convlstmcuda_cell.weights._grad

        self.assertTrue(torch.allclose(weight_grad, weight_grad_cpp), 'The two weights gradients are not equal')
        

    def test_bptt(self):
        time_steps = 100 #random.randint(1,20)
        criterion = torch.nn.MSELoss()

        # python model
        start = time.perf_counter()
        output = self.convlstm_cell(self.input, self.state)        
        for _ in range(time_steps):
            output = self.convlstm_cell(output[0], self.state)

        ground_truth = torch.full_like(output[0], 20)
        
        loss = criterion(output[0], ground_truth)
        loss.backward()
        grad = self.convlstm_cell.conv.weight._grad
        print('Python time: {}'.format(time.perf_counter() - start))

        # c++ model
        start = time.perf_counter()
        output_cpp = self.convlstmcuda_cell(self.input, self.state)
        for _ in range(time_steps):
            output_cpp = self.convlstmcuda_cell(output_cpp[0], self.state)

        loss_cpp = criterion(output_cpp[0], ground_truth)
        loss_cpp.backward()
        grad_cpp = self.convlstmcuda_cell.weights._grad
        print('C++ time: {}'.format(time.perf_counter() - start))

        self.assertTrue(torch.allclose(grad, grad_cpp), 'The two gradients are not equal')





if __name__ == '__main__':
    unittest.main()