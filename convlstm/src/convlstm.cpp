#include <torch/extension.h>

#include <iostream>
#include <vector> 
#include <array>
#include <cassert>

const int64_t stride   = 1;
const int64_t padding  = 2;
const int64_t dilation  = 1;

/**
 * --------------------------------------------------------------------------
 * | WARNING!!! It only works for stride=1, padding=2, dilation=1, groups=1 |
 * --------------------------------------------------------------------------
 */
std::vector<at::Tensor> convlstm_forward(const at::Tensor &input, const at::Tensor &weights, const at::Tensor &bias,
                                         const at::Tensor &old_h, const at::Tensor &old_cell) {   
  const auto X = at::cat({input, old_h}, /*dim=*/1);  // assuming channels-first ordering

  const auto gate_weights = at::conv2d(X, weights, bias, stride, padding);  // FIXME: had to put a fixed size padding because of some integer overflow issues occurring when computing weights.size(2)/int64_t(2)
  const auto gate_size = gate_weights.size(1)/4;
  const auto input_gate  = at::sigmoid(gate_weights.narrow(/*dim=*/1, 0, gate_size));
  const auto forget_gate = at::sigmoid(gate_weights.narrow(/*dim=*/1, gate_size, gate_size));
  const auto output_gate = at::sigmoid(gate_weights.narrow(/*dim=*/1, 2*gate_size, gate_size));
  const auto candidate_cell = at::tanh(gate_weights.narrow(/*dim=*/1, 3*gate_size, gate_size));

  const auto c_next = forget_gate * old_cell + input_gate * candidate_cell;
  const auto h_next = output_gate * at::tanh(c_next);

  return {h_next, c_next, input_gate, forget_gate, output_gate, 
          candidate_cell, X, gate_weights};
}


at::Tensor d_sigmoid(at::Tensor z) {
  auto s = at::sigmoid(z);
  return (1 - s) * s;
}

// tanh'(z) = 1 - tanh^2(z)
at::Tensor d_tanh(at::Tensor z) {
  return 1 - z.tanh().pow(2);
}


/**
 * Computes the gradient of Y = at::conv2d(X, W) function
 *                      $$ Y = X \star W $$
 * 
 * ----------------------------------------------------------------------------
 * | WARNING!!! It does not work for generic 2D convolution. It works only for |
 * | stride=1, padding=2, dilation=1, groups=1                                 |
 *  ---------------------------------------------------------------------------
 * 
 * @conv_out_grad: gradient of the loss function wrt the output of conv2d, 
 *                 i.e. $$ \frac{\partial \mathcal{L}}{\partial Y} $$
 * @input: input of conv2d, i.e. $X$
 * @weights: tensor of weights used in conv2d, i.e. $W$
 * @return: a tuple containing the weights gradients and the convolution input gradients,
 *          i.e. $$ < \frac{\partial \mathcal{L}}{\partial W}, \frac{\partial
 *          \mathcal{L}}{X} > $$
 */
std::tuple<at::Tensor, at::Tensor> d_conv2d(const at::Tensor &conv_out_grad, const at::Tensor &input, const at::Tensor &weights) {
    
  const auto batch_size   = input.size(0);
  const auto out_channels = conv_out_grad.size(1);
  const auto in_channels  = input.size(1);
  
  std::vector<int64_t> d_weights_sizes{out_channels, in_channels, weights.size(2), weights.size(3)};
  auto d_weights = at::zeros(d_weights_sizes, weights.type());
  for (auto batch_index = 0; batch_index < batch_size; ++batch_index) {           // Compute loss function derivatives wrt weights for each data sample in the batch: $$ \frac{\partial L}{\partial W_{j,i}} = X_i \star \frac{\partial L}{\partial Y_j} $$
    const auto X  = input.narrow(0, batch_index, 1);
    const auto dY = conv_out_grad.narrow(0, batch_index, 1).permute({1,0,2,3});   // permute from (1, out_channels, ker_h, ker_w) to (out_channels, 1, ker_h, ker_w) 

    std::vector<at::Tensor> d_weights_in(in_channels);
    auto d_weights_batch = at::zeros_like(d_weights);
    for (auto i = 0; i < in_channels; ++i) {                      // for each input channels compute the gradient wrt weights obtaining a 
      d_weights_in[i] = at::conv2d(X.narrow(1, i, 1), dY, {},     //
                                   stride, padding).squeeze();    // (1, out_channels, ker_h, ker_w) shaped tensor and remove the first dimension
    }
    d_weights += at::stack(d_weights_in, /*dim=*/1);
    
    //const auto X  = input.narrow(0, batch_index, 1).permute({1,0,2,3});
    //const auto dY = conv_out_grad.narrow(0, batch_index, 1).permute({1,0,2,3});
    //d_weights += at::conv2d(dY, X, {}, stride, padding);
  }

  /*
   * rotate the kernels of 180° (see here
   * https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e 
   * and here https://grzegorzgwardys.wordpress.com/2016/04/22/8/)
   * 
   * conv_transpose2d() performs the convolution rotating the kernel by 180°
   */
  auto dX = at::conv_transpose2d(conv_out_grad, weights, {}, stride, padding);

  return {d_weights, dX};
}


std::vector<at::Tensor> convlstm_backward(const at::Tensor &grad_h, const at::Tensor &grad_cell,  //these are the gradients coming from timestep t+1
                                          const at::Tensor &new_cell, const at::Tensor &old_cell,  // all variables that are part of the cell state
                                          const at::Tensor &input_gate, const at::Tensor &forget_gate, 
                                          const at::Tensor &output_gate, const at::Tensor &candidate_cell,
                                          const at::Tensor &X, const at::Tensor &gate_weights,
                                          const at::Tensor &weights) {
  auto d_output_gate   = at::tanh(new_cell) * grad_h;
  const auto d_tanh_new_cell = output_gate * grad_h;
  const auto d_new_cell      = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;
  
  const auto d_old_cell = forget_gate    * d_new_cell;
  auto d_input_gate     = candidate_cell * d_new_cell;
  auto d_forget_gate    = old_cell       * d_new_cell;
  auto d_candidate_cell = input_gate     * d_new_cell;

  auto gate_size = gate_weights.size(1)/4;
  d_input_gate     *= d_sigmoid(gate_weights.narrow(1, 0, gate_size));
  d_forget_gate    *= d_sigmoid(gate_weights.narrow(1, gate_size, gate_size));
  d_output_gate    *= d_sigmoid(gate_weights.narrow(1, 2*gate_size, gate_size));
  d_candidate_cell *= d_tanh(gate_weights.narrow(1, 3*gate_size, gate_size));

  const auto d_gates = at::cat({d_input_gate, d_forget_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);
  
  const auto gradients = d_conv2d(d_gates, X, weights);
  const auto d_weights = std::get<0>(gradients);
  const auto d_X       = std::get<1>(gradients);

  auto d_bias = d_gates.sum(/*dim=*/{2,3}, /*keepdim=*/false).sum(/*dim=*/0); // CHECK: devo sommare o mediare sulla batch dimension? Pare che la somma vada bene
  
  const auto input_size = d_X.size(1) - grad_h.size(1);
  const auto d_input = d_X.slice(/*dim=*/1, 0, input_size);
  const auto d_old_h = d_X.slice(/*dim=*/1, input_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward",  &convlstm_forward,  "ConvLSTM forward");
  m.def("backward", &convlstm_backward, "ConvLSTM backward");
}