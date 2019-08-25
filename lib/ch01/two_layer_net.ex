defmodule ForwardOut do
  defstruct outs: nil, loss: nil
end

defmodule TwoLayerNet do
  import Matrex
  import Affine
  import Sigmoid
  import SoftmaxWithLoss
  import Param
  import Grad

  '''
  x: Input
  y: Ground Truth
  weights: List of weights of each layer
  '''
  def predict(%Matrex{} = x, %Matrex{} = y, [_, _] = params, lambda) do
    feed_forward(x, y, params, lambda)
  end

  def feed_forward(%Matrex{} = x, %Matrex{} = y, [_, _] = params, lambda) do
    layer1_out = Affine.forward(x, Enum.at(params, 0))
    layer2_out = Sigmoid.forward(layer1_out)
    layer3_out = Affine.forward(layer2_out, Enum.at(params, 2))
    %{out: layer4_out, loss: loss} = SoftmaxWithLoss.forward(layer3_out, y, Enum.at(params, 3), lambda)
    %ForwardOut{outs: [layer1_out, layer2_out, layer3_out, layer4_out], loss: loss}
  end

  def back_propagate(%Matrex{} = x, %Matrex{} = y,
        [param1, param2, param3, param4],
        %ForwardOut{outs: [out1, out2, out3, out4]}) do
    grad4
      = %Grad{dx: dx4, dw: _, db: _}
      = SoftmaxWithLoss.backward(x, y, param4, out4)
    grad3
      = %Grad{dx: dx3, dw: _, db: _}
      = Affine.backward(x, param3, dx4)
    grad2
      = %Grad{dx: dx2, dw: _, db: _}
      = Sigmoid.backward(x, out2, dx3)
    grad1
      = %Grad{dx: dx1, dw: _, db: _}
      = Affine.backward(x, param1, dx2)
    [grad1, grad2, grad3, grad4]
  end
end

