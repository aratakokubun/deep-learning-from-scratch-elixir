defmodule TwoLayerNet do
  import Matrex
  import Affine
  import Sigmoid
  import SoftmaxWithLoss
  import Param
  import Grad

  def predict(%Matrex{} = x, %Matrex{} = y, params) do
    feed_forward(x, y, params)
  end

  def feed_forward(%Matrex{} = x, %Matrex{} = y, %{layer1: param1, layer3: param3}) do
    layer1_out = Affine.forward(x, param1)
    layer2_out = Sigmoid.forward(layer1_out)
    layer3_out = Affine.forward(layer2_out, param3)
    %{out: layer4_out, loss: loss} = SoftmaxWithLoss.forward(layer3_out, y)
    %{outs: [layer1_out, layer2_out, layer3_out, layer4_out], loss: loss}
  end

  def back_propagate(%Matrex{} = x, %Matrex{} = y,
        %{layer1: param1, layer2: nil, layer3: param3},
        %{outs: [out1, out2, out3, out4]},
        dout \\ 1.0) do
    grad4 = %Grad{dx: dout4, dw: _, db: _} = SoftmaxWithLoss.backward(x, y, out4, dout)
    grad3 = %Grad{dx: dout3, dw: _, db: _} = Affine.backward(x, param3, dout4)
    grad2 = %Grad{dx: dout2, dw: _, db: _} = Sigmoid.backward(x, out2, dout3)
    grad1 = Affine.backward(x, param1, dout2)
    %{layer1: grad1, layer2: grad2, layer3: grad3, layer4: grad4}
  end
end

