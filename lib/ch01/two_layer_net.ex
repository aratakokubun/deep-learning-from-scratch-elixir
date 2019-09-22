defmodule TwoLayerNet do
  require Matrex
  require Affine
  require Sigmoid
  require SoftmaxWithLoss
  require Param
  require Grad

  def predict(%Matrex{} = x, %{layer1: param1, layer2: _, layer3: param3, layer4: _}) do
    Affine.forward(x, param1)
    |> Sigmoid.forward()
    |> Affine.forward(param3)
  end

  def feed_forward(%Matrex{} = x, %Matrex{} = y, %{layer1: param1, layer2: _, layer3: param3, layer4: _}) do
    out1 = Affine.forward(x, param1)
    out2 = Sigmoid.forward(out1)
    out3 = Affine.forward(out2, param3)
    %{out: out4, loss: loss} = SoftmaxWithLoss.forward(out3, y)
    %{outs: %{layer1: out1, layer2: out2, layer3: out3, layer4: out4}, loss: loss}
  end

  def back_propagate(%Matrex{} = x, %Matrex{} = y,
        %{layer1: param1, layer2: _,    layer3: param3, layer4: _   },
        %{layer1: _,      layer2: out2, layer3: _,      layer4: out4},
        dout \\ 1.0) do
    grad4 = %Grad{dx: dout4, dw: _, db: _} = SoftmaxWithLoss.backward(y, out4, dout)
    grad3 = %Grad{dx: dout3, dw: _, db: _} = Affine.backward(out2, param3, dout4)
    grad2 = %Grad{dx: dout2, dw: _, db: _} = Sigmoid.backward(out2, dout3)
    grad1 = Affine.backward(x, param1, dout2)
    %{layer1: grad1, layer2: grad2, layer3: grad3, layer4: grad4}
  end
end

