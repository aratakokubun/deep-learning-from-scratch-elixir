import Np

defmodule Sigmoid do
  def forward(x, _, _) do
    Np.sigmoid(x)
  end
end

defmodule Affine do
  def forward(x, w, b) do
    Np.add(Np.dot(x, w), b)
  end
end

defmodule SoftmaxWithLoss do
  # TODO
end

defmodule Net do
  @layers [&Affine.forward/3, &Sigmoid.forward/3, &Affine.forward/3]
  def net_2layers, do: @layers

  def predict(x, [], [], []), do: x
  def predict(x, [layer_head | layer_tail], [weight_head | weight_tail], [bias_head | bias_tail]) do
    predict(layer_head.(x, weight_head, bias_head), layer_tail, weight_tail, bias_tail)
  end
end
