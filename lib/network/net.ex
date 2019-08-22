import Np

defmodule Sigmoid do
  def forward(x, _, _), do: Np.sigmoid(x)
  def backward(x, w, out, dout) do
    Np.mul(dout, Np.sub(1, out))
    |> Np.mul(dout)
  end
end

defmodule Affine do
  def forward(x, w, b), do: Np.add(Np.dot(x, w), b)
  def backward(x, w, _, dout) do
    dx = Np.dot(dout, Np.transpose(w))
    dw = Np.transpose(x) |> Np.dot(dout)
    db = Np.sum(dout, :row, false)
    %{dx: dx, dw: dw, db: db}
  end
end

defmodule SoftmaxWithLoss do
  import Functions

  def forward(x, t) do
    y = x
    |> Np.mul(-1)
    |> Np.exp
    |> Np.add(1)
    |> (&(Np.div(1, &1))).()

    Np.c_([Np.sub(1, y), y])
    |> Functions.cross_entropy_error(t)
  end
end

defmodule Matmul do
  def forward(x, w), do: Np.dot(x, w)
  def backward(x, w, _, dout) do
    dx = Np.dot(dout, Np.transpose(w))
    dw = x |> Np.transpose |> Np.dot(dout)
    %{dx: dx, dw: dw}
  end
end

defmodule Net do
  def net_2layers, do: [&Affine.forward/3, &Sigmoid.forward/3, &Affine.forward/3]

  def predict(x, [], [], []), do: x
  def predict(x, [layer_head | layer_tail], [weight_head | weight_tail], [bias_head | bias_tail]) do
    predict(layer_head.(x, weight_head, bias_head), layer_tail, weight_tail, bias_tail)
  end
end
