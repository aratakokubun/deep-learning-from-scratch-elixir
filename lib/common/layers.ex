import Matrex
import MatrexUtils

defmodule Param do
  defstruct w: nil, b: nil
end

defmodule Grad do
  defstruct dx: nil, dw: nil, db: nil
end

defmodule Sigmoid do
  def forward(%Matrex{} = x) do
    Matrex.apply(x, :sigmoid)
  end
  def backward(%Matrex{} = x, %Matrex{} = out, %Matrex{} = dout) do
    dout
    |> Matrex.multiply(Matrex.subtract(1, out))
    |> Matrex.multiply(out)
    |> (&%Grad{dx: &1}).()
  end
end

defmodule Affine do
  def forward(%Matrex{} = x, %Param{w: weight, b: bias}) do
    Matrex.dot(x, weight)
    |> MatrexUtils.add(bias)
  end
  def backward(%Matrex{} = x, %Param{w: weight}, %Matrex{} = dout) do
    %Grad{
      dx: Matrex.dot_nt(dout, weight),
      dw: Matrex.dot_tn(x, dout),
      db: MatrexUtils.sum(dout, :rows)
    }
  end
end

defmodule SoftmaxWithLoss do
  def forward(%Matrex{} = x, %Matrex{} = y) do
    # Forward propagation Cross entropy error
    # Reference: https://ja.wikipedia.org/wiki/%E4%BA%A4%E5%B7%AE%E3%82%A8%E3%83%B3%E3%83%88%E3%83%AD%E3%83%94%E3%83%BC#%E4%BA%A4%E5%B7%AE%E3%82%A8%E3%83%B3%E3%83%88%E3%83%AD%E3%83%94%E3%83%BC%E8%AA%A4%E5%B7%AE%E9%96%A2%E6%95%B0%E3%81%A8%E3%83%AD%E3%82%B8%E3%82%B9%E3%83%86%E3%82%A3%E3%83%83%E3%82%AF%E5%9B%9E%E5%B8%B0
    batch_size = y[:rows]
    h = Matrex.apply(x, :sigmoid)

    loss = y
           |> Matrex.multiply(Matrex.apply(h, :log))
           |> Matrex.neg()
           |> Matrex.subtract(
                Matrex.subtract(1, y)
                |> Matrex.multiply(
                     Matrex.subtract(1, h)
                     |> Matrex.apply(:log)
                   )
              )
           |> Matrex.sum()
           |> Kernel./(batch_size)
    %{out: h, loss: loss}
  end

  def backward(%Matrex{} = x, %Matrex{} = y, %Matrex{} = out, dout \\ 1.0) do
    batch_size = y[:rows]
    dx = out
         |> Matrex.subtract(y)
         |> Matrex.multiply(dout)
         |> Matrex.divide(batch_size)
    %Grad{dx: dx}
  end
end

defmodule SoftmaxWithLossLeguralize do
  def forward(%Matrex{} = x, %Matrex{} = y, %Param{w: weight}, lambda) when is_number(lambda) do
    # Forward propagation Cross entropy error
    # Reference: https://ja.wikipedia.org/wiki/%E4%BA%A4%E5%B7%AE%E3%82%A8%E3%83%B3%E3%83%88%E3%83%AD%E3%83%94%E3%83%BC#%E4%BA%A4%E5%B7%AE%E3%82%A8%E3%83%B3%E3%83%88%E3%83%AD%E3%83%94%E3%83%BC%E8%AA%A4%E5%B7%AE%E9%96%A2%E6%95%B0%E3%81%A8%E3%83%AD%E3%82%B8%E3%82%B9%E3%83%86%E3%82%A3%E3%83%83%E3%82%AF%E5%9B%9E%E5%B8%B0
    batch_size = y[:rows]
    h = Matrex.dot_and_apply(x, weight, :sigmoid)

    loss = y
           |> Matrex.dot_tn(Matrex.apply(h, :log), -1.0)
           |> Matrex.subtract(
                Matrex.subtract(1, y)
                |> Matrex.dot_tn(
                     Matrex.subtract(1, h)
                     |> Matrex.apply(:log)
                   )
              )
           |> Matrex.scalar()
           |> _regularize(weight, batch_size, lambda)
    %{out: h, loss: loss}
  end

  defp _regularize(:nan, _, _), do: :nan
  defp _regularize(sum_diffs, %Matrex{} = weight, batch_size, lambda), do: sum_diffs / batch_size + _regularization(weight, batch_size, lambda)

  defp _regularization(%Matrex{} = weight, batch_size, lambda) do
    # L1 Regularization
    Matrex.ones(weight[:rows], weight[:cols])
    |> Matrex.set(1, 1, 0)
    |> Matrex.dot_tn(Matrex.square(weight))
    |> Matrex.scalar()
    |> Kernel.*(lambda / (2 * batch_size))
  end

  def backward(%Matrex{} = x, %Matrex{} = y, %Param{w: weight}, %Matrex{} = out, lambda) do
    batch_size = y[:rows]
    l = Matrex.ones(weight[:rows], weight[:cols])
        |> Matrex.set(1, 1, 0)
    dx = Matrex.subtract(out, y)
    x
    |> Matrex.dot_tn(dx)
    |> Matrex.add(Matrex.multiply(weight, l), 1.0, lambda)
    |> Matrex.divide(batch_size)
    |> (&%Grad{dx: dx, dw: &1}).()
  end
end

defmodule Matmul do
  def forward(%Matrex{} = x, %Param{w: weight}) do
    Matrex.dot(x, weight)
  end
  def backward(%Matrex{} = x, %Param{w: weight}, %Matrex{} = dout) do
    dx = weight
         |> Matrex.transpose()
         |> Matrex.multiply(dout)
    dw = x
         |> Matrex.transpose()
         |> Matrex.multiply(dout)
    %Grad{dx: dx, dw: dw}
  end
end
