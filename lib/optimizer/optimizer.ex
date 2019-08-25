defmodule Sgd do
  import Param
  import Grad
  import Matrex

  def optimize([_, _] = params, [_, _] = grads, lr) do
    Stream.zip(params, grads)
    |> Stream.map(fn {p, g} -> _optimize(p, g, lr) end)
    |> Enum.to_list
  end
  defp _optimize(%Param{w: w, b: nil}, %Grad{dw: dw, db: nil}, lr) do
    %Param{w: Matrex.subtract(w, Matrex.multiply(lr, dw)), b: nil}
  end
  defp _optimize(%Param{w: w, b: b}, %Grad{dw: dw, db: db}, lr) do
    %Param{
      w: Matrex.subtract(w, Matrex.multiply(lr, dw)),
      b: Matrex.subtract(b, Matrex.multiply(lr, db))
    }
  end
end