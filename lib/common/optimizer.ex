defmodule Sgd do
  require Param
  require Grad
  require Matrex
  require MatrexUtils

  def optimize(%{} = params, %{} = grads, lr) do
    cond do
      Map.keys(params) == Map.keys(grads) -> {:ok, _optimize(params, grads, lr)}
      true                                -> {:error, "Keys of params and gras do not match."}
    end
  end
  defp _optimize(%{} = params, %{} = grads, lr) do
    Stream.zip(params, grads)
    |> Stream.map(fn {{key, p}, {key, g}} -> {key, _optimize_each(p, g, lr)} end)
    |> Map.new
  end
  defp _optimize_each(%Param{w: nil, b: nil}, %Grad{dx: _, dw: nil, db: nil}, _) do
    %Param{w: nil, b: nil}
  end
  defp _optimize_each(%Param{w: w, b: nil}, %Grad{dx: _, dw: dw, db: nil}, lr) do
    %Param{
      w: Matrex.subtract(w, Matrex.multiply(dw, lr)),
      b: nil
    }
  end
  defp _optimize_each(%Param{w: w, b: b}, %Grad{dx: _, dw: dw, db: db}, lr) do
    %Param{
      w: Matrex.subtract(w, Matrex.multiply(dw, lr)),
      b: Matrex.subtract(b, Matrex.multiply(db, lr))
    }
  end
end