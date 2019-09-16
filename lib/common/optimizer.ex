defmodule Sgd do
  import Param
  import Grad
  import Matrex
  import MatrexUtils

  def optimize(%{} = params, %{} = grads, lr) do
    cond do
      Map.keys(params) == Map.keys(grads) -> {:ok, _optimize(params, grads, lr)}
      true                                -> {:error, "Keys of params and gras do not match."}
    end
  end
  defp _optimize(%{} = params, %{} = grads, lr) do
    Stream.zip(params, grads)
    |> Stream.map(fn {{key, p}, {key, g}} -> {key, _optimize_each(p, g, lr)} end)
    |> Enum.to_list
  end
  defp _optimize_each(nil, nil, lr) do
    %Param{w: nil, b: nil}
  end
  defp _optimize_each(%Param{w: w, b: nil}, %Grad{dx: _, dw: dw, db: nil}, lr) do
    %Param{
      w: Matrex.subtract(w,
        dw
        |> MatrexUtils.broad_cast(:columns, w[:rows])
        |> Matrex.multiply(lr)),
      b: nil
    }
  end
  defp _optimize_each(%Param{w: w, b: b}, %Grad{dx: _, dw: dw, db: db}, lr) do
    %Param{
      w: Matrex.subtract(w,
        dw
        |> MatrexUtils.broad_cast(:columns, w[:rows])
        |> Matrex.multiply(lr)),
      b: Matrex.subtract(b,
        dw
        |> MatrexUtils.broad_cast(:columns, b[:rows])
        |> Matrex.multiply(lr))
    }
  end
end