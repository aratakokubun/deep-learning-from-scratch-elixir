defmodule Functions do
  import Np

  def cross_entropy_error(y, t) do
    {y, t}
    |> _reshape2d
    |> _reshape_1hot_vec
    |> _cross_entropy_error
  end

  defp _reshape2d({y, t}) do
    cond do
      Np.ndim(y) == 1          -> {Np.reshape(y, 1, Np.size(y)), Np.reshape(t, 1, Np.size(t))}
      true                     -> {y, t}
    end
  end
  defp _reshape_1hot_vec({y, t}) do
    cond do
      Np.size(t) == Np.size(y) -> {y, Np.argmax(t, :row)}
      true                     -> {y, t}
    end
  end
  defp _cross_entropy_error(y, t) do
    # TODO
  end
end