defmodule Random do
  import Np
  alias Statistics.Distributions.Normal, as: Normal

  def randn() do
    Normal.rand()
  end

  def randn(row) do
    1..row
    |> Enum.map(fn _ -> Normal.rand() end)
    |> Np.new()
  end

  def randn(row, col) do
    1..row*col
    |> Enum.map(fn _ -> Normal.rand() end)
    |> Enum.chunk_every(col)
    |> Np.new()
  end

  def sigmoid(%Array{array: v, shape: {_, nil}}) do
    v
    |> Enum.map(&(1/(1+ :math.exp(-1 * &1))))
    |> Np.new
  end
end
