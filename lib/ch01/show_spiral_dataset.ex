defmodule ShowSpiral do
  import Matrex
  import Spiral
  import Expyplot.Plot

  @marker ["o", "x", "^"]

  def plot() do
    {plots, ts} = Spiral.load_data()
    IO.inspect(plots[:size])
    IO.inspect(ts[:size])

    ts
    |> Matrex.to_list_of_lists()
    |> Enum.map(fn one_hot -> _to_marker(one_hot) end)
    |> Enum.zip(Matrex.to_list_of_lists(plots))
    |> Enum.each(
         fn {marker, [x, y]} -> Expyplot.Plot.scatter(x, y, [s: 'None', marker: marker]) end)
    Expyplot.Plot.show()
  end

  defp _to_marker(one_hot, marker \\ @marker) do
    Enum.zip(one_hot, marker)
    |> Enum.reduce("",
         fn ({x, marker}, acc) -> acc <> String.duplicate(marker, Kernel.trunc(x)) end)
  end
end