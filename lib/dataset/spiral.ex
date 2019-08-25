defmodule Spiral do
  import Matrex

  def load_data(seed \\ {1566, 664769, 600157}) do
    :rand.seed(:exs1024, seed)
    samples_per_cls = 100
    dim = 2
    cls_num = 3

    x = Matrex.zeros(samples_per_cls*cls_num, dim)
    t = Matrex.zeros(samples_per_cls*cls_num, cls_num)

    Enum.to_list(0..samples_per_cls*cls_num-1)
    |> Enum.map(fn n -> {samples_per_cls, cls_num, n, rem(n, cls_num), floor(n / cls_num)} end)
    |> Enum.reduce({[], []}, &_compose_matrix/2)
    |> (fn {xs, ts} -> {Matrex.new(xs), Matrex.new(ts)} end).()
  end

  defp _compose_matrix({samples_per_cls, cls_num, n, cls_index, sample_index}, {xs, ts}) do
    rate = sample_index / samples_per_cls
    radius = 1.0 * rate
    theta = cls_index * 4.0 + 4.0 * rate + :rand.uniform() * 0.2
    ts_row = Matrex.zeros(1, cls_num)
             |> Matrex.set(1, cls_index + 1, 1)
             |> Matrex.to_list_of_lists
    {
      xs ++ [[radius * :math.sin(theta), radius * :math.cos(theta)]],
      ts ++ ts_row
    }
  end
end