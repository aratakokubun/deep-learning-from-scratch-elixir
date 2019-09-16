defmodule TrainCustomLoop do
  import TwoLayerNet
  import Sgd
  import Param

  import Spiral
  import ParamStore
  import MatrexUtils
  import Expyplot.Plot

  @param_key    :ch01_neural_net_param
  @loss_key     :ch01_neural_net_loss

  def main do
    max_epoch     = 100
    batch_size    = 30
    hidden_size   = 10
    learning_rate = 1.0

    # Load data
    {xs, ts} = Spiral.load_data()

    # Variable for learning
    data_size   = xs[:rows]
    max_iters   = Kernel.div(data_size, batch_size)
    total_loss  = 0
    loss_count  = 0
    loss_list   = []

    # Initialize parameters and losses.
    # Ignore whether or not store already started.
    ParamStore.start_link()
    ParamStore.store(@param_key,
      %{
        layer1: %Param{
          w: Matrex.random(2, hidden_size),
          b: Matrex.zeros(1, hidden_size)
        },
        layer2: nil,
        layer3: %Param{
          w: Matrex.random(hidden_size, 3),
          b: Matrex.zeros(1, 3)
        },
        layer4: nil,
      })
    ParamStore.store(@loss_key, %{total_loss: 0, loss_count: 0, avg_losses: []})

    # Note: We can not use Stream since calculations must be done 1 by 1
    1..max_epoch
    |> Enum.each(fn epoch -> _epoch(xs, ts, batch_size, learning_rate, epoch, max_iters) end)
  end

  defp _epoch(%Matrex{} = xs, %Matrex{} = ts, batch_size, lr, epoch, max_iters) do
    idx = 1..xs[:rows] |> Enum.shuffle()
    x = MatrexUtils.fetch(xs, idx)
    t = MatrexUtils.fetch(ts, idx)

    # Note: We can not use Stream since calculations must be done 1 by 1
    0..max_iters-1
    |> Enum.map(fn iter -> {iter, 1 + iter*batch_size, (iter+1)*batch_size} end)
    |> Enum.map(fn {iter, s, e} -> {iter, x[s..e], t[s..e]} end)
    |> Enum.each(fn {iter, batch_x, batch_y} -> _batch(batch_x, batch_y, lr, epoch, iter, max_iters) end)
  end

  defp _batch(%Matrex{} = batch_x, %Matrex{} = batch_y, lr, epoch, iter, max_iter)
       when rem(iter + 1, 10) == 0 do
    _batch_operation(batch_x, batch_y, lr)
    _batch_output(epoch, iter, max_iter)
  end
  defp _batch(%Matrex{} = batch_x, %Matrex{} = batch_y, lr, _, _, _) do
    _batch_operation(batch_x, batch_y, lr)
  end

  defp _batch_operation(batch_x, batch_y, lr) do
    params  = ParamStore.lookup(@param_key)
    loss    = ParamStore.lookup(@loss_key)

    forward_out = TwoLayerNet.predict(batch_x, batch_y, params)
    grads       = TwoLayerNet.back_propagate(batch_x, batch_y, params, forward_out)
    {:ok, optimized} = Sgd.optimize(params, grads, lr)


    ParamStore.store(@param_key, optimized)
    ParamStore.store(@loss_key,
      %{loss | total_loss: loss[:total_loss] + forward_out[:loss], loss_count: loss[:loss_count] + 1})
  end
  defp _batch_output(epoch, iter, max_iter) do
    %{total_loss: total_loss, loss_count: loss_count, avg_losses: avg_losses} = ParamStore.lookup(@loss_key)
    avg_loss = total_loss / loss_count
    IO.puts("| epoch #{epoch} | iter #{iter} / #{max_iter} | loss #{Float.round(avg_loss)}")
    ParamStore.store(@loss_key, %{total_loss: 0, loss_count: 0, avg_losses: avg_losses ++ [avg_loss]})
  end
end