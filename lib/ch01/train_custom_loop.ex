defmodule TrainCustomLoop do
  import TwoLayerNet
  import Sgd

  def main do
    params = nil
    batches = nil
    _batch(params, batches, [], [])
  end

  defp _batch([_, _] = params, [min_batch | batches], acc_params, acc_grads) do
    a = [_, _] = TwoLayerNet.predict(params)
    b = [_, _] = TwoLayerNet.back_propagate(a, params)
    c = [_, _] = Sgd.optimize(b, params)
    _batch(c, batches, [acc_params] ++ [c], [acc_grads] ++ [b])
  end
  defp _batch(_, [], acc_params, acc_grads), do: {acc_params, acc_grads}
end