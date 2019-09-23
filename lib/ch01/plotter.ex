defmodule Plotter do
  require TwoLayerNet

  require Matrex
  require MatrexUtils
  require Expyplot.Plot
  require ShowSpiral

  def plot(%Matrex{} = xs, params, [_|_] = loss_list) do
    plot_learning_curve(loss_list)
    plot_boundary(xs, params)
  end

  def plot_learning_curve([_|_] = loss_list) do
    Expyplot.Plot.plot([1..length(loss_list) |> Enum.to_list(), loss_list], %{label: 'train'})
    Expyplot.Plot.xlabel("iterations (x10)")
    Expyplot.Plot.ylabel("loss")
    Expyplot.Plot.show()
  end

  def plot_boundary(%Matrex{} = xs, params) do
    h = 0.01
    {x_min, x_max} = {Matrex.column(xs, 1)[:min] - 0.1, Matrex.column(xs, 1)[:max] + 0.1}
    {y_min, y_max} = {Matrex.column(xs, 2)[:min] - 0.1, Matrex.column(xs, 2)[:max] + 0.1}
    {xx, yy} = MatrexUtils.meshgrid(MatrexUtils.arrange(x_min, x_max, h), MatrexUtils.arrange(y_min, y_max, h))
    {mesh_x, mesh_y} = {MatrexUtils.flattened(xx), MatrexUtils.flattened(yy)}
    score = Matrex.concat(mesh_x, mesh_y) |> TwoLayerNet.predict(params)
    predict_cls = MatrexUtils.argmax(score, :columns)
    z = Matrex.reshape(Matrex.new([predict_cls]), length(xx), Kernel.div(length(predict_cls), length(xx)))
    Expyplot.Plot.contourf([xx, yy, Matrex.to_list_of_lists(z)])
    # TODO: Corresponding function for plt.axis("off")?
    ShowSpiral.plot()
  end
end
