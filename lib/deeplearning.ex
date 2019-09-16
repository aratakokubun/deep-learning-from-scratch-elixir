defmodule Deeplearning do
  use Application
  import ParamStore

  def start(_type, _args) do
    import Supervisor.Spec, warn: false

    children = [
      worker(ParamStore, [])
    ]

    opts = [strategy: :one_for_one, name: ParamStore]
    Supervisor.start_link(children, opts)
  end
end
