defmodule Deeplearning.MixProject do
  use Mix.Project

  def project do
    [
      app: :deeplearning,
      version: "0.1.0",
      elixir: "~> 1.9",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:matrex, "~> 0.6.0", override: true},
      {:expyplot, "~> 1.2.2", override: true},
      {:earmark, "~> 1.3.2", override: true},
      {:ex_doc, "~> 0.20.2", override: true},
    ]
  end
end
