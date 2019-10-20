defmodule ParamStore do
  @moduledoc """
  OTP Server to store learning parameters.
  """

  use GenServer

  def init(state), do: {:ok, state}

  def start_link(state \\ %{}) do
    GenServer.start_link(__MODULE__, state, name: __MODULE__)
  end

  def store(key, value) do
    GenServer.cast(__MODULE__, {:store, key, value})
  end

  def lookup(key) do
    GenServer.call(__MODULE__, {:lookup, key})
  end

  def dump do
    GenServer.call(__MODULE__, :dump)
  end

  def handle_cast({:store, key, value}, state) do
    {:noreply, Map.put(state, key, value)}
  end

  def handle_call({:lookup, key}, _from, state) do
    {:reply, state[key], state}
  end

  def handle_call(:dump, _from, state) do
    {:reply, state, state}
  end
end