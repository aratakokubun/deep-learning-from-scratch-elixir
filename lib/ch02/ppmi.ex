defmodule Ppmi do
  require Util

  def main do
    {corpus, word_to_id, id_to_word} =
      "You say goodbye and I say hello."
      |> Util.preprocess()
    co = Util.create_co_matrix(corpus, Map.keys(word_to_id) |> length())

    ppmi = Util.ppmi(co)
    IO.puts("Covariance matrix:")
    IO.inspect(co)
    IO.puts(String.duplicate("-", 50))
    IO.puts("PPMI:")
    IO.inspect(ppmi)
  end
end
