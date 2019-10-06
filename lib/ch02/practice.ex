defmodule Practice do
  import Util

  def main do
    {corpus, word_to_id, id_to_word} =
      "You say goodbye and I say hello."
      |> Util.preprocess()
    Util.create_co_matrix(corpus, Map.keys(word_to_id) |> length())
  end
end