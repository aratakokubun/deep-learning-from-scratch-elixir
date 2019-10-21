defmodule MostSimilar do
  require Util

  def main do
    {corpus, word_to_id, id_to_word} =
      "You say goodbye and I say hello."
      |> Util.preprocess()
    co = Util.create_co_matrix(corpus, Map.keys(word_to_id) |> length())

    Util.most_similar("you", word_to_id, id_to_word, co, 5)
  end
end
