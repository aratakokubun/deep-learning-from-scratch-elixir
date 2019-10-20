defmodule Similarity do
  require Util

  def main do
    {corpus, word_to_id, id_to_word} =
      "You say goodbye and I say hello."
      |> Util.preprocess()
    co = Util.create_co_matrix(corpus, Map.keys(word_to_id) |> length())

    co_you = word_to_id |> Map.get("you") |> (&co[&1]).()
    co_i   = word_to_id |> Map.get("i")   |> (&co[&1]).()
    Util.cos_similarity(co_you, co_i)
  end
end