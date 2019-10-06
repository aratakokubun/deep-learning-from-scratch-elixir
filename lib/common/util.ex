defmodule Util do
  require Matrex
  require MatrexUtils

  def preprocess(text) do
    words =
      text
      |> String.downcase()
      |> String.replace(".", " .")
      |> String.split(" ")

    {word_to_id, id_to_word} = Enum.reduce(
      words,
      {Map.new, Map.new},
      fn word, {word_to_id, id_to_word} -> _append(word_to_id, id_to_word, word) end)

    corpus =
      words
      |> Enum.map(fn word -> word_to_id[word] end)
      |> MatrexUtils.new()

    {corpus, word_to_id, id_to_word}
  end

  defp _append(word_to_id, id_to_word, word) do
    case Map.has_key?(word_to_id, word) do
      true  -> {word_to_id, id_to_word}
      false -> with new_id = Map.keys(word_to_id) |> length() |> Kernel.+(1) do
                   {Map.put(word_to_id, word, new_id), Map.put(id_to_word, new_id, word)}
               end
    end
  end

  def create_co_matrix(corpus, vocab_size, window_size \\ 1) do
    corpus_size = corpus[:rows]
    co_matrix = Matrex.zeros({vocab_size, vocab_size})
    corpus
    |> Enum.with_index()
    |> Enum.reduce(
        co_matrix,
        fn ({word_id, index}, acc) -> _to_word_vec(acc, window_size, corpus_size, Kernel.trunc(word_id), index+1) end)
  end

  defp _to_word_vec(co_matrix, window_size, corpus_size, word_id, idx) do
    -window_size..window_size
    |> Enum.reduce(co_matrix, fn (index, acc) -> _to_co(acc, window_size, corpus_size, word_id, idx, idx + index) end)
  end
  defp _to_co(co_matrix, window_size, corpus_size, word_id, idx, window_idx)
       when window_idx < 1 or corpus_size < window_idx or window_idx == idx, do: co_matrix
  defp _to_co(co_matrix, window_size, corpus_size, word_id, idx, window_idx),
       do: Matrex.set(co_matrix, word_id, window_idx, co_matrix[word_id][window_idx] + 1)
end