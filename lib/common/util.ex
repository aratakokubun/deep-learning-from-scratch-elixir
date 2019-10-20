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
    corpus_size = corpus[:cols]
    co_matrix = Matrex.zeros({vocab_size, vocab_size})
    corpus
    |> Enum.with_index()
    |> Enum.reduce(
        co_matrix,
        # Note: Matrex manages variables as float so that the word id must be trunc to integer here.
        fn ({word_id, index}, acc) ->
          _create_co_row(acc, corpus, Kernel.trunc(word_id), index + 1, window_size, corpus_size) end)
  end

  defp _create_co_row(co_matrix, corpus, word_id, word_index, window_size, corpus_size) do
    -window_size..window_size
    |> Enum.reduce(
         co_matrix,
         fn (index_diff, acc) -> _create_co_cell(acc, corpus, word_id, word_index, word_index + index_diff, corpus_size) end)
  end
  defp _create_co_cell(co_matrix, _, _, word_index, window_index, corpus_size)
       when window_index < 1 or corpus_size <= window_index or window_index == word_index, do: co_matrix
  defp _create_co_cell(co_matrix, corpus, word_id, _, window_index, _) do
    with window_word_id = corpus[window_index] |> Kernel.trunc() do
      co_matrix
      |> Matrex.set(word_id, window_word_id, co_matrix[word_id][window_word_id] + 1)
    end
  end

  def cos_similarity(x, y, eps \\ 1.0e-8) do
    nx = MatrexUtils.l2_normalize(x, eps)
    ny = MatrexUtils.l2_normalize(y, eps)
    Matrex.multiply(nx, ny)
    |> Matrex.sum()
  end
end