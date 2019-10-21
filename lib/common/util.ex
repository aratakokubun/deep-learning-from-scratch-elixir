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
       when window_index < 1 or corpus_size < window_index or window_index == word_index, do: co_matrix
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

  def most_similar(query, word_to_id, id_to_word, word_matrix, top \\ 5) do

    case Map.has_key?(word_to_id, query) do
      true  -> {:ok, _most_similar(query, word_to_id, id_to_word, word_matrix, top)}
      false -> {:error, "#{query} not found in dictionary"}
    end
  end

  defp _most_similar(query, word_to_id, id_to_word, word_matrix, top) do
     query_id = word_to_id[query]
     query_vec = word_matrix[query_id]

     vocab_size = Map.keys(id_to_word) |> length()
     1..vocab_size
     |> Enum.map(fn word_index -> {cos_similarity(word_matrix[word_index], query_vec), word_index} end)
     |> Enum.filter(fn {_, i} -> id_to_word[i] != query end)
     |> Enum.sort(fn ({s1, _}, {s2, _}) -> s1 >= s2 end)
     |> Enum.take(top)
     |> Enum.each(fn {s, i} -> IO.puts("#{id_to_word[i]}: #{s}") end)
  end

  @doc """
  Calculate 'P'ositive 'P'ointwise 'M'lutiple 'I'nformation.
  @param verbose: Put progress on console if true.
  @param eps: Tiny value to avoid 0 division. Do not modify this if required.
  """
  def ppmi(
      co = %Matrex{data: <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32, _::binary>>},
      verbose \\ false, eps \\ 1.0e-8) do
    count_all_occur = Matrex.sum(co)
    word_to_all_occur = MatrexUtils.sum(co, :rows)
    count_total_loop = rows * columns

    1..rows
    |> Enum.map(fn row -> Enum.map(1..columns,
      fn col ->
        with count = row * columns + col do
          _ppmi_cell(co, count_all_occur, word_to_all_occur, row, col, eps, verbose, count_total_loop, count)
        end
      end)
    end)
    |> Matrex.new()
  end

  def _ppmi_cell(
        co = %Matrex{}, count_all_occur, word_to_all_occur, target_row, target_col, eps,
        verbose, count_total_loop, count_current_loop)
      when verbose and rem(count_current_loop, Kernel.trunc(count_total_loop / 100)) == 0 do
    _ppmi_cell_inner(co, count_all_occur, word_to_all_occur, target_row, target_col, eps)
    |> _ppmi_verbose(count_total_loop, count_current_loop)
  end
  def _ppmi_cell(
        co = %Matrex{}, count_all_occur, word_to_all_occur, target_row, target_col, eps,
        _, _, _) do
    _ppmi_cell_inner(co, count_all_occur, word_to_all_occur, target_row, target_col, eps)
  end
  def _ppmi_cell_inner(co = %Matrex{}, count_all_occur, word_to_all_occur, target_row, target_col, eps) do
    co[target_row][target_col] * count_all_occur / (word_to_all_occur[target_row] * word_to_all_occur[target_col])
    |> Kernel.+(eps)
    |> :math.log2()
    |> max(0)
  end
  def _ppmi_verbose(ppmi_cell, count_total_loop, count_current_loop) do
    IO.puts("#{100 * count_current_loop / count_total_loop} % done.")
    ppmi_cell
  end
end