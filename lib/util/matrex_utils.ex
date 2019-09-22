defmodule MatrexUtils do
  require Matrex
  @binary_per_data 4

  @doc """
    Adds two matrices which have either matched rows or columns
  """
  def add(%Matrex{} = first, %Matrex{} = second, alpha \\ 1.0, beta \\ 1.0), do: _add(first, second, alpha, beta)

  def _add(
        %Matrex{
          data:
            <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32, _data1::binary>>
        } = first,
        %Matrex{
          data:
            <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32, _data2::binary>>
        } = second,
        alpha,
        beta
      ), do: Matrex.add(first, second, alpha, beta)
  def _add(
        %Matrex{
          data:
            <<rows::unsigned-integer-little-32, columns1::unsigned-integer-little-32, _data1::binary>>
        } = first,
        %Matrex{
          data:
            <<rows::unsigned-integer-little-32, columns2::unsigned-integer-little-32, _data2::binary>>
        } = second,
        alpha,
        beta
      ) when columns1 > columns2 and rem(columns1, columns2) == 0 do
    second
    |> broad_cast(:columns, columns1)
    |> Matrex.add(first, alpha, beta)
  end
  def _add(
        %Matrex{
          data:
            <<rows::unsigned-integer-little-32, columns1::unsigned-integer-little-32, _data1::binary>>
        } = first,
        %Matrex{
          data:
            <<rows::unsigned-integer-little-32, columns2::unsigned-integer-little-32, _data2::binary>>
        } = second,
        alpha,
        beta
      ) when columns2 > columns1 and rem(columns2, columns1) == 0 do
    first
    |> broad_cast(:columns, columns2)
    |> Matrex.add(second, alpha, beta)
  end
  def _add(
        %Matrex{
          data:
            <<rows1::unsigned-integer-little-32, columns::unsigned-integer-little-32, _data1::binary>>
        } = first,
        %Matrex{
          data:
            <<rows2::unsigned-integer-little-32, columns::unsigned-integer-little-32, _data2::binary>>
        } = second,
        alpha,
        beta
      ) when rows1 > rows2 and rem(rows1, rows2) == 0 do
    second
    |> broad_cast(:rows, rows1)
    |> Matrex.add(first, alpha, beta)
  end
  def _add(
        %Matrex{
          data:
            <<rows1::unsigned-integer-little-32, columns::unsigned-integer-little-32, _data1::binary>>
        } = first,
        %Matrex{
          data:
            <<rows2::unsigned-integer-little-32, columns::unsigned-integer-little-32, _data2::binary>>
        } = second,
        alpha,
        beta
      ) when rows2 > rows1 and rem(rows2, rows1) == 0 do
    first
    |> broad_cast(:rows, rows2)
    |> Matrex.add(second, alpha, beta)
  end

  def broad_cast(
         %Matrex{
           data:
             <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32, body::binary>>},
         :columns,
         target_columns
       ) when rem(target_columns, columns) == 0 do
    new_body = 1..rows
               |> Stream.map(fn index -> _parse_binary(columns, body, index) end)
               |> Stream.map(
                    fn binary -> binary
                                 |> List.duplicate(Kernel.div(target_columns, columns))
                                 |> Enum.reduce(fn new, acc -> acc <> new end)
                    end)
               |> Enum.reduce(fn new, acc -> acc <> new end)
    %Matrex{data: <<rows::unsigned-integer-little-32,
              target_columns::unsigned-integer-little-32,
              new_body::binary>>}
  end
  def broad_cast(
         %Matrex{
           data:
             <<rows::unsigned-integer-little-32, columns::unsigned-integer-little-32, body::binary>>},
         :rows,
          target_rows
       ) when rem(target_rows, rows) == 0 do
    new_body = body
               |> List.duplicate(Kernel.div(target_rows, rows))
               |> Enum.reduce(fn new, acc -> acc <> new end)
    %Matrex{data: <<target_rows::unsigned-integer-little-32,
              columns::unsigned-integer-little-32,
              new_body::binary>>}
  end

  @doc """
    Get sum of rows or cols and compose them to Matrex.
  """
  def sum(%Matrex{} = x, :rows) do
    <<rows::unsigned-integer-little-32,
      columns::unsigned-integer-little-32,
      body::binary>> = x.data
    new_body = 1..columns
               |> Stream.map(
                    fn col_index -> 1..rows
                                    |> Stream.map(fn row_index -> _parse_binary(columns, body, row_index, col_index) end)
                                    |> Stream.map(fn <<val::float-little-32>> -> val end)
                                    |> Enum.sum()
                    end)
               |> Enum.reduce(<<>>, fn val, acc -> acc <> <<val::float-little-32>> end)
    %Matrex{data: <<1::unsigned-integer-little-32,
              columns::unsigned-integer-little-32,
              new_body::binary>>}
  end

  def sum(%Matrex{} = x, :columns) do
    <<rows::unsigned-integer-little-32,
      columns::unsigned-integer-little-32,
      body::binary>> = x.data
    new_body = 1..rows
               |> Stream.map(
                    fn row_index -> 1..columns
                                    |> Stream.map(fn col_index -> _parse_binary(columns, body, row_index, col_index) end)
                                    |> Stream.map(fn <<val::float-little-32>> -> val end)
                                    |> Enum.sum()
                    end)
               |> Enum.reduce(<<>>, fn val, acc -> acc <> <<val::float-little-32>> end)
    %Matrex{data: <<rows::unsigned-integer-little-32,
              1::unsigned-integer-little-32,
              new_body::binary>>}
  end

  @doc """
    Fetch data of specified list of rows and compose them to Matrex.
  """
  def fetch(%Matrex{} = x, [_| _] = row_indices) do
    <<_::unsigned-integer-little-32,
             columns::unsigned-integer-little-32,
             body::binary>> = x.data
    new_body = row_indices
               |> Enum.map(fn index -> _parse_binary(columns, body, index) end)
               |> Enum.reduce(<<>>, fn binary, acc -> acc <> binary end)
    %Matrex{data: <<length(row_indices)::unsigned-integer-little-32,
                    columns::unsigned-integer-little-32,
                    new_body::binary>>}
  end

  defp _parse_binary(columns, body, row_index) do
    binary_part(body, (row_index - 1) * columns * @binary_per_data, columns * @binary_per_data)
  end
  defp _parse_binary(columns, body, row_index, col_index) do
    binary_part(body, ((row_index - 1) * columns + col_index - 1) * @binary_per_data, @binary_per_data)
  end

  @doc """
  Create a mesh grid 2d array in the rectangle.
  """
  def meshgrid([_| _] = x_range, [_| _] = y_range) do
    {
      List.duplicate(x_range, length(y_range)),
      y_range |> Enum.map(fn elem -> List.duplicate(elem, length(x_range)) end)
    }
  end

  @doc """
  Create a list in the range with the step.
  """
  def arrange(s, e, step) when s < e do
    Stream.iterate(s, &(&1 + step))
    |> Enum.take(Kernel.trunc((e - s) / step))
  end

  @doc """
  Return argmax for row/column indices.
  """
  def argmax(%Matrex{} = x, :rows) do
    1..x[:columns]
    |> Enum.map(fn column_index -> Matrex.column(x, column_index)[:argmax] end)
  end
  def argmax(%Matrex{} = x, :columns) do
    1..x[:rows]
    |> Enum.map(fn row_index -> x[row_index][:argmax] end)
  end

  @doc """
  Create a matrex of flattened list.
  """
  def flattened([_ | _] = list) do
    list
    |> List.flatten()
    |> (&Matrex.reshape(&1, length(&1), 1)).()
  end
end