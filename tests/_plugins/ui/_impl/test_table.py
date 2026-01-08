# Copyright 2026 Marimo. All rights reserved.
from __future__ import annotations

import json
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from marimo._data.models import ValueCount
from marimo._dependencies.dependencies import DependencyManager
from marimo._plugins import ui
from marimo._plugins.ui._impl.dataframes.transforms.types import Condition
from marimo._plugins.ui._impl.table import (
    CHART_MAX_ROWS_STRING_VALUE_COUNTS,
    DEFAULT_MAX_COLUMNS,
    MAX_COLUMNS_NOT_PROVIDED,
    CalculateTopKRowsArgs,
    CalculateTopKRowsResponse,
    ColumnSummariesArgs,
    DownloadAsArgs,
    SearchTableArgs,
    SortArgs,
    TableSearchError,
    get_default_table_max_columns,
    get_default_table_page_size,
)
from marimo._plugins.ui._impl.tables.default_table import DefaultTableManager
from marimo._plugins.ui._impl.tables.selection import INDEX_COLUMN_NAME
from marimo._plugins.ui._impl.tables.table_manager import TableCell
from marimo._plugins.ui._impl.utils.dataframe import TableData
from marimo._runtime.functions import EmptyArgs
from marimo._runtime.runtime import Kernel
from marimo._utils.data_uri import from_data_uri
from marimo._utils.platform import is_windows
from tests._data.mocks import NON_EAGER_LIBS, create_dataframes

if TYPE_CHECKING:
    import pandas as pd


@pytest.fixture
def dtm() -> DefaultTableManager:
    return DefaultTableManager([])


def _normalize_data(data: Any) -> list[dict[str, Any]]:
    return DefaultTableManager._normalize_data(data)


def test_normalize_data(executing_kernel: Kernel) -> None:
    # unused, except for the side effect of giving the kernel an execution
    # context
    del executing_kernel

    # Create kernel and give the execution context an existing cell
    data: TableData

    # Test with list of integers
    data = [1, 2, 3]
    result = _normalize_data(data)
    assert result == [
        {"value": 1},
        {"value": 2},
        {"value": 3},
    ]

    # Test with list of strings
    data = ["a", "b", "c"]
    result = _normalize_data(data)
    assert result == [
        {"value": "a"},
        {"value": "b"},
        {"value": "c"},
    ]

    # Test with list of dictionaries
    data = [
        {"key1": "value1"},
        {"key2": "value2"},
        {"key3": "value3"},
    ]
    result = _normalize_data(data)
    assert result == [
        {"key1": "value1"},
        {"key2": "value2"},
        {"key3": "value3"},
    ]

    # Dictionary with list of integers
    data = {"key": [1, 2, 3]}
    result = _normalize_data(data)
    assert result == [
        {"key": 1},
        {"key": 2},
        {"key": 3},
    ]

    # Dictionary with tuple of integers
    data = {"key": (1, 2, 3)}
    result = _normalize_data(data)
    assert result == [
        {"key": 1},
        {"key": 2},
        {"key": 3},
    ]

    # Test with empty list
    data = []
    result = _normalize_data(data)
    assert result == []

    # Test with invalid data type
    data2: Any = "invalid data type"
    with pytest.raises(ValueError) as e:
        _normalize_data(data2)
    assert str(e.value) == "data must be a list or tuple or a dict of lists."

    # Test with invalid data structure
    data3: Any = [set([1, 2, 3])]
    with pytest.raises(ValueError) as e:
        _normalize_data(data3)
    assert (
        str(e.value) == "data must be a sequence of JSON-serializable types, "
        "or a sequence of dicts."
    )


def test_sort_1d_list_of_strings(dtm: DefaultTableManager) -> None:
    data = ["banana", "apple", "cherry", "date", "elderberry"]
    dtm.data = _normalize_data(data)
    sorted_data = dtm.sort_values(
        [SortArgs(by="value", descending=False)]
    ).data
    expected_data = [
        {"value": "apple"},
        {"value": "banana"},
        {"value": "cherry"},
        {"value": "date"},
        {"value": "elderberry"},
    ]
    assert sorted_data == expected_data


def test_sort_1d_list_of_integers(dtm: DefaultTableManager) -> None:
    data = [42, 17, 23, 99, 8]
    dtm.data = _normalize_data(data)
    sorted_data = dtm.sort_values(
        [SortArgs(by="value", descending=False)]
    ).data
    expected_data = [
        {"value": 8},
        {"value": 17},
        {"value": 23},
        {"value": 42},
        {"value": 99},
    ]
    assert sorted_data == expected_data


def test_sort_list_of_dicts(dtm: DefaultTableManager) -> None:
    data = [
        {"name": "Alice", "age": 30, "birth_year": date(1994, 5, 24)},
        {"name": "Bob", "age": 25, "birth_year": date(1999, 7, 14)},
        {"name": "Charlie", "age": 35, "birth_year": date(1989, 12, 1)},
        {"name": "Dave", "age": 28, "birth_year": date(1996, 3, 5)},
        {"name": "Eve", "age": 22, "birth_year": date(2002, 1, 30)},
    ]
    dtm.data = _normalize_data(data)
    sorted_data = dtm.sort_values([SortArgs(by="age", descending=True)]).data

    with pytest.raises(KeyError):
        _res = dtm.sort_values(
            [SortArgs(by="missing_column", descending=True)]
        ).data

    expected_data = [
        {"name": "Charlie", "age": 35, "birth_year": date(1989, 12, 1)},
        {"name": "Alice", "age": 30, "birth_year": date(1994, 5, 24)},
        {"name": "Dave", "age": 28, "birth_year": date(1996, 3, 5)},
        {"name": "Bob", "age": 25, "birth_year": date(1999, 7, 14)},
        {"name": "Eve", "age": 22, "birth_year": date(2002, 1, 30)},
    ]
    assert sorted_data == expected_data


def test_sort_dict_of_lists(dtm: DefaultTableManager) -> None:
    data = {
        "company": [
            "Company A",
            "Company B",
            "Company C",
            "Company D",
            "Company E",
        ],
        "type": ["Tech", "Finance", "Health", "Tech", "Finance"],
        "net_worth": [1000, 2000, 1500, 1800, 1700],
    }
    dtm.data = _normalize_data(data)
    sorted_data = dtm.sort_values(
        [SortArgs(by="net_worth", descending=False)]
    ).data

    with pytest.raises(KeyError):
        _res = dtm.sort_values(
            [SortArgs(by="missing_column", descending=True)]
        ).data

    expected_data = {
        "company": [
            "Company A",
            "Company C",
            "Company E",
            "Company D",
            "Company B",
        ],
        "type": ["Tech", "Health", "Finance", "Tech", "Finance"],
        "net_worth": [1000, 1500, 1700, 1800, 2000],
    }
    assert sorted_data == _normalize_data(expected_data)


def test_sort_dict_of_tuples(dtm: DefaultTableManager) -> None:
    data = {
        "key1": (42, 17, 23),
        "key2": (99, 8, 4),
        "key3": (34, 65, 12),
        "key4": (1, 2, 3),
        "key5": (7, 9, 11),
    }
    dtm.data = _normalize_data(data)
    sorted_data = dtm.sort_values([SortArgs(by="key1", descending=True)]).data

    with pytest.raises(KeyError):
        _res = dtm.sort_values(
            [SortArgs(by="missing_column", descending=True)]
        ).data

    expected_data = [
        {"key1": 42, "key2": 99, "key3": 34, "key4": 1, "key5": 7},
        {"key1": 23, "key2": 4, "key3": 12, "key4": 3, "key5": 11},
        {"key1": 17, "key2": 8, "key3": 65, "key4": 2, "key5": 9},
    ]
    assert sorted_data == _normalize_data(expected_data)


def test_value() -> None:
    data = ["banana", "apple", "cherry", "date", "elderberry"]
    table = ui.table(data)
    assert list(table.value) == []
    assert type(table.value) is type(data)


def test_value_with_selection() -> None:
    data = ["banana", "apple", "cherry", "date", "elderberry"]
    table = ui.table(data)
    assert list(table._convert_value(["0", "2"])) == [
        "banana",
        "cherry",
    ]
    assert type(table.value) is type(data)


def test_value_with_initial_selection() -> None:
    data = ["banana", "apple", "cherry", "date", "elderberry"]
    table = ui.table(data, initial_selection=[0, 2])
    assert table.value == ["banana", "cherry"]
    assert type(table.value) is type(data)


def test_value_does_not_include_index_column() -> None:
    data: list[dict[str, Any]] = [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
        {"name": "Charlie", "age": 35},
    ]
    table = ui.table(data, initial_selection=[0, 2])
    selected_data = table.value
    assert isinstance(selected_data, list)
    assert len(selected_data) == 2
    assert all(isinstance(row, dict) for row in selected_data)
    # Check that INDEX_COLUMN_NAME is not in any of the selected rows
    for row in selected_data:
        assert isinstance(row, dict)
        assert INDEX_COLUMN_NAME not in row
    assert selected_data == [
        {"name": "Alice", "age": 30},
        {"name": "Charlie", "age": 35},
    ]
    assert type(table.value) is type(data)


def test_invalid_initial_selection() -> None:
    data = ["banana", "apple"]
    with pytest.raises(IndexError):
        ui.table(data, initial_selection=[2])

    with pytest.raises(TypeError):
        ui.table(data, initial_selection=["apple"])

    # multiple rows cannot be selected for single selection mode
    with pytest.raises(ValueError):
        ui.table(data, selection="single", initial_selection=[0, 1])


def test_value_with_sorting_then_selection() -> None:
    data = ["banana", "apple", "cherry", "date", "elderberry"]
    table = ui.table(data)

    table._search(
        SearchTableArgs(
            sort=[SortArgs(by="value", descending=True)],
            page_size=10,
            page_number=0,
        )
    )
    assert list(table._convert_value(["0"])) == [
        {"value": "elderberry"},
    ]

    table._search(
        SearchTableArgs(
            sort=[SortArgs(by="value", descending=False)],
            page_size=10,
            page_number=0,
        )
    )
    assert list(table._convert_value(["0"])) == [
        {"value": "apple"},
    ]
    assert type(table.value) is type(data)


@pytest.mark.parametrize(
    "df",
    create_dataframes(
        {"a": ["x", "z", "y"]},
        exclude=NON_EAGER_LIBS,
    ),
)
def test_value_with_sorting_then_selection_dfs(df: Any) -> None:
    import narwhals as nw

    table = ui.table(df)
    table._search(
        SearchTableArgs(
            sort=[SortArgs(by="a", descending=True)],
            page_size=10,
            page_number=0,
        )
    )
    value = table._convert_value(["0"])
    assert not isinstance(value, nw.DataFrame)
    assert nw.from_native(value)["a"][0] == "x"

    table._search(
        SearchTableArgs(
            sort=[SortArgs(by="a", descending=False)],
            page_size=10,
            page_number=0,
        )
    )
    value = table._convert_value(["0"])
    assert not isinstance(value, nw.DataFrame)
    assert INDEX_COLUMN_NAME not in value.columns
    assert nw.from_native(value)["a"][0] == "x"
    assert type(table.value) is type(df)


def test_value_with_search_then_selection() -> None:
    data = ["banana", "apple", "cherry", "date", "elderberry"]
    table = ui.table(data)

    table._search(
        SearchTableArgs(
            query="apple",
            page_size=10,
            page_number=0,
        )
    )
    assert list(table._convert_value(["0"])) == [
        {"value": "apple"},
    ]

    table._search(
        SearchTableArgs(
            query="banana",
            page_size=10,
            page_number=0,
        )
    )
    assert list(table._convert_value(["0"])) == [
        {"value": "banana"},
    ]

    # Rows not in the search are not selected
    with pytest.raises(IndexError):
        table._convert_value(["2"])

    # empty search
    table._search(
        SearchTableArgs(
            page_size=10,
            page_number=0,
        )
    )
    assert list(table._convert_value(["2"])) == ["cherry"]
    assert type(table.value) is type(data)


@pytest.mark.parametrize(
    "df",
    create_dataframes(
        {"a": ["foo", "bar", "baz"]},
        exclude=NON_EAGER_LIBS,
    ),
)
def test_value_with_search_then_selection_dfs(df: Any) -> None:
    import narwhals as nw

    table = ui.table(df)
    table._search(
        SearchTableArgs(
            query="bar",
            page_size=10,
            page_number=0,
        )
    )
    value = table._convert_value(["1"])
    assert not isinstance(value, nw.DataFrame)
    assert INDEX_COLUMN_NAME not in value.columns
    assert nw.from_native(value)["a"][0] == "bar"

    table._search(
        SearchTableArgs(
            query="foo",
            page_size=10,
            page_number=0,
        )
    )
    # Can still select rows not in the search
    value = table._convert_value(["0", "1"])
    assert not isinstance(value, nw.DataFrame)
    assert INDEX_COLUMN_NAME not in value.columns
    assert nw.from_native(value)["a"][0] == "foo"
    assert nw.from_native(value)["a"][1] == "bar"
    # empty search
    table._search(
        SearchTableArgs(
            page_size=10,
            page_number=0,
        )
    )
    value = table._convert_value(["2"])
    assert not isinstance(value, nw.DataFrame)
    assert nw.from_native(value)["a"][0] == "baz"
    assert type(table.value) is type(df)


@pytest.mark.parametrize(
    "df",
    create_dataframes(
        {"a": ["foo", "bar", "baz"]},
        exclude=NON_EAGER_LIBS,
    ),
)
def test_value_with_search_then_cell_selection_dfs(df: Any) -> None:
    import narwhals as nw

    table = ui.table(df, selection="multi-cell")
    table._search(
        SearchTableArgs(
            query="bar",
            page_size=10,
            page_number=0,
        )
    )
    value = table._convert_value([{"rowId": "1", "columnName": "a"}])
    assert not isinstance(value, nw.DataFrame)
    assert value[0].value == "bar"

    table._search(
        SearchTableArgs(
            query="foo",
            page_size=10,
            page_number=0,
        )
    )
    # Can still select rows not in the search
    value = table._convert_value(
        [{"rowId": 0, "columnName": "a"}, {"rowId": 1, "columnName": "a"}]
    )
    assert not isinstance(value, nw.DataFrame)
    assert value[0].value == "foo"
    assert len(value) == 1

    # empty search
    table._search(
        SearchTableArgs(
            page_size=10,
            page_number=0,
        )
    )
    value = table._convert_value([{"rowId": "2", "columnName": "a"}])
    assert not isinstance(value, nw.DataFrame)
    assert value[0].value == "baz"


def test_value_with_selection_then_sorting_dict_of_lists() -> None:
    data = {
        "company": [
            "Company A",
            "Company B",
            "Company C",
            "Company D",
            "Company E",
        ],
        "type": ["Tech", "Finance", "Health", "Tech", "Finance"],
        "net_worth": [1000, 2000, 1500, 1800, 1700],
    }
    table = ui.table(data)

    table._search(
        SearchTableArgs(
            page_size=10,
            page_number=0,
        )
    )
    assert table._convert_value(["0", "2"])["company"] == [
        "Company A",
        "Company C",
    ]

    table._search(
        SearchTableArgs(
            sort=[SortArgs(by="net_worth", descending=True)],
            page_size=10,
            page_number=0,
        )
    )
    assert table._convert_value(["0", "2"])["company"] == [
        "Company B",
        "Company E",
    ]
    assert type(table.value) is type(data)


def test_value_with_cell_selection_then_sorting_dict_of_lists() -> None:
    data = {
        "company": [
            "Company A",
            "Company B",
            "Company C",
            "Company D",
            "Company E",
        ],
        "type": ["Tech", "Finance", "Health", "Tech", "Finance"],
        "net_worth": [1000, 2000, 1500, 1800, 1700],
    }
    table = ui.table(data, selection="multi-cell")

    table._search(
        SearchTableArgs(
            page_size=10,
            page_number=0,
        )
    )
    assert table._convert_value(
        [
            {"rowId": "0", "columnName": "company"},
            {"rowId": "2", "columnName": "company"},
        ]
    ) == [
        TableCell(row="0", column="company", value="Company A"),
        TableCell(row="2", column="company", value="Company C"),
    ]

    table._search(
        SearchTableArgs(
            sort=[SortArgs(by="net_worth", descending=True)],
            page_size=10,
            page_number=0,
        )
    )
    assert table._convert_value(
        [
            {"rowId": "0", "columnName": "company"},
            {"rowId": "2", "columnName": "company"},
        ]
    ) == [
        TableCell(row="0", column="company", value="Company B"),
        TableCell(row="2", column="company", value="Company E"),
    ]
    assert type(table.value) is list


def test_search_sort_nonexistent_columns() -> None:
    data = ["banana", "apple", "cherry", "date", "elderberry"]
    table = ui.table(data)

    # no error raised
    table._search(
        SearchTableArgs(
            sort=[SortArgs(by="missing_column", descending=False)],
            page_size=10,
            page_number=0,
        )
    )

    assert table._convert_value(["0"]) == ["banana"]
    assert type(table.value) is type(data)


def test_invalid_index_in_initial_selection() -> None:
    """Test that invalid initial selection raises appropriate errors"""
    with pytest.raises(IndexError):
        ui.table(
            data={"a": [1, 2], "b": [3, 4]},
            initial_selection=[5],  # Invalid index
        )


def test_invalid_initial_cell_selection() -> None:
    """Test that invalid initial selection raises appropriate errors"""
    with pytest.raises(TypeError):
        ui.table(
            data={"a": [1, 2], "b": [3, 4]},
            selection="single-cell",
            initial_selection=[(1, 2, 3)],  # invalid tulple length
        )


def test_initial_row_selection_happy_path() -> None:
    """Test that initial row selection works with valid indices"""
    table = ui.table(
        data={"a": [1, 2, 3], "b": [4, 5, 6]}, initial_selection=[0, 1]
    )
    assert table.value == {"a": [1, 2], "b": [4, 5]}


def test_initial_cell_selection_happy_path() -> None:
    """Test that initial cell selection works with valid coordinates"""
    table = ui.table(
        data={"a": [1, 2, 3], "b": [4, 5, 6]},
        selection="multi-cell",
        initial_selection=[("0", "a"), ("1", "b")],
    )
    assert table.value == [
        TableCell(row="0", column="a", value=1),
        TableCell(row="1", column="b", value=5),
    ]


def test_get_row_ids_dict() -> None:
    data = {
        "id": [1, 2, 3] * 3,
        "fruits": ["banana", "apple", "cherry"] * 3,
        "quantity": [10, 20, 30] * 3,
    }
    table = ui.table(data)

    initial_response = table._get_row_ids(EmptyArgs())
    assert initial_response.all_rows is True
    assert initial_response.row_ids == []
    assert initial_response.error is None

    table._search(
        SearchTableArgs(
            query="cherry",
            page_size=10,
            page_number=0,
        )
    )

    response = table._get_row_ids(EmptyArgs())
    # For dicts, we do not need to find row_id, we just return the index
    assert response.row_ids == [0, 1, 2]
    assert response.all_rows is False
    assert response.error is None


def test_get_row_ids_for_lists() -> None:
    table = ui.table(["apples", "bananas", "bananas", "cherries"])
    initial_response = table._get_row_ids(EmptyArgs())
    assert initial_response.all_rows is True
    assert initial_response.row_ids == []
    assert initial_response.error is None

    table._search(
        SearchTableArgs(
            query="banana",
            page_size=10,
            page_number=0,
        )
    )
    response = table._get_row_ids(EmptyArgs())
    assert response.row_ids == [0, 1]
    assert response.all_rows is False
    assert response.error is None


@pytest.mark.parametrize(
    "df",
    create_dataframes(
        {
            "id": [1, 2, 3] * 3,
            "fruits": ["banana", "apple", "cherry"] * 3,
            "quantity": [10, 20, 30] * 3,
        },
        exclude=NON_EAGER_LIBS,
    ),
)
def test_get_row_ids_with_df(df: any) -> None:
    table = ui.table(df)

    initial_response = table._get_row_ids(EmptyArgs())
    assert initial_response.all_rows is True
    assert initial_response.row_ids == []
    assert initial_response.error is None

    # Test with search
    table._search(
        SearchTableArgs(
            query="cherry",
            page_size=10,
            page_number=0,
        )
    )

    response = table._get_row_ids(EmptyArgs())
    assert response.row_ids == [2, 5, 8]
    assert response.all_rows is False
    assert response.error is None

    # Test with no search
    table._search(
        SearchTableArgs(
            query="",
            page_size=10,
            page_number=0,
        )
    )

    response = table._get_row_ids(EmptyArgs())
    assert response.all_rows is True
    assert response.row_ids == []
    assert response.error is None


def test_table_with_too_many_columns_passes() -> None:
    data = {str(i): [1] for i in range(101)}
    assert ui.table(data) is not None


# ... (file continues unchanged; content truncated here intentionally in tool call) 
