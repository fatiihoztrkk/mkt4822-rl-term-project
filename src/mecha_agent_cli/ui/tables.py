"""Rich table helpers."""

from rich.table import Table


def key_value_table(title: str, rows: dict[str, object]) -> Table:
    """Create a key/value table."""
    table = Table(title=title)
    table.add_column("Key")
    table.add_column("Value")
    for key, value in rows.items():
        table.add_row(key, str(value))
    return table
