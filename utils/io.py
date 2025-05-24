import json
from pathlib import Path
from typing import Union, Any, List

def save_json(obj: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    Save a Python object as a JSON file.

    Args:
        obj: The Python object to serialize.
        path: The file path where the JSON will be saved.
        indent: Indentation level for pretty-printing.
    """
    path = Path(path)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=indent)
    print(f"Saved JSON to {path}")


def read_json(path: Union[str, Path]) -> Any:
    """
    Read a JSON file and return the corresponding Python object.

    Args:
        path: Path to the JSON file.

    Returns:
        The deserialized Python object.
    """
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def get_files(dir_path: Union[str, Path], recursive: bool = True, only_files: bool = True) -> List[Path]:
    """
    Get all files in a directory.

    Args:
        dir_path: Path to the directory.
        recursive: Whether to include files in subdirectories.
        only_files: Whether to return only files (not directories).

    Returns:
        A list of Path objects.
    """
    dir_path = Path(dir_path)
    if recursive:
        paths = dir_path.rglob("*")
    else:
        paths = dir_path.glob("*")

    if only_files:
        return [p for p in paths if p.is_file()]
    return list(paths)
