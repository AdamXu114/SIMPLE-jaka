"""String/regex list matching helpers.

Ported from sim2real-jaka ``utils/strings.py`` (trimmed to what the Jaka MF
policy stack uses). Used to map regex-keyed yaml dicts (e.g. ``joint_kp``)
onto concrete joint/body name lists.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple, Union


def resolve_matching_names_values(
    data: Dict[str, Any],
    list_of_strings: Sequence[str],
    preserve_order: bool = False,
    strict: bool = True,
) -> Tuple[List[int], List[str], List[Any]]:
    """Match regex keys in ``data`` against ``list_of_strings``.

    Returns ``(indices, names, values)``. When ``preserve_order`` is True the
    ordering follows ``list_of_strings``; when False it follows the dict keys.
    """
    if not isinstance(data, dict):
        raise TypeError(f"Input argument `data` should be a dictionary. Received: {data}")
    index_list: List[int] = []
    names_list: List[str] = []
    values_list: List[Any] = []
    key_idx_list: List[int] = []
    target_strings_match_found: List[Any] = [None for _ in range(len(list_of_strings))]
    keys_match_found: List[List[Any]] = [[] for _ in range(len(data))]

    for target_index, potential_match_string in enumerate(list_of_strings):
        for key_index, (re_key, value) in enumerate(data.items()):
            if re.fullmatch(re_key, potential_match_string):
                if target_strings_match_found[target_index]:
                    raise ValueError(
                        f"Multiple matches for '{potential_match_string}':"
                        f" '{target_strings_match_found[target_index]}' and '{re_key}'!"
                    )
                target_strings_match_found[target_index] = re_key
                index_list.append(target_index)
                names_list.append(potential_match_string)
                values_list.append(value)
                key_idx_list.append(key_index)
                keys_match_found[key_index].append(potential_match_string)

    if preserve_order:
        reordered_index_list = [None] * len(index_list)
        global_index = 0
        for key_index in range(len(data)):
            for key_idx_position, key_idx_entry in enumerate(key_idx_list):
                if key_idx_entry == key_index:
                    reordered_index_list[key_idx_position] = global_index
                    global_index += 1
        index_list_reorder = [None] * len(index_list)
        names_list_reorder = [None] * len(index_list)
        values_list_reorder = [None] * len(index_list)
        for idx, reorder_idx in enumerate(reordered_index_list):
            index_list_reorder[reorder_idx] = index_list[idx]
            names_list_reorder[reorder_idx] = names_list[idx]
            values_list_reorder[reorder_idx] = values_list[idx]
        index_list = index_list_reorder
        names_list = names_list_reorder
        values_list = values_list_reorder

    if strict and not all(keys_match_found):
        msg = "\n"
        for key, value in zip(data.keys(), keys_match_found):
            msg += f"\t{key}: {value}\n"
        msg += f"Available strings: {list_of_strings}\n"
        raise ValueError(f"Not all regular expressions matched: {msg}")

    return index_list, names_list, values_list


def resolve_matching_names(
    keys: Union[str, Sequence[str]],
    list_of_strings: Sequence[str],
    preserve_order: bool = False,
) -> Tuple[List[int], List[str]]:
    """Match regex list ``keys`` against ``list_of_strings`` → (indices, names)."""
    if isinstance(keys, str):
        keys = [keys]
    index_list: List[int] = []
    names_list: List[str] = []
    key_idx_list: List[int] = []
    target_strings_match_found: List[Any] = [None for _ in range(len(list_of_strings))]
    keys_match_found: List[List[Any]] = [[] for _ in range(len(keys))]

    for target_index, potential_match_string in enumerate(list_of_strings):
        for key_index, re_key in enumerate(keys):
            if re.fullmatch(re_key, potential_match_string):
                if target_strings_match_found[target_index]:
                    raise ValueError(
                        f"Multiple matches for '{potential_match_string}':"
                        f" '{target_strings_match_found[target_index]}' and '{re_key}'!"
                    )
                target_strings_match_found[target_index] = re_key
                index_list.append(target_index)
                names_list.append(potential_match_string)
                key_idx_list.append(key_index)
                keys_match_found[key_index].append(potential_match_string)

    if preserve_order:
        reordered_index_list = [None] * len(index_list)
        global_index = 0
        for key_index in range(len(keys)):
            for key_idx_position, key_idx_entry in enumerate(key_idx_list):
                if key_idx_entry == key_index:
                    reordered_index_list[key_idx_position] = global_index
                    global_index += 1
        index_list_reorder = [None] * len(index_list)
        names_list_reorder = [None] * len(index_list)
        for idx, reorder_idx in enumerate(reordered_index_list):
            index_list_reorder[reorder_idx] = index_list[idx]
            names_list_reorder[reorder_idx] = names_list[idx]
        index_list = index_list_reorder
        names_list = names_list_reorder

    if not all(keys_match_found):
        msg = "\n"
        for key, value in zip(keys, keys_match_found):
            msg += f"\t{key}: {value}\n"
        msg += f"Available strings: {list_of_strings}\n"
        raise ValueError(f"Not all regular expressions matched: {msg}")

    return index_list, names_list


def match_param(name: str, param_dict: Sequence[Tuple[str, float]]) -> float:
    """Fallback single-name matcher for (value,) mappings kept as dicts."""
    import re as _re

    if isinstance(param_dict, dict):
        for pattern, value in param_dict.items():
            if pattern == ".*":
                continue
            if _re.fullmatch(pattern, name):
                return value
        if ".*" in param_dict:
            return param_dict[".*"]
        raise KeyError(f"No value for joint: {name}")
    raise TypeError(f"Unsupported param_dict type: {type(param_dict)}")


__all__ = [
    "resolve_matching_names_values",
    "resolve_matching_names",
    "match_param",
]
