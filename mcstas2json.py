from __future__ import annotations

import ast
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# =============================================================================
# Expression evaluation
# =============================================================================

_ALLOWED_MATH_NAMES: dict[str, Any] = {
    "pi": math.pi,
    "PI": math.pi,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "abs": abs,
}


def _safe_float(
    expression: str,
    variables: dict[str, Any] | None = None,
) -> float:
    """
    Safely evaluate a scalar McStas expression.

    Supported examples:
        L1 / 2
        2 * cm
        sin(pi / 4)
        ROT1 + 15
        guide[0][4]
    """
    variables = variables or {}

    names = {
        **_ALLOWED_MATH_NAMES,
        **variables,
    }

    expression = expression.strip()

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError(
            f"Invalid expression: {expression!r}"
        ) from error

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Name,
        ast.Call,
        ast.Load,
        ast.Subscript,
    )

    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(
                f"Unsupported expression: {expression!r}"
            )

        if isinstance(node, ast.Name) and node.id not in names:
            raise ValueError(
                f"Unknown symbol {node.id!r} in expression {expression!r}. "
                "Provide it through variables=, DEFINE INSTRUMENT, DECLARE, "
                "or INITIALIZE."
            )

        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in _ALLOWED_MATH_NAMES
            ):
                raise ValueError(
                    f"Unsupported function call in expression: "
                    f"{expression!r}"
                )

        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Slice):
                raise ValueError(
                    f"Array slices are not supported: {expression!r}"
                )

    try:
        result = eval(
            compile(tree, "<mcstas-expression>", "eval"),
            {"__builtins__": {}},
            names,
        )
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError(
            f"Could not evaluate expression {expression!r}: {error}"
        ) from error

    try:
        return float(result)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Expression {expression!r} did not resolve to a scalar value. "
            f"Result: {result!r}"
        ) from error


def _parse_vector(
    text: str,
    variables: dict[str, Any] | None,
) -> np.ndarray:
    """Parse a McStas vector, for example: '0, ROT1, guide[0][4]'."""
    values = [value.strip() for value in text.split(",")]

    if len(values) != 3:
        raise ValueError(
            f"Expected a 3-vector, got: ({text})"
        )

    return np.array(
        [_safe_float(value, variables) for value in values],
        dtype=float,
    )


# =============================================================================
# Text parsing helpers
# =============================================================================

def _remove_mcstas_comments(text: str) -> str:
    """
    Remove C/McStas comments.

    Handles:
        /* block comments */
        // line comments
        %{ and %} McStas C block delimiters
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)

    # Convert %{ to { and %} to }, and remove leading % on comment lines.
    text = re.sub(r"(?m)^(\s*)%\s?", r"\1", text)

    return text


def _split_top_level_commas(text: str) -> list[str]:
    """Split text at commas outside (), [] and {}."""
    parts: list[str] = []
    current: list[str] = []

    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0

    for char in text:
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1

        if (
            char == ","
            and paren_depth == 0
            and bracket_depth == 0
            and brace_depth == 0
        ):
            value = "".join(current).strip()
            if value:
                parts.append(value)
            current = []
        else:
            current.append(char)

    value = "".join(current).strip()

    if value:
        parts.append(value)

    return parts


def _split_top_level_semicolons(text: str) -> list[str]:
    """Split text at semicolons outside (), [] and {}."""
    statements: list[str] = []
    current: list[str] = []

    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0

    for char in text:
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1

        if (
            char == ";"
            and paren_depth == 0
            and bracket_depth == 0
            and brace_depth == 0
        ):
            value = "".join(current).strip()
            if value:
                statements.append(value)
            current = []
        else:
            current.append(char)

    value = "".join(current).strip()

    if value:
        statements.append(value)

    return statements


def _extract_brace_block(
    text: str,
    keyword: str,
) -> str | None:
    """
    Extract content from the first {...} block following a keyword.

    Example:

        DECLARE
        %{
            double x = 1;
        %}
    """
    match = re.search(
        rf"\b{re.escape(keyword)}\b",
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    brace_start = text.find("{", match.end())

    if brace_start == -1:
        return None

    depth = 0

    for index in range(brace_start, len(text)):
        char = text[index]

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1

            if depth == 0:
                return text[brace_start + 1:index]

    raise ValueError(
        f"Unclosed brace block after {keyword!r}."
    )


# =============================================================================
# McStas variable extraction
# =============================================================================

def _read_instrument_defaults(
    mcstas_text: str,
    supplied_variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read default numerical values from DEFINE INSTRUMENT."""
    supplied_variables = supplied_variables or {}

    values: dict[str, Any] = {
        "pi": math.pi,
        "PI": math.pi,
        "cm": 0.01,
        "mm": 0.001,
        "m": 1.0,
        "deg2rad": math.pi / 180.0,
        "DEG2RAD": math.pi / 180.0,
        "rad2deg": 180.0 / math.pi,
        "RAD2DEG": 180.0 / math.pi,
    }

    match = re.search(
        r"\bDEFINE\s+INSTRUMENT\s+"
        r"[A-Za-z_][A-Za-z0-9_]*\s*\(",
        mcstas_text,
        flags=re.IGNORECASE,
    )

    if match is None:
        values.update(supplied_variables)
        return values

    start = match.end()
    end = start
    depth = 1

    while end < len(mcstas_text) and depth > 0:
        if mcstas_text[end] == "(":
            depth += 1
        elif mcstas_text[end] == ")":
            depth -= 1
        end += 1

    parameter_text = mcstas_text[start:end - 1]
    pending: dict[str, str] = {}

    for declaration in _split_top_level_commas(parameter_text):
        if "=" not in declaration:
            continue

        name, expression = declaration.split("=", 1)
        name = name.strip()
        expression = expression.strip()

        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue

        if name not in supplied_variables:
            pending[name] = expression

    unresolved = dict(pending)

    while unresolved:
        progress = False

        for name, expression in list(unresolved.items()):
            try:
                values[name] = _safe_float(expression, values)
            except ValueError:
                continue

            del unresolved[name]
            progress = True

        if not progress:
            break

    values.update(supplied_variables)

    return values


def _parse_c_array_initializer(
    initializer: str,
    variables: dict[str, Any],
) -> list[Any]:
    """
    Parse a C-style array initializer into nested Python lists.

    Example:
        {{1, 2}, {3, 4}}

    Returns:
        [[1.0, 2.0], [3.0, 4.0]]
    """
    initializer = initializer.strip()

    if not (
        initializer.startswith("{")
        and initializer.endswith("}")
    ):
        raise ValueError(
            "Expected a C-style array initializer enclosed in braces. "
            f"Got: {initializer[:100]!r}"
        )

    inner = initializer[1:-1].strip()

    if not inner:
        return []

    result: list[Any] = []

    for item in _split_top_level_commas(inner):
        item = item.strip()

        if not item:
            continue

        if item.startswith("{"):
            result.append(
                _parse_c_array_initializer(item, variables)
            )
        else:
            result.append(
                _safe_float(item, variables)
            )

    return result


def _read_declared_double_defaults(
    mcstas_text: str,
    base_variables: dict[str, Any] | None = None,
    supplied_variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Read initialized scalar and array double declarations from DECLARE.

    Supports scalar declarations:

        double angle = 15.0;
        double L1 = 10.0, L2 = L1 + 1.0;

    Supports arrays:

        double guide[19][10] = {
            {0.0662, 0.0482, 0.0652, 0.0522, 0.5, 0, 0, 0, 0, 1},
            ...
        };
    """
    supplied_variables = supplied_variables or {}

    values: dict[str, Any] = dict(base_variables or {})
    values.update(supplied_variables)

    declare_block = _extract_brace_block(
        mcstas_text,
        "DECLARE",
    )

    if declare_block is None:
        return values

    declare_block = _remove_mcstas_comments(declare_block)

    pending_scalars: dict[str, str] = {}
    pending_arrays: dict[str, str] = {}

    for statement in _split_top_level_semicolons(declare_block):
        double_match = re.match(
            r"^\s*(?:(?:static|const|volatile)\s+)*"
            r"double\s+(.+?)\s*$",
            statement,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if double_match is None:
            continue

        declaration = double_match.group(1).strip()

        array_match = re.match(
            r"^\s*"
            r"([A-Za-z_][A-Za-z0-9_]*)"
            r"(?:\s*\[[^\]]+\])+"
            r"\s*=\s*"
            r"(\{.*\})"
            r"\s*$",
            declaration,
            flags=re.DOTALL,
        )

        if array_match is not None:
            name, initializer = array_match.groups()

            if name not in supplied_variables:
                pending_arrays[name] = initializer.strip()

            continue

        for scalar_declaration in _split_top_level_commas(declaration):
            scalar_match = re.match(
                r"^\s*"
                r"([A-Za-z_][A-Za-z0-9_]*)"
                r"\s*=\s*"
                r"(.+?)"
                r"\s*$",
                scalar_declaration,
                flags=re.DOTALL,
            )

            if scalar_match is None:
                continue

            name, expression = scalar_match.groups()

            if name not in supplied_variables:
                pending_scalars[name] = expression.strip()

    unresolved_scalars = dict(pending_scalars)
    unresolved_arrays = dict(pending_arrays)

    while unresolved_scalars or unresolved_arrays:
        progress = False

        for name, expression in list(unresolved_scalars.items()):
            try:
                values[name] = _safe_float(expression, values)
            except ValueError:
                continue

            del unresolved_scalars[name]
            progress = True

        for name, initializer in list(unresolved_arrays.items()):
            try:
                values[name] = _parse_c_array_initializer(
                    initializer,
                    values,
                )
            except ValueError:
                continue

            del unresolved_arrays[name]
            progress = True

        if not progress:
            break

    unresolved_names = [
        *unresolved_scalars.keys(),
        *unresolved_arrays.keys(),
    ]

    if unresolved_names:
        warnings.warn(
            "Could not statically resolve these DECLARE values: "
            + ", ".join(unresolved_names),
            stacklevel=2,
        )

    return values


def _read_initialize_assignments(
    mcstas_text: str,
    base_variables: dict[str, Any] | None = None,
    supplied_variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Resolve simple assignments in INITIALIZE.

    Supported examples:

        ROT1 = ANA_POS;
        ROT2 = ROT1 + 15;
        distance = L1 + guide[0][4];

    Complex C statements, loops, conditionals, pointers, structs, file I/O,
    and custom function calls are ignored.
    """
    supplied_variables = supplied_variables or {}

    values: dict[str, Any] = dict(base_variables or {})
    values.update(supplied_variables)

    initialize_block = _extract_brace_block(
        mcstas_text,
        "INITIALIZE",
    )

    if initialize_block is None:
        return values

    initialize_block = _remove_mcstas_comments(initialize_block)

    pending: dict[str, str] = {}

    for statement in _split_top_level_semicolons(initialize_block):
        statement = statement.strip()

        assignment_match = re.match(
            r"^\s*"
            r"([A-Za-z_][A-Za-z0-9_]*)"
            r"\s*=\s*"
            r"([^=].*?)"
            r"\s*$",
            statement,
            flags=re.DOTALL,
        )

        if assignment_match is None:
            continue

        name, expression = assignment_match.groups()

        if name not in supplied_variables:
            pending[name] = expression.strip()

    unresolved = dict(pending)

    while unresolved:
        progress = False

        for name, expression in list(unresolved.items()):
            try:
                values[name] = _safe_float(expression, values)
            except ValueError:
                continue

            del unresolved[name]
            progress = True

        if not progress:
            break

    if unresolved:
        warnings.warn(
            "Could not statically resolve these INITIALIZE assignments: "
            + ", ".join(unresolved.keys()),
            stacklevel=2,
        )

    return values


# =============================================================================
# Transformation mathematics
# =============================================================================

def _rx(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)

    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def _ry(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)

    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def _rz(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)

    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _mcstas_rotation(rotation_xyz_deg: np.ndarray) -> np.ndarray:
    """
    Convert McStas ROTATED (x, y, z) into a rotation matrix.

    Convention:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    roll, pitch, yaw = rotation_xyz_deg

    return _rz(yaw) @ _ry(pitch) @ _rx(roll)


def _homogeneous(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Create a 4x4 homogeneous transform matrix."""
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _euler_zyx_deg(
    rotation: np.ndarray,
) -> tuple[float, float, float]:
    """
    Extract yaw, pitch, roll from:

        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    sin_pitch = max(-1.0, min(1.0, -rotation[2, 0]))

    pitch = math.asin(sin_pitch)
    cos_pitch = math.cos(pitch)

    if abs(cos_pitch) > 1e-10:
        yaw = math.atan2(
            rotation[1, 0],
            rotation[0, 0],
        )

        roll = math.atan2(
            rotation[2, 1],
            rotation[2, 2],
        )
    else:
        # Gimbal-lock convention: roll is set to zero.
        yaw = math.atan2(
            -rotation[0, 1],
            rotation[1, 1],
        )

        roll = 0.0

    return (
        math.degrees(yaw),
        math.degrees(pitch),
        math.degrees(roll),
    )


# =============================================================================
# McStas component parsing
# =============================================================================

@dataclass
class McStasComponent:
    name: str
    component_type: str
    at: np.ndarray
    at_relative_to: str | None
    rotated: np.ndarray
    rotation_relative_to: str | None
    previous_name: str | None


def _parse_mcstas_components(
    mcstas_text: str,
    variables: dict[str, Any] | None,
) -> list[McStasComponent]:
    """
    Parse all COMPONENT blocks.

    McStas components do not require semicolon separators. A component block
    extends from one line beginning with COMPONENT to the next COMPONENT line.
    """
    text = _remove_mcstas_comments(mcstas_text)

    component_start_pattern = re.compile(
        r"(?im)^[ \t]*\bCOMPONENT\b"
    )

    component_starts = list(component_start_pattern.finditer(text))

    if not component_starts:
        return []

    component_header_pattern = re.compile(
        r"\bCOMPONENT\s+"
        r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
        r"([A-Za-z_][A-Za-z0-9_]*)",
        flags=re.IGNORECASE | re.DOTALL,
    )

    at_pattern = re.compile(
        r"\bAT\s*\(\s*([^)]+?)\s*\)"
        r"(?:\s+RELATIVE\s+"
        r"([A-Za-z_][A-Za-z0-9_]*|PREVIOUS))?",
        flags=re.IGNORECASE | re.DOTALL,
    )

    rotated_pattern = re.compile(
        r"\bROTATED\s*\(\s*([^)]+?)\s*\)"
        r"(?:\s+RELATIVE\s+"
        r"([A-Za-z_][A-Za-z0-9_]*|PREVIOUS))?",
        flags=re.IGNORECASE | re.DOTALL,
    )

    components: list[McStasComponent] = []
    previous_name: str | None = None

    for index, component_start in enumerate(component_starts):
        block_start = component_start.start()

        if index + 1 < len(component_starts):
            block_end = component_starts[index + 1].start()
        else:
            block_end = len(text)

        component_block = text[block_start:block_end]

        header_match = component_header_pattern.search(component_block)

        if header_match is None:
            continue

        component_name, component_type = header_match.groups()

        at_match = at_pattern.search(component_block)

        if at_match is None:
            raise ValueError(
                f"Component {component_name!r} has no AT (...) clause.\n\n"
                f"Component block:\n{component_block}"
            )

        rotated_match = rotated_pattern.search(component_block)

        at_vector = _parse_vector(
            at_match.group(1),
            variables,
        )

        at_relative_to = at_match.group(2)

        if rotated_match is None:
            rotation_vector = np.zeros(3, dtype=float)
            rotation_relative_to = None
        else:
            rotation_vector = _parse_vector(
                rotated_match.group(1),
                variables,
            )

            rotation_relative_to = rotated_match.group(2)

        components.append(
            McStasComponent(
                name=component_name,
                component_type=component_type,
                at=at_vector,
                at_relative_to=at_relative_to,
                rotated=rotation_vector,
                rotation_relative_to=rotation_relative_to,
                previous_name=previous_name,
            )
        )

        previous_name = component_name

    return components


# =============================================================================
# Resolve McStas RELATIVE coordinate frames
# =============================================================================

def _resolve_world_transforms(
    components: list[McStasComponent],
) -> dict[str, np.ndarray]:
    """
    Resolve every component transform into McStas global/source coordinates.

    Supports:
      - AT (...) RELATIVE component
      - ROTATED (...) RELATIVE component
      - RELATIVE PREVIOUS
      - chained component references
      - forward references
    """
    components_by_name: dict[str, McStasComponent] = {}

    for component in components:
        if component.name in components_by_name:
            raise ValueError(
                f"Duplicate McStas component name: {component.name!r}"
            )

        components_by_name[component.name] = component

    resolved: dict[str, np.ndarray] = {}
    resolving_stack: list[str] = []

    def resolve_reference(
        reference: str | None,
        previous_name: str | None,
    ) -> np.ndarray:
        if reference is None:
            return np.eye(4, dtype=float)

        if reference.upper() == "PREVIOUS":
            if previous_name is None:
                raise ValueError(
                    "RELATIVE PREVIOUS was used for the first component."
                )

            reference = previous_name

        if reference not in components_by_name:
            available = ", ".join(components_by_name.keys())

            raise ValueError(
                f"Reference component {reference!r} was not found.\n\n"
                f"Available components:\n{available}"
            )

        return resolve_component(reference)

    def resolve_component(component_name: str) -> np.ndarray:
        if component_name in resolved:
            return resolved[component_name]

        if component_name in resolving_stack:
            cycle_start = resolving_stack.index(component_name)
            cycle = resolving_stack[cycle_start:] + [component_name]

            raise ValueError(
                "Circular RELATIVE dependency:\n  "
                + " -> ".join(cycle)
            )

        resolving_stack.append(component_name)

        component = components_by_name[component_name]

        position_parent = resolve_reference(
            component.at_relative_to,
            component.previous_name,
        )

        rotation_parent = resolve_reference(
            component.rotation_relative_to,
            component.previous_name,
        )

        world_position = (
            position_parent[:3, 3]
            + position_parent[:3, :3] @ component.at
        )

        world_rotation = (
            rotation_parent[:3, :3]
            @ _mcstas_rotation(component.rotated)
        )

        transform = _homogeneous(
            world_rotation,
            world_position,
        )

        resolved[component_name] = transform
        resolving_stack.pop()

        return transform

    for component in components:
        resolve_component(component.name)

    return resolved


# =============================================================================
# JSON / NeXus output helpers
# =============================================================================

def _safe_nexus_name(name: str) -> str:
    """Convert a McStas identifier into a safe NeXus path segment."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _attribute(
    name: str,
    dtype: str,
    values: Any,
) -> dict[str, Any]:
    """Create a JSON attribute."""
    return {
        "name": name,
        "dtype": dtype,
        "values": values,
    }


def _string_dataset(
    name: str,
    value: str,
) -> dict[str, Any]:
    """Create a basic string dataset."""
    return {
        "module": "dataset",
        "config": {
            "name": name,
            "values": value,
            "dtype": "string",
        },
    }


def _transformation_dataset(
    name: str,
    value: float,
    vector: list[float],
    transformation_type: str,
    units: str,
    depends_on: str,
) -> dict[str, Any]:
    """Create one NeXus transformation dataset."""
    return {
        "module": "dataset",
        "config": {
            "name": name,
            "values": float(value),
            "dtype": "double",
        },
        "attributes": [
            _attribute(
                "vector",
                "double",
                vector,
            ),
            _attribute(
                "depends_on",
                "string",
                depends_on,
            ),
            _attribute(
                "transformation_type",
                "string",
                transformation_type,
            ),
            _attribute(
                "units",
                "string",
                units,
            ),
        ],
    }


def _nexus_transformation_group(
    component_name: str,
    transform_in_sample_frame: np.ndarray,
    instrument_path: str,
    zero_tolerance: float,
) -> tuple[dict[str, Any], str]:
    """
    Create an NXtransformations group.

    Transformations with no effect are omitted:

        abs(value) <= zero_tolerance

    Returns:
        transformations_group, final_depends_on_path

    The latter is required for the enclosing component's depends_on dataset.
    It is "." where no transformations have been emitted.
    """
    safe_name = _safe_nexus_name(component_name)

    transformation_path = (
        f"{instrument_path}/{safe_name}/transformations"
    )

    translation = transform_in_sample_frame[:3, 3]

    yaw, pitch, roll = _euler_zyx_deg(
        transform_in_sample_frame[:3, :3]
    )

    children: list[dict[str, Any]] = []
    depends_on = "."

    translations = (
        ("x_translation", translation[0], [1.0, 0.0, 0.0]),
        ("y_translation", translation[1], [0.0, 1.0, 0.0]),
        ("z_translation", translation[2], [0.0, 0.0, 1.0]),
    )

    for name, value, vector in translations:
        if abs(value) <= zero_tolerance:
            continue

        children.append(
            _transformation_dataset(
                name=name,
                value=value,
                vector=vector,
                transformation_type="translation",
                units="m",
                depends_on=depends_on,
            )
        )

        depends_on = f"{transformation_path}/{name}"

    rotations = (
        ("yaw", yaw, [0.0, 0.0, 1.0]),
        ("pitch", pitch, [0.0, 1.0, 0.0]),
        ("roll", roll, [1.0, 0.0, 0.0]),
    )

    for name, value, vector in rotations:
        if abs(value) <= zero_tolerance:
            continue

        children.append(
            _transformation_dataset(
                name=name,
                value=value,
                vector=vector,
                transformation_type="rotation",
                units="deg",
                depends_on=depends_on,
            )
        )

        depends_on = f"{transformation_path}/{name}"

    transformations_group = {
        "name": "transformations",
        "type": "group",
        "children": children,
        "attributes": [
            _attribute(
                "NX_class",
                "string",
                "NXtransformations",
            )
        ],
    }

    return transformations_group, depends_on


def _nexus_class_for_component(component_type: str) -> str:
    """
    Map a McStas component type to an appropriate NeXus class.

    Unknown component types become NXcomponent.
    """
    component_type_lower = component_type.casefold()

    if "monitor" in component_type_lower:
        return "NXmonitor"

    if "detector" in component_type_lower:
        return "NXdetector"

    if "source" in component_type_lower:
        return "NXsource"

    if "sample" in component_type_lower:
        return "NXsample"

    if "slit" in component_type_lower:
        return "NXslit"

    if "chopper" in component_type_lower:
        return "NXdisk_chopper"

    if "guide" in component_type_lower:
        return "NXguide"

    if "collimator" in component_type_lower:
        return "NXcollimator"

    return "NXcomponent"


def _component_to_nexus_group(
    component: McStasComponent,
    transform_in_sample_frame: np.ndarray,
    instrument_path: str,
    zero_tolerance: float,
) -> dict[str, Any]:
    """
    Convert one McStas component into a NeXus-style JSON group.

    The original McStas component type is retained in the dataset:

        mcstas_component_type
    """
    transformations, final_depends_on = _nexus_transformation_group(
        component_name=component.name,
        transform_in_sample_frame=transform_in_sample_frame,
        instrument_path=instrument_path,
        zero_tolerance=zero_tolerance,
    )

    return {
        "name": _safe_nexus_name(component.name),
        "type": "group",
        "attributes": [
            _attribute(
                "NX_class",
                "string",
                _nexus_class_for_component(component.component_type),
            )
        ],
        "children": [
            _string_dataset(
                "mcstas_component_type",
                component.component_type,
            ),
            transformations,
            _string_dataset(
                "depends_on",
                final_depends_on,
            ),
        ],
    }


# =============================================================================
# Public conversion function
# =============================================================================

def mcstas_to_sample_relative_json(
    mcstas_file: str | Path,
    sample_component: str,
    *,
    variables: dict[str, Any] | None = None,
    instrument_path: str = "/entry/instrument",
    include_sample: bool = False,
    zero_tolerance: float = 1e-12,
    exclude_component_types: list[str] | tuple[str, ...] | None = ("Arm",),
    exclude_components: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """
    Convert a McStas .instr file to sample-relative NeXus-style JSON groups.

    Parameters
    ----------
    mcstas_file:
        Path to the McStas instrument file.

    sample_component:
        Exact McStas COMPONENT name representing the sample, e.g. "Sample".

    variables:
        Optional explicit scalar or array values. These override values
        obtained from DEFINE INSTRUMENT, DECLARE and INITIALIZE.

    instrument_path:
        Base NeXus path used by `depends_on` references.

    include_sample:
        If True, write the sample itself to the output.

    zero_tolerance:
        A translation/rotation with abs(value) <= zero_tolerance is omitted.

    exclude_component_types:
        Component types to exclude from JSON. The default excludes all Arm
        components:

            ("Arm",)

        Matching is case-insensitive. Set this to an empty tuple/list to
        include Arms:

            exclude_component_types=()

    exclude_components:
        Exact McStas component names to exclude from JSON.

        Example:

            exclude_components=[
                "Source",
                "ISCS",
                "InstrArm",
            ]

    Notes
    -----
    Excluded components are still parsed and used for RELATIVE transform
    resolution. They are only removed from the returned JSON list.

    Variable precedence
    -------------------
    1. Explicit `variables=`
    2. INITIALIZE assignments
    3. DECLARE scalar and array double declarations
    4. DEFINE INSTRUMENT defaults
    5. Built-in constants: pi, cm, mm, m, deg2rad and rad2deg
    """
    mcstas_text = Path(mcstas_file).read_text(
        encoding="utf-8",
    )

    resolved_variables = _read_instrument_defaults(
        mcstas_text,
        supplied_variables=variables,
    )

    resolved_variables = _read_declared_double_defaults(
        mcstas_text,
        base_variables=resolved_variables,
        supplied_variables=variables,
    )

    resolved_variables = _read_initialize_assignments(
        mcstas_text,
        base_variables=resolved_variables,
        supplied_variables=variables,
    )

    components = _parse_mcstas_components(
        mcstas_text,
        variables=resolved_variables,
    )

    if not components:
        raise ValueError(
            "No McStas COMPONENT declarations were found."
        )

    # Resolve all transforms before filtering. Excluded components may be
    # required as RELATIVE coordinate-frame references.
    world_transforms = _resolve_world_transforms(components)

    if sample_component not in world_transforms:
        available = ", ".join(world_transforms.keys())

        raise ValueError(
            f"Sample component {sample_component!r} was not found.\n\n"
            f"Available components:\n{available}"
        )

    excluded_types = {
        component_type.casefold()
        for component_type in (exclude_component_types or ())
    }

    excluded_names = set(exclude_components or ())

    # Transform global/source-coordinate transforms into sample coordinates:
    #
    # T_sample_component =
    #     inverse(T_world_sample) @ T_world_component
    #
    sample_inverse = np.linalg.inv(
        world_transforms[sample_component]
    )

    output: list[dict[str, Any]] = []

    for component in components:
        if component.name == sample_component and not include_sample:
            continue

        if component.name in excluded_names:
            continue

        if component.component_type.casefold() in excluded_types:
            continue

        transform_in_sample_frame = (
            sample_inverse
            @ world_transforms[component.name]
        )

        output.append(
            _component_to_nexus_group(
                component=component,
                transform_in_sample_frame=transform_in_sample_frame,
                instrument_path=instrument_path,
                zero_tolerance=zero_tolerance,
            )
        )

    return output


# =============================================================================
# Example use
# =============================================================================

if __name__ == "__main__":
    result = mcstas_to_sample_relative_json(
        "ESS_MIRACLES_kafkatubes.instr",
        sample_component="Sample",
        variables={
            "L_source_sample": 15.0,
        },

        # Arm components are already excluded by default.
        exclude_component_types=(
            "Arm",
            # "Slit",
            # "Beamstop",
        ),

        # Exclude individual component names if required.
        exclude_components=[
            # "Source",
            # "ISCS",
            # "InstrArm",
        ],

        instrument_path="/entry/instrument",
        include_sample=False,
        zero_tolerance=1e-12,
    )

    with open(
        "sample_relative_components.json",
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(result, output_file, indent=2)

    print("Wrote sample_relative_components.json")