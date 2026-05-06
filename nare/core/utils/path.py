"""Path validation for preventing hallucinated file paths."""

import os
import re
import logging
from typing import List, Tuple

def extract_file_paths(text: str) -> List[str]:
    """Extract potential file paths from text.

    Looks for patterns like:
    - path/to/file.py
    - django/db/models/fields/__init__.py
    - astropy/io/ascii/rst.py
    """

    patterns = [
        r'[a-z_][a-z0-9_/]*\.py',
        r'[a-z_][a-z0-9_/]*\.js',
        r'[a-z_][a-z0-9_/]*\.java',
        r'[a-z_][a-z0-9_/]*\.cpp',
        r'[a-z_][a-z0-9_/]*\.ts',
    ]

    paths = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        paths.extend(matches)

    return list(set(paths))

def validate_paths(paths: List[str], project_root: str = ".") -> Tuple[List[str], List[str]]:
    """Validate that file paths exist in the project.

    Returns:
        (valid_paths, invalid_paths)
    """
    valid = []
    invalid = []

    for path in paths:
        full_path = os.path.join(project_root, path)
        if os.path.exists(full_path):
            valid.append(path)
        else:
            invalid.append(path)

    return valid, invalid

def check_solution_paths(solution: str, project_root: str = ".") -> dict:
    """Check if solution contains hallucinated file paths.

    Returns:
        {
            'has_paths': bool,
            'valid_paths': List[str],
            'invalid_paths': List[str],
            'hallucination_detected': bool
        }
    """
    paths = extract_file_paths(solution)

    if not paths:
        return {
            'has_paths': False,
            'valid_paths': [],
            'invalid_paths': [],
            'hallucination_detected': False
        }

    valid, invalid = validate_paths(paths, project_root)

    return {
        'has_paths': True,
        'valid_paths': valid,
        'invalid_paths': invalid,
        'hallucination_detected': len(invalid) > 0
    }

def suggest_corrections(invalid_paths: List[str], project_root: str = ".") -> dict:
    """Suggest corrections for invalid paths.

    Looks for similar existing paths (e.g., fields.py vs fields/__init__.py)
    """
    suggestions = {}

    for invalid_path in invalid_paths:

        dir_path = invalid_path.replace('.py', '')
        init_path = os.path.join(dir_path, '__init__.py')
        full_init = os.path.join(project_root, init_path)

        if os.path.exists(full_init):
            suggestions[invalid_path] = init_path
            continue

        parent = os.path.dirname(invalid_path)
        if parent and os.path.isdir(os.path.join(project_root, parent)):

            try:
                files = os.listdir(os.path.join(project_root, parent))
                py_files = [f for f in files if f.endswith('.py')]
                if py_files:
                    suggestions[invalid_path] = f"{parent}/ contains: {', '.join(py_files[:3])}"
            except Exception as e:
                logging.warning(f"[Path] Failed to suggest alternatives for {invalid_path}: {e}")

    return suggestions
