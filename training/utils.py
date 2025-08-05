"""
This module contains utility functions shared across the training package.
"""

def create_safe_group_name(subject_name: str, grade_level: str) -> str:
    """
    Creates a filesystem-safe name from a subject and grade level.

    Example: ("Biologi", "Kelas XII") -> "biologi--kelas_xii"

    Returns:
        str: A sanitized string suitable for filenames or directory names.
    """
    safe_subject = subject_name.replace(' ', '_').lower()
    safe_grade = grade_level.replace(' ', '_').lower()
    return f'{safe_subject}--{safe_grade}'