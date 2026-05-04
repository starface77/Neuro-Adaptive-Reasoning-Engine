"""
NARE CLI — Neuro-Adaptive Reasoning Engine

Interactive command-line interface with real-time streaming,
autonomy modes, and professional design.
"""

__all__ = ["run"]

# Avoid circular imports - only expose main entry point
def run():
    """Main entry point for NARE CLI."""
    from .app import main
    main()
