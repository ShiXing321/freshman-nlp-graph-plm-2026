"""Compatibility entry point for the final submission coverage audit."""
from check_submission import main
if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
