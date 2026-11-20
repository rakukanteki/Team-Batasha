"""Keep test flights out of flightlogs/: every run logged with the default folder goes to a temp folder instead."""
import os
import tempfile

# Set before any test module imports Systems.RunLogger, which reads it once at import.
os.environ.setdefault("BATASHA_LOG_DIR", tempfile.mkdtemp(prefix="batasha_test_logs_"))
