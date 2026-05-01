"""Shared constants for labels and column metadata."""

# Binary labels used in training CSVs
LABEL_BOT = 0
LABEL_HUMAN = 1

# Columns stored in features.csv that are not model inputs
METADATA_COLUMNS = ("user_id", "session_id", "source_file", "label")
