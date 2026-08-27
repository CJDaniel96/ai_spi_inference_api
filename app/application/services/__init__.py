"""Application services used by independently running pipeline workers.

Modules are intentionally not imported here: each worker should load only the
dependencies required by its own stage (for example, only ingest needs OpenCV).
"""
