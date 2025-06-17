
import os
from pathlib import Path

# Create folder structure for the Intelligent Document Q&A System
base_dir = Path("/mnt/data/intelligent_qa_system")

folders = [
    "app/api",
    "app/processing",
    "app/qa_engine",
    "app/memory",
    "app/data/synthetic_docs",
    "app/data/chunks",
    "app/evaluation",
    "frontend",
    "notebooks",
    "configs",
    "logs"
]

# Create directories
for folder in folders:
    os.makedirs(base_dir / folder, exist_ok=True)

# Return the structure created
sorted([str(path.relative_to(base_dir)) for path in base_dir.rglob("*") if path.is_dir()])