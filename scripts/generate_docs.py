"""Generate API documentation using pdoc."""
import os
import shutil
import subprocess
from pathlib import Path

def generate_api_docs():
    """Generate API documentation using pdoc."""
    docs_dir = Path("docs/api")
    
    # Remove existing docs
    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    
    # Create docs directory
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate docs
    result = subprocess.run(
        ["pdoc", "-o", str(docs_dir), "--docformat", "numpy", "--math", "recsys"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("Error generating documentation:")
        print(result.stderr)
        return False
    
    print(f"Documentation generated at {docs_dir.absolute()}")
    return True

if __name__ == "__main__":
    generate_api_docs()
