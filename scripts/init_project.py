#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the project directory structure."""
    directories = [
        "app/api",
        "app/core/agents",
        "app/core/parsers",
        "app/core/retrievers",
        "app/models",
        "app/utils",
        "config",
        "data/documents",
        "data/vector_store",
        "tests",
        "scripts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def create_init_files():
    """Create __init__.py files in Python packages."""
    python_dirs = [
        "app",
        "app/api",
        "app/core",
        "app/core/agents",
        "app/core/parsers",
        "app/core/retrievers",
        "app/models",
        "app/utils",
        "tests"
    ]
    
    for directory in python_dirs:
        init_file = Path(directory) / "__init__.py"
        init_file.touch()
        print(f"Created file: {init_file}")

def create_gitkeep_files():
    """Create .gitkeep files in empty directories."""
    empty_dirs = [
        "data/documents",
        "data/vector_store"
    ]
    
    for directory in empty_dirs:
        gitkeep_file = Path(directory) / ".gitkeep"
        gitkeep_file.touch()
        print(f"Created file: {gitkeep_file}")

def copy_env_example():
    """Copy .env.example if it doesn't exist."""
    if not Path(".env").exists() and Path(".env.example").exists():
        shutil.copy(".env.example", ".env")
        print("Created .env file from .env.example")

def main():
    """Initialize the project structure."""
    print("Initializing project structure...")
    
    # Create directory structure
    create_directory_structure()
    
    # Create __init__.py files
    create_init_files()
    
    # Create .gitkeep files
    create_gitkeep_files()
    
    # Copy .env file if needed
    copy_env_example()
    
    print("\nProject initialization complete!")
    print("\nNext steps:")
    print("1. Create and activate a virtual environment:")
    print("   python -m venv venv")
    print("   source venv/bin/activate  # On Windows: .\\venv\\Scripts\\activate")
    print("\n2. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n3. Set up your environment variables in .env")
    print("\n4. Start the development environment:")
    print("   docker-compose up -d")

if __name__ == "__main__":
    main() 