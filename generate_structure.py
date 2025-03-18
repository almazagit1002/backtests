import os
import argparse
from pathlib import Path

def generate_project_structure(root_dir, output_file="project_structure.md", 
                               ignore_dirs=None, ignore_files=None, 
                               max_depth=None, include_files=True):
    """
    Generate a Markdown representation of a project's directory structure.
    
    Args:
        root_dir (str): Root directory of the project
        output_file (str): Output Markdown file name
        ignore_dirs (list): List of directory names to ignore
        ignore_files (list): List of file extensions or patterns to ignore
        max_depth (int): Maximum directory depth to traverse
        include_files (bool): Whether to include files in the output
        
    Returns:
        str: Path to the generated Markdown file
    """
    if ignore_dirs is None:
        ignore_dirs = ['.git', '__pycache__', 'venv', 'env', '.venv', '.env', 'node_modules', '.idea', '.vscode']
    
    if ignore_files is None:
        ignore_files = ['.pyc', '.pyo', '.pyd', '.DS_Store', '.gitignore', '__pycache__', '.pytest_cache']
    
    root_path = Path(root_dir).resolve()
    output_path = Path(output_file).resolve()
    
    def should_ignore(path, is_dir=False):
        # Check if the path should be ignored
        name = path.name
        
        if is_dir and name in ignore_dirs:
            return True
        
        if not is_dir:
            if name in ignore_files:
                return True
            if any(name.endswith(ext) for ext in ignore_files if ext.startswith('.')):
                return True
        
        return False
    
    def process_directory(directory, depth=0):
        # Stop if we've reached the maximum depth
        if max_depth is not None and depth > max_depth:
            return []
        
        lines = []
        indent = "  " * depth
        
        try:
            items = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            
            for item in items:
                if should_ignore(item, item.is_dir()):
                    continue
                
                if item.is_dir():
                    lines.append(f"{indent}- **{item.name}/**")
                    lines.extend(process_directory(item, depth + 1))
                elif include_files:
                    lines.append(f"{indent}- {item.name}")
        except PermissionError:
            lines.append(f"{indent}  ⚠️ *Permission denied*")
        
        return lines
    
    # Generate the structure
    structure_lines = ["# Project Structure", "", f"Root directory: `{root_path.name}`", ""]
    structure_lines.extend(process_directory(root_path))
    
    # Write to file with UTF-8 encoding
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(structure_lines))
    
    print(f"Project structure generated and saved to {output_path}")
    return str(output_path)

def generate_structure_for_mkdocs():
    """Command-line interface for the project structure generator"""
    parser = argparse.ArgumentParser(description="Generate a Markdown representation of a project's structure for MkDocs")
    parser.add_argument('--root', '-r', default='.', help='Root directory of the project (default: current directory)')
    parser.add_argument('--output', '-o', default='docs/project_structure.md', help='Output Markdown file (default: docs/project_structure.md)')
    parser.add_argument('--ignore-dirs', '-id', nargs='+', help='Additional directories to ignore')
    parser.add_argument('--ignore-files', '-if', nargs='+', help='Additional file extensions or patterns to ignore')
    parser.add_argument('--max-depth', '-d', type=int, help='Maximum directory depth to traverse')
    parser.add_argument('--no-files', action='store_true', help='Exclude files from the output')
    
    args = parser.parse_args()
    
    # Combine default ignore lists with user-provided ones
    ignore_dirs = ['.git', '__pycache__', 'venv', 'env', '.venv', '.env', 'node_modules', '.idea', '.vscode']
    ignore_files = ['.pyc', '.pyo', '.pyd', '.DS_Store', '.gitignore', '__pycache__', '.pytest_cache']
    
    if args.ignore_dirs:
        ignore_dirs.extend(args.ignore_dirs)
    
    if args.ignore_files:
        ignore_files.extend(args.ignore_files)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate the structure
    generate_project_structure(
        args.root, 
        args.output,
        ignore_dirs=ignore_dirs,
        ignore_files=ignore_files,
        max_depth=args.max_depth,
        include_files=not args.no_files
    )

if __name__ == "__main__":
    generate_structure_for_mkdocs()