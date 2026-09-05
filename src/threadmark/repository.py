from pathlib import Path
from git import Repo

SUPPORTED_EXTENSIONS = {
    ".py",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
}


def clone_repository(repo_url: str, destination_root: str) -> str:
    """Clone a Git repository locally and return its repository path."""
    repo_url = repo_url.rstrip("/")

    repo_name = repo_url.split("/")[-1] 
    # Split by / to get an array of strings
    # Then access the last element using [-1]

    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4] # Slice (remove) the .git (4 characters)

    # Convert the destination into path
    root = Path(destination_root)
    root.mkdir(parents=True, exist_ok=True)

    repo_path = root / repo_name # Combine, e.g. ...root/repo_name

    if repo_path.exists():
        return str(repo_path)

    Repo.clone_from(repo_url, repo_path)

    return str(repo_path)

def find_source_files(repo_path: str) -> list[str]:
    root = Path(repo_path)
    source_files = []
    
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS:
            source_files.append(str(path.relative_to(root)))
            
    return source_files    


def read_source_file(repo_path: str, relative_path: str) -> list[tuple[int, str]]:
    root = Path(repo_path)
    file_path = root / relative_path
    
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    output = []
    
    for line_number, line in enumerate(lines, start=1):
        output.append((line_number, line))
        
    return output
