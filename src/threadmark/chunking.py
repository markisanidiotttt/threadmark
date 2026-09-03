from dataclasses import dataclass
from threadmark.repository import (
    clone_repository,
    find_source_files,
    read_source_file,
)

@dataclass
class CodeChunk:
    file_path: str
    start_line: int
    end_line: int
    content: str
    
    
def chunk_source_file(
    file_path: str,
    lines: list[tuple[int, str]],
    chunk_size: int = 40,
    overlap: int = 10,
) -> list[CodeChunk]:
    
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    
    chunks = []
    step = chunk_size - overlap
    
    for start_index in range(0, len(lines), step):
        chunk_lines = lines[start_index:start_index + chunk_size]
        
        if not chunk_lines:
            break
        
        start_line = chunk_lines[0][0]
        end_line = chunk_lines[-1][0]
        
        context_lines = []
        
        for line_number, line in chunk_lines:
            context_lines.append(line)
            
        content = "\n".join(context_lines)
        
        chunk = CodeChunk(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            content=content,
        )
        
        chunks.append(chunk)
        
        if start_index + chunk_size >= len(lines):
            break
        
    return chunks


def chunk_repository(repo_path: str) -> list[CodeChunk]:
    source_files = find_source_files(repo_path)
    
    all_chunks = []

    for file_path in source_files:
        lines = read_source_file(repo_path, file_path)
        file_chunks = chunk_source_file(file_path, lines)
        
        all_chunks.extend(file_chunks)    
    
    return all_chunks


if __name__ == "__main__":
    repo_path = clone_repository(
        "https://github.com/nartnek/RiftPredict",
        "data/repos",
    )
    # PYTHONPATH=src python -m threadmark.chunking

    chunks = chunk_repository(repo_path)

    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks:
        print(
            f"{chunk.file_path}:"
            f"{chunk.start_line}-{chunk.end_line}"
        )
    