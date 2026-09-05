import ast
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
    symbol_name: str | None = None
    
    
def chunk_lines_fixed(
    file_path: str,
    lines: list[tuple[int, str]],
    chunk_size: int = 40,
    overlap: int = 10,
    symbol_name: str | None = None,
) -> list[CodeChunk]:
    """Breaks input file into a list of fixed-line chunks"""
    
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
            symbol_name=symbol_name,
        )
        
        chunks.append(chunk)
        
        if start_index + chunk_size >= len(lines):
            break
        
    return chunks


def chunk_repository_fixed(repo_path: str) -> list[CodeChunk]:
    """Chunk Python files using fixed windows."""
    
    source_files = find_source_files(repo_path)
    
    all_chunks = []

    for file_path in source_files:
        lines = read_source_file(repo_path, file_path)
        file_chunks = chunk_lines_fixed(file_path, lines)
        
        all_chunks.extend(file_chunks)    
    
    return all_chunks




# Chunker with ast-awareness
def chunk_repository_ast(repo_path: str) -> list[CodeChunk]:
    """Chunk Python files using AST symbol boundaries and fixed windows for large symbols."""
    
    source_files = find_source_files(repo_path)

    all_chunks = []

    for file_path in source_files:
        lines = read_source_file(
            repo_path,
            file_path,
        )

        if file_path.endswith(".py"):
            try:
                file_chunks = chunk_python_file_ast(
                    file_path,
                    lines,
                )
            except SyntaxError:
                file_chunks = chunk_lines_fixed(
                    file_path,
                    lines,
                )
        else:
            file_chunks = chunk_lines_fixed(
                file_path,
                lines,
            )

        all_chunks.extend(file_chunks)

    return all_chunks


def chunk_python_file_ast(
    file_path: str,
    lines: list[tuple[int, str]],
    chunk_size: int = 40,
    overlap: int = 10,
) -> list[CodeChunk]:
    
    
    source = "\n".join(
        line
        for _, line in lines
    )
    
    tree = ast.parse(source)
    
    chunks = []
    
    for node in tree.body:
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            continue
        
        symbol_lines = lines[
            node.lineno - 1:
            node.end_lineno
        ]
        
        if isinstance(node, ast.ClassDef):
            symbol_name = f"class {node.name}"
        else:
            symbol_name = node.name
            
        symbol_chunks = chunk_lines_fixed(
            file_path=file_path,
            lines=symbol_lines,
            chunk_size=chunk_size,
            overlap=overlap,
            symbol_name=symbol_name,
        )
        
        chunks.extend(symbol_chunks)
        
    return chunks

    