# file: code_duplication_analyzer.py
import os
import ast
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Set
from itertools import combinations
import token as tk
from tokenize import tokenize, untokenize
from io import BytesIO

# ================= CONFIG =================
THRESHOLD = 0.75  # 75% similarity → yellow (review needed)
MIN_TOKENS = 15   # ignore tiny methods (adjust as needed)
# =========================================

class MethodInfo:
    def __init__(self, name: str, file: str, start_line: int, end_line: int, code: str, tokens: List[tk.TokenInfo]):
        self.name = name
        self.file = file
        self.start_line = start_line
        self.end_line = end_line
        self.code = code
        self.tokens = tokens
        self.hash = self._hash_tokens()

    def _hash_tokens(self):
        # Normalize tokens: keep only type and string, ignore whitespace/comment differences
        normalized = [(t.type, t.string) for t in self.tokens if t.type not in (tk.INDENT, tk.DEDENT, tk.NEWLINE, tk.COMMENT, tk.NL)]
        return hashlib.sha256(str(normalized).encode()).hexdigest()

def extract_methods_from_file(file_path: Path) -> List[MethodInfo]:
    with open(file_path, 'rb') as f:
        tokens = list(tokenize(f.readline))
    
    code = Path(file_path).read_text()
    tree = ast.parse(code)
    methods = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith('_') and node.name != '__init__': 
                continue  # skip private-ish
            start = node.lineno
            end = node.end_lineno
            method_code = '\n'.join(code.splitlines()[start-1:end])
            methods.append(MethodInfo(node.name, str(file_path), start, end, method_code, tokens))
    return methods

def normalize_tokens(tokens: List[tk.TokenInfo]) -> Tuple[str, ...]:
    result = []
    for t in tokens:
        if t.type in (tk.NAME, tk.NUMBER, tk.STRING):
            result.append((t.type, "X"))  # anonymize identifiers & literals
        elif t.type not in (tk.INDENT, tk.DEDENT, tk.NEWLINE, tk.COMMENT, tk.NL, tk.ENDMARKER):
            result.append((t.type, t.string))
    return tuple(result)

def token_similarity(m1: MethodInfo, m2: MethodInfo) -> float:
    if len(m1.tokens) < MIN_TOKENS or len(m2.tokens) < MIN_TOKENS:
        return 0.0
    
    seq1 = normalize_tokens(m1.tokens)
    seq2 = normalize_tokens(m2.tokens)
    
    set1, set2 = set(seq1), set(seq2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def analyze_project(root: str = ".") -> Tuple[List[MethodInfo], List[Tuple[MethodInfo, MethodInfo, float]]]:
    root_path = Path(root)
    all_methods: List[MethodInfo] = []
    for pyfile in root_path.rglob("*.py"):
        if "venv" in str(pyfile) or "env" in str(pyfile) or ".git" in str(pyfile):
            continue
        all_methods.extend(extract_methods_from_file(pyfile))
    
    pairs = []
    for m1, m2 in combinations(all_methods, 2):
        sim = token_similarity(m1, m2)
        if sim >= 0.3:  # only keep reasonably similar pairs to reduce noise
            pairs.append((m1, m2, sim))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    return all_methods, pairs

def generate_mermaid(all_methods: List[MethodInfo], similar_pairs: List[Tuple[MethodInfo, MethodInfo, float]]) -> str:
    lines = ['flowchart TD']
    lines.append('    subgraph " "')
    
    # Define node styles
    for method in all_methods:
        node_id = f"n_{method.file.replace('/', '_').replace('.', '_')}__{method.name}"
        label = f"{method.name}\\n({Path(method.file).name}:{method.start_line})"
        
        # Determine if this method has any high-similarity pair
        is_suspicious = any(sim >= THRESHOLD for m1, m2, sim in similar_pairs 
                          if m1.name == method.name or m2.name == method.name)
        
        style = "fill:#90EE90 stroke:#333" if not is_suspicious else "fill:#FFD700 stroke:#333,stroke-width:3px"
        lines.append(f'    {node_id}[["{label}"]]:::{style}')
    
    # Edges
    for m1, m2, sim in similar_pairs:
        if sim < THRESHOLD:
            continue  # only draw edges for things worth reviewing
        id1 = f"n_{m1.file.replace('/', '_').replace('.', '_')}__{m1.name}"
        id2 = f"n_{m2.file.replace('/', '_').replace('.', '_')}__{m2.name}"
        label = f"{sim:.2f}"
        lines.append(f'    {id1} -- "{label}" ---> {id2}')
    
    lines.append('    end')
    lines.append('    classDef fill:#90EE90 stroke:#333')
    lines.append('    classDef fill:#FFD700 stroke:#333,stroke-width:3px')
    
    return '\n'.join(lines)

# =============== MAIN ===============
if __name__ == "__main__":
    print("Scanning project for duplicated methods...")
    methods, pairs = analyze_project(".")
    
    print(f"Found {len(methods)} methods, {len(pairs)} similar pairs above 30%")
    print(f"Threshold for review (yellow): {THRESHOLD*100}%")
    
    mermaid = generate_mermaid(methods, pairs)
    
    with open("duplication_report.mmd", "w", encoding="utf-8") as f:
        f.write(mermaid)
    
    print("\nMermaid diagram written to duplication_report.mmd")
    print("Open it in https://mermaid.live or any Mermaid-compatible viewer")
    print("\nTop 10 most similar pairs:")
    for m1, m2, sim in pairs[:10]:
        print(f"  {sim:.3f} → {m1.name} ({Path(m1.file).name}:{m1.start_line}) ↔ {m2.name} ({Path(m2.file).name}:{m2.start_line})")