# chroma_ast_importer_v2.py
import json
import chromadb
from chromadb.utils import embedding_functions
import argparse
from pathlib import Path

def build_rich_text_for_method(method, class_name=None, file_path=""):
    """Build a dense, embeddable string from your rich AST data"""
    name = method['name']
    full_name = f"{class_name}.{name}" if class_name else name

    args = ", ".join([
        f"{a['name']}{f': {a['annotation']}' if a['annotation'] else ''}{f' = {a['default']}' if a['default'] else ''}"
        for a in method['arguments']
    ])

    returns = f" -> {method['returns']}" if method['returns'] else ""

    docstring = (method['docstring'] or "").strip()
    comment = (method.get('comment') or "").strip()

    # Local variables (top 10 most meaningful)
    vars_preview = "; ".join([
        f"{v['name']}{f': {v['annotation']}' if v.get('annotation') else ''}{f' = {v['value_preview']}' if v.get('value_preview') else ''}"
        for v in method['variables'][:10]
    ])

    # Top called functions
    calls = [c['function'] for c in method['calls'][:8]]
    calls_str = f"calls: {', '.join(calls)}" if calls else ""

    # Top attribute accesses
    attrs = [f"{a['object']}.{a['attribute']}" for a in method['attributes'][:8]]
    attrs_str = f"uses: {', '.join(attrs)}" if attrs else ""

    text = f"""
Function: {full_name}
File: {file_path}
Signature: def {name}({args}){returns}
Docstring: {docstring.split(chr(10))[0] if docstring else "None"}
Comment: {comment or "None"}
Variables: {vars_preview or "None"}
{calls_str}
{attrs_str}
""".strip()

    return text

def import_from_your_json(json_path: str, collection):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    metadatas = []
    ids = []

    for file_path, file_data in data.items():
        # Standalone functions
        for func in file_data.get('functions', []):
            text = build_rich_text_for_method(func, file_path=file_path)
            func_id = f"{file_path}::{func['name']}"

            documents.append(text)
            metadatas.append({
                "name": func['name'],
                "file": file_path,
                "type": "function",
                "docstring": func['docstring'] or "",
                "source": "rich_ast"
            })
            ids.append(func_id)

        # Class methods
        for cls in file_data.get('classes', []):
            class_name = cls['name']
            for method in cls.get('methods', []):
                text = build_rich_text_for_method(method, class_name, file_path)
                method_id = f"{file_path}::{class_name}.{method['name']}"

                documents.append(text)
                metadatas.append({
                    "name": f"{class_name}.{method['name']}",
                    "class": class_name,
                    "file": file_path,
                    "type": "method",
                    "docstring": method['docstring'] or "",
                    "source": "rich_ast"
                })
                ids.append(method_id)

    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(documents)} rich documents from {json_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_input", help="Your extractor's JSON output")
    parser.add_argument("--persist_dir", default="./chroma_rich_db")
    parser.add_argument("--collection", default="codebase_rich")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.persist_dir)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name=args.collection,
        embedding_function=embedding_fn
    )

    import_from_your_json(args.json_input, collection)
    print(f"Done! Collection '{args.collection}' ready with rich semantic docs.")

if __name__ == "__main__":
    main()