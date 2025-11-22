# chroma_ast_importer.py
import json
import chromadb
from chromadb.utils import embedding_functions
from typing import Dict, Any, List
import os
import argparse

# You can swap this for OpenAI, Cohere, etc.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # local & fast (sentence-transformers)

def extract_functions_from_ast_json(data: Any, file_path: str = "") -> List[Dict[str, Any]]:
    """
    Recursively walk the AST JSON and pull out function/method definitions
    with all available metadata.
    Expects a structure similar to what libcst or a custom exporter gives:
    - node type
    - name
    - params
    - returns (annotation)
    - body
    - docstring (usually in body[0] if it's an Expr with Constant string)
    - leading comments, etc.
    """
    functions = []

    def walk(node: Dict[str, Any]):
        if not isinstance(node, dict):
            return

        node_type = node.get("node_type") or node.get("type", "")

        # FunctionDef or AsyncFunctionDef
        if node_type in ("FunctionDef", "AsyncFunctionDef"):
            name = node.get("name", "unknown")
            args = node.get("args", {})
            returns = node.get("returns", None)

            # Extract docstring (common pattern: first statement is Expr(value=Constant(string)))
            docstring = ""
            body = node.get("body", [])
            if body and body[0].get("node_type") == "Expr":
                value = body[0].get("value", {})
                if value.get("node_type") == "Constant" and isinstance(value.get("value"), str):
                    docstring = value["value"].strip()

            # Optional: pull leading comments if your exporter preserved them
            leading_comments = node.get("leading_comments", [])

            # Build a rich text representation for embedding
            arg_str = ""
            if args:
                parameters = args.get("args", []) + args.get("kwonlyargs", [])
                arg_list = []
                for a in parameters:
                    arg_name = a.get("arg", "")
                    annotation = a.get("annotation")
                    if annotation:
                        ann_str = annotation.get("value") if isinstance(annotation, dict) else str(annotation)
                        arg_list.append(f"{arg_name}: {ann_str}")
                    else:
                        arg_list.append(arg_name)
                # Add *args, **kwargs if present
                if args.get("vararg"):
                    arg_list.append(f"*{args['vararg'].get('arg')}")
                if args.get("kwarg"):
                    arg_list.append(f"**{args['kwarg'].get('arg')}")
                arg_str = ", ".join(arg_list)

            return_annotation = ""
            if returns:
                if isinstance(returns, dict):
                    return_annotation = returns.get("value", "")
                else:
                    return_annotation = str(returns)

            full_text = f"""
Function: {name}
File: {file_path}
Args: {arg_str}
Returns: {return_annotation}
Docstring:
{docstring}

Leading comments:
{chr(10).join(c.get('value', '') for c in leading_comments) if leading_comments else 'None'}
            """.strip()

            metadata = {
                "name": name,
                "file": file_path,
                "type": "async" if node_type == "AsyncFunctionDef" else "function",
                "args": json.dumps(args),
                "returns": return_annotation,
                "docstring": docstring,
                "source": "ast_json",
            }

            functions.append({
                "id": f"{file_path}::{name}",
                "document": full_text,
                "metadata": metadata
            })

        # Also handle methods inside ClassDef
        elif node_type == "ClassDef":
            class_name = node.get("name", "")
            class_file_id = f"{file_path}::{class_name}" if file_path else class_name
            body = node.get("body", [])
            for stmt in body:
                if isinstance(stmt, dict) and stmt.get("node_type") in ("FunctionDef", "AsyncFunctionDef"):
                    stmt["name"] = f"{class_name}.{stmt.get('name', 'unknown')}"
                    walk(stmt)

        # Recurse into lists (body, etc.) and dict children
        for key, value in node.items():
            if isinstance(value, dict):
                walk(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        walk(item)

    walk(data)
    return functions


def main():
    parser = argparse.ArgumentParser(description="Import AST JSON into ChromaDB")
    parser.add_argument("json_path", help="Path to the exported AST JSON file or directory")
    parser.add_argument("--collection", default="python_codebase", help="Chroma collection name")
    parser.add_argument("--persist_dir", default="./chroma_db", help="Chroma persist directory")
    parser.add_argument("--embedding", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.persist_dir)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=args.embedding)

    collection = client.get_or_create_collection(
        name=args.collection,
        embedding_function=embedding_fn
    )

    total = 0
    json_files = []
    if os.path.isdir(args.json_path):
        for root, _, files in os.walk(args.json_path):
            for f in files:
                if f.endswith(".json"):
                    json_files.append(os.path.join(root, f))
    else:
        json_files = [args.json_path]

    for json_file in json_files:
        print(f"Processing {json_file}...")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_path = data.get("file_path", os.path.basename(json_file))
        functions = extract_functions_from_ast_json(data, file_path=file_path)

        if not functions:
            print(f"  No functions found in {json_file}")
            continue

        ids = [func["id"] for func in functions]
        documents = [func["document"] for func in functions]
        metadatas = [func["metadata"] for func in functions]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        total += len(functions)
        print(f"  → Added {len(functions)} functions/methods")

    print(f"\nDone! Imported {total} functions into collection '{args.collection}'")


if __name__ == "__main__":
    main()