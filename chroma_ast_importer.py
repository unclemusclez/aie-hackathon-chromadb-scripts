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
    Extract functions and methods from JSON structure.
    Handles two formats:
    1. AST format: nodes with node_type fields (FunctionDef, ClassDef, etc.)
    2. Code analysis format: top-level dict with file paths, each containing classes/functions arrays
    """
    functions = []

    # Check if this is the code analysis format (file paths as top-level keys)
    if isinstance(data, dict) and not any(k in data for k in ["node_type", "type"]):
        # This is the code analysis format - iterate over file paths
        for file_key, file_data in data.items():
            if not isinstance(file_data, dict):
                continue
            
            # Use the file key as the file path, or extract from file_data
            current_file_path = file_data.get("file_path", file_key)
            
            # Extract standalone functions
            file_functions = file_data.get("functions", [])
            for func in file_functions:
                if not isinstance(func, dict):
                    continue
                functions.append(_build_function_entry(func, current_file_path, is_method=False))
            
            # Extract methods from classes
            classes = file_data.get("classes", [])
            for cls in classes:
                if not isinstance(cls, dict):
                    continue
                class_name = cls.get("name", "")
                methods = cls.get("methods", [])
                for method in methods:
                    if not isinstance(method, dict):
                        continue
                    # Prefix method name with class name
                    method_copy = method.copy()
                    method_copy["name"] = f"{class_name}.{method.get('name', 'unknown')}"
                    functions.append(_build_function_entry(method_copy, current_file_path, is_method=True))
        
        return functions

    # Otherwise, try the AST format (original logic)
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


def _build_function_entry(func: Dict[str, Any], file_path: str, is_method: bool = False) -> Dict[str, Any]:
    """Build a function entry from the code analysis format."""
    name = func.get("name", "unknown")
    docstring = func.get("docstring", "") or ""
    returns = func.get("returns", "")
    is_async = func.get("is_async", False)
    arguments = func.get("arguments", [])
    comment = func.get("comment", "")
    line = func.get("line", "")
    end_line = func.get("end_line", "")
    decorators = func.get("decorators", [])
    variables = func.get("variables", [])
    calls = func.get("calls", [])
    attributes = func.get("attributes", [])
    
    # Build argument string
    arg_list = []
    for arg in arguments:
        arg_name = arg.get("name", "")
        annotation = arg.get("annotation")
        default = arg.get("default")
        if annotation:
            arg_str = f"{arg_name}: {annotation}"
        else:
            arg_str = arg_name
        if default is not None:
            arg_str += f" = {default}"
        arg_list.append(arg_str)
    arg_str = ", ".join(arg_list)
    
    # Build decorators string
    decorators_str = ", ".join(decorators) if decorators else "None"
    
    # Build full text representation (for embedding/search)
    func_type = "async method" if (is_async and is_method) else ("async function" if is_async else ("method" if is_method else "function"))
    
    # Format variables as a simple list for document text
    variables_list = []
    for var in variables:
        var_name = var.get("name", "")
        var_type = var.get("value_type", "")
        var_preview = var.get("value_preview", "")
        var_comment = var.get("comment", "")
        var_entry = var_name
        if var_type:
            var_entry += f" ({var_type})"
        if var_preview:
            var_entry += f": {var_preview}"
        if var_comment:
            var_entry += f" # {var_comment}"
        variables_list.append(var_entry)
    variables_text = ", ".join(variables_list) if variables_list else "None"
    
    # Format function calls as a simple list
    calls_list = []
    for call in calls:
        func_name = call.get("function", "")
        line_num = call.get("line", "")
        args_count = call.get("args_count", 0)
        keywords = call.get("keywords", [])
        call_entry = func_name
        if args_count > 0:
            call_entry += f"({args_count} args"
            if keywords:
                call_entry += f", keywords: {', '.join(keywords)}"
            call_entry += ")"
        calls_list.append(call_entry)
    calls_text = ", ".join(calls_list) if calls_list else "None"
    
    # Format attributes as a simple list
    attrs_list = []
    for attr in attributes:
        obj = attr.get("object", "")
        attr_name = attr.get("attribute", "")
        attrs_list.append(f"{obj}.{attr_name}")
    attributes_text = ", ".join(attrs_list) if attrs_list else "None"
    
    # Build document text in the format requested
    comment_display = comment if comment else "None"
    docstring_display = docstring if docstring else ""
    returns_display = returns if returns else ""
    
    full_text = f"""Function: {name}
File: {file_path}
Args: {arg_str}
Returns: {returns_display}
Docstring:
{docstring_display}
Comment: {comment_display}
variables: {variables_text}
function calls: {calls_text}
attributes: {attributes_text}""".strip()
    
    # Build metadata (for filtering/display)
    metadata = {
        "name": name,
        "file": file_path,
        "type": func_type,
        "returns": returns or "",
        "docstring": docstring,
        "is_async": str(is_async),
        "is_method": str(is_method),
        "line": str(line) if line else "",
        "end_line": str(end_line) if end_line else "",
        "decorators": json.dumps(decorators),
        "variables_count": str(len(variables)),
        "calls_count": str(len(calls)),
        "attributes_count": str(len(attributes)),
        "source": "code_analysis_json",
    }
    
    # Add variables, calls, and attributes as JSON strings for detailed access
    metadata["variables"] = json.dumps(variables)
    metadata["calls"] = json.dumps(calls)
    metadata["attributes"] = json.dumps(attributes)
    
    return {
        "id": f"{file_path}::{name}",
        "document": full_text,
        "metadata": metadata
    }


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

        # For the code analysis format, file_path is extracted inside extract_functions_from_ast_json
        # For AST format, we need to provide it
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