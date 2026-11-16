"""
AST-based code chunker for RAG pipelines.

Why AST-based chunking instead of fixed-size text splitting:
Fixed-size splitters cut code at arbitrary character counts, which frequently
slices a function in half or separates a method from the class it belongs to.
This module instead walks Python's Abstract Syntax Tree (AST) and chunks along
*semantic* boundaries — modules, functions, classes, and methods — so each
chunk is a complete, syntactically valid, self-contained unit of code that
also carries the surrounding context (e.g. a method chunk includes its
class's header and attributes) needed to make sense of it in isolation.

Pipeline overview (see chunk_project at the bottom):
    find_python_files      -> walk the project, list every .py file
    read_python_file       -> read each file's source text
    extract_code_structure -> split a file's top-level AST nodes into
                               (module-level code, functions, classes)
    build_*_chunk           -> turn each of those into human/LLM-readable text
    split_if_large          -> fall back to character-based splitting only
                               when a single node is too big to embed whole
    create_code_chunk       -> wrap the text + metadata into a LangChain Document
"""

from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ast

# Directories we never want to walk into: virtual envs, VCS metadata, caches,
# and JS dependency folders that sometimes sit inside a polyglot repo.
EXCLUDED_DIRECTORIES = {
    ".venv",
    "venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
}

# Upper bound (in characters, not tokens) for a single chunk's text before we
# fall back to splitting it further. This is a rough proxy for an embedding
# model's token limit — roughly 4 characters per token for English/code text.
# Tune this down if your embedding model has a small context window (e.g.
# 512 tokens), or replace the character count in split_if_large with a real
# tokenizer (tiktoken, the model's own tokenizer, etc.) for precision.
MAX_CHUNK_CHARS = 3000

# LangChain's recursive splitter: it tries to cut on paragraph breaks first,
# then line breaks, then spaces, only falling back to splitting mid-word as
# a last resort. chunk_overlap keeps a little shared context between the
# pieces of an oversized node so retrieval doesn't lose continuity at the
# seam. This is only ever invoked on already-semantically-scoped text (a
# single function/method/module chunk that's too long), never on a whole
# file — so it's a safety net, not the primary chunking strategy.
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHUNK_CHARS,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
)


def find_python_files(project_path):
    """
    Recursively list every .py file under project_path, skipping anything
    inside EXCLUDED_DIRECTORIES (at any depth — e.g. src/app/.venv/foo.py
    is excluded because ".venv" appears as one of its path parts).

    Returns a list of Path objects *relative* to project_path, so they can
    be used later both as a display/metadata name and re-joined with
    project_path to get the real file location.
    """
    project_path = Path(project_path)
    paths = project_path.rglob("*.py")

    final_paths = []
    for path in paths:
        # path.parts is a tuple of every path segment, e.g.
        # ('src', 'app', '.venv', 'foo.py') — checking membership against
        # EXCLUDED_DIRECTORIES catches excluded folders regardless of depth.
        if any(part in EXCLUDED_DIRECTORIES for part in path.parts):
            continue
        relative_path = path.relative_to(project_path)
        final_paths.append(relative_path)

    return final_paths


def read_python_file(path):
    """
    Read a source file as UTF-8 text.

    Falls back to errors="ignore" on a UnicodeDecodeError rather than
    raising, so one file with unusual/corrupted encoding doesn't abort
    the whole project's chunking run — we'd rather chunk that file with a
    few dropped bytes than skip it (or the whole project) entirely.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def extract_code_structure(source):
    """
    Parse source into an AST and bucket its *top-level* statements into
    three groups:
      - module_nodes: anything that isn't a top-level def/class (imports,
        constants, `if __name__ == "__main__":` blocks, etc.)
      - functions: top-level `def` / `async def` statements
      - classes: top-level `class` statements

    Note this only looks at tree.body (the top level of the file) — a
    function defined inside another function, or a class nested inside a
    class, is *not* pulled out here; it stays embedded inside its parent's
    source text as-is. See get_class_methods for the equivalent one-level
    walk used inside classes.
    """
    tree = ast.parse(source)

    module_nodes = []
    functions = []
    classes = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node)
        elif isinstance(node, ast.ClassDef):
            classes.append(node)
        else:
            module_nodes.append(node)

    return module_nodes, functions, classes


def get_node_source(source, node):
    """
    Return the exact source text for a single AST node, decorators included.

    ast.get_source_segment (stdlib) gives us a node's source text, but its
    range starts at the `def`/`class` keyword — it does NOT include any
    @decorator lines above it. Since decorators are semantically part of
    the function/class (e.g. @staticmethod, @app.route(...)), we detect
    them via node.decorator_list and, if present, manually slice the
    source starting from the first decorator's line instead.
    """
    if hasattr(node, "decorator_list") and node.decorator_list:
        lines = source.splitlines(keepends=True)
        start_line = min(decorator.lineno for decorator in node.decorator_list)
        # end_lineno is provided by the AST for the node itself, so we only
        # need to move the *start* earlier to cover the decorators.
        return "".join(lines[start_line - 1:node.end_lineno])

    return ast.get_source_segment(source, node)


def get_class_header(source, class_node):
    """
    Return just the class's declaration — decorators (if any) through the
    line containing the closing `:` of `class Foo(Bar):` — WITHOUT the
    class body. This is what gets prepended to every method chunk so a
    method retrieved on its own still shows what class/base class it
    belongs to, without repeating the entire class body in every chunk.

    Implementation note: a class signature can span multiple lines (e.g.
    long base-class lists or generic parameters), so we scan line-by-line
    from the class's start until we find one containing ":" and treat that
    as the end of the header. This is a simple heuristic — it would
    misfire on a class whose signature legitimately contains a ":" before
    the real one (e.g. inside a subscripted generic type on its own line),
    which is a rare edge case in practice.
    """
    lines = source.splitlines(keepends=True)

    start_line = class_node.lineno
    if class_node.decorator_list:
        start_line = min(decorator.lineno for decorator in class_node.decorator_list)

    class_start = class_node.lineno - 1  # convert 1-indexed lineno to 0-indexed

    for i in range(class_start, class_node.end_lineno):
        line = lines[i]
        if ":" in line:
            return "".join(lines[start_line - 1:i + 1])

    # Fallback if no ":" was found within the class's line range (should not
    # normally happen for valid Python, but avoids a hard failure).
    return "".join(lines[start_line - 1:class_node.lineno])


def extract_class_context(source, class_node):
    """
    Return the source of every class-body statement that ISN'T a method or
    a nested class — i.e. class-level attributes like `MAX_USERS = 100`,
    type annotations, docstrings, or other class-body statements.

    This is what lets a method chunk reference `self.MAX_USERS` and still
    be understandable: the constant's definition rides along in the same
    chunk as "Class context", even though the constant lives elsewhere in
    the class body from the method itself.
    """
    context_nodes = [
        child
        for child in class_node.body
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    context_parts = []
    for node in context_nodes:
        node_source = get_node_source(source, node)
        if node_source:
            context_parts.append(node_source)

    return "\n\n".join(context_parts)


def get_class_methods(class_node):
    """
    Return the direct-child def/async def nodes of a class — i.e. its
    methods. Like extract_code_structure, this only looks one level deep:
    a method defined inside another method is not surfaced separately.
    """
    return [
        child
        for child in class_node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def create_code_chunk(text, chunk_type, file_path, name=None, class_name=None,
                       start_line=None, end_line=None):
    """
    Wrap a piece of chunk text into a LangChain Document, the standard unit
    LangChain's retrievers/vector stores expect.

    metadata fields:
      source     - the file the chunk came from (relative path)
      type       - "module" | "function" | "class_overview" | "method"
      name       - function/method/class name (None for module chunks... well,
                   set to the file's stem so it's still identifiable)
      class_name - the owning class, only set for "method" chunks
      start_line / end_line - where in the source file this chunk's code
                   lives, useful for jumping back to the real file from a
                   search result

    Keeping this metadata means downstream retrieval can filter/display
    results (e.g. "show me only chunks from UserService" or "link back to
    sample.py:12-18") instead of getting back an anonymous blob of text.
    """
    return Document(
        page_content=text,
        metadata={
            "source": str(file_path),
            "type": chunk_type,
            "name": name,
            "class_name": class_name,
            "start_line": start_line,
            "end_line": end_line,
        },
    )


def split_if_large(text):
    """
    Safety-net splitter. Every build_*_chunk function below produces one
    semantically-scoped piece of text (one function, one method-with-context,
    etc.). Most of the time that's already a reasonably sized chunk — but an
    unusually long function/method (or a class with many attributes) could
    still exceed what an embedding model can handle in one pass.

    If the text fits under MAX_CHUNK_CHARS, return it unchanged as a
    single-item list. Otherwise, hand it to the RecursiveCharacterTextSplitter
    to break it into smaller overlapping pieces. This is only a fallback —
    the primary chunk boundaries are always the AST ones.
    """
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    return _splitter.split_text(text)


def build_method_chunk(source, class_node, method_node):
    """
    Build the text for a single method chunk. Format:

        Class: <ClassName>
        <class header, e.g. "class UserService(BaseService):">

        Class context:
        <class-level attributes, if any>

        Method: <method_name>
        <the method's own full source, including its decorators>

    The point of prepending the class header + class context is that a
    method chunk retrieved in isolation (e.g. by a semantic search hit)
    still tells the reader/LLM what class it belongs to, what it inherits
    from, and what constants/attributes it might reference — without
    requiring every method chunk to also carry every *other* method's body.
    """
    class_header = get_class_header(source, class_node)
    class_context = extract_class_context(source, class_node)
    method_source = get_node_source(source, method_node)

    parts = [f"Class: {class_node.name}", class_header]

    if class_context:
        parts.append("Class context:\n" + class_context)

    parts.append(f"Method: {method_node.name}")
    parts.append(method_source)

    return "\n\n".join(parts)


def build_class_overview_chunk(source, class_node):
    """
    Build a single "birds-eye view" chunk for a class: its header, its
    class-level attributes, and the *signatures* (not bodies) of all its
    methods. Format:

        Class: <ClassName>
        <class header>

        Class attributes:
        <class-level attributes, if any>

        Methods:
            def method_one(self, x)
            async def method_two(self, y)

    This exists alongside the per-method chunks (build_method_chunk) so
    that a broad query like "what does UserService do" or "what methods
    does this class expose" can match one compact chunk instead of relying
    on the retriever to piece together every individual method chunk.

    ast.unparse (Python 3.9+) turns the method's argument-list AST node
    back into source-like text (e.g. "self, name, role='user'"); on older
    Python versions where ast.unparse doesn't exist, we fall back to "..."
    rather than failing.
    """
    class_header = get_class_header(source, class_node)
    class_context = extract_class_context(source, class_node)

    method_signatures = []
    for method in get_class_methods(class_node):
        prefix = "async def" if isinstance(method, ast.AsyncFunctionDef) else "def"
        try:
            args = ast.unparse(method.args)
        except AttributeError:
            args = "..."  # Python < 3.9 fallback
        method_signatures.append(f"     {prefix} {method.name}({args})")

    parts = [f"Class: {class_node.name}", class_header]

    if class_context:
        parts.append("Class attributes:\n" + class_context)
    if method_signatures:
        parts.append("Methods:\n" + "\n".join(method_signatures))

    return "\n\n".join(parts)


def build_function_chunk(source, function_node):
    """
    Build the text for a single top-level function chunk: just a short
    "Function: <name>" label followed by the function's full source
    (decorators included, via get_node_source). Unlike methods, top-level
    functions don't need any extra class context prepended — they're
    already self-contained.

    Returns None if get_node_source somehow couldn't retrieve the source
    (e.g. malformed node), so the caller can skip it rather than emit an
    empty/garbage chunk.
    """
    function_source = get_node_source(source, function_node)
    if not function_source:
        return None
    return f"Function: {function_node.name}\n\n{function_source}"


def build_module_chunk(source, module_nodes):
    """
    Concatenate the source of every top-level, non-function/class statement
    in a file — imports, constants, `if __name__ == "__main__":` blocks,
    etc. — into one chunk. This captures file-level context (what the
    module imports, what globals it defines) that would otherwise be
    dropped entirely, since extract_code_structure routes these statements
    away from the functions/classes buckets.
    """
    parts = []
    for node in module_nodes:
        node_source = get_node_source(source, node)
        if node_source:
            parts.append(node_source)
    return "\n\n".join(parts)


def chunk_file(file_path, project_root):
    """
    Full single-file pipeline: read the file, parse it, and produce a flat
    list of Document chunks covering its module-level code, its top-level
    functions, and its classes (one overview chunk + one chunk per method
    for each class).

    file_path is relative (as returned by find_python_files); project_root
    is joined with it to get the real filesystem path to read from, while
    file_path itself is what gets stored in each chunk's metadata — so
    metadata stays portable/relative rather than leaking an absolute local
    path.

    Errors are caught per-file rather than per-project: a single file with
    a SyntaxError (e.g. Python 2 source, a half-edited file) or an unreadable
    file just gets skipped with a printed warning, so one bad file doesn't
    abort chunking for the rest of the project.
    """
    full_path = project_root / file_path

    try:
        source = read_python_file(full_path)
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"Skipping {file_path}: syntax error ({e})")
        return []
    except OSError as e:
        print(f"Skipping {file_path}: could not read file ({e})")
        return []

    module_nodes, functions, classes = extract_code_structure(source)
    chunks = []

    # --- Module-level code (imports, constants, top-level side effects) ---
    if module_nodes:
        module_text = build_module_chunk(source, module_nodes)
        if module_text:
            # split_if_large almost always returns [module_text] unchanged;
            # it only kicks in for a file with an unusually large amount of
            # top-level code.
            for piece in split_if_large(module_text):
                chunks.append(create_code_chunk(
                    piece, "module", file_path,
                    name=file_path.stem,
                    start_line=module_nodes[0].lineno,
                    end_line=module_nodes[-1].end_lineno,
                ))

    # --- Top-level functions: one chunk each (further split if oversized) ---
    for func in functions:
        func_text = build_function_chunk(source, func)
        if not func_text:
            continue
        for piece in split_if_large(func_text):
            chunks.append(create_code_chunk(
                piece, "function", file_path,
                name=func.name,
                start_line=func.lineno,
                end_line=func.end_lineno,
            ))

    # --- Classes: one overview chunk, plus one context-rich chunk per method ---
    for cls in classes:
        overview_text = build_class_overview_chunk(source, cls)
        for piece in split_if_large(overview_text):
            chunks.append(create_code_chunk(
                piece, "class_overview", file_path,
                name=cls.name,
                start_line=cls.lineno,
                end_line=cls.end_lineno,
            ))

        for method in get_class_methods(cls):
            method_text = build_method_chunk(source, cls, method)
            for piece in split_if_large(method_text):
                chunks.append(create_code_chunk(
                    piece, "method", file_path,
                    name=method.name,
                    class_name=cls.name,
                    start_line=method.lineno,
                    end_line=method.end_lineno,
                ))

    return chunks


def chunk_project(project_path):
    """
    Top-level entry point: find every .py file under project_path (minus
    excluded directories) and chunk each one, returning a single flat list
    of Document chunks across the whole project — ready to hand to a vector
    store's `.add_documents(...)`.
    """
    project_path = Path(project_path)
    file_paths = find_python_files(project_path)

    all_chunks = []
    for file_path in file_paths:
        all_chunks.extend(chunk_file(file_path, project_path))

    return all_chunks


if __name__ == "__main__":
    # Quick manual smoke test: chunk the current directory and print a
    # preview of the first few chunks with their metadata, so you can
    # eyeball that the boundaries/context look right before wiring this
    # into an embedding + vector-store step.
    chunks = chunk_project(".")
    print(f"Produced {len(chunks)} chunks")

    for chunk in chunks[:3]:
        print("=" * 50)
        print(chunk.metadata)
        print(chunk.page_content[:300])