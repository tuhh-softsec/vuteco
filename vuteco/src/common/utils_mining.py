import hashlib
import os
import re

import git
import javalang as jl
from common.constants import CODE_COL
from cwe2.database import Database
from pycvesearch import CVESearch
from tqdm import tqdm

CVE_SEARCH = CVESearch("https://cve.circl.lu")
CWE_DB = Database()
JUNIT_OVERRIDABLE_METHODS = [
    "countTestCases",
    "createResult",
    "getName",
    "run",
    "runBare",
    "runTest",
    "setName",
    "setUp",
    "tearDown",
    "toString"
]
JUNIT_FIXTURE_ANNOTATIONS = [
    "Override",
    "Before",
    "BeforeEach",
    "After",
    "AfterEach"
]


def retrieve_tests_from_file(filepath: str) -> list:
    with open(filepath, errors="ignore") as fin:
        read_jf = fin.read()
    return retrieve_tests_from_class(read_jf)


def retrieve_tests_from_class(text: str) -> list:
    tests = []
    try:
        cu = jl.parse.parse(text)
    except:
        return tests
    # types contain only the top level classes
    for class_node in list(cu.types):
        if not isinstance(class_node, jl.tree.ClassDeclaration):
            continue
        top_level_method_nodes = []
        for a, b in list(class_node.filter(jl.tree.MethodDeclaration)):
            if b in class_node.methods:
                top_level_method_nodes.append((a, b))
        for tm_node in get_test_methods(class_node, top_level_method_nodes):
            method_text = get_method_text_fixed(tm_node, text)
            # method_text, _, _ = get_method_text_broken(tm_node, top_level_method_nodes, text.splitlines())
            # if re.compile(r"assert|fail|verify|expect", re.IGNORECASE).search(method_text):
            tests.append((cu, class_node, tm_node, method_text.replace("\n\t", "\n").strip()))
    return list(dict.fromkeys(tests))


def get_class_fqn(cu, class_node):
    prefix = f"{cu.package.name}." if cu.package else ""
    return f"{prefix}{class_node.name}"


def get_java_files_in_dir(start_dir: str):
    return [os.path.join(root, file) for root, _, files in os.walk(start_dir) for file in files if file.endswith('.java')]


def blob_to_text(blob):
    try:
        blob_bytes = blob.data_stream.read() if blob is not None else None
        return blob_bytes.decode("utf-8", "ignore")
    except (AttributeError, ValueError):
        return None


def get_test_methods(class_node: jl.tree.ClassDeclaration, method_nodes: list) -> list[jl.tree.MethodDeclaration]:
    test_method_nodes = []
    # If the class is abstract, ignore it
    if "abstract" in [m.lower() for m in class_node.modifiers]:
        return test_method_nodes
    for _, node in method_nodes:
        # if the method is overriding one of those belonging to TestCase
        if node.name in JUNIT_OVERRIDABLE_METHODS:
            continue
        annotations = [s.name for s in node.annotations]
        # if at least one JUnit annotation is found, skip
        if len(set(JUNIT_FIXTURE_ANNOTATIONS) & set(annotations)) > 0:
            continue
        # Must return void if not @TestFactory
        if node.return_type is not None and "TestFactory" not in annotations:
            continue
        method_modifiers = [m.lower() for m in node.modifiers]
        # no static, abstract or private methods
        if "static" in method_modifiers:
            continue
        if "abstract" in method_modifiers:
            continue
        if "private" in method_modifiers:
            continue
        # "test" annotations okay or "test" in superclass name
        if any(re.compile(r"test", re.IGNORECASE).search(a) for a in annotations):
            test_method_nodes.append(node)
        elif class_node.extends:
            ext_name = class_node.extends.name if isinstance(class_node.extends, jl.tree.ReferenceType) else class_node.extends[0].name
            if re.compile(r"test", re.IGNORECASE).search(ext_name):
                test_method_nodes.append(node)
    return test_method_nodes


def get_method_text_fixed(method_node: jl.ast.Node, source_text: str):
    source_tokens = list(jl.tokenizer.tokenize(source_text))
    brace_count = 0
    started = False
    last_tok = None
    for tok in source_tokens:
        if tok.position.line < method_node.position.line:
            continue
        if isinstance(tok, jl.tokenizer.Separator) and tok.value == '{':
            brace_count += 1
            if not started:
                started = True
        if isinstance(tok, jl.tokenizer.Separator) and tok.value == '}':
            brace_count -= 1
            if started and brace_count == 0:
                last_tok = tok
                break
    method_text = "\n".join(source_text.splitlines()[method_node.position.line - 1: last_tok.position.line])
    return method_text


"""
Credits to "anjandash" and "Canonize" (https://github.com/c2nes/javalang/issues/49)
"""


def get_method_text_broken(method_node: jl.ast.Node, tree: list, codelines: list[str]) -> tuple[str, int, int]:
    startpos = None
    endpos = None
    startline = None
    endline = None
    for path, node in tree:
        if startpos is not None and method_node not in path:
            endpos = node.position
            endline = node.position.line if node.position is not None else None
            break
        if startpos is None and node == method_node:
            startpos = node.position
            startline = node.position.line if node.position is not None else None
    # last_endline_index = None
    if startpos is None:
        return "", None, None
    else:
        startline_index = startline - 1
        endline_index = endline - 1 if endpos is not None else None

        # 1. check for and fetch annotations
        # if last_endline_index is not None:
        #    for line in codelines[(last_endline_index + 1):(startline_index)]:
        #        if "@" in line:
        #            startline_index = startline_index - 1
        meth_text = "<ST>".join(codelines[startline_index:endline_index])
        meth_text = meth_text[:meth_text.rfind("}") + 1]

        # 2. remove trailing rbrace for last methods & any external content/comments
        if meth_text.count("{") > meth_text.count("}"):
            for _ in range(meth_text.count("{") - meth_text.count("}")):
                meth_text += "<ST>}"
        if meth_text.count("}") > meth_text.count("{"):
            for _ in range(meth_text.count("}") - meth_text.count("{")):
                meth_text = meth_text[:meth_text.rfind("}")]
                meth_text = meth_text[:meth_text.rfind("}") + 1]

        meth_lines = meth_text.split("<ST>")
        meth_text = "\n".join(meth_lines)
        last_endline_index = startline_index + (len(meth_lines) - 1)

        return meth_text, (startline_index + 1), (last_endline_index + 1)


def get_java_files_in_revision(revision: git.Commit) -> list[str]:
    return [a.abspath for a in revision.tree.list_traverse() if ".java" in a.abspath]


def get_current_file_names_touched_in(revisions: list[git.Commit]) -> list[str]:
    files = []
    for rev in revisions:
        if len(rev.parents) == 0:
            files_touched = []
        else:
            files_touched = [item.a_path for item in rev.diff(rev.parents[0]) if item.a_path]
        # print(f"Files touched in {rev}: {files_touched}")
        for file_touched in files_touched:
            logout = rev.repo.git.log('--follow', "--name-only", '--pretty=format:""', '--', file_touched)
            file_names = list(dict.fromkeys([s for s in logout.splitlines() if ".java" in s]))
            if len(file_names) == 0:
                continue
            # print(f"Names: {file_names}")
            file_cur_name = file_names[0]
            file_cur_path = os.path.join(rev.repo.working_dir, file_cur_name)
            # print(f"Check if {file_cur_path} exists")
            if file_cur_path in files:
                continue
            if not os.path.exists(file_cur_path):
                continue
            files.append(file_cur_path)
    return list(dict.fromkeys(files))


def collect_tests_in_revision(revision: git.Commit, print_unreadable_files: bool = False) -> list[dict]:
    all_tests = []
    java_files = get_java_files_in_revision(revision)
    if len(java_files) == 0:
        return all_tests
    for jf_path in tqdm(java_files, desc=f"  - Collecting test methods in Java files"):
        jf_relpath = os.path.relpath(jf_path, revision.repo.working_dir)
        try:
            tests = retrieve_tests_from_file(jf_path)
        except:
            if print_unreadable_files:
                print(f"Failed to read file {jf_path}. Skipping it.")
            continue
        for jf_cu, class_node, tm_node, test_code in tests:
            class_fqn = get_class_fqn(jf_cu, class_node)
            # In case we need to export them somewhere finalize this code
            # with tempfile.TemporaryDirectory() as tmp_dirname:
            #    dest_file = os.path.join(tmp_dirname, os.path.basename(revision.repo.working_dir), revision.hexsha, class_fqn, f"{tm_node.name}.txt")
            #    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            #    with open(dest_file, 'w') as fout:
            #        fout.write(test_code)
            test_dict = {
                "file": jf_relpath,
                "class": class_fqn,
                "method": tm_node.name,
                CODE_COL: test_code,
                "startline": tm_node.position.line
            }
            all_tests.append(test_dict)
    return all_tests


def make_hash(*fields: str):
    m = hashlib.sha256()
    for f in fields:
        m.update(f.encode())
    return m.hexdigest()


def extract_identifiers(test_code: str, must_separate: bool) -> list[str]:
    ids: list[str] = []
    try:
        tokens = jl.tokenizer.tokenize(test_code, ignore_errors=True)
        method_decl = jl.parser.Parser(tokens).parse_member_declaration()
        if isinstance(method_decl, jl.parser.tree.MethodDeclaration):
            method_name: str = method_decl.name.removeprefix("test")
            var_names: list[str] = [n.name for _, n in method_decl.filter(jl.parser.tree.VariableDeclarator) if n.name != ""]
            if must_separate:
                ids.extend(separate(method_name))
                for vn in var_names:
                    ids.extend(separate(vn))
            else:
                if method_name != "":
                    ids.append(method_name)
                ids.extend(var_names)
    except Exception:
        pass
    return ids


def separate(input: str) -> list[str]:
    results = []
    # First, split by underscore, then look for any hump
    pieces_method_name = [p for i in range(0, input.count("_") + 1) for p in input.rsplit("_", i) if re.match(r"[a-zA-Z_].*", p)]
    for p in pieces_method_name:
        humps = [m.group(0) for m in re.finditer(r'.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', p)]
        results.extend(humps)
    return list(dict.fromkeys(results))


def get_cve_descr_date(cve_id: str):
    api_out = CVE_SEARCH.id(cve_id)
    if api_out is None:
        return None, None
    descriptions = api_out.get("containers", {}).get("cna", {}).get("descriptions", [])
    eng_descriptions = [d for d in descriptions if d["lang"] == "en"]
    if len(eng_descriptions) > 0:
        descr = re.sub(r'\n+', ' ', eng_descriptions[0]["value"])
    else:
        descr = None
    pub_date = api_out.get("cveMetadata", {}).get("datePublished")
    return descr, pub_date


def get_cwe_info(cve_id: str):
    api_out = CVE_SEARCH.id(cve_id)
    if api_out is None:
        return []
    if "problemTypes" not in api_out["containers"]["cna"]:
        return []
    return [d for d in api_out["containers"]["cna"]["problemTypes"][0]["descriptions"] if d["lang"] == "en" and d["type"] == "CWE"]


def get_cwe_name(cwe_id: str):
    try:
        cwe_info = CWE_DB.get(cwe_id.removeprefix("CWE-"))
        return cwe_info.name if cwe_info else None
    except:
        return None


def normalize_repo_url(url: str):
    return url.rstrip("/").removesuffix(".git")


def join_cwe_id_info(cwe_id: str, cwe_name: str):
    return f"{cwe_id} - {cwe_name}"


def prepend_text(base: str, to_prepend: str):
    return f"{f'{to_prepend}. ' if to_prepend else ''}{base}"


def chunk_list(lst: list, chunk_size=16) -> list:
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]