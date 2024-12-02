import javalang
import re
from typing import Any, List, Union, Dict
import dataclasses

from etestgen.data.data import DataMUT2E, DataNE2E


@dataclasses.dataclass
class LLMResults:
    id: int = None
    project: str = None
    input: DataNE2E = None
    cname: str = None
    mname: str = None
    module_i: int = None
    prompt: Union[str, List[str]] = None
    gold: Union[List[str], str] = None
    prior_stmts: List[List[str]] = None
    topk: List[str] = None
    metrics: Dict[str, float] = None


@dataclasses.dataclass
class LLMOutput:
    responses: List[str] = None  # responses
    prompt: Union[str, List[Dict[str, str]]] = None


@dataclasses.dataclass
class LLMConfig:
    model: str = None
    model_path: str = None
    data_cls: str = None
    prompt_style: str = None
    device: str = None
    num_gpus: int = None
    load_8bit: bool = None
    eval_method: str = None
    temperature: float = 0.0
    debug: bool = None
    sample_size: int = 1
    stop: List[str] = None


def extract_java_methods(
    code_text: str, target_method_name: str = None
) -> Union[str, dict]:
    """Extract the java methods from the generated code."""

    tree = javalang.parse.parse(code_text)
    lex = -1
    methods = {}

    for _, method_node in tree.filter(javalang.tree.MethodDeclaration):
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

        method_text, startline, endline, lex = get_method_text(
            startpos,
            endpos,
            startline,
            endline,
            lex,
            code_text.splitlines(keepends=True),
        )
        methods[method_node.name] = method_text

    if target_method_name and target_method_name in methods:
        return methods[target_method_name]
    elif len(methods) == 1:
        v = list(methods.values())[0]
        return v
    elif len(methods) > 1:
        for method_name, method_text in methods.items():
            if "test" in method_name:
                return method_text
    return None


def extract_test_from_raw_code(raw_code: str, method_name: str) -> str:
    if "public class" not in raw_code:
        adhoc_test = "public class adhoc_Test {\n " + raw_code + " }\n"
    else:
        adhoc_test = raw_code
    try:
        predicted_test = extract_java_methods(
            code_text=adhoc_test, target_method_name=method_name
        )
    except:
        predicted_test = raw_code
    if predicted_test is None:
        predicted_test = raw_code
    return predicted_test


def extract_code_from_response(response: str) -> str:
    """
    Extract the code from the response of the LLM, excluding natural language descriptions.
    """
    try:
        # response = response.replace("```java", "```")
        if "```java" in response:
            generated_code = response.split("```java")[1].split("```")[0].strip()
        else:
            generated_code = response
    except IndexError:
        generated_code = response
    if generated_code == "":
        generated_code = response.replace("```", "")
    if not generated_code.strip().startswith("@Test"):
        java_class_code = extract_code_from_model(generated_code)
        try:
            generated_code = postprocess_outputs(java_class_code)
        except:
            generated_code = java_class_code
    return generated_code.strip()


def get_method_text(
    startpos: Any,
    endpos: Any,
    startline: int,
    endline: int,
    last_endline_index: int,
    codelines: List[str],
):
    if startpos is None:
        return "", None, None, None
    else:
        startline_index = startline - 1
        endline_index = endline - 1 if endpos is not None else None

        # 1. check for and fetch annotations
        if last_endline_index is not None:
            for line in codelines[(last_endline_index + 1) : (startline_index)]:
                if "@" in line:
                    startline_index = startline_index - 1
        meth_text = "<ST>".join(codelines[startline_index:endline_index])
        meth_text = meth_text[: meth_text.rfind("}") + 1]

        # 2. remove trailing rbrace for last methods & any external content/comments
        # if endpos is None and
        if not abs(meth_text.count("}") - meth_text.count("{")) == 0:
            # imbalanced braces
            brace_diff = abs(meth_text.count("}") - meth_text.count("{"))

            for _ in range(brace_diff):
                meth_text = meth_text[: meth_text.rfind("}")]
                meth_text = meth_text[: meth_text.rfind("}") + 1]

        meth_lines = meth_text.split("<ST>")
        meth_text = "".join(meth_lines)
        last_endline_index = startline_index + (len(meth_lines) - 1)

        return (
            meth_text,
            (startline_index + 1),
            (last_endline_index + 1),
            last_endline_index,
        )


def extract_code_from_model(gen_code: str):
    code_lines = gen_code.splitlines()
    class_start_line = 0
    for i in range(len(code_lines)):
        if "public class" in code_lines[i]:
            class_start_line = i
            break
    gen_code = "\n".join(code_lines[class_start_line + 1 :])
    return gen_code


def postprocess_outputs(text: str):

    re_template = r"void\s+(\w+)\s*\((.*?)\)\s*(throws\s+[\w,\s]+)?\s*\{([^}]*)\}"
    match = re.search(re_template, text, flags=re.MULTILINE)
    
    if match is None and "@Test" in text:
        test_annotations = text.split("@Test")
        if len(test_annotations) <= 2:
            updated_text = text.split("}\n")[0]
            not_test = "Copyright" in updated_text or (
                "test" not in updated_text and "Test" not in updated_text
            )
            if not_test:
                print("\t\t\t\t\t\tERROR: new file, no test generated")
                return "" if not_test else updated_text
            return test_annotations[0] + "@Test" + test_annotations[1].split("@")[0]
    elif match is None:
        return text
    else:
        matched_text = match.group()
        curr_text = matched_text.encode("utf-8").decode("unicode_escape")
        new_text = text.split(curr_text)[0] + curr_text
        if "Copyright" in new_text:
            print("\t\t\t\t\t\tERROR: new file, no test generated")
            return ""
        return new_text
