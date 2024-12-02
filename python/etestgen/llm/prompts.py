import seutil as su
from jsonargparse import CLI
from typing import *

from etestgen.data.data import TData, DataMUT2E, DataNE2E
from etestgen.data.utils import load_dataset, save_dataset
from etestgen.macros import Macros


def format_stack_trace_prompt(dt: DataMUT2E) -> str:
    """
    Format the stack trace prompt.
        1. ignore stack trace if it is greater than 5 methods
    """

    method_calls = "```java\n"
    called_ms_seq = []
    mut_name = dt.mut_key.split("#")[1]
    for ms_call in dt.call_stacks:
        if ms_call.code is None:
            continue
        if ms_call.code not in called_ms_seq:
            called_ms_seq.append(ms_call)
    # Ignore uninformative stack trace
    if ignore_stack_trace(called_ms_seq):
        return ""
    # handle edge cases
    if len(called_ms_seq) == 0:
        stack_trace_prompt = (
            f"""The '{dt.etype}' can be thrown within the method {mut_name}."""
        )
        return stack_trace_prompt
    if len(called_ms_seq) == 1:
        called_ms = called_ms_seq[0]
        if called_ms.name + "#" + called_ms.desc == "#".join(dt.mut_key.split("#")[1:]):
            stack_trace_prompt = (
                f"""The '{dt.etype}' can be thrown within the method {mut_name}."""
            )
            return stack_trace_prompt

    # remove the mut to avoid repeated printing
    called_ms = called_ms_seq[0]
    if called_ms.name + "#" + called_ms.desc == "#".join(dt.mut_key.split("#")[1:]):
        called_ms_seq = called_ms_seq[1:]
    for i, called_ms in enumerate(called_ms_seq):
        method_calls += f"//method-{i}\n{called_ms.code}\n"
    #

    method_calls += "```\n"
    last_called_ms = called_ms_seq[-1].name
    stack_trace_prompt = f"""The '{dt.etype}' can be thrown within the '{last_called_ms}' method by invoking the '{mut_name}' method. '{last_called_ms}' can be invoked indirectly by the '{mut_name}' through a sequence of method calls. Consider the following sequence of method calls:\n{method_calls}"""

    return stack_trace_prompt


def LLM_stack_trace_prompt(
    dt: Union[DataMUT2E, DataNE2E], stack_trace: List, ignore_num: int = 6
) -> str:
    """
    Format the stack trace prompt.
        1. ignore stack trace if it is greater than 'ignore_num' methods
        2. include indication of line number
    """
    etype = dt.etype.split(".")[-1]
    method_calls = "```java\n"
    called_method_seq = []
    called_line_nums = []
    mut_name = dt.mut_key.split("#")[1]
    if len(stack_trace) > ignore_num or len(stack_trace) == 0:
        return ""
    last_called_method_name = stack_trace[-1][0]["method"].split("#")[1]

    for ms_call in stack_trace:
        raw_code = ms_call[0]["method_node"]
        line_num = int(ms_call[1])
        called_method_seq.append(raw_code)
        called_line_nums.append(line_num)

    if len(called_method_seq) > 1:
        # add line number to stack trace method
        for i, called_ms in enumerate(called_method_seq[1:]):
            called_ms = add_called_comment_to_method(called_ms, called_line_nums[i + 1])
            method_calls += f"//method-{i}\n{called_ms}\n"
        #
        method_calls += "```\n"
        stack_trace_prompt = f"""The '{etype}' can be thrown within the '{last_called_method_name}' method by invoking the '{mut_name}' method.' {last_called_method_name}' can be invoked indirectly by the '{mut_name}' through a sequence of method calls. Consider the following sequence of method calls:\n{method_calls}"""
    else:
        stack_trace_prompt = (
            f"""The '{etype}' can be thrown within the tested method '{mut_name}'.\n"""
        )

    return stack_trace_prompt


def stack_trace_prompt(dt: DataMUT2E, ignore_num: int = 6) -> str:
    """
    Format the stack trace prompt.
        1. ignore stack trace if it is greater than 'ignore_num' methods
        2. include indication of line number
    """

    method_calls = "```java\n"
    called_method_seq = []
    called_line_nums = []
    etype = dt.etype.split(".")[-1]
    mut_name = dt.mut_key.split("#")[1]
    if dt.e_stack_trace:
        last_called_method_name = dt.e_stack_trace[-1][0]["method"].split("#")[1]
        stack_trace = dt.e_stack_trace
    elif len(dt.call_stacks) == 0:
        return ""
    else:
        stack_trace = dt.call_stacks
        last_called_method_name = stack_trace[-1][0]["method"].split("#")[1]
    if len(stack_trace) > ignore_num:
        return ""
    for ms_call in stack_trace:
        raw_code = ms_call[0]["method_node"]
        line_num = int(ms_call[1])
        called_method_seq.append(raw_code)
        called_line_nums.append(line_num)

    if len(called_method_seq) > 1:
        # add line number to stack trace method
        for i, called_ms in enumerate(called_method_seq[1:]):
            called_ms = add_called_comment_to_method(called_ms, called_line_nums[i + 1])
            method_calls += f"//method-{i}\n{called_ms}\n"
        #
        method_calls += "```\n"
        stack_trace_prompt = f"""The '{etype}' can be thrown within the '{last_called_method_name}' method by invoking the '{mut_name}' method.' {last_called_method_name}' can be invoked indirectly by the '{mut_name}' through a sequence of method calls. Consider the following sequence of method calls:\n{method_calls}"""
    else:
        stack_trace_prompt = (
            f"""The '{etype}' can be thrown within the tested method '{mut_name}'.\n"""
        )

    return stack_trace_prompt


def add_called_line_to_method(called_ms, line_num) -> str:
    """Insert the line number to the method call."""
    if line_num <= 0:
        return called_ms
    try:
        method_lines = called_ms.splitlines()
        method_lines[line_num] = f">>> {method_lines[line_num]}"
    except IndexError:
        print(
            "The called line number is larger than method itself: possibly an anonymous method."
        )
        return called_ms
    return "\n".join(method_lines)


def add_called_comment_to_method(called_ms, line_num) -> str:
    """Insert the line number to the method call."""
    if line_num <= 0:
        return called_ms
    try:
        method_lines = called_ms.splitlines()
        method_lines[line_num] = f"{method_lines[line_num]}  // this line is executed"
    except IndexError:
        print(
            "The called line number is larger than method itself: possibly an anonymous method."
        )
        return called_ms
    return "\n".join(method_lines)


def raw_stack_trace_prompt(dt: DataNE2E) -> str:
    mut_name = dt.mut_key.split("#")[1]
    method_calls = "```java\n"
    called_ms_seq = []
    # mut_name = dt.mut_key.split("#")[1]
    for ms_call in dt.call_stacks:
        if ms_call.code is None:
            continue
        if ms_call.code not in called_ms_seq:
            called_ms_seq.append(ms_call)

    # handle edge cases
    if len(called_ms_seq) <= 1:
        stack_trace_prompt = (
            f"""The '{dt.etype}' can be thrown within the \n```java\n{dt.mut}```\n"""
        )
        return stack_trace_prompt

    # remove the mut to avoid repeated printing
    for i, called_ms in enumerate(called_ms_seq[1:]):
        method_calls += f"//method-{i}\n{called_ms.code}\n"
    #

    method_calls += "```\n"
    last_called_ms = called_ms_seq[-1].name
    stack_trace_prompt = f"""The '{dt.etype}' can be thrown within the '{last_called_ms}' method by invoking '{mut_name}' through a sequence of method calls. Consider the following sequence of method calls:\n{method_calls}"""

    return stack_trace_prompt


def ignore_stack_trace(called_ms_seq: List) -> bool:
    """
    Decide if the stack trace should be ignored.
    """
    if len(called_ms_seq) > 5:
        return True
    else:
        return False


def test_stack_trace_prompt():
    dataset = load_dataset(Macros.data_dir / "mut2e", clz=DataMUT2E)

    for dt in dataset:
        prompt = format_stack_trace_prompt(dt)
        print(prompt)
        # breakpoint()


if __name__ == "__main__":
    su.log.setup(Macros.log_file)
    CLI(test_stack_trace_prompt, as_positional=False)


# def mut2e_w_stack_trace(dt):
#     instruct = f"""Please complete an exceptional behaviour test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nHere is a related test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method '{etest_name}' to compete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
