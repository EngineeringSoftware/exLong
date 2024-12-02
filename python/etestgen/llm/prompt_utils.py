from typing import *
import tiktoken
import re
import seutil as su
import dataclasses

from etestgen.data.data import TData
from etestgen.macros import Macros

logger = su.log.get_logger(__name__)


one_shot_example_dict = {
    "mut_name": "parseMessageML",
    "mut_class": "MessageMLContext",
    "exception": "InvalidInputException",
    "etest_name": "testUIActionWithoutButtonChild",
    "etest_class": "UIActionTest",
    "test_sign": "@Test public void testUIActionWithoutButtonChild ( ) throws Exception {",
    "test_stmts": 'String inputMessageML = "<messageML>" + "<ui-action trigger="click" action="open-im" user-ids="[123,456]" side-by-side="true"></ui-action>" + "</messageML>";',
    "gold": '```java\n@Test\n    public void testUIActionWithoutButtonChild() throws Exception {\n        String inputMessageML = "<messageML>"\n            + "<ui-action trigger="click" action="open-im" user-ids="[123,456]" side-by-side="true"></ui-action>"\n            + "</messageML>";\n        expectedException.expect(InvalidInputException.class);\n        expectedException.expectMessage(\n            "The "ui-action" element must have at least one child that is any of "\n                + "the following elements: [button, uiaction].");\n        context.parseMessageML(inputMessageML, null, MessageML.MESSAGEML_VERSION);\n    }\n```\n',
}


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613") -> int:
    """
    Return the number of tokens used by a list of messages.
    """

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


@dataclasses.dataclass
class OpenAILLMInput:
    system_content: str
    user_content: str


def get_context(data: TData, input: str):
    """Inputs: mut, mut_class, exception, etest_name, etest_class, test_sign"""
    if input == "mut":
        return data.mut
    elif input == "mut_class":
        return data.mut_key.split("#")[0].split("/")[-1]
    elif input == "constructors":
        return "\n".join(data.constructors)
    elif input == "mut_desc":
        return "#".join(data.mut_key.split("#")[1:])
    elif input == "exception":
        return data.etype
    elif input == "etest_class":
        return data.test_e_key.split("#")[0].split("/")[-1]
    elif input == "etest_name":
        return data.test_e_key.split("#")[1]
    elif input == "mut_name":
        return data.mut_key.split("#")[1]
    elif input == "test_sign":
        return data.test_sign_code + " { "
    elif input == "test_stmts":
        return "\n".join(
            [stmt for stmt in data.test_stmt_code]
        )  # sum(data.test_stmt_toks, [])
    elif input == "types_local":
        if data.types_local_simplified[len(data.test_stmts)] == []:
            return None
        return ", ".join(data.types_local_simplified[len(data.test_stmts)])
    elif input == "test_context":
        return data.test_context.replace("adhoc_", "")
    elif input == "types_absent":
        if data.types_absent_simplified[len(data.test_stmts)] == []:
            return None
        return ", ".join(data.types_absent_simplified[len(data.test_stmts)])
    elif input == "fields_set":
        if data.fields_set[len(data.test_stmts)] == []:
            return None
        return ", ".join(data.fields_set[len(data.test_stmts)])
    elif input == "fields_notset":
        if data.fields_notset[len(data.test_stmts)] == []:
            return None
        return ", ".join(data.fields_notset[len(data.test_stmts)])
    elif input == "setup_teardown":
        methods_in_context = []
        for m in data.setup_methods:  # used
            methods_in_context.append(m.get_code())
        for m in data.teardown_methods:  # used
            methods_in_context.append(m.get_code())
        if len(methods_in_context) == 0:
            return None
        return "\n".join(methods_in_context)
    elif input == "last_called_method":
        lcm = data.resolve_last_called_method(len(data.test_stmts))
        if lcm:
            return lcm.get_tokens()
        else:
            return None
    else:
        raise ValueError(f"Unknown input {input}")


def zero_shot_chatgpt_prompt(data: TData) -> List[Dict[str, str]]:
    """
    ChatGPT-based model's zero-shot prompt
    """

    system_content = "You are a helpful programming assistant and an expert Java programmer. You are helping a user writing exceptional-behavior tests for their Java code. "
    setup = f"""Please complete an exceptional behaviour test method in Java to test the method '{get_context(data, 'mut_name')}' in class '{get_context(data, 'mut_class')}' for the exception '{get_context(data, 'exception')}'.\nThe method to be tested is defined as:\n```java\n{get_context(data, 'mut')}\n```\nPlease only give the new exceptional-behavior test method '{get_context(data, 'etest_name')}' which conforms to the @Test(expected = SomeException.class) pattern to compete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n"""
    test_context = ""
    if get_context(data, "test_context"):
        test_context += f"""```java\n{get_context(data, "test_context")}\n```\n"""
    user_content = setup + test_context
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": remove_empty_newlines(user_content)},
    ]
    num_toks = num_tokens_from_messages(messages)
    prompt_data = {"messages": messages, "num_toks": num_toks}
    return prompt_data


def zero_shot_chatgpt_prompt_no_test_method_name(data: TData) -> List[Dict[str, str]]:
    """
    ChatGPT-based model's zero-shot prompt without providing the test method name.
    """

    system_content = "You are a helpful programming assistant and an expert Java programmer. You are helping a user writing exceptional-behavior tests for their Java code. "
    setup = f"""Please complete an exceptional behaviour test method in Java to test the method '{get_context(data, 'mut_name')}' in class '{get_context(data, 'mut_class')}' for the exception '{get_context(data, 'exception')}'.\nThe method to be tested is defined as:\n```java\n{get_context(data, 'mut')}\n```\nPlease only give the new exceptional-behavior test method 'exceptionTest' to compete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n"""
    test_context = ""
    if get_context(data, "test_context"):
        test_context += f"""```java\n{get_context(data, "test_context")}\n```\n"""
    user_content = setup + test_context
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": remove_empty_newlines(user_content)},
    ]
    num_toks = num_tokens_from_messages(messages)
    prompt_data = {"messages": messages, "num_toks": num_toks}
    return prompt_data


def zero_shot_w_traces_chatgpt_prompt(data: TData) -> List[Dict[str, str]]:
    """
    ChatGPT-based model's zero-shot prompt with provided traces to trigger exceptions.
    """

    NO_TRACE = False
    system_content = "You are a helpful programming assistant and an expert Java programmer. You are helping a user writing exceptional-behavior tests for their Java code. "
    mut = get_context(data, "mut")
    mut_name = get_context(data, "mut_name")
    mut_desc = get_context(data, "mut_desc")
    exception = get_context(data, "exception")

    traces_prompt = f"```java\n//tested method\n{mut}\n"
    call_stacks = [c_ms for c_ms in data.call_stacks if c_ms.code is not None]
    if (len(call_stacks) == 1 and mut_desc == call_stacks[0].namedesc) or len(
        call_stacks
    ) == 0:
        last_called_method = mut_name
        NO_TRACE = True
    else:
        method_id = 0
        for called_ms in call_stacks:
            if mut_desc == called_ms.namedesc:
                continue
            traces_prompt += f"\n//method {method_id}\n" + called_ms.code + "\n"
            method_id += 1
        last_called_method = called_ms.name
        if last_called_method == mut_name:
            NO_TRACE = True
    traces_prompt = traces_prompt + "```\n"
    if NO_TRACE:
        user_content = f"""Please complete an exceptional behaviour test method in Java to test the method '{get_context(data, 'mut_name')}' in class '{get_context(data, 'mut_class')}' for the exception '{get_context(data, 'exception')}'.\nThe method to be tested is defined as:\n```java\n{get_context(data, 'mut')}\n```\nPlease only give the new exceptional-behavior test method '{get_context(data, 'etest_name')}' to compete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n"""
    else:
        user_content = f"""Please complete an exceptional behaviour test method in Java to test the method '{get_context(data, 'mut_name')}' in class '{get_context(data, 'mut_class')}' for the exception '{exception}' which is thrown within the '{last_called_method}'.\nPlease note that the '{last_called_method}' method can be invoked indirectly by the '{mut_name}' method through a sequence of method calls. Consider the following sequence of method calls:\n{traces_prompt}\nPlease only give the new exceptional-behavior test method '{get_context(data, 'etest_name')}' to compete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n"""
    test_context = ""
    if get_context(data, "test_context"):
        test_context += f"""```java\n{get_context(data, "test_context")}\n```\n"""
    user_content = user_content + test_context
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": remove_empty_newlines(user_content)},
    ]
    num_toks = num_tokens_from_messages(messages)
    prompt_data = {"messages": messages, "num_toks": num_toks}
    return prompt_data


def zero_shot_w_traces_chatgpt_prompt_without_test_method(
    data: TData,
) -> List[Dict[str, str]]:
    """
    ChatGPT-based model's zero-shot prompt with provided traces to trigger exceptions but without test method name
    """

    NO_TRACE = False
    system_content = "You are a helpful programming assistant and an expert Java programmer. You are helping a user writing exceptional-behavior tests for their Java code. "
    mut = get_context(data, "mut")
    mut_name = get_context(data, "mut_name")
    mut_desc = get_context(data, "mut_desc")
    exception = get_context(data, "exception")

    traces_prompt = f"```java\n//tested method\n{mut}\n"
    call_stacks = [c_ms for c_ms in data.call_stacks if c_ms.code is not None]
    if (len(call_stacks) == 1 and mut_desc == call_stacks[0].namedesc) or len(
        call_stacks
    ) == 0:
        last_called_method = mut_name
        NO_TRACE = True
    else:
        method_id = 0
        for called_ms in call_stacks:
            if mut_desc == called_ms.namedesc:
                continue
            traces_prompt += f"\n//method {method_id}\n" + called_ms.code + "\n"
            method_id += 1
        last_called_method = called_ms.name
        if last_called_method == mut_name:
            NO_TRACE = True
    traces_prompt = traces_prompt + "```\n"
    if NO_TRACE:
        user_content = f"""Please complete an exceptional behaviour test method in Java to test the method '{get_context(data, 'mut_name')}' in class '{get_context(data, 'mut_class')}' for the exception '{get_context(data, 'exception')}'.\nThe method to be tested is defined as:\n```java\n{get_context(data, 'mut')}\n```\nPlease only give the new exceptional-behavior test method 'exceptionTest' to compete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n"""
    else:
        user_content = f"""Please complete an exceptional behaviour test method in Java to test the method '{get_context(data, 'mut_name')}' in class '{get_context(data, 'mut_class')}' for the exception '{exception}' which is thrown within the '{last_called_method}'.\nPlease note that the '{last_called_method}' method can be invoked indirectly by the '{mut_name}' method through a sequence of method calls. Consider the following sequence of method calls:\n{traces_prompt}\nPlease only give the new exceptional-behavior test method 'exceptionTest' to compete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n"""
    test_context = ""
    if get_context(data, "test_context"):
        test_context += f"""```java\n{get_context(data, "test_context")}\n```\n"""
    user_content = user_content + test_context
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": remove_empty_newlines(user_content)},
    ]
    num_toks = num_tokens_from_messages(messages)
    prompt_data = {"messages": messages, "num_toks": num_toks}
    return prompt_data


def zero_shot_full_context_example(data: TData) -> str:
    """mut, test class context, constructor, teco context"""

    setup = f"""You are a Java programmer who is completing the following test method starting from the below given lines of code..\n```java\n{get_context(data, 'test_sign')}
    {get_context(data, 'test_stmts')}\n```\nYour task is to complete the exceptional behaviour test in Java to test the method '{get_context(data, 'mut_name')}' in class '{get_context(data, 'mut_class')}' to test the exception '{get_context(data, 'exception')}'.\n\nThe method to be tested is defined as:\n```java\n{get_context(data, 'mut')}\n```\n"""
    test_context = ""
    if get_context(data, "constructors"):
        test_context += f"""The constructors for class {get_context(data, 'mut_class')} are:\n ```java\n{get_context(data, "constructors")}\n```\n"""
    if get_context(data, "test_context"):
        test_context += f"""The existing code in the test class is:\n ```java\n{get_context(data, "test_context")}\n```\n"""
    if get_context(data, "fields_set"):  # The following fields
        test_context += f"""The following fields have been initialized: {get_context(data, "fields_set")}.\n"""
    if get_context(data, "fields_notset"):
        test_context += f"""The following fields from test class and the class under test have not been initialized: {get_context(data, "fields_notset")}.\n"""
    if get_context(data, "setup_teardown"):
        test_context += f"""The existing setup teardown methods are:\n ```java\n{get_context(data, "setup_teardown")}.\n```\n"""
    # if get_context(data, "last_called_method"):
    #     test_context += f"""The last called method is: {get_context(data, "last_called_method")}.\n"""
    if get_context(data, "types_local"):
        test_context += f"""The types for the existing local variables are: {get_context(data, "types_local")}.\n"""
    if get_context(data, "types_absent"):
        test_context += f"""The types for the variables that are needed but have not been prepared are: {get_context(data, "types_absent")}.\n"""
    return setup + test_context


def zero_shot_teco_context_example(data: TData) -> str:
    setup = f"""Please complete the exceptional behaviour test in Java to test the method {get_context(data, 'mut_name')} in class {get_context(data, 'mut_class')} to test the exception {get_context(data, 'exception')}.\nThe name for the test is {get_context(data, 'etest_name')} from class {get_context(data, 'etest_class')}."""
    tear_down = f"""These are the first few statments in the test, please start from the following code and complete the test method:\n```java\n{get_context(data, 'test_sign')}
    {get_context(data, 'test_stmts')}```\n"""
    test_context = ""
    if get_context(data, "fields_set"):  # The following fields
        test_context += f"""The following fields have been initialized: {get_context(data, "fields_set")}.\n"""
    if get_context(data, "fields_notset"):
        test_context += f"""The following fields from test class and the class under test have not been initialized: {get_context(data, "fields_notset")}.\n"""
    if get_context(data, "setup_teardown"):
        test_context += f"""The existing setup teardown methods are: {get_context(data, "setup_teardown")}.\n"""
    if get_context(data, "last_called_method"):
        test_context += f"""The last called method is: {get_context(data, "last_called_method")}.\n"""
    if get_context(data, "types_local"):
        test_context += f"""The types for the local variables are: {get_context(data, "types_local")}.\n"""
    if get_context(data, "types_absent"):
        test_context += f"""The types for the variables that are needed but have not been prepared are: {get_context(data, "types_absent")}.\n"""
    return setup + test_context + tear_down


def zero_shot_prompt(data: TData) -> str:
    return f"""Please complete the exceptional behaviour test in Java for the method {get_context(data, 'mut_name')} in class {get_context(data, 'mut_class')} to test the exception {get_context(data, 'exception')}.\nThe name for the test is {get_context(data, 'etest_name')} from class {get_context(data, 'etest_class')}.\nThese are the first few statments in the test, please start from the following code and complete the test method:\n```java\n{get_context(data, 'test_sign')}
    {get_context(data, 'test_stmts')}```\n"""


def one_shot_example() -> str:
    return f"""Please complete the exceptional behaviour test in Java for the method {one_shot_example_dict.get('mut_name')} in class {one_shot_example_dict.get('mut_class')} to test the exception {one_shot_example_dict.get('exception')}.\nThe name for the test is {one_shot_example_dict.get( 'etest_name')} from class {one_shot_example_dict.get('etest_class')}.\nThese are the first few statments in the test, please start from the following code and complete the test method:\n```java\n{one_shot_example_dict.get('test_sign')}
    {one_shot_example_dict.get('test_stmts')}\n```\nCompleted test:\n{one_shot_example_dict.get('gold')}\n\n"""


def one_shot_full_context_example() -> str:
    # TODO
    pass


def exception_condition_prompt_w_call_traces(data: TData) -> List[Dict[str, str]]:
    """
    Ask LLM for what conditions will cause the exception given the called methods.
    """

    system_content = "You are a helpful programming assistant and an expert Java programmer. You are helping a user writing exceptional-behavior tests for their Java code. "
    mut = get_context(data, "mut")
    mut_name = get_context(data, "mut_name")
    exception = get_context(data, "exception")

    traces_prompt = f"```java\n//tested method\n{mut}\n```\n"
    for i, called_ms in enumerate(data.call_stacks):
        traces_prompt += f"```java\n//method{i}\n" + called_ms.code + "\n```\n"
    last_called_method = called_ms
    user_content = f"""Please provide the specific conditions under which the {exception} is **thrown** within the '{last_called_method.name}' method (referred to as the 'method {i}') by invoking the '{mut_name}' method (referred to as 'tested method '). Please note that the '{last_called_method.name}' method can be invoked indirectly by the '{mut_name}' method through a sequence of method calls. Consider the following sequence of method calls:\n{traces_prompt}\nPlease number list all the possible conditions (e.g., '1. 2. ...) without any sub-bullet points or detailed examples."""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return messages


def zero_shot_openai_prompt_with_condition(data: TData) -> OpenAILLMInput:
    system_content = "You are a Java programmer who wants to write tests that test exceptional behaviors."
    setup = f"""Please write the exceptional behaviour test in Java to test the method '{get_context(data, 'mut_name')}' in class '{get_context(data, 'mut_class')}' to test the exception '{get_context(data, 'exception')}'.\nThe method to be tested is defined as:\n```java\n{get_context(data, 'mut')}\n```\nTo trigger the exception, the test should satisfy the following condition: {data.condition}\n"""
    condition = f"""To test the {get_context(data, 'exception')} thrown by the {get_context(data, 'mut_name')} method, {data.condition.replace('If', '').replace('if', '').strip()}"""
    test_context = ""
    if get_context(data, "constructors"):
        test_context += f"""The constructors for class {get_context(data, 'mut_class')} are:\n ```java\n{get_context(data, "constructors")}\n```\n"""
    if get_context(data, "test_context"):
        test_context += f"""The existing code in the test class is:\n ```java\n{get_context(data, "test_context")}\n```\n"""
    # if get_context(data, "fields_set"):  # The following fields
    #     test_context += f"""The following fields have been initialized: {get_context(data, "fields_set")}.\n"""
    # if get_context(data, "fields_notset"):
    #     test_context += f"""The following fields from test class and the class under test have not been initialized: {get_context(data, "fields_notset")}.\n"""
    # if get_context(data, "setup_teardown"):
    #     test_context += f"""The existing setup teardown methods are:\n ```java\n{get_context(data, "setup_teardown")}.\n```\n"""
    # if get_context(data, "types_local"):
    #     test_context += f"""The types for the existing local variables are: {get_context(data, "types_local")}.\n"""
    # if get_context(data, "types_absent"):
    #     test_context += f"""The types for the variables that are needed but have not been prepared are: {get_context(data, "types_absent")}.\n"""
    user_content = setup + test_context + condition
    return OpenAILLMInput(system_content, user_content)


def one_shot_prompt(data: TData) -> str:
    return one_shot_example() + zero_shot_prompt(data) + "Completed test:\n"


def extract_code_from_response(response: str) -> str:
    try:
        response = response.replace("```java", "")
        generated_code = response.replace("```", "")
    except IndexError:
        return response
    return generated_code


hard_code_prompt = """Please write an exceptional behaviour tests in Java for the following method in 
    class `MessageMLContext` to test the exception `InvalidInputException`:
    ```java
    /**
     * Parse the text contents of the message and optionally EntityJSON into a MessageMLV2 message. Expands Freemarker templates and generates document tree structures for serialization into output formats with the respective get() methods.
     * 
     * @param message string containing a MessageMLV2 message with optional Freemarker templates
     * @param entityJson string containing EntityJSON data
     * @param version string containing the version of the message format 
     * @throws InvalidInputException thrown on invalid MessageMLV2 input
     * @throws ProcessingException thrown on errors generating the document tree
     * @throws IOException thrown on invalid EntityJSON input
     */
    public void parseMessageML(String message, String entityJson, String version)
        throws InvalidInputException, IOException, ProcessingException {
        this.presentationML = null;
        this.messageML = messageMLParser.parse(message, entityJson, version);
        this.entityJson = messageMLParser.getEntityJson();
        this.biContext = messageMLParser.getBiContext();
        this.markdownRenderer = new MarkdownRenderer(messageML.asMarkdown());
    }
    ```
    The name for the test is `testTableCellInvalidRowSpan` from class `TableTest`:
    """

gpt_result = """
'Here is an example of a test case in Java for the `parseMessageML` method, testing the `InvalidInputException` for an invalid row span value in a table cell:\n\n```java\nimport org.junit.Test;\nimport static org.junit.Assert.*;\nimport org.junit.Before;\n\npublic class TableTest {\n\n    private MessageMLContext messageMLContext;\n\n    @Before\n    public void setUp() {\n        messageMLContext = new MessageMLContext();\n    }\n\n    @Test(expected = InvalidInputException.class)\n    public void testTableCellInvalidRowSpan() throws Exception {\n        String message = "<messageML><table><tr><td rowspan=\\"0\\">cell 1</td></tr></table></messageML>";\n        String entityJson = "{}";\n        String version = "2.0";\n        messageMLContext.parseMessageML(message, entityJson, version);\n    }\n}\n```\n\nThis test case uses the `@Test` annotation to define a unit test for the `parseMessageML` method with an invalid row span value. It expects an `InvalidInputException` to be thrown when the test is executed. \n\nThe `@Before` annotation is used to set up the `MessageMLContext` instance for each test case.\n\nIn the test method, the input message contains a table with a cell that has an invalid row span value of 0. The `parseMessageML` method is called with this input, and the expected exception is declared in the `@Test` annotation using the `expected` parameter.\n\nIf the `InvalidInputException` is thrown during the execution of this test, the test passes. If any other exception is thrown or no exception is thrown at all, the test fails.'
"""


def get_prompt(prompt: str, data: TData) -> OpenAILLMInput:
    """
    Returns the prompt of a certain style.

    Parameters:
    - prompt (str): The style of the prompt.
    - data (TData): Data to be passed to the corresponding prompt function.

    Returns:
    - OpenAILLMInput: The prompt which contains system message and user message.
    """

    prompt_mapping: Dict[str, Callable[[TData], str]] = {
        "chatgpt_zero_shot_traces": zero_shot_w_traces_chatgpt_prompt,
        "chatgpt_zero_shot_sampling": zero_shot_chatgpt_prompt,
        "chatgpt_zero_shot_no_test_name": zero_shot_chatgpt_prompt_no_test_method_name,
        "chatgpt_zero_shot_traces_no_test_name": zero_shot_w_traces_chatgpt_prompt_without_test_method,
        "condition_with_traces": exception_condition_prompt_w_call_traces,
    }

    try:
        return prompt_mapping[prompt](data)
    except KeyError:
        raise ValueError(f"Unknown prompt style {prompt}")


def remove_empty_newlines(s: str) -> str:
    # Replace multiple newlines with a single newline
    return re.sub(r"\n\s*\n", "\n", s)
