import seutil as su
from jsonargparse import CLI
from tqdm import tqdm
import random
from typing import List

random.seed(42)
from etestgen.llm.prompts import (
    add_called_comment_to_method,
    format_stack_trace_prompt,
    raw_stack_trace_prompt,
)
from etestgen.llm.prompt_utils import num_tokens_from_messages, get_context
from etestgen.codellama.DataProcessor import find_the_most_revelant_netest
from etestgen.collector.MetricsCollector import remove_duplicate_stack_trace
from etestgen.data.data import TData, DataMUT2E, DataNE2E, parse_data_cls
from etestgen.data.utils import load_dataset
from etestgen.macros import Macros
from etestgen.llm.prompts import (
    LLM_stack_trace_prompt,
    format_stack_trace_prompt,
    add_called_line_to_method,
    raw_stack_trace_prompt,
    stack_trace_prompt,
)


class PromptProcessor:
    def __init__(self, config_file: str):
        self.config = su.io.load(config_file)
        self.data_dir = (
            Macros.work_dir / "setup" / self.config["setup"] / self.config["model_name"]
        )
        su.io.mkdir(self.data_dir)
        self.dataset = load_dataset(
            Macros.data_dir / self.config["data_file"],
            clz=parse_data_cls(self.config["data_type"]),
        )
        # copy the data files to setup dir
        setup_dir = Macros.work_dir / "setup" / self.config["setup"]
        su.io.mkdir(setup_dir / "eval")
        su.bash.run(
            f"cp -r {Macros.data_dir / self.config['data_file']} {setup_dir}/eval/test",
            0,
        )

    def process_prompts(self, which: str):
        if which == "gpt4-reason":
            self.create_reasons_prompt()
        elif which == "mut2e-zero-shot":
            self.create_mut2e_prompt()
        elif which == "gpt-ne2e-no-name-one-shot":
            self.create_ne2e_no_name_one_shot()
        elif which == "gpt-ne2e-with-name-one-shot":
            self.create_ne2e_with_name_one_shot()
        elif which == "ne2e":
            self.create_ne2e_prompt()
        elif which == "gpt3.5-nestack2e":
            self.create_nestack2e_prompt()
        elif which == "gpt4-cot":
            self.create_cot_prompt()
        elif which == "gpt4-nestack2e-few-shot":
            self.create_nestack2e_few_shot_prompt()
        elif which == "gpt3.5-stack2e-zero-shot":
            self.create_stack2e_zero_shot_prompt()
        elif which == "gpt4-conditionnestack2e-one-shot-no-name":
            self.create_condition_ne_stack_2e_no_name_one_shot()
        elif which == "gpt4-conditionnestack2e-one-shot-with-name":
            self.create_condition_ne_stack_2e_with_name_one_shot()
        else:
            raise RuntimeError(f"Unknown prompt type: {which}")

    ####
    # Helper functions
    ####
    def create_stack2e_zero_shot_prompt(self):
        dataset = load_dataset(
            Macros.work_dir / self.config["data_file"],
            clz=parse_data_cls(self.config["data_type"]),
        )
        processed_prompts = []
        for dt in tqdm(dataset, total=len(dataset)):
            processed_prompts.append(self.process_gpt_stack2e_prompt(dt))
        #
        su.io.dump(self.data_dir / "prompt.jsonl", processed_prompts)

    def create_cot_prompt(self):
        dataset = load_dataset(
            Macros.work_dir / self.config["data_file"],
            clz=parse_data_cls(self.config["data_type"]),
        )
        processed_prompts = []
        for dt in tqdm(dataset, total=len(dataset)):
            processed_prompts.append(self.create_cot_prompt_from_data(dt))
        #
        su.io.dump(self.data_dir / "prompt.jsonl", processed_prompts)

    def create_condition_ne2e_no_name_one_shot(self):
        processed_prompts = []
        # few shot messages
        example = """Please complete an exceptional behaviour test method in Java to test the method 'create' in class 'ProtocolSwitch' for the exception 'IllegalArgumentException'.\nThe method to be tested is defined as:\n```java\npublic static ProtocolSwitch create(int value) {\n    ProtocolSwitch status = new ProtocolSwitch();\n    status.setBs(toBitSet(value));\n    return status;\n}\n```\nThe exception will be triggered if the following condition is satisfied:\n```java\nvalue < 0 || value > Byte.MAX_VALUE\n```\nHere is one related non-exceptional test method for reference:\n```java\n@Test\npublic void test_createUsingIndex() {\n    for (int i = 0; i < 7; ++i) {\n        Assert.assertTrue(ProtocolSwitch.create(new int[] { i }).isOn(i));\n    }\n    int size = 7;\n    int[] a = new int[size];\n    for (int i = 0; i < size; ++i) {\n        a[i] = i;\n    }\n    ProtocolSwitch status = ProtocolSwitch.create(a);\n    for (int i = 0; i < size; ++i) {\n        Assert.assertTrue(status.isOn(i));\n    }\n}\n```\nPlease only give the new exceptional-behavior test method to compete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\npackage com.alipay.remoting.inner.utiltest;\nimport java.util.BitSet;\nimport org.junit.After;\nimport org.junit.AfterClass;\nimport org.junit.Assert;\nimport org.junit.Before;\nimport org.junit.BeforeClass;\nimport org.junit.Test;\nimport com.alipay.remoting.config.switches.ProtocolSwitch;\npublic class ProtocolSwitchTest {\n    @BeforeClass\n    public static void initClass() {\n    }\n    @Before\n    public void init() {\n    }\n    @After\n    public void stop() {\n    }\n    @AfterClass\n    public static void afterClass() {\n    }\n}\n```\n"""
        test_e = """@Test\npublic void test_createUsingByte() {\n    Assert.assertFalse(ProtocolSwitch.create(0).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(1).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(2).isOn(1));\n    Assert.assertTrue(ProtocolSwitch.create(4).isOn(2));\n    Assert.assertTrue(ProtocolSwitch.create(8).isOn(3));\n    Assert.assertTrue(ProtocolSwitch.create(3).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(3).isOn(1));\n    Assert.assertFalse(ProtocolSwitch.create(3).isOn(2));\n    Assert.assertTrue(ProtocolSwitch.create(64).isOn(6));\n    Assert.assertFalse(ProtocolSwitch.create(64).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(127).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(127).isOn(1));\n    Assert.assertTrue(ProtocolSwitch.create(127).isOn(6));\n    Assert.assertFalse(ProtocolSwitch.create(127).isOn(7));\n    try {\n        ProtocolSwitch.create(255);\n        Assert.fail(\"Should not reach here!\");\n    } catch (IllegalArgumentException e) {\n    }\n}"""
        ground_truth = f"""```java\n{test_e}\n```\n"""

        messages = [
            {
                "role": "system",
                "content": "As a knowledgeable programming assistant and proficient Java developer, your role is to assist a user in creating exceptional-behavior tests for their Java code.",
            },
            {"role": "user", "content": example},
            {"role": "assistant", "content": ground_truth},
        ]
        few_shot_example = messages
        for data in tqdm(self.dataset, total=len(self.dataset)):
            mut_name = data.mut_key.split("#")[1]
            mut_class = data.mut_key.split("#")[0].split(".")[-1]
            exception_type = data.etype.split(".")[-1]
            mut_code = data.mut
            etest_name = data.test_e_key.split("#")[1]
            etest_context = data.test_context.replace("adhoc_", "").replace(
                "/*TEST PLACEHOLDER*/", ""
            )
            etest_class = data.test_e_key.split("#")[0].split("/")[-1]
            # process condition prompt
            if type(data.condition) == str:
                data.condition = [data.condition]
            data.condition = list(set([cond for cond in data.condition if cond != ""]))
            if len(data.condition) > 0:
                condition = random.choice(data.condition)
                condition_prompt = f"""The exception will be triggered if the following condition is satisfied:\n```java\n{condition}\n```\n"""
            else:
                condition_prompt = ""
            # make prompt
            if not data.test_ne:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}Please only give the new exceptional-behavior test method to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
            else:
                ne_test = random.choice(data.test_ne)
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""

            new_messages = [
                {"role": "user", "content": instruct},
            ]
            messages = few_shot_example + new_messages
            num_toks = num_tokens_from_messages(messages)
            prompt_data = {"messages": messages, "num_toks": num_toks}
            processed_prompts.append(prompt_data)
        su.io.dump(self.data_dir / "prompt.jsonl", processed_prompts)

    def create_conditions_ne2e_no_name_one_shot(self, selected_ids: List = None):
        """
        Sample 5 conditions from all possible conditions
        """
        processed_prompts = []
        # few shot messages
        example = """Please complete an exceptional behaviour test method in Java to test the method 'create' in class 'ProtocolSwitch' for the exception 'IllegalArgumentException'.\nThe method to be tested is defined as:\n```java\npublic static ProtocolSwitch create(int value) {\n    ProtocolSwitch status = new ProtocolSwitch();\n    status.setBs(toBitSet(value));\n    return status;\n}\n```\nThe exception will be triggered if the following condition is satisfied:\n```java\nvalue < 0 || value > Byte.MAX_VALUE\n```\nHere is one related non-exceptional test method for reference:\n```java\n@Test\npublic void test_createUsingIndex() {\n    for (int i = 0; i < 7; ++i) {\n        Assert.assertTrue(ProtocolSwitch.create(new int[] { i }).isOn(i));\n    }\n    int size = 7;\n    int[] a = new int[size];\n    for (int i = 0; i < size; ++i) {\n        a[i] = i;\n    }\n    ProtocolSwitch status = ProtocolSwitch.create(a);\n    for (int i = 0; i < size; ++i) {\n        Assert.assertTrue(status.isOn(i));\n    }\n}\n```\nPlease only give the new exceptional-behavior test method to compete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\npackage com.alipay.remoting.inner.utiltest;\nimport java.util.BitSet;\nimport org.junit.After;\nimport org.junit.AfterClass;\nimport org.junit.Assert;\nimport org.junit.Before;\nimport org.junit.BeforeClass;\nimport org.junit.Test;\nimport com.alipay.remoting.config.switches.ProtocolSwitch;\npublic class ProtocolSwitchTest {\n    @BeforeClass\n    public static void initClass() {\n    }\n    @Before\n    public void init() {\n    }\n    @After\n    public void stop() {\n    }\n    @AfterClass\n    public static void afterClass() {\n    }\n}\n```\n"""
        test_e = """@Test\npublic void test_createUsingByte() {\n    Assert.assertFalse(ProtocolSwitch.create(0).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(1).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(2).isOn(1));\n    Assert.assertTrue(ProtocolSwitch.create(4).isOn(2));\n    Assert.assertTrue(ProtocolSwitch.create(8).isOn(3));\n    Assert.assertTrue(ProtocolSwitch.create(3).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(3).isOn(1));\n    Assert.assertFalse(ProtocolSwitch.create(3).isOn(2));\n    Assert.assertTrue(ProtocolSwitch.create(64).isOn(6));\n    Assert.assertFalse(ProtocolSwitch.create(64).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(127).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(127).isOn(1));\n    Assert.assertTrue(ProtocolSwitch.create(127).isOn(6));\n    Assert.assertFalse(ProtocolSwitch.create(127).isOn(7));\n    try {\n        ProtocolSwitch.create(255);\n        Assert.fail(\"Should not reach here!\");\n    } catch (IllegalArgumentException e) {\n    }\n}"""
        ground_truth = f"""```java\n{test_e}\n```\n"""

        messages = [
            {
                "role": "system",
                "content": "As a knowledgeable programming assistant and proficient Java developer, your role is to assist a user in creating exceptional-behavior tests for their Java code.",
            },
            {"role": "user", "content": example},
            {"role": "assistant", "content": ground_truth},
        ]
        few_shot_example = messages
        tmp_dataset = load_dataset(
            Macros.work_dir / "data" / "gt-ne2e-test", clz=DataNE2E
        )
        for i, data in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            if selected_ids and data.id not in selected_ids:
                continue
            mut_name = data.mut_key.split("#")[1]
            mut_class = data.mut_key.split("#")[0].split(".")[-1]
            exception_type = data.etype.split(".")[-1]
            mut_code = data.mut
            etest_name = data.test_e_key.split("#")[1]
            etest_context = data.test_context.replace("adhoc_", "").replace(
                "/*TEST PLACEHOLDER*/", ""
            )
            etest_class = data.test_e_key.split("#")[0].split("/")[-1]
            # process condition prompt
            if type(data.condition) == str:
                data.condition = [data.condition]
            data.condition = list(set([cond for cond in data.condition if cond != ""]))
            for condition in data.condition:
                condition_prompt = f"""The exception will be triggered if the following condition is satisfied:\n```java\n{condition}\n```\n"""
                # make prompt
                if not data.test_ne:
                    instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}Please only give the new exceptional-behavior test method to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                else:
                    ne_test = random.choice(data.test_ne)
                    instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""

                new_messages = [
                    {"role": "user", "content": instruct},
                ]
                messages = few_shot_example + new_messages
                num_toks = num_tokens_from_messages(messages)
                prompt_data = {
                    "id": data.id,
                    "messages": messages,
                    "num_toks": num_toks,
                }
                processed_prompts.append(prompt_data)
        su.io.dump(self.data_dir / "prompt.jsonl", processed_prompts)

    def create_condition_ne2e_with_name_one_shot(self):
        processed_prompts = []
        # few shot messages
        example = """Please complete an exceptional behaviour test method in Java to test the method 'create' in class 'ProtocolSwitch' for the exception 'IllegalArgumentException'.\nThe method to be tested is defined as:\n```java\npublic static ProtocolSwitch create(int value) {\n    ProtocolSwitch status = new ProtocolSwitch();\n    status.setBs(toBitSet(value));\n    return status;\n}\n```\nThe exception will be triggered if the following condition is satisfied:\n```java\nvalue < 0 || value > Byte.MAX_VALUE\n```\nHere is one related non-exceptional test method for reference:\n```java\n@Test\npublic void test_createUsingIndex() {\n    for (int i = 0; i < 7; ++i) {\n        Assert.assertTrue(ProtocolSwitch.create(new int[] { i }).isOn(i));\n    }\n    int size = 7;\n    int[] a = new int[size];\n    for (int i = 0; i < size; ++i) {\n        a[i] = i;\n    }\n    ProtocolSwitch status = ProtocolSwitch.create(a);\n    for (int i = 0; i < size; ++i) {\n        Assert.assertTrue(status.isOn(i));\n    }\n}\n```\nPlease only give the new exceptional-behavior test method 'test_createUsingByte' to compete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\npackage com.alipay.remoting.inner.utiltest;\nimport java.util.BitSet;\nimport org.junit.After;\nimport org.junit.AfterClass;\nimport org.junit.Assert;\nimport org.junit.Before;\nimport org.junit.BeforeClass;\nimport org.junit.Test;\nimport com.alipay.remoting.config.switches.ProtocolSwitch;\npublic class ProtocolSwitchTest {\n    @BeforeClass\n    public static void initClass() {\n    }\n    @Before\n    public void init() {\n    }\n    @After\n    public void stop() {\n    }\n    @AfterClass\n    public static void afterClass() {\n    }\n}\n```\n"""
        test_e = """@Test\npublic void test_createUsingByte() {\n    Assert.assertFalse(ProtocolSwitch.create(0).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(1).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(2).isOn(1));\n    Assert.assertTrue(ProtocolSwitch.create(4).isOn(2));\n    Assert.assertTrue(ProtocolSwitch.create(8).isOn(3));\n    Assert.assertTrue(ProtocolSwitch.create(3).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(3).isOn(1));\n    Assert.assertFalse(ProtocolSwitch.create(3).isOn(2));\n    Assert.assertTrue(ProtocolSwitch.create(64).isOn(6));\n    Assert.assertFalse(ProtocolSwitch.create(64).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(127).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(127).isOn(1));\n    Assert.assertTrue(ProtocolSwitch.create(127).isOn(6));\n    Assert.assertFalse(ProtocolSwitch.create(127).isOn(7));\n    try {\n        ProtocolSwitch.create(255);\n        Assert.fail(\"Should not reach here!\");\n    } catch (IllegalArgumentException e) {\n    }\n}"""
        ground_truth = f"""```java\n{test_e}\n```\n"""

        messages = [
            {
                "role": "system",
                "content": "As a knowledgeable programming assistant and proficient Java developer, your role is to assist a user in creating exceptional-behavior tests for their Java code.",
            },
            {"role": "user", "content": example},
            {"role": "assistant", "content": ground_truth},
        ]
        few_shot_example = messages
        for data in tqdm(self.dataset, total=len(self.dataset)):
            mut_name = data.mut_key.split("#")[1]
            mut_class = data.mut_key.split("#")[0].split(".")[-1]
            exception_type = data.etype.split(".")[-1]
            mut_code = data.mut
            etest_name = data.test_e_key.split("#")[1]
            etest_context = data.test_context.replace("adhoc_", "").replace(
                "/*TEST PLACEHOLDER*/", ""
            )
            etest_class = data.test_e_key.split("#")[0].split("/")[-1]
            # process condition prompt
            data.condition = [cond for cond in data.condition if cond != ""]
            if len(data.condition) > 0:
                condition = random.choice(data.condition)
                assert condition != ""
                condition_prompt = f"""The exception will be triggered if the following condition is satisfied:\n```java\n{condition}\n```\n"""
            else:
                condition_prompt = ""
            # make prompt
            if not data.test_ne:
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}Please only give the new exceptional-behavior test method '{etest_name}' to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
            else:
                ne_test = random.choice(data.test_ne)
                instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method '{etest_name}' to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""

            new_messages = [
                {"role": "user", "content": instruct},
            ]
            messages = few_shot_example + new_messages
            num_toks = num_tokens_from_messages(messages)
            prompt_data = {"messages": messages, "num_toks": num_toks}
            processed_prompts.append(prompt_data)
        su.io.dump(self.data_dir / "prompt.jsonl", processed_prompts)

    def create_ne2e_with_name_one_shot(self):
        """
        Create NE2E prompts for GPT models w. nEBT.
        """
        dataset = load_dataset(
            Macros.data_dir / self.config["data_file"],
            clz=parse_data_cls(self.config["data_type"]),
        )
        # copy the data files to setup dir
        setup_dir = Macros.work_dir / "setup" / self.config["setup"]
        su.io.mkdir(setup_dir / "eval")
        processed_prompts = []
        # few shot messages
        example = """Please complete an exceptional behavior test method in Java to test the method 'build' in class 'io.reinert.requestor.core.uri.UriBuilderImpl' for the exception 'IllegalArgumentException'.\nThe method to be tested is defined as:\n```java\n@Override\n    public Uri build(Map<String, ?> values) {\n        final List<String> parsedSegments = new ArrayList<String>();\n        final LinkedHashMap<String, LinkedHashMap<String, Uri.Param>> parsedMatrixParams =\n                matrixParams != null && values != null && values.size() > 0 ?\n                        new LinkedHashMap<String, LinkedHashMap<String, Uri.Param>>() : null;\n\n        if (segments != null) {\n            for (final String segment : segments) {\n                final String parsed = parsePart(values, segment);\n\n                // Replace the template segment for the parsed one if necessary\n                if (parsedMatrixParams != null && matrixParams.containsKey(segment)) {\n                    parsedMatrixParams.put(parsed, matrixParams.get(segment));\n                }\n\n                parsedSegments.add(parsed);\n            }\n        }\n\n        final String parsedFrag = parsePart(values, fragment);\n\n        return new UriImpl(scheme, user, password, host, port, parsedSegments,\n                parsedMatrixParams != null ? parsedMatrixParams : matrixParams, queryParams, parsedFrag);\n    }\n```\nHere is a related non-exceptional test method for reference:\n```java\n@Test\npublic void build_ProperTemplateValuesMap_ShouldBuildSuccessfully() {\n    String expected = \"http://user:pwd@localhost:8888/server/1/any;class=2;class=5;class=6\" + \"/child;group=A;subGroup=A.1;subGroup=A.2?age=12&name=Aa&name=Zz#firstserver\";\n    Map<String, Object> params = new HashMap<String, Object>();\n    params.put(\"a\", \"server\");\n    params.put(\"b\", 1);\n    params.put(\"c\", \"any\");\n    params.put(\"d\", \"first\");\n    String uri = UriBuilder.newInstance().scheme(\"http\").user(\"user\").password(\"pwd\").host(\"localhost\").port(8888).path(\"/{a}/{b}\").segment(\"{c}\").matrixParam(\"class\", 2, 5, 6).segment(\"child\").matrixParam(\"group\", \"A\").matrixParam(\"subGroup\", \"A.1\", \"A.2\").queryParam(\"age\", 12).queryParam(\"name\", \"Aa\", \"Zz\").fragment(\"{d}{a}\").build(params).toString();\n    assertEquals(expected, uri);\n}\n```\nPlease only give the new exceptional-behavior test method 'build_InsufficientTemplateValuesMap_ShouldThrowIllegalArgumentException' using the '@Test(expected=exception.class)' pattern to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\npackage io.reinert.requestor.core.uri;\n\nimport java.util.HashMap;\nimport java.util.Map;\nimport org.junit.Test;\nimport static org.junit.Assert.assertEquals;\n\npublic class UriBuilderJreTest {\n    \n}\n\n```\n"""
        test_e = """@Test(expected = IllegalArgumentException.class)\npublic void build_InsufficientTemplateValuesMap_ShouldThrowIllegalArgumentException() {\n    Map<String, Object> params = new HashMap<String, Object>();\n    params.put(\"a\", \"server\");\n    params.put(\"b\", 1);\n    params.put(\"c\", \"any\");\n    UriBuilder.newInstance().path(\"{a}/{b}\").segment(\"{c}\").fragment(\"{d}{a}\").build(params);\n}\n"""
        ground_truth = f"""```java\n{test_e}\n```\n"""

        messages = [
            {
                "role": "system",
                "content": "As a knowledgeable programming assistant and proficient Java developer, your role is to assist a user in creating exceptional-behavior tests for their Java code. The Junit test should use the @Test(expected = exception.class) pattern.",
            },
            {"role": "user", "content": example},
            {"role": "assistant", "content": ground_truth},
        ]
        few_shot_example = messages
        for data in tqdm(dataset, total=len(dataset)):
            mut_name = data.mut_key.split("#")[1]
            mut_class = data.mut_key.split("#")[0].split(".")[-1]
            exception_type = data.etype.split(".")[-1]
            mut_code = data.mut
            etest_name = data.test_e_key.split("#")[1]
            etest_context = data.test_context.replace("adhoc_", "").replace(
                "/*TEST PLACEHOLDER*/", ""
            )
            etest_class = data.test_e_key.split("#")[0].split("/")[-1]
            # make prompt
            if not data.test_ne:
                instruct = f"""Please write an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nPlease only give the new exceptional-behavior test method '{etest_name}' using the '@Test(expected=exception.class)' pattern to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the test method:\n```java\n{etest_context}\n```\n"""
            else:
                ne_test = random.choice(data.test_ne)
                instruct = f"""Please write an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nHere is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method '{etest_name}' using the '@Test(expected=exception.class)' pattern to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the test method:\n```java\n{etest_context}\n```\n"""

            new_messages = [
                {"role": "user", "content": instruct},
            ]
            messages = few_shot_example + new_messages
            num_toks = num_tokens_from_messages(messages)
            prompt_data = {"id": data.id, "messages": messages, "num_toks": num_toks}
            processed_prompts.append(prompt_data)
        su.io.dump(self.data_dir / "prompt.jsonl", processed_prompts)

    def create_ne2e_no_name_one_shot(self):
        """
        Create NE2E prompts for GPT models w.o. nEBT.
        """
        dataset = load_dataset(
            Macros.data_dir / self.config["data_file"],
            clz=parse_data_cls(self.config["data_type"]),
        )
        # copy the data files to setup dir
        setup_dir = Macros.work_dir / "setup" / self.config["setup"]
        su.io.mkdir(setup_dir / "eval")
        processed_prompts = []
        # few shot messages
        example = """Please complete an exceptional behavior test method in Java to test the method 'build' in class 'io.reinert.requestor.core.uri.UriBuilderImpl' for the exception 'IllegalArgumentException'.\nThe method to be tested is defined as:\n```java\n@Override\n    public Uri build(Map<String, ?> values) {\n        final List<String> parsedSegments = new ArrayList<String>();\n        final LinkedHashMap<String, LinkedHashMap<String, Uri.Param>> parsedMatrixParams =\n                matrixParams != null && values != null && values.size() > 0 ?\n                        new LinkedHashMap<String, LinkedHashMap<String, Uri.Param>>() : null;\n\n        if (segments != null) {\n            for (final String segment : segments) {\n                final String parsed = parsePart(values, segment);\n\n                // Replace the template segment for the parsed one if necessary\n                if (parsedMatrixParams != null && matrixParams.containsKey(segment)) {\n                    parsedMatrixParams.put(parsed, matrixParams.get(segment));\n                }\n\n                parsedSegments.add(parsed);\n            }\n        }\n\n        final String parsedFrag = parsePart(values, fragment);\n\n        return new UriImpl(scheme, user, password, host, port, parsedSegments,\n                parsedMatrixParams != null ? parsedMatrixParams : matrixParams, queryParams, parsedFrag);\n    }\n```\nHere is a related non-exceptional test method for reference:\n```java\n@Test\npublic void build_ProperTemplateValuesMap_ShouldBuildSuccessfully() {\n    String expected = \"http://user:pwd@localhost:8888/server/1/any;class=2;class=5;class=6\" + \"/child;group=A;subGroup=A.1;subGroup=A.2?age=12&name=Aa&name=Zz#firstserver\";\n    Map<String, Object> params = new HashMap<String, Object>();\n    params.put(\"a\", \"server\");\n    params.put(\"b\", 1);\n    params.put(\"c\", \"any\");\n    params.put(\"d\", \"first\");\n    String uri = UriBuilder.newInstance().scheme(\"http\").user(\"user\").password(\"pwd\").host(\"localhost\").port(8888).path(\"/{a}/{b}\").segment(\"{c}\").matrixParam(\"class\", 2, 5, 6).segment(\"child\").matrixParam(\"group\", \"A\").matrixParam(\"subGroup\", \"A.1\", \"A.2\").queryParam(\"age\", 12).queryParam(\"name\", \"Aa\", \"Zz\").fragment(\"{d}{a}\").build(params).toString();\n    assertEquals(expected, uri);\n}\n```\nPlease only give the new exceptional-behavior test method using the '@Test(expected=exception.class)' pattern to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\npackage io.reinert.requestor.core.uri;\n\nimport java.util.HashMap;\nimport java.util.Map;\nimport org.junit.Test;\nimport static org.junit.Assert.assertEquals;\n\npublic class UriBuilderJreTest {\n    \n}\n\n```\n"""
        test_e = """@Test(expected = IllegalArgumentException.class)\npublic void build_InsufficientTemplateValuesMap_ShouldThrowIllegalArgumentException() {\n    Map<String, Object> params = new HashMap<String, Object>();\n    params.put(\"a\", \"server\");\n    params.put(\"b\", 1);\n    params.put(\"c\", \"any\");\n    UriBuilder.newInstance().path(\"{a}/{b}\").segment(\"{c}\").fragment(\"{d}{a}\").build(params);\n}\n"""
        ground_truth = f"""```java\n{test_e}\n```\n"""

        messages = [
            {
                "role": "system",
                "content": "As a knowledgeable programming assistant and proficient Java developer, your role is to assist a user in creating exceptional-behavior tests for their Java code. The Junit test should use the @Test(expected = exception.class) pattern.",
            },
            {"role": "user", "content": example},
            {"role": "assistant", "content": ground_truth},
        ]
        few_shot_example = messages
        for data in tqdm(dataset, total=len(dataset)):
            mut_name = data.mut_key.split("#")[1]
            mut_class = data.mut_key.split("#")[0].split(".")[-1]
            exception_type = data.etype.split(".")[-1]
            mut_code = data.mut
            etest_context = data.test_context.replace("adhoc_", "").replace(
                "/*TEST PLACEHOLDER*/", ""
            )
            etest_class = data.test_e_key.split("#")[0].split("/")[-1]
            # make prompt
            if not data.test_ne:
                instruct = f"""Please write an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nPlease only give the new exceptional-behavior test method using the '@Test(expected=exception.class)' pattern to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the test method:\n```java\n{etest_context}\n```\n"""
            else:
                ne_test = random.choice(data.test_ne)
                instruct = f"""Please write an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nHere is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method using the '@Test(expected=exception.class)' pattern to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the test method:\n```java\n{etest_context}\n```\n"""

            new_messages = [
                {"role": "user", "content": instruct},
            ]
            messages = few_shot_example + new_messages
            num_toks = num_tokens_from_messages(messages)
            prompt_data = {"id": data.id, "messages": messages, "num_toks": num_toks}
            processed_prompts.append(prompt_data)
        su.io.dump(self.data_dir / "prompt.jsonl", processed_prompts)

    def create_condition_ne_stack_2e_no_name_one_shot(self):
        example = """Please complete an exceptional behavior test method in Java to test the method 'build' in class 'io.reinert.requestor.core.uri.UriBuilderImpl' for the exception 'IllegalArgumentException'.\nThe method to be tested is defined as:\n```java\n    @Override\n    public Uri build(Map<String, ?> values) {\n        final List<String> parsedSegments = new ArrayList<String>();\n        final LinkedHashMap<String, LinkedHashMap<String, Uri.Param>> parsedMatrixParams =\n                matrixParams != null && values != null && values.size() > 0 ?\n                        new LinkedHashMap<String, LinkedHashMap<String, Uri.Param>>() : null;\n\n        if (segments != null) {\n            for (final String segment : segments) {\n                final String parsed = parsePart(values, segment);\n\n                // Replace the template segment for the parsed one if necessary\n                if (parsedMatrixParams != null && matrixParams.containsKey(segment)) {\n                    parsedMatrixParams.put(parsed, matrixParams.get(segment));\n                }\n\n                parsedSegments.add(parsed);\n            }\n        }\n\n        final String parsedFrag = parsePart(values, fragment); // this line is executed\n\n        return new UriImpl(scheme, user, password, host, port, parsedSegments,\n                parsedMatrixParams != null ? parsedMatrixParams : matrixParams, queryParams, parsedFrag);\n    }\n```\nThe following condition should be satisfied to trigger the exception:\n```java\nfragment.indexOf("{") > -1 && fragment.indexOf("}", fragment.indexOf("{")) > -1 && values.get(param) == null\n```\nThe 'IllegalArgumentException' can be thrown within the 'parsePart' method by invoking the 'build' method.' parsePart' can be invoked indirectly by the 'build' through a sequence of method calls. Consider the following sequence of method calls:\n```java\n//method-0\nprivate String parsePart(Map<String, ?> templateValues, String segment) {\n        int cursor = segment.indexOf("{");\n        while (cursor > -1) {\n            int closingBracket = segment.indexOf("}", cursor);\n            if (closingBracket > -1) {\n                final String param = segment.substring(cursor + 1, closingBracket);\n                final Object value = templateValues.get(param);\n\n                if (value == null)\n                    throw new IllegalArgumentException("Uri could no be built: The template param '" + param + "' " +  // this line is executed\n                            "could not be resolved.");\n\n                segment = segment.substring(0, cursor) + value.toString() + segment.substring(closingBracket + 1);\n                cursor = segment.indexOf("{", closingBracket + 1);\n            } else {\n                cursor = -1;\n            }\n        }\n\n        return segment;\n    }\n```\nHere is a related non-exceptional test method for reference:\n```java\n@Test\npublic void build_ProperTemplateValuesMap_ShouldBuildSuccessfully() {\n    String expected = "http://user:pwd@localhost:8888/server/1/any;class=2;class=5;class=6" + "/child;group=A;subGroup=A.1;subGroup=A.2?age=12&name=Aa&name=Zz#firstserver";\n    Map<String, Object> params = new HashMap<String, Object>();\n    params.put("a", "server");\n    params.put("b", 1);\n    params.put("c", "any");\n    params.put("d", "first");\n    String uri = UriBuilder.newInstance().scheme("http").user("user").password("pwd").host("localhost").port(8888).path("/{a}/{b}").segment("{c}").matrixParam("class", 2, 5, 6).segment("child").matrixParam("group", "A").matrixParam("subGroup", "A.1", "A.2").queryParam("age", 12).queryParam("name", "Aa", "Zz").fragment("{d}{a}").build(params).toString();\n    assertEquals(expected, uri);\n}\n```\nPlease only give the new exceptional-behavior test method 'build_InsufficientTemplateValuesMap_ShouldThrowIllegalArgumentException' using the '@Test(expected=exception.class)' pattern to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\npackage io.reinert.requestor.core.uri;\n\nimport java.util.HashMap;\nimport java.util.Map;\nimport org.junit.Test;\nimport static org.junit.Assert.assertEquals;\n\npublic class UriBuilderJreTest {\n    \n}\n\n```\n"""
        test_e = """@Test(expected = IllegalArgumentException.class)\npublic void build_InsufficientTemplateValuesMap_ShouldThrowIllegalArgumentException() {\n    Map<String, Object> params = new HashMap<String, Object>();\n    params.put(\"a\", \"server\");\n    params.put(\"b\", 1);\n    params.put(\"c\", \"any\");\n    UriBuilder.newInstance().path(\"{a}/{b}\").segment(\"{c}\").fragment(\"{d}{a}\").build(params);\n}\n"""
        ground_truth = f"""```java\n{test_e}\n```\n"""

        few_shot_example = [
            {
                "role": "system",
                "content": "As a knowledgeable programming assistant and proficient Java developer, your role is to assist a user in creating exceptional-behavior tests for their Java code. The Junit test should use the @Test(expected = exception.class) pattern.",
            },
            {"role": "user", "content": example},
            {"role": "assistant", "content": ground_truth},
        ]
        processed_prompts = []
        for data in tqdm(self.dataset, total=len(self.dataset)):
            mut_name = data.mut_key.split("#")[1]
            mut_class = data.mut_key.split("#")[0].split(".")[-1]
            exception_type = data.etype.split(".")[-1]
            if data.e_stack_trace:
                mut_code = add_called_comment_to_method(
                    data.mut, data.e_stack_trace[0][1]
                )
            condition = data.condition
            if condition and condition != "":
                condition_prompt = f"The following condition should be satisfied to trigger the exception:\n```java\n{condition}\n```\n"
            else:
                condition_prompt = ""
            stack_trace_prompt = LLM_stack_trace_prompt(
                data, data.e_stack_trace
            ).strip()
            etest_context = data.test_context.replace("adhoc_", "").replace(
                "/*TEST PLACEHOLDER*/", ""
            )
            etest_class = data.test_e_key.split("#")[0].split("/")[-1]
            # make prompt
            if not data.test_ne:
                instruct = f"""Please write an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\n{stack_trace_prompt}\nPlease only give the new exceptional-behavior test method using the '@Test(expected=exception.class)' pattern to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the test method:\n```java\n{etest_context}\n```\n"""
            else:
                ne_test = random.choice(data.test_ne)
                instruct = f"""Please write an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\n{stack_trace_prompt}\nHere is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method using the '@Test(expected=exception.class)' pattern to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the test method:\n```java\n{etest_context}\n```\n"""

            new_messages = [
                {"role": "user", "content": instruct},
            ]
            messages = few_shot_example + new_messages
            num_toks = num_tokens_from_messages(messages)
            prompt_data = {"id": data.id, "messages": messages, "num_toks": num_toks}
            processed_prompts.append(prompt_data)
        su.io.dump(self.data_dir / "prompt.jsonl", processed_prompts)

    def create_condition_ne_stack_2e_with_name_one_shot(self):
        example = """Please complete an exceptional behavior test method in Java to test the method 'build' in class 'io.reinert.requestor.core.uri.UriBuilderImpl' for the exception 'IllegalArgumentException'.\nThe method to be tested is defined as:\n```java\n    @Override\n    public Uri build(Map<String, ?> values) {\n        final List<String> parsedSegments = new ArrayList<String>();\n        final LinkedHashMap<String, LinkedHashMap<String, Uri.Param>> parsedMatrixParams =\n                matrixParams != null && values != null && values.size() > 0 ?\n                        new LinkedHashMap<String, LinkedHashMap<String, Uri.Param>>() : null;\n\n        if (segments != null) {\n            for (final String segment : segments) {\n                final String parsed = parsePart(values, segment);\n\n                // Replace the template segment for the parsed one if necessary\n                if (parsedMatrixParams != null && matrixParams.containsKey(segment)) {\n                    parsedMatrixParams.put(parsed, matrixParams.get(segment));\n                }\n\n                parsedSegments.add(parsed);\n            }\n        }\n\n        final String parsedFrag = parsePart(values, fragment); // this line is executed\n\n        return new UriImpl(scheme, user, password, host, port, parsedSegments,\n                parsedMatrixParams != null ? parsedMatrixParams : matrixParams, queryParams, parsedFrag);\n    }\n```\nThe following condition should be satisfied to trigger the exception:\n```java\nfragment.indexOf("{") > -1 && fragment.indexOf("}", fragment.indexOf("{")) > -1 && values.get(param) == null\n```\nThe 'IllegalArgumentException' can be thrown within the 'parsePart' method by invoking the 'build' method.' parsePart' can be invoked indirectly by the 'build' through a sequence of method calls. Consider the following sequence of method calls:\n```java\n//method-0\nprivate String parsePart(Map<String, ?> templateValues, String segment) {\n        int cursor = segment.indexOf("{");\n        while (cursor > -1) {\n            int closingBracket = segment.indexOf("}", cursor);\n            if (closingBracket > -1) {\n                final String param = segment.substring(cursor + 1, closingBracket);\n                final Object value = templateValues.get(param);\n\n                if (value == null)\n                    throw new IllegalArgumentException("Uri could no be built: The template param '" + param + "' " +  // this line is executed\n                            "could not be resolved.");\n\n                segment = segment.substring(0, cursor) + value.toString() + segment.substring(closingBracket + 1);\n                cursor = segment.indexOf("{", closingBracket + 1);\n            } else {\n                cursor = -1;\n            }\n        }\n\n        return segment;\n    }\n```\nHere is a related non-exceptional test method 'build_ProperTemplateValuesMap_ShouldBuildSuccessfully' for reference:\n```java\n@Test\npublic void build_ProperTemplateValuesMap_ShouldBuildSuccessfully() {\n    String expected = "http://user:pwd@localhost:8888/server/1/any;class=2;class=5;class=6" + "/child;group=A;subGroup=A.1;subGroup=A.2?age=12&name=Aa&name=Zz#firstserver";\n    Map<String, Object> params = new HashMap<String, Object>();\n    params.put("a", "server");\n    params.put("b", 1);\n    params.put("c", "any");\n    params.put("d", "first");\n    String uri = UriBuilder.newInstance().scheme("http").user("user").password("pwd").host("localhost").port(8888).path("/{a}/{b}").segment("{c}").matrixParam("class", 2, 5, 6).segment("child").matrixParam("group", "A").matrixParam("subGroup", "A.1", "A.2").queryParam("age", 12).queryParam("name", "Aa", "Zz").fragment("{d}{a}").build(params).toString();\n    assertEquals(expected, uri);\n}\n```\nPlease only give the new exceptional-behavior test method 'build_InsufficientTemplateValuesMap_ShouldThrowIllegalArgumentException' using the '@Test(expected=exception.class)' pattern to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\npackage io.reinert.requestor.core.uri;\n\nimport java.util.HashMap;\nimport java.util.Map;\nimport org.junit.Test;\nimport static org.junit.Assert.assertEquals;\n\npublic class UriBuilderJreTest {\n    \n}\n\n```\n"""
        test_e = """@Test(expected = IllegalArgumentException.class)\npublic void build_InsufficientTemplateValuesMap_ShouldThrowIllegalArgumentException() {\n    Map<String, Object> params = new HashMap<String, Object>();\n    params.put(\"a\", \"server\");\n    params.put(\"b\", 1);\n    params.put(\"c\", \"any\");\n    UriBuilder.newInstance().path(\"{a}/{b}\").segment(\"{c}\").fragment(\"{d}{a}\").build(params);\n}\n"""
        ground_truth = f"""```java\n{test_e}\n```\n"""

        few_shot_example = [
            {
                "role": "system",
                "content": "As a knowledgeable programming assistant and proficient Java developer, your role is to assist a user in creating exceptional-behavior tests for their Java code. The Junit test should use the @Test(expected = exception.class) pattern.",
            },
            {"role": "user", "content": example},
            {"role": "assistant", "content": ground_truth},
        ]
        processed_prompts = []
        for data in tqdm(self.dataset, total=len(self.dataset)):
            mut_name = data.mut_key.split("#")[1]
            mut_class = data.mut_key.split("#")[0].split(".")[-1]
            exception_type = data.etype.split(".")[-1]
            etest_name = data.test_e_key.split("#")[1]
            if data.e_stack_trace:
                mut_code = add_called_comment_to_method(
                    data.mut, data.e_stack_trace[0][1]
                )
            condition = data.condition
            if condition and condition != "":
                condition_prompt = f"The following condition should be satisfied to trigger the exception:\n```java\n{condition}\n```\n"
            else:
                condition_prompt = ""
            stack_trace_prompt = LLM_stack_trace_prompt(
                data, data.e_stack_trace
            ).strip()
            etest_context = data.test_context.replace("adhoc_", "").replace(
                "/*TEST PLACEHOLDER*/", ""
            )
            etest_class = data.test_e_key.split("#")[0].split("/")[-1]
            # make prompt
            if not data.test_ne:
                instruct = f"""Please write an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\n{stack_trace_prompt}\nPlease only give the new exceptional-behavior test method '{etest_name}' using the '@Test(expected=exception.class)' pattern to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the test method:\n```java\n{etest_context}\n```\n"""
            else:
                ne_test = random.choice(data.test_ne)
                instruct = f"""Please write an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{condition_prompt}\n{stack_trace_prompt}\nHere is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method '{etest_name}' using the '@Test(expected=exception.class)' pattern to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the test method:\n```java\n{etest_context}\n```\n"""
            new_messages = [
                {"role": "user", "content": instruct},
            ]
            messages = few_shot_example + new_messages
            num_toks = num_tokens_from_messages(messages)
            prompt_data = {"id": data.id, "messages": messages, "num_toks": num_toks}
            processed_prompts.append(prompt_data)
        su.io.dump(self.data_dir / "prompt.jsonl", processed_prompts)

    def create_mut2e_few_shot_prompt(self):
        dataset = load_dataset(
            Macros.work_dir / self.config["data_file"],
            clz=parse_data_cls(self.config["data_type"]),
        )
        processed_prompts = []
        for dt in tqdm(dataset, total=len(dataset)):
            processed_prompts.append(self.create_mut2e_one_shot_prompt_from_data(dt))
        #
        su.io.dump(self.data_dir / "prompt.jsonl", processed_prompts)

    def create_nestack2e_few_shot_prompt(self):
        processed_prompts = []
        for dt in tqdm(self.dataset, total=len(self.dataset)):
            processed_prompts.append(
                self.create_nestack2e_one_shot_prompt_from_data(dt)
            )
        #
        su.io.dump(self.data_dir / "prompt.jsonl", processed_prompts)

    def create_mut2e_one_shot_prompt_from_data(self, data: DataMUT2E):
        few_shot_messages = self.few_shot_mut2e_example()
        mut_name = data.mut_key.split("#")[1]
        mut_class = data.mut_key.split("#")[0].split(".")[-1]
        exception_type = data.etype.split(".")[-1]
        mut_code = data.mut
        etest_context = data.test_context.replace("adhoc_", "").replace(
            "/*TEST PLACEHOLDER*/", ""
        )
        etest_class = data.test_e_key.split("#")[0].split("/")[-1]
        # make prompt
        if not data.test_ne:
            instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
        else:
            ne_test = random.choice(data.test_ne)
            instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nHere is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""

        new_messages = [
            {"role": "user", "content": instruct},
        ]
        messages = few_shot_messages + new_messages
        num_toks = num_tokens_from_messages(messages)
        prompt_data = {"messages": messages, "num_toks": num_toks}
        return prompt_data

    def create_nestack2e_one_shot_prompt_from_real_data(
        self, selected_ids: List = None
    ):
        """
        Sample 5 stack traces from all possible stack traces
        """
        processed_prompts = []
        # few shot messages
        few_shot_example = self.few_shot_nestack2e_example()
        for data in tqdm(self.dataset, total=len(self.dataset)):
            if selected_ids and data.id not in selected_ids:
                continue
            mut_name = data.mut_key.split("#")[1]
            mut_class = data.mut_key.split("#")[0].split(".")[-1]
            exception_type = data.etype.split(".")[-1]
            mut_code = data.mut
            etest_name = data.test_e_key.split("#")[1]
            etest_context = data.test_context.replace("adhoc_", "").replace(
                "/*TEST PLACEHOLDER*/", ""
            )
            etest_class = data.test_e_key.split("#")[0].split("/")[-1]
            stack_trace_prompts = set()
            for stack_trace in data.e_stack_trace:
                if len(stack_trace) > 6:
                    stack_trace = remove_duplicate_stack_trace(stack_trace)
                if len(stack_trace) > 6:
                    continue
                mut_code = add_called_comment_to_method(
                    data.mut, line_num=stack_trace[0][1]
                )
                stack_trace_prompt = LLM_stack_trace_prompt(data, stack_trace)
                stack_trace_prompts.add(stack_trace_prompt)
            # sample at most 5 stack_traces
            sample_stack_trace_prompts = random.sample(
                list(stack_trace_prompts), min(5, len(stack_trace_prompts))
            )
            if len(sample_stack_trace_prompts) == 0:
                sample_stack_trace_prompts = [""]
            for stack_trace_prompt in sample_stack_trace_prompts:
                # make prompt
                if not data.test_ne:
                    if self.config["with_name"]:
                        instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{stack_trace_prompt}Please only give the new exceptional-behavior test method '{etest_name}' to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                    else:
                        instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{stack_trace_prompt}Please only give the new exceptional-behavior test method to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                else:
                    ne_test = random.choice(data.test_ne)
                    if self.config["with_name"]:
                        instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{stack_trace_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method '{etest_name}' to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                    else:
                        instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{stack_trace_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                new_messages = [
                    {"role": "user", "content": instruct},
                ]
                messages = few_shot_example + new_messages
                num_toks = num_tokens_from_messages(messages)
                prompt_data = {
                    "id": data.id,
                    "messages": messages,
                    "num_toks": num_toks,
                }
                processed_prompts.append(prompt_data)
        su.io.dump(self.data_dir / "prompt.jsonl", processed_prompts)

    def create_nestack2e_one_shot_prompt_from_gt_data(self, selected_ids: List = None):
        """
        Use ground truth stack trace to prompt
        """
        processed_prompts = []
        # few shot messages
        few_shot_example = self.few_shot_nestack2e_example()
        for data in tqdm(self.dataset, total=len(self.dataset)):
            if selected_ids and data.id not in selected_ids:
                continue
            mut_name = data.mut_key.split("#")[1]
            mut_class = data.mut_key.split("#")[0].split(".")[-1]
            exception_type = data.etype.split(".")[-1]
            mut_code = data.mut
            etest_name = data.test_e_key.split("#")[1]
            etest_context = data.test_context.replace("adhoc_", "").replace(
                "/*TEST PLACEHOLDER*/", ""
            )
            etest_class = data.test_e_key.split("#")[0].split("/")[-1]

            stack_trace = data.e_stack_trace
            mut_code = add_called_comment_to_method(
                data.mut, line_num=stack_trace[0][1]
            )
            stack_trace_prompt = LLM_stack_trace_prompt(data, stack_trace)
            # make prompt
            if not data.test_ne:
                if self.config["with_name"]:
                    instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{stack_trace_prompt}Please only give the new exceptional-behavior test method '{etest_name}' to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                else:
                    instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{stack_trace_prompt}Please only give the new exceptional-behavior test method to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
            else:
                ne_test = random.choice(data.test_ne)
                if self.config["with_name"]:
                    instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{stack_trace_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method '{etest_name}' to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
                else:
                    instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{stack_trace_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
            new_messages = [
                {"role": "user", "content": instruct},
            ]
            messages = few_shot_example + new_messages
            num_toks = num_tokens_from_messages(messages)
            prompt_data = {
                "id": data.id,
                "messages": messages,
                "num_toks": num_toks,
            }
            processed_prompts.append(prompt_data)
        su.io.dump(self.data_dir / "prompt.jsonl", processed_prompts)

    def create_cot_prompt_from_data(self, data: DataNE2E):
        few_shot_messages = self.cot_few_shot_nestack2e_example()
        mut_name = data.mut_key.split("#")[1]
        mut_class = data.mut_key.split("#")[0].split(".")[-1]
        exception_type = data.etype.split(".")[-1]

        if data.e_stack_trace:
            mut_code = add_called_line_to_method(data.mut, data.e_stack_trace[0][1])
        elif len(data.call_stacks) > 0:
            # pick the shortest stack trace for eval
            shortest_trace_len = float("inf")
            for trace in data.call_stacks:
                if len(trace) < shortest_trace_len:
                    shortest_trace_len = len(trace)
                    stack_trace = trace
            #
            if len(stack_trace) > 1:
                mut_code = add_called_line_to_method(data.mut, stack_trace[0][1])
            else:
                mut_code = data.mut
            data.call_stacks = stack_trace
        else:
            mut_code = data.mut
        etest_name = data.test_e_key.split("#")[1]
        etest_context = data.test_context.replace("adhoc_", "").replace(
            "/*TEST PLACEHOLDER*/", ""
        )
        etest_class = data.test_e_key.split("#")[0].split("/")[-1]
        call_stack_prompt = stack_trace_prompt(
            data, ignore_num=self.config["stack_trace_max_length"]
        )

        # make prompt
        if not data.test_ne:
            instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{call_stack_prompt}Please only give the new exceptional-behavior test method to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
        else:
            # ne_test = find_the_most_revelant_netest(nestack2e_dt)[1]
            ne_test = random.choice(data.test_ne)
            instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{call_stack_prompt}Here is a related non-exceptional test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""
        new_messages = [
            {"role": "user", "content": instruct},
        ]
        messages = few_shot_messages + new_messages
        num_toks = num_tokens_from_messages(messages)
        prompt_data = {"messages": messages, "num_toks": num_toks}
        return prompt_data

    def select_shortest_stack_trace(self, dt: DataNE2E):
        """Return the shortest stack trace"""
        stack_traces = dt.stack_traces
        shortest_stack_trace = stack_traces[0]
        for stack_trace in stack_traces:
            if len(stack_trace) < len(shortest_stack_trace):
                shortest_stack_trace = stack_trace
        dt.call_stacks = shortest_stack_trace
        return dt

    def few_shot_mut2e_example(self):
        example = """Please complete an exceptional behaviour test method in Java to test the method 'create' in class 'ProtocolSwitch' for the exception 'IllegalArgumentException'.\nThe method to be tested is defined as:\n```java\npublic static ProtocolSwitch create(int value) {\n    ProtocolSwitch status = new ProtocolSwitch();\n    status.setBs(toBitSet(value));\n    return status;\n}\n```\nPlease only give the new exceptional-behavior test method to compete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\npackage com.alipay.remoting.inner.utiltest;\nimport java.util.BitSet;\nimport org.junit.After;\nimport org.junit.AfterClass;\nimport org.junit.Assert;\nimport org.junit.Before;\nimport org.junit.BeforeClass;\nimport org.junit.Test;\nimport com.alipay.remoting.config.switches.ProtocolSwitch;\npublic class ProtocolSwitchTest {\n    @BeforeClass\n    public static void initClass() {\n    }\n    @Before\n    public void init() {\n    }\n    @After\n    public void stop() {\n    }\n    @AfterClass\n    public static void afterClass() {\n    }\n}\n```\n"""
        # thought_example = """##STEP 1: Analyze the conditions to triger exception:\nAccording to the provided method calls, the exception 'IllegalArgumentException' will be triggered if the condition: value < 0 || value > Byte.MAX_VALUE is satisified.\n"""
        test_e = """@Test\npublic void test_createUsingByte() {\n    Assert.assertFalse(ProtocolSwitch.create(0).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(1).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(2).isOn(1));\n    Assert.assertTrue(ProtocolSwitch.create(4).isOn(2));\n    Assert.assertTrue(ProtocolSwitch.create(8).isOn(3));\n    Assert.assertTrue(ProtocolSwitch.create(3).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(3).isOn(1));\n    Assert.assertFalse(ProtocolSwitch.create(3).isOn(2));\n    Assert.assertTrue(ProtocolSwitch.create(64).isOn(6));\n    Assert.assertFalse(ProtocolSwitch.create(64).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(127).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(127).isOn(1));\n    Assert.assertTrue(ProtocolSwitch.create(127).isOn(6));\n    Assert.assertFalse(ProtocolSwitch.create(127).isOn(7));\n    try {\n        ProtocolSwitch.create(255);\n        Assert.fail(\"Should not reach here!\");\n    } catch (IllegalArgumentException e) {\n    }\n}"""
        ground_truth = f"""```java\n{test_e}\n```\n"""

        messages = [
            {
                "role": "system",
                "content": "As a knowledgeable programming assistant and proficient Java developer, your role is to assist a user in creating exceptional-behavior tests for their Java code.",
            },
            {"role": "user", "content": example},
            {"role": "assistant", "content": ground_truth},
        ]
        return messages

    def few_shot_nestack2e_example(self):
        """
        create few-shot example without cot.
        """
        example = """Please complete an exceptional behaviour test method in Java to test the method 'create' in class 'ProtocolSwitch' for the exception 'IllegalArgumentException'.\nThe method to be tested is defined as:\n```java\npublic static ProtocolSwitch create(int value) {\n    ProtocolSwitch status = new ProtocolSwitch();\n    status.setBs(toBitSet(value));  // this line is called\n    return status;\n}\n```\nThe 'IllegalArgumentException' can be thrown within the 'toBitSet' method by invoking the 'create' method. Consider the following sequence of method calls:\n```java\n//method-0\npublic static BitSet toBitSet(int value) {\n    if (value < 0 || value > Byte.MAX_VALUE) {\n        throw new IllegalArgumentException("The value " + value + " is out of byte range, should be limited between [0] to [" + Byte.MAX_VALUE + "]");    // this line is called\n    }\n    BitSet bs = new BitSet();\n    int index = 0;\n    while (value != 0) {\n        if (value % 2 != 0) {\n            bs.set(index);\n        }\n        ++index;\n        value = (byte) (value >> 1);\n    }\n    return bs;\n}\n```\nHere is a related test method for reference:\n```java\n@Test\npublic void test_createUsingIndex() {\n    for (int i = 0; i < 7; ++i) {\n        Assert.assertTrue(ProtocolSwitch.create(new int[] { i }).isOn(i));\n    }\n    int size = 7;\n    int[] a = new int[size];\n    for (int i = 0; i < size; ++i) {\n        a[i] = i;\n    }\n    ProtocolSwitch status = ProtocolSwitch.create(a);\n    for (int i = 0; i < size; ++i) {\n        Assert.assertTrue(status.isOn(i));\n    }\n}\n```\nPlease only give the new exceptional-behavior test method to compete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\npackage com.alipay.remoting.inner.utiltest;\nimport java.util.BitSet;\nimport org.junit.After;\nimport org.junit.AfterClass;\nimport org.junit.Assert;\nimport org.junit.Before;\nimport org.junit.BeforeClass;\nimport org.junit.Test;\nimport com.alipay.remoting.config.switches.ProtocolSwitch;\npublic class ProtocolSwitchTest {\n    @BeforeClass\n    public static void initClass() {\n    }\n    @Before\n    public void init() {\n    }\n    @After\n    public void stop() {\n    }\n    @AfterClass\n    public static void afterClass() {\n    }\n}\n```\n"""
        # thought_example = """##STEP 1: Analyze the conditions to triger exception:\nAccording to the provided method calls, the exception 'IllegalArgumentException' will be triggered if the condition: value < 0 || value > Byte.MAX_VALUE is satisified.\n"""
        test_e = """@Test\npublic void test_createUsingByte() {\n    Assert.assertFalse(ProtocolSwitch.create(0).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(1).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(2).isOn(1));\n    Assert.assertTrue(ProtocolSwitch.create(4).isOn(2));\n    Assert.assertTrue(ProtocolSwitch.create(8).isOn(3));\n    Assert.assertTrue(ProtocolSwitch.create(3).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(3).isOn(1));\n    Assert.assertFalse(ProtocolSwitch.create(3).isOn(2));\n    Assert.assertTrue(ProtocolSwitch.create(64).isOn(6));\n    Assert.assertFalse(ProtocolSwitch.create(64).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(127).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(127).isOn(1));\n    Assert.assertTrue(ProtocolSwitch.create(127).isOn(6));\n    Assert.assertFalse(ProtocolSwitch.create(127).isOn(7));\n    try {\n        ProtocolSwitch.create(255);\n        Assert.fail(\"Should not reach here!\");\n    } catch (IllegalArgumentException e) {\n    }\n}"""
        ground_truth = f"""```java\n{test_e}\n```\n"""

        messages = [
            {
                "role": "system",
                "content": "As a knowledgeable programming assistant and proficient Java developer, your role is to assist a user in creating exceptional-behavior tests for their Java code.",
            },
            {"role": "user", "content": example},
            {"role": "assistant", "content": ground_truth},
        ]
        return messages

    def cot_few_shot_example(self):
        """
        Create cot few-shot prompt for openai models.
        """
        cot_example = """Please complete an exceptional behaviour test method in Java to test the method 'create' in class 'ProtocolSwitch' for the exception 'IllegalArgumentException'.\nThe method to be tested is defined as:\n```java\npublic static ProtocolSwitch create(int value) {\n    ProtocolSwitch status = new ProtocolSwitch();\n>>>    status.setBs(toBitSet(value));\n    return status;\n}\n```\nThe 'IllegalArgumentException' can be thrown within the 'toBitSet' method by invoking the 'create' method. Consider the following sequence of method calls:\n```java\n//method-0\npublic static BitSet toBitSet(int value) {\n    if (value < 0 || value > Byte.MAX_VALUE) {\n>>>        throw new IllegalArgumentException("The value " + value + " is out of byte range, should be limited between [0] to [" + Byte.MAX_VALUE + "]");\n    }\n    BitSet bs = new BitSet();\n    int index = 0;\n    while (value != 0) {\n        if (value % 2 != 0) {\n            bs.set(index);\n        }\n        ++index;\n        value = (byte) (value >> 1);\n    }\n    return bs;\n}\n```\nHere is a related test method for reference:\n```java\n@Test\npublic void test_createUsingIndex() {\n    for (int i = 0; i < 7; ++i) {\n        Assert.assertTrue(ProtocolSwitch.create(new int[] { i }).isOn(i));\n    }\n    int size = 7;\n    int[] a = new int[size];\n    for (int i = 0; i < size; ++i) {\n        a[i] = i;\n    }\n    ProtocolSwitch status = ProtocolSwitch.create(a);\n    for (int i = 0; i < size; ++i) {\n        Assert.assertTrue(status.isOn(i));\n    }\n}\n```\nPlease only give the new exceptional-behavior test method to compete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\npackage com.alipay.remoting.inner.utiltest;\nimport java.util.BitSet;\nimport org.junit.After;\nimport org.junit.AfterClass;\nimport org.junit.Assert;\nimport org.junit.Before;\nimport org.junit.BeforeClass;\nimport org.junit.Test;\nimport com.alipay.remoting.config.switches.ProtocolSwitch;\npublic class ProtocolSwitchTest {\n    @BeforeClass\n    public static void initClass() {\n    }\n    @Before\n    public void init() {\n    }\n    @After\n    public void stop() {\n    }\n    @AfterClass\n    public static void afterClass() {\n    }\n}\n```\n"""
        thought_example = """##STEP 1: Analyze the conditions to triger exception:\nAccording to the provided method calls, the exception 'IllegalArgumentException' will be triggered if the condition: value < 0 || value > Byte.MAX_VALUE is satisified.\n"""
        test_e = """@Test\npublic void test_createUsingByte() {\n    Assert.assertFalse(ProtocolSwitch.create(0).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(1).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(2).isOn(1));\n    Assert.assertTrue(ProtocolSwitch.create(4).isOn(2));\n    Assert.assertTrue(ProtocolSwitch.create(8).isOn(3));\n    Assert.assertTrue(ProtocolSwitch.create(3).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(3).isOn(1));\n    Assert.assertFalse(ProtocolSwitch.create(3).isOn(2));\n    Assert.assertTrue(ProtocolSwitch.create(64).isOn(6));\n    Assert.assertFalse(ProtocolSwitch.create(64).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(127).isOn(0));\n    Assert.assertTrue(ProtocolSwitch.create(127).isOn(1));\n    Assert.assertTrue(ProtocolSwitch.create(127).isOn(6));\n    Assert.assertFalse(ProtocolSwitch.create(127).isOn(7));\n    try {\n        ProtocolSwitch.create(255);\n        Assert.fail(\"Should not reach here!\");\n    } catch (IllegalArgumentException e) {\n    }\n}"""
        ground_truth = f"""##STEP2: Generate test:\n```java\n{test_e}\n```\n"""

        model_output = f"""{thought_example}\n{ground_truth}"""
        messages = [
            {
                "role": "system",
                "content": "As a knowledgeable programming assistant and proficient Java developer, your role is to assist a user in creating exceptional-behavior tests for their Java code. Follow these steps to guide the user through the process:\n\nStep 1: Analyze the conditions to trigger the exception\n\nBegin by carefully analyzing the conditions that should be met to trigger the exception in the Java code. Provide detailed insights into the scenarios and inputs that will lead to the desired exceptional behavior. Use the prefix '##STEP 1: Analyze the conditions to trigger exception:' for this analysis.\n\nStep 2: Generate test code\n\nSubsequently, leverage your expertise to generate comprehensive test code that aligns with the identified conditions. Deliver the generated test code under the specified test file using the prefix '##STEP 2: Generate test:'",
            },
            {"role": "user", "content": cot_example},
            {"role": "assistant", "content": model_output},
        ]
        return messages

    def create_mut2e_prompt(self):
        """
        Create mut2e zero-shot prompt for openai models.
        """

        dataset = load_dataset(
            Macros.work_dir / self.config["data_file"],
            clz=parse_data_cls(self.config["data_type"]),
        )
        processed_prompts = []
        for dt in tqdm(dataset, total=len(dataset)):
            processed_prompts.append(self.process_gpt_mut2e_prompt(dt))
        #
        su.io.dump(self.data_dir / "prompt.jsonl", processed_prompts)

    def create_nestack2e_prompt(self):
        dataset = load_dataset(
            Macros.work_dir / self.config["data_file"],
            clz=parse_data_cls(self.config["data_type"]),
        )
        processed_prompts = []
        for dt in tqdm(dataset, total=len(dataset)):
            processed_prompts.append(self.process_gpt_nestack2e_prompt(dt))
        #
        su.io.dump(self.data_dir / "prompt.jsonl", processed_prompts)

    def create_reasons_prompt(self):
        dataset = load_dataset(
            Macros.work_dir / self.config["data_file"],
            clz=parse_data_cls(self.config["data_type"]),
        )
        processed_prompts = []
        for dt in tqdm(dataset, total=len(dataset)):
            processed_prompts.append(self.process_gpt_reason_prompt(dt))
        #
        su.io.dump(self.data_dir / "prompt.jsonl", processed_prompts)

    def process_gpt_mut2e_prompt(self, data: DataMUT2E) -> dict:
        system_content = "You are a helpful programming assistant and an expert Java programmer. You are helping a user writing exceptional-behavior tests for their Java code."

        mut_name = data.mut_key.split("#")[1]
        mut_class = data.mut_key.split("#")[0].split(".")[-1]
        exception_type = data.etype.split(".")[-1]
        mut_code = data.mut

        etest_name = data.test_e_key.split("#")[1]
        etest_context = data.test_context.replace("adhoc_", "").replace(
            "/*TEST PLACEHOLDER*/", ""
        )
        etest_class = data.test_e_key.split("#")[0].split("/")[-1]

        instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\nPlease only give the new exceptional-behavior test method '{etest_name}' to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": instruct},
        ]
        num_toks = num_tokens_from_messages(messages)
        if num_toks > 16000:
            print(f"THE PROMPT IS TOO LONG for data-{data.id}!!")
        prompt_data = {"messages": messages, "num_toks": num_toks}
        return prompt_data

    def process_gpt_stack2e_prompt(self, data: DataMUT2E) -> dict:
        system_content = "You are a helpful programming assistant and an expert Java programmer. You are helping a user writing exceptional-behavior tests for their Java code."

        mut_name = data.mut_key.split("#")[1]
        mut_class = data.mut_key.split("#")[0].split(".")[-1]
        exception_type = data.etype.split(".")[-1]
        if data.e_stack_trace:
            mut_code = add_called_line_to_method(data.mut, data.e_stack_trace[0][1])
        elif len(data.call_stacks) > 0:
            shortest_trace_len = float("inf")
            for trace in data.call_stacks:
                if len(trace) < shortest_trace_len:
                    shortest_trace_len = len(trace)
                    stack_trace = trace
            #
            if len(stack_trace) > 1:
                mut_code = add_called_line_to_method(data.mut, stack_trace[0][1])
            else:
                mut_code = data.mut
            data.call_stacks = stack_trace
        else:
            mut_code = data.mut
        etest_name = data.test_e_key.split("#")[1]
        etest_context = data.test_context.replace("adhoc_", "").replace(
            "/*TEST PLACEHOLDER*/", ""
        )
        etest_class = data.test_e_key.split("#")[0].split("/")[-1]
        call_stack_prompt = stack_trace_prompt(data)

        instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{call_stack_prompt}Please only give the new exceptional-behavior test method '{etest_name}' to complete the following test class '{etest_class}'. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{etest_context}\n```\n"""

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": instruct},
        ]
        num_toks = num_tokens_from_messages(messages)
        if num_toks > 5000:
            print(data.project, num_toks)
        prompt_data = {"messages": messages, "num_toks": num_toks}
        return prompt_data

    def process_gpt_cot_prompt(self, data: TData) -> dict:
        # provide one shot as example
        system_content = "You are a helpful programming assistant and an expert Java programmer. You are helping a user writing exceptional-behavior tests for their Java code."
        test_context = data.test_context.replace("adhoc_", "")
        setup = f"""Please complete an exceptional behaviour test method in Java to test the method '{get_context(data, 'mut_name')}' in class '{get_context(data, 'mut_class')}' for the exception '{get_context(data, 'exception')}'.\nThe method to be tested is defined as:\n```java\n{get_context(data, 'mut')}\n```\nPlease only give the new exceptional-behavior test method '{get_context(data, 'etest_name')}' which conforms to the @Test(expected = SomeException.class) pattern to compete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n"""

        if test_context:
            test_context += f"""```java\n{test_context}\n```\n"""
        user_content = setup + test_context
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        num_toks = num_tokens_from_messages(messages)
        prompt_data = {"messages": messages, "num_toks": num_toks}
        return prompt_data

    def process_gpt_nestack2e_prompt(self, data: DataNE2E):
        system_content = "You are a helpful programming assistant and an expert Java programmer. You are helping a user writing exceptional-behavior tests for their Java code."
        mut_name = data.mut_key.split("#")[1]
        mut_class = data.mut_key.split("#")[0].split("/")[-1]
        exception_type = data.etype
        mut_code = data.mut
        call_stack_prompt = raw_stack_trace_prompt(data)
        _, ne_test, ne_test_context = find_the_most_revelant_netest(data)
        # process ne_test_context
        ne_test_context = ne_test_context.replace("adhoc_", "").replace(
            "/*TEST PLACEHOLDER*/", ""
        )
        instruct = f"""Please complete an exceptional behavior test method in Java to test the method '{mut_name}' in class '{mut_class}' for the exception '{exception_type}'.\nThe method to be tested is defined as:\n```java\n{mut_code}\n```\n{call_stack_prompt}Here is a related test method for reference:\n```java\n{ne_test}\n```\nPlease only give the new exceptional-behavior test method to complete the following test class. Do NOT use extra libraries or define new helper methods. Return **only** the code in the completion:\n```java\n{ne_test_context}\n```\n"""
        user_content = instruct
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        num_toks = num_tokens_from_messages(messages)
        prompt_data = {"messages": messages, "num_toks": num_toks}
        return prompt_data

    def process_gpt_reason_prompt(self, data: DataMUT2E) -> dict:
        """
        ChatGPT-based model's zero-shot prompt
        """

        # prepare component
        system_content = "You are a helpful programming assistant and an expert Java programmer. You are helping a user writing exceptional-behavior tests for their Java code. "
        mut_name = data.mut_key.split("#")[1]
        etest_name = data.test_e_key.split("#")[1]
        call_stack_prompt = format_stack_trace_prompt(data)
        instruction = f"Please provide concise and general instructions on steps to trigger the '{data.etype}' by calling method '{mut_name}' to help generate the unit test named '{etest_name}'."

        user_content = f"{call_stack_prompt}\n{instruction}"
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        num_toks = num_tokens_from_messages(messages)
        prompt_data = {"messages": messages, "num_toks": num_toks}
        return prompt_data


if __name__ == "__main__":
    su.log.setup(Macros.log_file)
    CLI(PromptProcessor, as_positional=False)
