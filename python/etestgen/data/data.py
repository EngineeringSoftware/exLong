import dataclasses
from typing import List, Type, Optional, Union, Dict, Tuple, Any
import functools
import seutil as su

from etestgen.data.structures import AST, simplify_type_name, MethodStructure
from etestgen.eval.compute_throws_coverage import TestMethod

BASIC_FIELDS = [
    "mut",
    "etype",
    "test_name",
    "test_body",
    "proj_name",
    "test_cid",
    "test_mid",
    "test_mkey",
    "mut_cid",
    "mut_mid",
    "mut_mkey",
]


@dataclasses.dataclass
class DataMUT2E:
    id: str = None
    fully_deserialized: bool = True
    mut_toks: List[str] = None
    mut: str = None
    em: List[str] = None  # method that throws the exception
    em_context: List[str] = None  # the lines around throw statement
    e_stack_trace: List[str] = None
    etype: str = None
    test_e: str = None
    test_e_toks: List[str] = None
    # test stmts (ast)
    test_stmts: List[AST] = None
    test_sign: AST = None
    test_context: str = None
    etest_pattern: str = None
    test_method: TestMethod = None
    # call traces which consists of possible method (id) calls to lead to exception
    call_stacks: List[Any] = None

    @functools.cached_property
    def test_sign_toks(self) -> List[str]:
        return self.test_sign.get_tokens()

    @functools.cached_property
    def test_sign_code(self) -> str:
        test_sign_string = " ".join(self.test_sign_toks)
        if "@ Test (" in test_sign_string:
            test_sign_string = (
                test_sign_string.replace("@ Test (", "@Test(")
                .replace("public", "\npublic")
                .replace("( )", "()")
            )
        else:
            test_sign_string = test_sign_string.replace("@ Test", "@Test\n").replace(
                "( )", "()"
            )
        return test_sign_string

    @functools.cached_property
    def test_stmt_toks(self) -> List[List[str]]:
        return [stmt.get_tokens() for stmt in self.test_stmts]

    @functools.cached_property
    def test_stmt_code(self) -> List[str]:
        return [stmt.get_code() for stmt in self.test_stmts]

    project: str = None
    module: str = None
    module_i: int = None
    mut_key: str = None
    em_key: str = None
    test_e_key: str = None
    condition: str = None

    # more fields
    # local variables' types before each statement + after the last statement
    types_local: List[List[str]] = None

    # list of fields that are set / not set in focal class
    fields_set: List[List[str]] = None
    fields_notset: List[List[str]] = None

    # class constructor
    constructors: List[str] = dataclasses.field(default_factory=list)

    # setup and teardown methods (asts, mids, names)
    setup_methods: List[AST] = None
    setup_mids: List[int] = None
    teardown_methods: List[AST] = None
    teardown_mids: List[int] = None

    # the last called (APP/TEST) method in previous statements
    # "None" either means that no method was called in the particular previous statement (i.e., should look into more prior statements OR no method has been called yet)
    last_called_methods: List[Optional[AST]] = None

    # for each statement, the statement from APP code with the most similar prior context
    similar_stmts: List[Optional[Tuple[float, AST]]] = None

    def resolve_last_called_method(self, stmt_i: int) -> Optional[AST]:
        """
        Resolve the last called method when predicting statement #stmt_i,
        considering the fact that "None" means no method call at last statement only.
        """
        while self.last_called_methods[stmt_i] is None:
            stmt_i -= 1
            if stmt_i < 0:
                return None
        return self.last_called_methods[stmt_i]

    @functools.cached_property
    def types_local_simplified(self) -> List[List[str]]:
        # types_local normalized to simple type names
        if self.types_local is None:
            return None
        else:
            return [
                [simplify_type_name(t) for t in types] for types in self.types_local
            ]

    # types that are needed for invoking the focal method and are not available yet
    types_absent: List[List[str]] = None

    @functools.cached_property
    def types_absent_simplified(self) -> List[List[str]]:
        # types_absent normalized to simple type names
        if self.types_absent is None:
            return None
        else:
            return [
                [simplify_type_name(t) for t in types] for types in self.types_absent
            ]

    # the runtime types and values of variables of interest at each statement, saved as delta (only changed variables)
    runtime_types_values_delta: List[Dict[str, Tuple[str, str]]] = None

    # the runtime types and values of variables of interest at each statement (all variables)
    @functools.cached_property
    def runtime_types_values_all(self) -> List[Dict[str, Tuple[str, str]]]:
        if self.runtime_types_values_delta is None:
            return None
        else:
            ret = []
            all_types_values = {}
            for delta_types_values in self.runtime_types_values_delta:
                all_types_values.update(delta_types_values)
                ret.append(all_types_values.copy())
            return ret

    def runtime_data_valid(self) -> bool:
        if (
            self.runtime_types_values_delta is None
            or len(self.runtime_types_values_delta) != len(self.test_stmts) + 1
        ):
            return False
        return True

    def cutoff(self, stmt_i: int):
        """
        Cut off this data to only the initial few statements.
        """
        self.test_stmts = self.test_stmts[:stmt_i]

        if self.types_local is not None:
            self.types_local = self.types_local[: stmt_i + 1]
        if self.types_absent is not None:
            self.types_absent = self.types_absent[: stmt_i + 1]
        if self.fields_set is not None:
            self.fields_set = self.fields_set[: stmt_i + 1]
        if self.fields_notset is not None:
            self.fields_notset = self.fields_notset[: stmt_i + 1]
        if self.last_called_methods is not None:
            self.last_called_methods = self.last_called_methods[: stmt_i + 1]
        if self.similar_stmts is not None:
            self.similar_stmts = self.similar_stmts[: stmt_i + 1]
        if self.runtime_types_values_delta is not None:
            self.runtime_types_values_delta = self.runtime_types_values_delta[
                : stmt_i + 1
            ]

    def finish_deserialization(self):
        if self.fully_deserialized:
            return

        fields: Dict[str, dataclasses.Field] = {
            f.name: f for f in dataclasses.fields(TData)
        }
        # del fields["id"]  TODD: should we del id?
        del fields["fully_deserialized"]

        for name, field in fields.items():
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, su.io.deserialize(v, clz=field.type))

        self.fully_deserialized = True


@dataclasses.dataclass
class DataNE2E(DataMUT2E):
    id: str = None
    project: str = None
    module: str = None
    module_i: int = None
    stack_traces: List[List[MethodStructure]] = None

    # mut2e
    mut_key: str = None
    mut_toks: List[str] = None
    mut: str = None
    constructors: List[str] = None

    # etest
    test_e_key: str = None
    e_stack_trace: List[str] = None
    test_e: str = None
    etype: str = None
    etest_context: str = None
    etest_sign: str = None
    etest_stmts: List[AST] = None

    # netest(s)
    netest_methods: List[TestMethod] = dataclasses.field(default_factory=list)
    test_ne_key: List[str] = dataclasses.field(default_factory=list)
    test_ne: List[str] = dataclasses.field(default_factory=list)
    netest_context: List[str] = dataclasses.field(default_factory=list)
    netest_sign: List[str] = dataclasses.field(default_factory=list)


# all possible data types
TData = Union[DataMUT2E, DataNE2E]


def parse_data_cls(data_cls: str) -> Type[TData]:
    if data_cls == "MUT2E":
        return DataMUT2E
    elif data_cls == "NE2E":
        return DataNE2E
    else:
        return DataNE2E
