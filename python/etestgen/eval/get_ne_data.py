import os
import shutil
import seutil as su
from collections import defaultdict
from jsonargparse import CLI
from typing import List, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm
from seutil.maven import MavenModule, MavenProject
from seutil.project import Project

from etestgen.data.tool import Tool
from etestgen.macros import Macros
from etestgen.data.utils import load_dataset, save_dataset
from etestgen.data.data import DataNE2E, parse_data_cls
from etestgen.eval.extract_etest_data_from_coverage import collect_module_source_code
from etestgen.eval.compute_throws_coverage import TestMethod
from etestgen.collector.DataCollector import extract_module_test_coverage

logger = su.log.get_logger(__name__)


class NEData:

    def __init__(self):
        # used by get_non_cover_ne2e_data
        self._test_class_2_context = {}
        self._test_class_2_methods = defaultdict(list)
        self._project_2_test_classes = defaultdict(set)
        self._current_test_class_name = ""
        self._mut2test = None
        self._test_methods = None

    def _clear(self):
        self._test_class_2_context = {}
        self._test_class_2_methods = defaultdict(list)
        self._project_2_test_classes = defaultdict(set)
        self._current_test_class_name = ""
        self._mut2test = None
        self._test_methods = None

    def _skip_module(self, maven_module: Any, module_i: int):
        skipped = None
        if maven_module.packaging == "pom":
            # skip parent/aggregator modules
            skipped = (module_i, maven_module.coordinate, "package==pom")
            return skipped
        if (not Path(maven_module.main_srcpath).exists()
                or not Path(maven_module.test_srcpath).exists()):
            # skip non-standard modules
            skipped = (module_i, maven_module.coordinate, "missing src")
            return skipped
        return skipped

    def get_pub_with_throw(
        self,
        repos_file: Path = Macros.work_dir / "repos" / "filtered" /
        "repos.json",
        coverage_dir: Path = Macros.work_dir / "coverage-new",
        data_dir: Path = Macros.data_dir,
    ):
        Tool.ensure_tool_versions()
        Tool.require_compiled()

        projects = su.io.load(repos_file, clz=List[Project])

        indexed_projects: Dict[str, Project] = {
            p.full_name: p
            for p in projects
        }
        rq2_public_methods = []
        index = 0

        for prj_name, project in indexed_projects.items():
            project_dir = coverage_dir / prj_name
            project.clone(Macros.downloads_dir / prj_name)

            for module_dir in project_dir.iterdir():
                if not module_dir.is_dir():
                    continue
                maven_module = su.io.load(module_dir / "module.yaml",
                                          clz=MavenModule)
                maven_module.project = project
                module_i = int(os.path.basename(module_dir))
                if self._skip_module(maven_module, module_i):
                    continue
                try:
                    method_records = su.io.load(module_dir /
                                                "method_records.jsonl")
                except FileNotFoundError:
                    method_records = collect_module_source_code(
                        indexed_projects[prj_name],
                        Macros.work_dir / "downloads",
                        maven_module.rel_path,
                    )
                    su.io.dump(module_dir / "method_records.jsonl",
                               method_records)

                for m in method_records:
                    if "public" in m["modifiers"] and m["exceptions"]:
                        for ex in m["exceptions"]:
                            dt = DataNE2E()
                            dt.id = index
                            dt.project = prj_name
                            dt.module_i = module_i
                            dt.module = maven_module
                            dt.module = maven_module.rel_path
                            dt.mut_key = m["method"]
                            dt.mut = m["method_node"]
                            ex_line = int(ex.split("@@")[1]) - int(
                                m["startLine"])
                            dt.e_stack_trace = [[m, ex_line]]
                            dt.etype = ex.split("@@")[0]
                            rq2_public_methods.append(dt)
                            index += 1

        save_dataset(data_dir / "rq2-public", rq2_public_methods, clz=DataNE2E)

    def _find_existing_test_classes(self, project_name: str, module_i: int):
        """
        Collect the existing test classes in the target module.
        Collect the existing test methods in the target module.
        """

        tests_file = su.io.load(
            Macros.work_dir / "coverage-new" / project_name / str(module_i) /
            "manual.tests.jsonl",
            clz=TestMethod,
        )
        test_class_names = set()
        for test in tests_file:
            test_class_names.add(test.cname)
            self._test_class_2_context[
                f"{project_name}.{module_i}.{test.cname}"] = (test.ccontext)
            self._test_class_2_methods[
                f"{project_name}.{module_i}.{test.cname}"].append(test)
        # update dict
        self._project_2_test_classes[f"{project_name}.{module_i}"].update(
            test_class_names)

    def _extract_test_file_context(self, project_name: str, module_i: int,
                                   mut_key: str) -> Tuple[str, TestMethod]:
        """
        Extract the test file context, and netest within given the Test Name by name matching.
        """

        mut_class_name = mut_key.split("#")[0]
        mut_class_no_innerclass = mut_key.split("#")[0].split("$")[0]
        if (f"{mut_class_name}Test"
                in self._project_2_test_classes[f"{project_name}.{module_i}"]):
            test_file_context = self._test_class_2_context[
                f"{project_name}.{module_i}.{mut_class_name}Test"]
            self._current_test_class_name = f"{mut_class_name}Test"
            gold_test_method = self._test_class_2_methods[
                f"{project_name}.{module_i}.{mut_class_name}Test"][0]
        elif (f"{mut_class_no_innerclass}Test"
              in self._project_2_test_classes[f"{project_name}.{module_i}"]):
            test_file_context = self._test_class_2_context[
                f"{project_name}.{module_i}.{mut_class_no_innerclass}Test"]
            gold_test_method = self._test_class_2_methods[
                f"{project_name}.{module_i}.{mut_class_no_innerclass}Test"][0]
            self._current_test_class_name = f"{mut_class_no_innerclass}Test"
        else:
            test_file_context = None
            self._current_test_class_name = ""
            gold_test_method = None
        return test_file_context, gold_test_method

    def _extract_netest(self, mut_key: str, project_name: str,
                        module_i: int) -> List[TestMethod]:
        assert self._test_methods is not None
        ne_tests = []
        mut_key = mut_key.replace(".", "/")
        if not self._mut2test or len(self._mut2test[mut_key]) == 0:
            if self._current_test_class_name != "":
                # find test methods within the test class
                existing_methods = self._test_class_2_methods[
                    f"{project_name}.{module_i}.{self._current_test_class_name}"]
                ne_tests = [m for m in existing_methods if m.pattern is None]
        else:
            ne_tests = [
                self._test_methods[tid] for tid in self._mut2test[mut_key]
                if self._test_methods[tid].pattern is None
            ]
        #
        return ne_tests

    def get_non_cover_ne2e_data(self, source_dataset: str = "rq2-public"):
        """
        Collect dataset for non covered methods. (rq2)
        + Mut
        + test file
        + netest
        + stack trace (w.o. line number)
        + condition
        """
        self._clear()
        # load collected dataset
        dataset = load_dataset(Macros.data_dir / source_dataset, clz=DataNE2E)
        module_2_test_coverage = {}
        new_dataset = []
        # stats
        no_netest, no_test_context = 0, 0
        with tqdm(total=len(dataset),
                  desc="Collecting non covered data") as pbar:
            for dt in dataset:
                project_name, module_i = dt.project, dt.module_i
                if f"{project_name}.{module_i}" not in self._project_2_test_classes:
                    self._find_existing_test_classes(project_name, module_i)
                #
                test_file_context, gold_test = self._extract_test_file_context(
                    project_name, module_i, dt.mut_key)
                # netest
                if f"{project_name}.{module_i}" not in module_2_test_coverage:
                    module_2_test_coverage[f"{project_name}.{module_i}"] = (
                        extract_module_test_coverage(
                            dt,
                            coverage_out_dir=Macros.work_dir / "coverage-new"))
                self._mut2test, self._test_methods = module_2_test_coverage[
                    f"{project_name}.{module_i}"]
                ne_tests = self._extract_netest(dt.mut_key, project_name,
                                                module_i)
                dt.test_ne = [t.raw_code for t in ne_tests]
                if not test_file_context and len(ne_tests) > 0:
                    dt.test_context = [t.ccontext for t in ne_tests]
                    dt.test_method = ne_tests[0]
                else:
                    dt.test_context = test_file_context
                    dt.test_method = gold_test
                # stats
                if len(ne_tests) == 0:
                    no_netest += 1

                if dt.test_context is None or len(dt.test_context) == 0:
                    no_test_context += 1
                else:
                    assert dt.test_method is not None, "No at least one ne-test"
                    new_dataset.append(dt)
                pbar.update(1)

        logger.info(
            f"No netest: {no_netest}\n No test context: {no_test_context}")
        save_dataset(Macros.data_dir / f"new-{source_dataset}", new_dataset)

    def collect_netest_data(
        self,
        target_projects: List[str] = None,
        mut2e_data_dir: Path = Macros.work_dir / "data" / "mut2e-new",
    ):
        """
        For each etest, collect the netest that cover the same method.
        """
        # load mut2e data
        with tqdm("Loading MUT2E data") as pbar:
            mut2e_dataset = load_dataset(mut2e_data_dir,
                                         clz=parse_data_cls("MUT2E"),
                                         pbar=pbar)

        ne2e_data_list = []
        normal_test_cover_mut, no_normal_test = 0, 0
        module_2_tests = {}
        ne2e_data_dir = mut2e_data_dir
        #
        with tqdm(total=len(mut2e_dataset),
                  desc="Pairing etest with netest") as pbar:
            for i, mut2e_dt in enumerate(mut2e_dataset):
                cur_prj = mut2e_dt.project
                if target_projects and cur_prj not in target_projects:
                    continue
                mut = mut2e_dt.mut_key
                cur_module = mut2e_dt.module_i
                if f"{cur_prj}.{cur_module}" not in module_2_tests:
                    module_2_tests[f"{cur_prj}.{cur_module}"] = (
                        extract_module_test_coverage(
                            mut2e_dt,
                            coverage_out_dir=Macros.work_dir / "coverage-new"))
                #
                mut2test, test_methods = module_2_tests[
                    f"{cur_prj}.{cur_module}"]
                ne_tests = []
                mut_key = mut.replace(".", "/")
                if not mut2test:
                    no_normal_test += 1
                else:
                    if len(mut2test[mut_key]) == 0:
                        no_normal_test += 1
                    else:
                        ne_tests = [
                            tid for tid in mut2test[mut_key]
                            if test_methods[tid].pattern is None
                        ]
                        if len(ne_tests) > 0:
                            normal_test_cover_mut += 1
                        else:
                            no_normal_test += 1
                data = DataNE2E(**mut2e_dt.__dict__)
                for ne_id in ne_tests:
                    test_ne = test_methods[ne_id]
                    data.netest_sign.append(test_ne.ast.get_sign())
                    data.test_ne.append(test_ne.raw_code)
                    data.netest_context.append(test_ne.context)
                    data.test_ne_key.append(
                        test_ne.cname.replace(".", "/") + "#" + test_ne.mname +
                        "#()V")
                #
                data.id = f"ne2e-{i}"
                ne2e_data_list.append(data)
                pbar.update(1)
        logger.info(
            f"Normal test cover mut: {normal_test_cover_mut}\n No normal tests cover the same method: {no_normal_test}"
        )
        shutil.rmtree(ne2e_data_dir)
        save_dataset(ne2e_data_dir, ne2e_data_list, append=True)


if __name__ == "__main__":
    su.log.setup(Macros.log_file, su.log.INFO)
    CLI(NEData, as_positional=False)
