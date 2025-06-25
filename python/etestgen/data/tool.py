import os
from pathlib import Path
from typing import Optional
import subprocess

import seutil as su

from etestgen.macros import Macros


class Tool:
    """Pointers to the Java programs used in the project, including the collector, the Randoop and Evosuite libraries, etc."""

    EXPECTED_MAVEN_VERSION = "3.8.3"
    EXPECTED_JAVA_VERSION = "1.8.0"

    @classmethod
    def ensure_tool_versions(cls):
        maven_version = su.bash.run(
            r"mvn --version | grep 'Apache Maven' | sed -nE 's/Apache Maven ([0-9.]+).*/\1/p'"
        ).stdout.strip()
        if cls.EXPECTED_MAVEN_VERSION not in maven_version:
            raise RuntimeError(
                f"Expected Maven version {cls.EXPECTED_MAVEN_VERSION}, but got {maven_version}"
            )
        java_version = su.bash.run(
            r"mvn --version | grep 'Java version:' | sed -nE 's/Java version: ([0-9.]+).*/\1/p'"
        ).stdout.strip()
        if cls.EXPECTED_JAVA_VERSION not in java_version:
            raise RuntimeError(
                f"Expected Java version {cls.EXPECTED_JAVA_VERSION}, but got {java_version}"
            )

    collector_dir: Path = Macros.project_dir / "collector"
    collector_version: str = "0.1-dev"

    core_name = "core"
    core_jar: str = str(
        collector_dir
        / core_name
        / "target"
        / f"{core_name}-{collector_version}-jar-with-dependencies.jar"
    )

    rt_name = "rt"
    rt_jar: str = str(
        collector_dir
        / rt_name
        / "target"
        / f"{rt_name}-{collector_version}-jar-with-dependencies.jar"
    )

    compiled = False

    @classmethod
    def require_compiled(cls):
        if not cls.compiled:
            with su.io.cd(cls.collector_dir):
                su.bash.run(f"mvn package -DskipTests", 0)
                cls.compiled = True

    randoop_jar: str = str(Macros.python_dir / "lib" / "randoop-all-4.3.1.jar")
    evosuite_jar: str = str(Macros.python_dir / "lib" / "evosuite-1.2.0.jar")
    junit4_classpath: str = os.pathsep.join(
        [
            str(Macros.python_dir / "lib" / "junit-4.13.2.jar"),
            str(Macros.python_dir / "lib" / "hamcrest-core-1.3.jar"),
        ]
    )


class DataCollector:
    data_collector_dir: Path = Macros.project_dir / "collector"
    data_collector_version: str = "0.1-dev"

    static_name = "static-collector"
    static_jar: str = str(
        data_collector_dir
        / static_name
        / "target"
        / f"{static_name}-{data_collector_version}-jar-with-dependencies.jar"
    )

    bcverifier_name = "bcverifier"
    bcverifier_jar: str = str(
        data_collector_dir
        / bcverifier_name
        / "target"
        / f"{bcverifier_name}-{data_collector_version}-jar-with-dependencies.jar"
    )

    adhoc_runner_name = "adhoc-runner"
    adhoc_runner_jar: str = str(
        data_collector_dir
        / adhoc_runner_name
        / "target"
        / f"{adhoc_runner_name}-{data_collector_version}-jar-with-dependencies.jar"
    )

    compiled = False

    @classmethod
    def require_compiled(cls):
        if not cls.compiled:
            with su.io.cd(cls.data_collector_dir):
                su.bash.run(f"mvn package -DskipTests", 0)
                cls.compiled = True

    @classmethod
    def run_static(
        cls,
        main: str,
        config: Optional[dict] = None,
        args: Optional[str] = None,
        timeout: Optional[int] = None,
        check_returncode: int = 0,
        jvm_args: str = "",
    ) -> subprocess.CompletedProcess:
        cls.require_compiled()

        if config is not None and args is not None:
            raise ValueError("Cannot specify both config and args")

        if config is not None:
            config_file = su.io.mktmp("dcstatic", ".json")
            su.io.dump(config_file, config)
            args = config_file

        if args is None:
            args = ""

        rr = su.bash.run(
            f"java {jvm_args} -cp {cls.static_jar} {main} {args}",
            check_returncode=check_returncode,
            timeout=timeout,
        )

        if config is not None:
            # delete temp input file
            su.io.rm(config_file)

        return rr
