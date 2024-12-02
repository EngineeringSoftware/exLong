from pathlib import Path
import os


class Macros:
    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    python_dir: Path = this_dir.parent
    log_file: Path = python_dir / "experiment.log"
    stack_trace_log_file: Path = python_dir / "stack.trace.experiment.log"
    debug_dir: Path = python_dir / "debug"
    project_dir: Path = python_dir.parent
    java_dir: Path = project_dir / "collector"

    scripts_dir: Path = project_dir / "scripts"
    doc_dir: Path = project_dir / "docs"
    results_dir: Path = project_dir / "results"
    work_dir: Path = project_dir / "_work"
    setup_dir: Path = work_dir / "setup"
    exp_dir: Path = work_dir / "exp"
    data_dir: Path = work_dir / "data"
    doc_dir: Path = project_dir / "docs"
    downloads_dir: Path = work_dir / "downloads"
    config_dir: Path = python_dir / "configs"
    # git@bitbucket.org:extestdata/extestdata.git
    extestdata_dir: Path = project_dir / ".." / "extestdata"
    jar_dir: Path = project_dir / "jars"
    randoop_jar: str = jar_dir / "randoop-4.3.2" / "randoop-all-4.3.2.jar"
    evosuite_jar = jar_dir / "evosuite-1.2.0.jar"
    evosuitefit_jar = jar_dir / "evosuitefit-1.0.7.jar"
    junit_jar = jar_dir / "junit-platform-console-standalone-1.9.0-RC1.jar"

    paper_dir: Path = project_dir / "papers" / "icse25"

    # TODO: we temporarily use some resources from teco
    teco_project_dir: Path = project_dir.parent / "teco-internal"
    teco_work_dir: Path = teco_project_dir / "_work"
    teco_downloads_dir: Path = teco_work_dir / "downloads"

    all_set = "all"
    train = "train"
    val = "val"
    test = "test"

    SKIPS = "-Djacoco.skip -Dcheckstyle.skip -Drat.skip -Denforcer.skip -Danimal.sniffer.skip -Dmaven.javadoc.skip -Dfindbugs.skip -Dwarbucks.skip -Dmodernizer.skip -Dimpsort.skip -Dpmd.skip -Dxjc.skip -Dair.check.skip-all -Dfmt.skip -Dgpg.skip -Dlicense.skipAddThirdParty -Dlicense.skip"
