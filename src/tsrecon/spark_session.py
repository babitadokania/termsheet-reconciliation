"""Local SparkSession factory with Windows-friendly defaults and Java version handling.

PySpark 3.5 supports Java 8/11/17 (21 works). Newer JDKs break Hadoop's login code:
- Java 18-23: fixed automatically with -Djava.security.manager=allow (via JDK_JAVA_OPTIONS).
- Java 24+:   cannot be fixed by a flag. A supported JDK (17/21/11) installed in a standard location
              is picked up automatically; otherwise a clear error explains how to install one.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import shutil
import subprocess
import sys

log = logging.getLogger(__name__)

SUPPORTED_JAVA = (17, 21, 11)  # preference order
WINDOWS_JDK_ROOTS = [
    r"C:\Program Files\Eclipse Adoptium",
    r"C:\Program Files\Java",
    r"C:\Program Files\Microsoft",
    r"C:\Program Files\Amazon Corretto",
    r"C:\Program Files\Zulu",
    r"C:\Program Files\BellSoft",
]
INSTALL_HELP = (
    "Install JDK 17 (recommended for PySpark 3.5):\n"
    "  Windows:  winget install EclipseAdoptium.Temurin.17.JDK   (or https://adoptium.net)\n"
    "            then reopen PowerShell - the pipeline finds JDK 17 automatically,\n"
    "            or set it explicitly:  $env:JAVA_HOME = 'C:\\Program Files\\Eclipse Adoptium\\jdk-17...'\n"
    "  macOS:    brew install openjdk@17        Linux: sudo apt install openjdk-17-jre"
)


class SparkSetupError(RuntimeError):
    pass


def _java_exe(java_home: str | None) -> str | None:
    if java_home:
        exe = os.path.join(java_home, "bin", "java.exe" if os.name == "nt" else "java")
        if os.path.exists(exe):
            return exe
    return shutil.which("java")


def java_major_version(java_exe: str) -> int | None:
    try:
        out = subprocess.run([java_exe, "-version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r'version "(\d+)(?:\.(\d+))?', out.stderr + out.stdout)
    if not m:
        return None
    major = int(m.group(1))
    return int(m.group(2) or 0) if major == 1 else major  # "1.8.0" -> 8


def _find_supported_jdk() -> str | None:
    """Look for an installed JDK 17/21/11 in the usual Windows locations."""
    if os.name != "nt":
        return None
    for version in SUPPORTED_JAVA:
        for root in WINDOWS_JDK_ROOTS:
            candidates: list[str] = []
            for pattern in (f"jdk-{version}*", f"jdk{version}*", f"zulu-{version}*", f"zulu{version}*"):
                candidates += glob.glob(os.path.join(root, pattern))
            for home in sorted(candidates, reverse=True):
                if os.path.exists(os.path.join(home, "bin", "java.exe")):
                    return home
    return None


def prepare_java() -> int | None:
    """Validate the Java runtime Spark will use; adjust JAVA_HOME / JVM options when needed."""
    exe = _java_exe(os.environ.get("JAVA_HOME"))
    if not exe:
        alt = _find_supported_jdk()
        if alt:
            os.environ["JAVA_HOME"] = alt
            exe = _java_exe(alt)
            log.info("Using JDK found at %s", alt)
        else:
            raise SparkSetupError("Java was not found. PySpark needs a Java runtime.\n" + INSTALL_HELP)

    major = java_major_version(exe) if exe else None
    log.info("Java runtime: %s (major version %s)", exe, major)

    if major is not None and major >= 24:
        alt = _find_supported_jdk()
        if alt:
            os.environ["JAVA_HOME"] = alt
            os.environ["PATH"] = os.path.join(alt, "bin") + os.pathsep + os.environ.get("PATH", "")
            log.warning("Java %d is not supported by PySpark 3.5 - switching to %s", major, alt)
            return java_major_version(_java_exe(alt) or "")
        raise SparkSetupError(
            f"Java {major} is too new for PySpark 3.5 (it removed APIs Hadoop needs).\n" + INSTALL_HELP
        )

    if major is not None and major >= 18:
        # Java 18-23: re-enable Subject.getSubject() used by Hadoop's UserGroupInformation
        flag = "-Djava.security.manager=allow"
        current = os.environ.get("JDK_JAVA_OPTIONS", "")
        if flag not in current:
            os.environ["JDK_JAVA_OPTIONS"] = f"{current} {flag}".strip()
        log.info("Java %d detected - enabled %s for Spark compatibility", major, flag)
    return major


def get_spark(master: str = "local[*]", app_name: str = "termsheet-recon"):
    prepare_java()
    # Make Spark's Python workers use the same interpreter/venv as the driver (essential on Windows)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    import pyspark
    from pyspark.sql import SparkSession

    # Point SPARK_HOME at the pip-installed PySpark so Windows' find-spark-home.cmd is skipped
    # (it mis-parses paths containing spaces/apostrophes and prints a spurious warning).
    os.environ.setdefault("SPARK_HOME", os.path.dirname(pyspark.__file__))

    try:
        spark = (
            SparkSession.builder.appName(app_name)
            .master(master)
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
            .config("spark.sql.ansi.enabled", "false")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .getOrCreate()
        )
    except Exception as exc:
        if "getSubject is not supported" in str(exc):
            raise SparkSetupError("Your Java version is too new for PySpark 3.5.\n" + INSTALL_HELP) from exc
        raise
    spark.sparkContext.setLogLevel("ERROR")
    log.info("Spark %s started (master=%s)", spark.version, master)
    return spark
