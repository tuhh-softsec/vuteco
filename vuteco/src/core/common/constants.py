import os
from enum import Enum

import torch


class FinderName(str, Enum):
    # Given a test case, determine if it's witnessing (different principles)
    CODEBERT_FND = "cb-fnd"
    CODET5PLUS_FND = "ct5p-fnd"
    UNIXCODER_FND = "uxc-fnd"
    CODELLAMA_FND = "cl-fnd"
    QWENCODER_FND = "qc-fnd"
    DEEPSEEKCODER_FND = "dsc-fnd"
    GREP_FND = "grep-fnd"
    VOCABULARY_FND = "vocab-fnd"
    FIX_FND = "fix-fnd"


class HeuristicName(str, Enum):
    # Given a set of fix commits, extract the changed and added tests
    FIX_HEU = "fix-heu"


class LinkerName(str, Enum):
    # Given a witnessing test and a description of a vuln, determine if related
    CODEBERT_LNK = "cb-lnk"
    CODET5PLUS_LNK = "ct5p-lnk"
    UNIXCODER_LNK = "uxc-lnk"
    CODELLAMA_LNK = "cl-lnk"
    QWENCODER_LNK = "qc-lnk"
    DEEPSEEKCODER_LNK = "dsc-lnk"
    GREP_LNK = "grep-lnk"
    TERMS_LNK = "terms-lnk"
    SIM_LNK = "sim-lnk"


class End2EndName(str, Enum):
    # Given a test case and a description of a vuln, determine if related
    CODEBERT_E2E = "cb-e2e"
    CODET5PLUS_E2E = "ct5p-e2e"
    UNIXCODER_E2E = "uxc-e2e"
    CODELLAMA_E2E = "cl-e2e"
    QWENCODER_E2E = "qc-e2e"
    DEEPSEEKCODER_E2E = "dsc-e2e"
    GREP_E2E = "grep-e2e"
    TERMS_E2E = "terms-e2e"
    SIM_E2E = "sim-e2e"
    FIX_E2E = "fix-e2e"


class FinderLinkerName(str, Enum):
    # Given a test case and a description of a vuln, determine if related
    CODEBERT_FL = "cb-fl"
    CODET5PLUS_FL = "ct5p-fl"
    UNIXCODER_FL = "uxc-fl"
    CODELLAMA_FL = "cl-fl"
    QWENCODER_FL = "qc-fl"
    DEEPSEEKCODER_FL = "dsc-fl"


class TechniqueName(str, Enum):
    CODEBERT = "cb"
    CODET5PLUS = "ct5p"
    UNIXCODER = "uxc"
    CODELLAMA = "cl"
    QWENCODER = "qc"
    DEEPSEEKCODER = "dsc"
    GREP = "grep"
    VOCABULARY = "vocab"
    FIX = "fix"


class CommonConfigKeys(str, Enum):
    SPLIT = "split"
    DATASET_SEED = "dataset_seed"


class LossFunction(str, Enum):
    BCE = "bce",
    WBCE = "wbce"


class MergeStyle(str, Enum):
    CONCAT_TEST_DESCR = "concat-td"
    CONCAT_DESCR_TEST = "concat-dt"
    LEARN = "learn"
    JAVADOC = "jd"


class E2EArchitectureStyle(str, Enum):
    LINKER_ONLY = "lo"
    FINDER_LINKER_MASK = "fl-mask"
    FINDER_LINKER_META = "fl-meta"
    FINDER_LINKER_FUSE = "fl-fuse"


class E2ETrainingType(str, Enum):
    PRETRAIN_ONLY = "pt"
    FINETUNE_ONLY = "ft"
    PRETRAIN_FINETUNE = "pt-ft"


class VutecoRevisionStyle(str, Enum):
    ALL = "all"
    HEAD = "head"
    INPUT_FILE = "from-file"


ALL = "all"
CODEBERT_FULL = "codebert"
CODET5PLUS_FULL = "codet5+"
UNIXCODER_FULL = "unixcoder"

DETERMINISM = True
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNSLOTH_TRAINING_KEY = "unsloth_training"

IS_SECURITY_COL = "is_security"
LABEL_COL = "label"
TEXT_COL = "text"
TEXT_1_COL = "text_1"
TEXT_2_COL = "text_2"
TEXT_2_1_COL = "text_2_1"
CODE_COL = "code"
CVE_ID_COL = "cveId"
CVE_DESCR_COL = "cveDesc"
CWE_INFO_COL = "cweInfo"
INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
INPUT_IDS_1 = "input_ids_1"
INPUT_IDS_2 = "input_ids_2"
ATTENTION_MASK_1 = "attention_mask_1"
ATTENTION_MASK_2 = "attention_mask_2"
FND_INPUT_IDS = "fnd_input_ids"
FND_ATTENTION_MASK = "fnd_attention_mask"
LNK_INPUT_IDS = "lnk_input_ids"
LNK_ATTENTION_MASK = "lnk_attention_mask"
LNK_INPUT_IDS_1 = "lnk_input_ids_1"
LNK_INPUT_IDS_2 = "lnk_input_ids_2"
LNK_ATTENTION_MASK_1 = "lnk_attention_mask_1"
LNK_ATTENTION_MASK_2 = "lnk_attention_mask_2"
JAVADOC_ONE_LINE = "/** Tests that following vulnerability is present: {} */ {}"
JAVADOC_MULTILINE = "/**\n * Tests that following vulnerability is present:\n * {}\n */\n{}"

PROJECT_KB_URL = "https://github.com/SAP/project-kb/"
PROJECTKB_BRANCH = "vulnerability-data"

DATA_DIRPATH = os.path.join(os.path.pardir, "data")
TEST4VUL_FILEPATH = os.path.join(DATA_DIRPATH, "test4vul.json")
VUL4J_DIRPATH = os.path.join(DATA_DIRPATH, "vul4j_tests")
VUL4J_FILEPATH = os.path.join(DATA_DIRPATH, "vul4j.csv")
VUL4J_TEST_FILEPATH = os.path.join(DATA_DIRPATH, "vul4j_tests.json")
DEFAULT_VUTECO_RESULT_DIRNAME = "vuteco_results"

EXPERIMENT_DIRPATH = os.path.join(DATA_DIRPATH, "experiment")
EXPERIMENT_SESSION_DIRPATH = os.path.join(EXPERIMENT_DIRPATH, "sessions")
EXPERIMENT_ANALYSIS_DIRPATH = os.path.join(EXPERIMENT_DIRPATH, "analysis")
ANALYSIS_PERFORMANCE_FILENAME = "performance.csv"
ROC_CURVES_DIRNAME = "roc_curves"
PR_CURVES_DIRNAME = "pr_curves"
RAW_FILENAME = "raw_output.txt"

TOOL_DIRPATH = os.path.join(os.path.pardir, "tools")
SPAT_DIRPATH = os.path.join(TOOL_DIRPATH, "spat")
SPAT_JAR_FILEPATH = os.path.join(SPAT_DIRPATH, "SPAT-1.0.jar")
SPAT_COMMAND = "java -jar {spat_jar} {seed} {rule_id} {in_dir} {out_dir} {rt_jar}"
SPAT_JVM_PATHS = ["/usr/lib/jvm", "/Library/Java/JavaVirtualMachines"]
SPAT_PREFIX = "public class Foo {\n"
SPAT_SUFFIX = "\n}"
JT_DIRPATH = os.path.join(TOOL_DIRPATH, "java-transformer")
JT_JAR_FILEPATH = os.path.join(JT_DIRPATH, "JavaTransformer.jar")
JT_COMMAND = "java -jar {jt_jar} {in_dir} {out_dir} {seed}"
JT_RULES = ["BooleanExchange", "LogStatement", "LoopExchange", "PermuteStatement", "ReorderCondition", "SwitchToIf", "TryCatch", "UnusedStatement", "VariableRenaming"]

VULNERABILITIES_DIRPATH = os.path.join(DATA_DIRPATH, "vulnerabilities")
RAW_VULNERABILITIES_DIRPATH = os.path.join(VULNERABILITIES_DIRPATH, "raw")
PROJECT_KB_REPO_DIRPATH = os.path.join(RAW_VULNERABILITIES_DIRPATH, "project-kb")
REEF_RAW_FILEPATH = os.path.join(RAW_VULNERABILITIES_DIRPATH, "query_Java.jsonl")
REPOSVUL_RAW_FILEPATH = os.path.join(RAW_VULNERABILITIES_DIRPATH, "ReposVul_java.jsonl")
PROCESSED_VULNERABILITIES_DIRPATH = os.path.join(VULNERABILITIES_DIRPATH, "processed")
PROJECT_KB_FILEPATH = os.path.join(PROCESSED_VULNERABILITIES_DIRPATH, "projectkb.json")
REEF_KB_FILEPATH = os.path.join(PROCESSED_VULNERABILITIES_DIRPATH, "reefkb.json")
REPOSVUL_KB_FILEPATH = os.path.join(PROCESSED_VULNERABILITIES_DIRPATH, "reposvulkb.json")
CONFIG_DIRPATH = os.path.join(os.path.pardir, "config")
VUTECO_KB_FILEPATH = os.path.join(CONFIG_DIRPATH, "vutecokb.json")
PATTERN_DIRPATH = os.path.join(CONFIG_DIRPATH, "patterns")
EXACT_PATTERN_FILEPATH = os.path.join(PATTERN_DIRPATH, "exact-match")
SUBSTRING_RED_PATTERN_FILEPATH = os.path.join(PATTERN_DIRPATH, "substring-match-red")
SUBSTRING_EXT_PATTERN_FILEPATH = os.path.join(PATTERN_DIRPATH, "substring-match-ext")
CODESHOVEL_DIRPATH = os.path.join(TOOL_DIRPATH, "codeshovel")
CODESHOVEL_JAR_FILEPATH = os.path.join(CODESHOVEL_DIRPATH, "codeshovel-1.0.0-SNAPSHOT.jar")
CODESHOVEL_COMMAND = "java -jar {codeshovel_jar} -repopath {repo_dir} -filepath {file_path} -methodname {method_name} -startline {start_line} -outfile {outfile_path}"

TRAINED_MODEL_DIRPATH = os.path.join(os.path.pardir, "trained_models")
EVALUATED_MODEL_DIRPATH = os.path.join(os.path.pardir, "evaluated_models")
FINAL = "final"
NAMES_JSON = "names.json"
CONFIG_JSON = "config.json"

REMOTE_BASEPATH = "emaiannone/vuteco-{model}"

INTHEWILD_DIRPATH = os.path.join(DATA_DIRPATH, "inthewild")
INTHEWILD_FILEPATH = os.path.join(INTHEWILD_DIRPATH, "inthewild.csv")
INTHEWILD_PREP_STATS_FILEPATH = os.path.join(INTHEWILD_DIRPATH, "preparation_stats.json")
INTHEWILD_MATCHING_DIRPATH = os.path.join(INTHEWILD_DIRPATH, "matching")
INTHEWILD_FINDING_DIRPATH = os.path.join(INTHEWILD_DIRPATH, "finding")
INTHEWILD_RESULTS_DIRNAME = "results"
INTHEWILD_TESTS_DIRNAME = "tests"
INTHEWILD_STATS_FILENAME = "stats.csv"
INTHEWILD_TOTAL_STATS_FILENAME = "total_stats.json"
INTHEWILD_MATCHES_DIRNAME = "matches"
INTHEWILD_FINDINGS_DIRNAME = "findings"
INTHEWILD_FINDINGS_TO_INSPECT_FILENAME = "findings_to_inspect.csv"
INTHEWILD_MATCHINGS_TO_INSPECT_FILENAME = "matchings_to_inspect.csv"
INTHEWILD_FINDINGS_INSPECTED_FILENAME = "findings_inspected.csv"
INTHEWILD_MATCHINGS_INSPECTED_FILENAME = "matchings_inspected.csv"

TESTING_REPOS = [
    "HtmlUnit/htmlunit",
    "junit-team/junit4",
]
REPOS_MERGE_FROM_TO = {
    "frameworks/base": "aosp-mirror/platform_frameworks_base",
    "asf/cxf": "apache/cxf",
    "asf/hive": "apache/hive",
    "asf/camel": "apache/camel",
    "asf/struts": "apache/struts",
    "asf/activemq": "apache/activemq",
    "asf/tapestry-5": "apache/tapestry-5",
    "asf/lucene-solr": "apache/lucene-solr",
    "asf/ambari": "apache/ambari",
    "asf/tomee": "apache/tomee",
    "asf/cxf-fediz": "apache/cxf-fediz",
    "asf/hbase": "apache/hbase",
    "asf/cordova-plugin-file-transfer": "apache/cordova-plugin-file-transfer",
    "apache/incubator-iotdb": "apache/iotdb",
    "microsoft/ChakraCore": "chakra-core/ChakraCore",
    "nilsteampassnet/TeamPass": "nilsteampassnet/teampass",
    "dlitz/pycrypto": "pycrypto/pycrypto",
    "tomato42/tlslite-ng": "tlsfuzzer/tlslite-ng",
    "resteasy/Resteasy": "resteasy/resteasy",
    "eclipse-vertx/vert.x": "eclipse/vert.x",
}
REPOS_TO_RENAME = {
    "asf/activemq-apollo": "apache/activemq-apollo"
}
