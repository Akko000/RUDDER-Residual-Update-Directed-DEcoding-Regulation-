# ---------- 基础路径 ----------
DATA_DIR            = "../dataset"

VQA_QUESTIONS_FILE  = f"{DATA_DIR}/v2_OpenEnded_mscoco_val2014_questions.json"
VQA_ANNOTATIONS_FILE= f"{DATA_DIR}/v2_mscoco_val2014_annotations.json"
IMAGE_DIR           = f"{DATA_DIR}/val2014"

# POPE
POPE_DIR            = f"{DATA_DIR}/POPE"
POPE_SPLITS         = ["adversarial", "popular", "random"]
#POPE_SPLITS         = ["random"]
# CHAIR
COCO_INSTANCES_FILE = f"{DATA_DIR}/annotations/instances_val2014.json"
CHAIR_SPLIT_FILE    = f"{DATA_DIR}/chair_image_ids_500.json"
CHAIR_NUM_IMAGES    = 100

# ---------- 结果路径 ----------

RESULTS_DIR_CHAIR   = "../exp_qwen_chair"
VIZ_OUT_DIR         = "../exp_llava15_viz"
# ---------- 模型 ----------
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
CACHE_DIR           = "../cache"
DEVICE              = "cuda:0"
DTYPE               = "bf16"          

BATCH_SIZE          = 10

MAX_NEW_TOKENS_CAP  = 512

SEEDS               = [42]


EGR_POOLINGS        = ["attn"]
INJECTION_LAYERS    = [25,26,27]
# CARD+Beta 网格
BETA_ALPHA_MAX      = [5]
BETA_K              = [5]
BETA_C              = [1.0]
GATE_CLAMP          = (0.1,1)


CHAIR_PROMPT = (
    "USER: <image>\nPlease help me describe the image in detail.\nASSISTANT:"
)

BAD_WORD_STRINGS = [
    "Instruction:", "Image:", "Question:", "Answer:",
    "User:", "Assistant:", "Subject:", "System:",
    "USER:", "ASSISTANT:" 
]

COCO_IMG_DIR       = f"{DATA_DIR}/val2014"
COCO_INSTANCES_JSON= f"{DATA_DIR}/annotations/instances_val2014.json"
