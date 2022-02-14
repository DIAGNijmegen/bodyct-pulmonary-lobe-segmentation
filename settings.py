# Flags
COPY_DATA = False
ON_PREMISE_LOCATION = None
# ON_PREMISE_LOCATION = 'd:/test/maindisk/xie/airlabel/'
RELOAD_CHECKPOINT = False
IS_CUDA = True
TEST_RESULTS_DUMP_DEBUG_NUM = 1
TEST_RESULTS_DUMP_HEATMAP = False
RELOAD_CHECKPOINT_PATH = None
# RELOAD_DICT_LIST = ["model_dict", "metric", "optimizer_dict", "scheduler_dict"]
RELOAD_DICT_LIST = ["model_dict", "metric"]





EXP_NAME = "CTSUNET"

# Training iterations and sizes.
RESAMPLE_MODE = "fixed_output_size"
USE_PROPOSAL = True
POST_METHOD="nn"
SCAN_PAD_NUM = 16

#TOPK_LOSS = "topk"
#TOPK_LOSS = "box"
STATES_STEPS = 500
STATES_AFTER_STEPS = 500

NUM_STEPS = 5000000
VALID_STEPS = 2000
VALID_AFTER_TRAIN_STEPS = 2000
NUM_WORKERS = 0
LOG_STEPS = 2
AUG_RATIO = 0.0

TRAIN_BATCH_SIZE = [1, 2]
VAL_BATCH_SIZE = [1, 4]
TEST_BATCH_SIZE = [1, 4]
VALID_SAMPLING_RATE = 1.0

TRAIN_PATCH_OVERLAPS = [(0.0, 0.0, 0.0)]
TEST_PATCH_OVERLAPS = [(0.0, 0.0, 0.0)]
TRAIN_STITCHES_PATCH_SIZE = [((116, 116, 116), (28, 28, 28), (28, 28, 28))]
TEST_STITCHES_PATCH_SIZE = [((132, 132, 132), (44, 44, 44))]


SAMPLING_RATE = 0.5
RESAMPLE_SIZE = (256, 256, 256)
# RESAMPLE_SIZE = (192, 192, 192)
SIZE_JITTERING = 1.4
TEST_SPACING = 1.4
LOSS_FACTORS = [1.0, 1.0, 1.0, 1.0]
RELABEL_MAPPING = {}
LABEL_NAME_MAPPING = {0: 'background',
                      1: 'LUL',
                      2: 'LLL',
                      3: 'RUL',
                      4: 'RLL',
                      5: 'RML'}
# LABEL_NAME_MAPPING = {0: 'background',
#                       1: 'lung_tissue',
#                       2: 'fissure'}
# LABEL_NAME_MAPPING = {0: 'background',
#                       # 1: 'lung_tissue',
#                       1: 'fissure'}
# CLASS_WEIGHTS = {0: 0.2, 1: 0.5, 2: 0.8}
CLASS_WEIGHTS = {0: 0.65, 1: 0.7, 2: 0.7, 3: 0.7, 4: 0.7, 5: 0.85}
NR_CLASS = 6
# thresholds
PAD_VALUE = -2048
WINDOWING_MAX = 400
WINDOWING_MIN = -1200


# MODEL = {
#     "net1_param": {
#         "n_layers": 3,
#         "in_ch_list": [1, 1, 2, 4, 12, 6, 3],
#         "base_ch_list": [1, 1, 2, 4, 4, 2, 1],
#         "end_ch_list": [1, 2, 4, 8, 4, 2, 1],
#
#         'out_chs': [NR_CLASS, 1],
#         "padding_list": [(1, 1, 1), (1, 1, 1), (1, 1, 1),
#                          (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)],
#         "checkpoint_layers": [0, 1, 1, 0, 1, 1, 0],
#         "dropout": 0.0,
#         "d_dim": 32,
#         "d_sim": 32,
#         "p_sim": 32,
#         "group_norm": False,
#         "merge_method": "softmax",
#         "non_local_drop_rate": 0.0,
#         "pool_strides": [1, 1, 1],
#         "upsample_ksize": (3, 3, 3),
#         "upsample_sf": (2, 2, 2)
#     },
#     "net2_param": {
#         "n_layers": 3,
#         "in_ch_list": [8, 1, 2, 4, 12, 6, 3],
#         "base_ch_list": [1, 1, 2, 4, 4, 2, 1],
#         "end_ch_list": [1, 2, 4, 8, 4, 2, 1],
#
#         'out_chs': [NR_CLASS, 1],
#         "padding_list": [(0, 0, 0), (0, 0, 0), (0, 0, 0),
#                          (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
#         "checkpoint_layers": [0, 1, 1, 0, 1, 1, 0],
#         "dropout": 0.0,
#         "d_dim": 32,
#         "d_sim": 32,
#         "p_sim": 32,
#         "group_norm": False,
#         "merge_method": "softmax",
#         "non_local_drop_rate": 0.0,
#         "pool_strides": [1, 1, 1],
#         "upsample_ksize": (3, 3, 3),
#         "upsample_sf": (2, 2, 2)
#     },
#     "ds_rate": 0.5,
#     "return_res": False,
# }

MODEL = {
    "net1_param": {
        "n_layers": 3,
        "in_ch_list": [1, 24, 48, 128, 384, 176, 72],
        "base_ch_list": [16, 24, 64, 128, 128, 48, 24],
        "end_ch_list": [24, 48, 128, 256, 128, 48, 24],

        'out_chs': [NR_CLASS, 1],
        "padding_list": [(1, 1, 1), (1, 1, 1), (1, 1, 1),
                         (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)],
        "checkpoint_layers": [0, 1, 1, 0, 1, 1, 0],
        "dropout": 0.0,
        "d_dim": 32,
        "d_sim": 32,
        "p_sim": 32,
        "group_norm": False,
        "merge_method": "softmax",
        "non_local_drop_rate": 0.0,
        "pool_strides": [1, 1, 1],
        "upsample_ksize": (3, 3, 3),
        "upsample_sf": (2, 2, 2)
    },
    "net2_param": {
        "n_layers": 3,
        "in_ch_list": [8, 32, 96, 192, 576, 288, 128],
        "base_ch_list": [24, 48, 96, 192, 192, 96, 48],
        "end_ch_list": [32, 96, 192, 384, 192, 96, 48],

        'out_chs': [NR_CLASS, 1],
        "padding_list": [(0, 0, 0), (0, 0, 0), (0, 0, 0),
                         (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
        "checkpoint_layers": [0, 1, 1, 0, 1, 1, 0],
        "dropout": 0.0,
        "d_dim": 32,
        "d_sim": 32,
        "p_sim": 32,
        "group_norm": False,
        "merge_method": "softmax",
        "non_local_drop_rate": 0.0,
        "pool_strides": [1, 1, 1],
        "upsample_ksize": (3, 3, 3),
        "upsample_sf": (2, 2, 2)
    },
    "ds_rate": 0.5,
    "return_res": False,
}

# MODEL = {
#     "net1_param": {
#         "n_layers": 3,
#         "in_ch_list": [1, 2, 4, 8, 24, 12, 6],
#         "base_ch_list": [2, 2, 4, 8, 8, 4, 2],
#         "end_ch_list": [2, 4, 8, 16, 8, 4, 2],
#
#         'out_chs': [NR_CLASS, 1],
#         "padding_list": [(1, 1, 1), (1, 1, 1), (1, 1, 1),
#                          (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)],
#         "checkpoint_layers": [0, 1, 1, 0, 1, 1, 0],
#         "dropout": 0.0,
#         "d_dim": 32,
#         "d_sim": 32,
#         "p_sim": 32,
#         "group_norm": False,
#         "merge_method": "softmax",
#         "non_local_drop_rate": 0.0,
#         "pool_strides": [1, 1, 1],
#         "upsample_ksize": (3, 3, 3),
#         "upsample_sf": (2, 2, 2)
#     },
#     "net2_param": {
#         "n_layers": 3,
#         "in_ch_list": [8, 2, 4, 8, 24, 12, 6],
#         "base_ch_list": [2, 2, 4, 8, 8, 4, 2],
#         "end_ch_list": [2, 4, 8, 16, 8, 4, 2],
#
#         'out_chs': [NR_CLASS, 1],
#         "padding_list": [(0, 0, 0), (0, 0, 0), (0, 0, 0),
#                          (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
#         "checkpoint_layers": [0, 1, 1, 0, 1, 1, 0],
#         "dropout": 0.0,
#         "d_dim": 32,
#         "d_sim": 32,
#         "p_sim": 32,
#         "group_norm": False,
#         "merge_method": "softmax",
#         "non_local_drop_rate": 0.0,
#         "pool_strides": [1, 1, 1],
#         "upsample_ksize": (3, 3, 3),
#         "upsample_sf": (2, 2, 2)
#     },
#     "ds_rate": 0.5,
#     "return_res": False,
# }

INITIALIZER = {
    "method": "engine.models.initializer.HeNorm",
    "mode": "fan_in"
}

# OPTIMIZER = {
#    "method" : "adabound.AdaBound",
#    "lr": 0.0001,
#    "final_lr": 0.01
# }
# OPTIMIZER = {
#    "method": "torch.optim.SGD",
#    "momentum": 0.9,
#    "lr": 0.0001,
# }

OPTIMIZER = {
    "method": "torch.optim.Adam",
    #"momentum": 0.9,
    "lr": 0.0001,
}


#OPTIMIZER = {
#    "method": "engine.models.optimizer.RAdam",
#    "lr": 0.0001,
#}

SCHEDULER = {
    "method": "torch.optim.lr_scheduler.ExponentialLR",
    "gamma": 0.9
}


LOSS_FUNC = {
    "method": "engine.metrics.supervision.GenralizedDiceTopK",
    "smooth": 0.1,
    "top_k": 0.3,
    "class_weights": None,
    "skip_channels": [],
    "cost_factors": [1.0, 1.0]
}


LOSS_FUNC1 = {
    "method": "engine.metrics.supervision.GenralizedDiceTopK",
    "smooth": 0.1,
    "top_k": 0.3,
    "skip_channels": [],
    "class_weights": None,
    "cost_factors": [1.0, 1.0]
}

LOSS_FUNC2 = {
    "method": "engine.metrics.supervision.BinaryDiceLoss",
    "smooth": 0.1,
}

LOSS_FUNC3 = {
    "method": "engine.metrics.supervision.GenralizedDiceTopK",
    "smooth": 0.1,
    "top_k": 0.3,
    "skip_channels": [],
    "class_weights": None,
    "cost_factors": [1.0, 1.0]
}

LOSS_FUNC4 = {
    "method": "engine.metrics.supervision.BinaryDiceLoss",
    "smooth": 1.0,
}

#
# LOSS_FUNC4 = {
#     "method": "engine.metrics.supervision.GenralizedDiceTopK",
#     "smooth": 0.1,
#     "top_k": 0.3,
#     "skip_channels": [],
#     "class_weights": None,
#     "cost_factors": [1.0, 1.0]
# }
#
# LOSS_FUNC5 = {
#     "method": "engine.metrics.supervision.BinaryDiceLoss",
#     "smooth": 1.0,
# }
#
# LOSS_FUNC6 = {
#     "method": "engine.metrics.supervision.GenralizedDiceTopK",
#     "smooth": 0.1,
#     "top_k": 0.3,
#     "skip_channels": [],
#     "class_weights": None,
#     "cost_factors": [1.0, 1.0]
# }

# OPTIMIZER = {
#     "method": "torch.optim.SGD",
#     "momentum": 0.9,
#     "lr": 0.0001,
# }
#
# # OPTIMIZER = {
# #    "method": "engine.models.optimizer.RAdam",
# #    "lr": 0.0001,
# # }
#
# # SCHEDULER = {
# #     "method": "torch.optim.lr_scheduler.ExponentialLR",
# #     "gamma": 0.9
# # }
#
# SCHEDULER = {
#     "method": "torch.optim.lr_scheduler.CyclicLR",
#     "base_lr": 0.00001,
#     "max_lr": 0.001,
#     "step_size_up": 1,
#     "step_size_down": 3
# }
#
# LOSS_FUNC = {
#     "method": "engine.metrics.supervision.GenralizedDice",
#     "smooth": 0.1,
#     "skip_channels": [],
#     # "class_weights": [CLASS_WEIGHTS[k] for k in sorted(CLASS_WEIGHTS.keys())],
#     "class_weights": None,
#     "use_smooth_weight": False
#
# }
#
#
# LOSS_FUNC1 = {
#     "method": "engine.metrics.supervision.GenralizedDice",
#     "smooth": 0.1,
#     "skip_channels": [],
#     # "class_weights": [CLASS_WEIGHTS[k] for k in sorted(CLASS_WEIGHTS.keys())],
#     "class_weights": None,
#     "use_smooth_weight": False
#
# }
#
# LOSS_FUNC2 = {
#     "method": "engine.metrics.supervision.BinaryDiceLoss",
#     "smooth": 0.1,
#
# }
#
# # LOSS_FUNC3 = {
# #     "method": "engine.metrics.supervision.BinaryDiceLoss",
# #     "smooth": 0.0,
# #
# # }
#
# LOSS_FUNC3 = {
#     "method": "engine.metrics.supervision.GenralizedDice",
#     "smooth": 0.1,
#     "skip_channels": [],
#     # "class_weights": [CLASS_WEIGHTS[k] for k in sorted(CLASS_WEIGHTS.keys())],
#     "class_weights": None,
#     "use_smooth_weight": False
# }
#
# LOSS_FUNC4 = {
#     "method": "engine.metrics.supervision.GenralizedDice",
#     "smooth": 0.0,
#     "skip_channels": [],
#     # "class_weights": [CLASS_WEIGHTS[k] for k in sorted(CLASS_WEIGHTS.keys())],
#     "class_weights": None,
#     "use_smooth_weight": False
# }
#
# LOSS_FUNC5 = {
#     "method": "engine.metrics.supervision.BinaryDiceLoss",
#     "smooth": 1.0,
# }
#
# LOSS_FUNC6 = {
#     "method": "engine.metrics.supervision.GenralizedDice",
#     "smooth": 0.1,
#     "skip_channels": [],
#     # "class_weights": [CLASS_WEIGHTS[k] for k in sorted(CLASS_WEIGHTS.keys())],
#     "class_weights": None,
#     "use_smooth_weight": False
# }

# loggers.
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'
        },
        "file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": "info.log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', "file_handler"],
            'level': 'INFO',
            'propagate': True
        },
    }
}

PROCESSOR_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'
        },
        "file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": "processor_info.log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', "file_handler"],
            'level': 'INFO',
            'propagate': True
        },
    }
}

# visualization
INSPECT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'
        },
        "file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": "inspect_info.log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', "file_handler"],
            'level': 'INFO',
            'propagate': True
        },
    }
}

# visualization


VISUALIZATION_COLOR_TABLE = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (100, 0, 0),
    (100, 100, 0),
    (100, 100, 100),
    (50, 200, 0),
    (50, 200, 200),
    (50, 50, 200),
    (200, 50, 200),
    (50, 200, 50),
]

VISUALIZATION_ALPHA = 0.3
VISUALIZATION_SPARSENESS = 150
VISUALIZATION_PORT = 6012

CRF_PARAM = {
    "sxyz": 15,
    "srgb": 10,
    "comp_bi": 8,
    "comp_gaussian": 6,
    "iteration": 2
}

INSPECT_PARAMETERS = {
    "watch_layers": {
        "unet1.bg": {"input": True, "stride": 1},
        "unet1.non_local_module": {"input": False, "stride": 1},
        "unet2.bg": {"input": False, "stride": 1},
        "unet2.non_local_module": {"input": False, "stride": 1}
    },
}
