import os
import torch

# SAM_CHECKPOINT_PATH = os.path.join('..','..', 'tb_classification', 'weights', 'sam_vit_h_4b8939.pth')

SAM_CHECKPOINT_PATH = os.path.join('..','..', 'SAM_WEIGHTS', 'sam_vit_h_4b8939.pth')


SAM_ENCODER_VERSION = "vit_h"
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# DEVICE = torch.device('cpu')

DATA_OUT = os.path.join('..', 'outputs/')

DATA = os.path.join('..', 'data/ted_long/')


# Name the output folder containing measurements here
NAME = 'TED_LONG_SAM_04012024'
CSV_NAME_SAM = NAME +'_SAM_pix.csv'
CSV_NAME_MP = NAME + '_MP_pix.csv'
CSV_NAME_GT = NAME + '_GT_pix.csv'
CSV_NAME_MAE = NAME + '_mm_MAE.csv'

FLIP = False

# XML_NAME = 'normal_ground_truth.xml'
XML_NAME = 'cfd1-300.xml'


# configure plotting settings
PLOT_CONFIG = {
    'outputs': {
        'iris_outline': True,
        'pupil_outline': False,
        'distance_annotation': True,
        'masks': {
            'individual': False,
            'total': True
        },
        'boxes': True,
        'cropped_image': False,
    },

    'paths': {
        'pupil_circle': f'../outputs/pupil_circle/{NAME}',
        'iris_circle': f'../outputs/iris_circle/{NAME}',
        'annotation': f'../outputs/distance_annotation/{NAME}',
        'l_iris_mask':'../outputs/mask_annotation/left_iris',
        'r_iris_mask':'../outputs/mask_annotation/right_iris',
        'left_pupil_mask': '../outputs/mask_annotation/left_pupil',
        'right_pupil_mask':'../outputs/mask_annotation/right_pupil',
        'l_sclera_mask':'../outputs/mask_annotation/left_sclera',
        'r_sclera_mask':'../outputs/mask_annotation/right_sclera',
        'l_brow_mask':'../outputs/mask_annotation/left_brow',
        'r_brow_mask' :'../outputs/mask_annotation/right_brow',
        'total' : f'../outputs/mask_annotation/total/{NAME}',
        'boxes' : '../outputs/bounding_boxes/',
        'crop_img_out' : '../outputs/cropped_images/',
    }}

# hyper parameters for tuning
HYPERPARAMETERS = {
    'preprocessing': ['gamma_correction', 'adaptive_histogram', 'contrast', 'none'],
    'gamma_values': [.4, 0.6, 0.8],
    'clipLimit': [3.0, 4.0],  
    'tileGridSize': [(6, 6), (8, 8), (10, 10)],  
    'jitter_percent': [0, .05, .1, .15,  .2], 
    'mask_percent': [0, 0.1, 0.2, 0.3, 0.4, 0.5]
}

# setup, only gets used when not tuning
SETUP = {
    'iris': {   
        'preprocess': { 
            'method': 'none', 
            'params': {
                'gamma': .8,  
                'jitter_percent': 0,  
                'mask_percent': .5  
            }
        }
    },
    'pupil': {
        'preprocess': { 
            'method': 'none',  
            'params': { 
                'jitter_percent': 0,  
                'mask_percent': .5
            }
        }
    },
    'sclera': {
        'preprocess': { 
            'method': 'none',  
            'params': {
                'jitter_percent': 0,  
                'mask_percent': .5
            }
        }
    },      
    'brows': {
        'preprocess': { 
            'method': 'none',  
            'params': {
                'jitter_percent': 0,  
                'mask_percent': .5
            }
        }
    }
}

ANALYZE_CONFIG = {
    'epicmap': False,
    'SAM': True,
    'compare':True,
    'ted_long': True,
    # 'preprocess' : True, # Toggle this for any image preprocessing step
    'tuning_mode': False, # toggle this for tuning
    'measure' : True, # turns off measurement mode when false and only shows masks and bbs. for troubleshooting
    'mp_canthi' : False, # use mp canthi points for analysis
    'jitter_percent': 0, # 
    'mask_percent': .5, # 
    'image_alter': { # this will only be accessible if preprocess is set to true
        'method': 'none',  # Or 'gamma_correction', 'adaptive_histogram', 'contrast'
        'params': {
            'gamma': 8.0,  # Used if method is 'gamma_correction'
            'clipLimit': 4.0,  # Used if method is 'adaptive_histogram'
            'tileGridSize': (8, 8)  # Used if method is 'adaptive_histogram'
        }
    }
}


# set up whether or not to write csv file 
WRITE_CONFIG = {
    'write' : True,
    # 'name' : CSV_NAME,
    'path' : f'../outputs/csvs/{NAME}/',
    'tuning_path' : '../outputs/param_tuning/',
    'xml_path' : '../data/ground_truth/'
}



# Function to create directories if they don't exist
def create_directories(config):
    for key, path in config.items():
        if 'path' in key or key in ['pupil_circle', 'iris_circle', 'annotation', 'total']:
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Directory created: {path}")
            else:
                print(f"Directory already exists: {path}")

# Check and create directories for WRITE_CONFIG and PLOT_CONFIG
create_directories(WRITE_CONFIG)
create_directories(PLOT_CONFIG['paths'])



#### THESE GET AUTOMATICALLY HANDLED
# Turn off measure mode if doing hyperparameter tuning
if ANALYZE_CONFIG['tuning_mode']:
    ANALYZE_CONFIG['measure'] = False,
    WRITE_CONFIG['Write']= False
    ANALYZE_CONFIG['epicmap'] = False,
    ANALYZE_CONFIG['compare'] = False

# Turn off plotting if not measuring
if not ANALYZE_CONFIG['measure']:
    PLOT_CONFIG['outputs']['iris_outline'] = False
    PLOT_CONFIG['outputs']['pupil_outline'] = False
    PLOT_CONFIG['outputs']['distance_annotation'] = False
    WRITE_CONFIG['Write']= False
