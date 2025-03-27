
##### data paths #####

# TODO adapt your paths to folder that contains the celer and zuco folders
path_to_celer = '/data/lenbol/data/'  # e.g., path_to_celer = 'data/' if 'data/celer/...'
path_to_zuco = '/data/lenbol/data/' # e.g., path_to_zuco = 'data/' if 'data/zuco/...'
path_to_emtec = '/data/lenbol/data/' # e.g., path_to_copco = 'data/' if 'data/copco/...'
path_to_bsc = '/data/lenbol/data/' # e.g., path_to_copco = 'data/' if 'data/BSC/...'

PATH_TO_FIX = f'{path_to_celer}CELER/data_v2.0/sent_fix.tsv'
PATH_TO_IA = f'{path_to_celer}CELER/data_v2.0/sent_ia.tsv'
SUB_METADATA_PATH = f'{path_to_celer}/CELER/participant_metadata/metadata.tsv'
PATH_TO_EMTEC_FIX = f'{path_to_emtec}/EMTeC/fixations_corrected.csv'
PATH_TO_EMTEC_STIM = f'{path_to_emtec}/EMTeC/stimuli.csv'
PATH_TO_BSC_WORD = f'{path_to_bsc}/BSC/BSC.Word.Info.v2.xlsx'
PATH_TO_BSC_FIX = f'{path_to_bsc}/BSC/BSC.EMD.txt'


##### model paths #####

# TODO adapt your paths 

# training of original ScanDL for modular use with seq2seq fixdur module
SCANDL_MODULE_TRAIN_PATH = ''
SCANDL_MODULE_INF_PATH = ''

# training of the fixation duration module
FIXDUR_MODULE_TRAIN_PATH = ''
FIXDUR_MODULE_INF_PATH = ''

# training and inference of the diffusion-only architecture
DIFFUSION_ONLY_TRAIN_PATH = ''
DIFFUSION_ONLY_INF_PATH = ''


# names for EMTeC 
# training of original ScanDL for modular use with seq2seq fixdur module
SCANDL_MODULE_TRAIN_PATH_EMTEC = ''
SCANDL_MODULE_INF_PATH_EMTEC = ''

# training of the fixation duration module
FIXDUR_MODULE_TRAIN_PATH_EMTEC = ''
FIXDUR_MODULE_INF_PATH_EMTEC = ''


# names for BSC
# training of original ScanDL for modular use with seq2seq fixdur module
SCANDL_MODULE_TRAIN_PATH_BSC = ''
SCANDL_MODULE_INF_PATH_BSC = ''

# training of the fixation duration module
FIXDUR_MODULE_TRAIN_PATH_BSC = ''
FIXDUR_MODULE_INF_PATH_BSC = ''


# training of ScanDL 2.0 on all EMTeC data for paragraph-level ScanDL 2.0
COMPLETE_SCANDL_MODULE_TRAIN_PATH_EMTEC = ''
COMPLETE_FIXDUR_MODULE_TRAIN_PATH_EMTEC = ''


# training of ScanDL 2.0 on all CELER data for sentence-level ScanDL 2.0
COMPLETE_SCANDL_MODULE_TRAIN_PATH_CELER = ''
COMPLETE_FIXDUR_MODULE_TRAIN_PATH_CELER = ''
