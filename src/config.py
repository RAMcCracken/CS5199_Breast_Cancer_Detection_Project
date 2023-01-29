"""
Variables set by the command line arguments dictating which parts of the program to execute.
Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
"""

# Constants
RANDOM_SEED = 111
IMG_SIZE = (160,160) 


output_path = '../output/'  # Output path for figures and images
saved_models = '../saved_models/'
dataset = "CBIS-DDSM"       # The dataset to use.
preproc = False             # Specifies whether to trigger the preprocessing pipeline 
data_augmentation = False   # Specifies whether to implement data augmentation
mammogram_type = "all"      # The type of mammogram (Calc or Mass).
model = "MobileNet"         # The model to use.
run_mode = "training"       # The type of running mode, either training or testing.
learning_rate = 1e-4        # The learning rate with the pre-trained ImageNet layers frozen.
fine_tune_learning_rate = 1e-5
max_epoch_frozen = 100      # Max number of epochs when original CNN layers are frozen.
max_epoch_unfrozen = 50     # Max number of epochs when original CNN layers are unfrozen.
verbose_mode = False        # Boolean used to print additional logs for debugging purposes.
name = "Sample"             # Name of experiment.
image_size = 160            # Resolution to resize images to in pixels
batch_size = 32             # Batch size 
hyperparam_tuning = False   # Turn on hyperparameter tuning 
test_mode = False           # In testmode the model entered in the name parameter is tested and evaluation results are given 
# is_grid_search = False    # Run the grid search algorithm to determine the optimal hyper-parameters for the model.
