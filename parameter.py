import os

################# Training #################
# Your path for pretrained vgg model
load_model = './pretrained_models/vgg16_20M.caffemodel.pth'
# Your path for COCO9213
img_root_coco = './Data/COCO9213-os/img/'
gt_root_coco = './Data/COCO9213-os/gt/'
# Your path for DUTS Class
img_root = './Data/DUTS_class/img/'
gt_root = './Data/DUTS_class/gt/'
# Your path for our synethsis data
img_syn_root = './Data/DUTS_class_syn/img_png_seamless_cloning_add_naive/img/'
img_ReverseSyn_root = './Data/DUTS_class_syn/img_png_seamless_cloning_add_naive_reverse/img/'
gt_ReverseSyn_root = './Data/DUTS_class_syn/img_png_seamless_cloning_add_naive_reverse/gt/'

# save model path
save_model_dir = './checkpoint/'
if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)

# settings
gpu_id = "0"
max_num = 14
dec_channels = 64

img_size = 256
scale_size = 288
batch_size = 1
lr = 0.01
epochs = 300
train_steps = 40000
lr_decay_gamma = 0.1
stepvalue1 = 20000
stepvalue2 = 30000
loss_weights = [1, 0.8, 0.8, 0.5, 0.5, 0.5]
bn_momentum = 0.001


################# Testing #################
# save output path
save_test_path_root = './Preds/'
if not os.path.exists(save_test_path_root):
    os.makedirs(save_test_path_root)

# testing your own trained model
test_model = 'iterations40000.pth'
test_model_dir = save_model_dir + test_model

# testing our pretrained CADC model
# test_model_dir = 'checkpoint/CADC.pth'

# test dataset path
test_dir_img = ['./Data/CoCA/image',
                './Data/CoSal2015/Image',
                './Data/CoSOD3k/Image',
                './Data/MSRC/Image',
                ]

