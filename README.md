# CADC

source codes of "Summarize and Search: Learning Consensus-aware Dynamic Convolution for Co-Saliency Detection" by Ni Zhang, Nian Liu, Junwei Han, and Ling Shao.

created by Ni Zhang, email: nnizhang.1995@gmail.com


## Requirement
1. Pytorch 1.6.0
2. Torchvision 0.3.0
3. [apex](https://github.com/NVIDIA/apex)


## Training
1. Download the pretrained vgg model [[baidu pan](https://pan.baidu.com/s/19cik8v7Ix5YOo7sdEosp9A) fetch code: dyt4 | [Google drive](https://drive.google.com/drive/folders/1ZKK7Le5veXJVD3DZ8OdrO9CdqL2QOFAl?usp=sharing)] and put it in `pretrained_model/` directory.
2. Download training images, including original DUTS Class [[baidu pan](https://pan.baidu.com/s/1MG_aJ-Q_7xpxAOkxM8obrA) fetch code: 6jkx | [Google drive](https://drive.google.com/file/d/1XCeHbuuhy17Q8q6oT-uIWHDIZyGy7cFk/view?usp=sharing)], COCO9213, and our synthesis data.

## Testing 
1. Download [test datasets](http://dpfan.net/CoSOD3K/), including CoCA, CoSOD3k, CoSal150, and MSRC.
2. Modify path parameters in `parameter.py`.
3. Run `test.py` and the predictions are generated in `Preds/` directory.

## Evaluation
We use [evaluation tool](http://dpfan.net/wp-content/uploads/CoSalBenchmark-EvaluationTools.zip) from [the project](http://dpfan.net/CoSOD3K/).


## Testing on Our Pretrained CADC Model
1. Download our final model `CADC.pth` [[baidu pan](https://pan.baidu.com/s/11A0zw3rW2N_JXbZ4xlL6eQ) fetch code: 6sae| [Google drive](https://drive.google.com/file/d/18eCfpfIIWveFuQM60lsyhN1J6I4gLbyY/view?usp=sharing)]
2. Modify path parameters in `parameter.py`.
3. Run `test.py` and the predictions are generated in `Preds/` directory.

Our saliency maps can be download from [[baidu pan](https://pan.baidu.com/s/1bkCrqsOzNAgH-VSM0p7fMA) fetch code: i59u | [Google drive](https://drive.google.com/file/d/1LBBQOBeasn6O2caccs5t1e26ilv-d62h/view?usp=sharing)].




