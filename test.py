import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from dataset import get_loader
import transforms as trans
import time
from parameter import *
from CoSODNet import CoSODNet


def test_net(net, batch_size):

    for dataset_idx in range(len(test_dir_img)):
        dataset_name = test_dir_img[dataset_idx].split('/')[-2]

        print('testing {}'.format(test_dir_img[dataset_idx]))

        test_loader = get_loader(test_dir_img[dataset_idx], img_size, 1, gt_root=None, mode='test', num_thread=1)
        print('''
           Starting testing:
               Batch size: {}
               Testing size: {}
           '''.format(batch_size, len(test_loader.dataset)))
        iter_num = len(test_loader.dataset) // batch_size
        for i, data_batch in enumerate(test_loader):
            print('{}/{}'.format(i, len(test_loader.dataset)))
            if (i + 1) > iter_num: break

            inputs = Variable(data_batch[0].squeeze(0).cuda())
            subpaths = data_batch[1]
            ori_sizes = data_batch[2]

            _, _, _, _, _, y_pred = net(inputs)

            output = F.sigmoid(y_pred)

            saved_root = save_test_path_root + dataset_name
            # save_final_path = saved_root + '/CADC_' + test_model + '/' + subpaths[0][0].split('/')[0] + '/'
            save_final_path = saved_root + '/CADC/' + subpaths[0][0].split('/')[0] + '/'
            os.makedirs(save_final_path, exist_ok=True)

            for inum in range(output.size(0)):
                pre = output[inum, :, :, :].data.cpu()

                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][1].item(),
                            ori_sizes[inum][0].item())

                transform = trans.Compose([
                    trans.ToPILImage(),
                    trans.Scale(ori_size)
                ])
                outputImage = transform(pre)
                filename = subpath.split('/')[1]
                outputImage.save(os.path.join(save_final_path, filename))


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    cudnn.benchmark = True

    start = time.time()

    net = CoSODNet(3, mode='test')
    net.cuda()
    print('Model has constructed!')

    net.load_state_dict(torch.load(test_model_dir))
    print('Model loaded from {}'.format(test_model_dir))

    test_net(net, 1)
    print('total time {}'.format(time.time()-start))
