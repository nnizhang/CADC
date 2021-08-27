import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from dataset import get_loader
import math
from parameter import *
from CoSODNet import CoSODNet
from apex import amp


def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
    fh = open(save_dir, 'a')
    epoch_total_loss = str(epoch_total_loss)
    epoch_loss = str(epoch_loss)
    fh.write('until_' + str(epoch) + '_run_iter_num' + str(whole_iter_num) + '\n')
    fh.write(str(epoch) + '_epoch_total_loss' + epoch_total_loss + '\n')
    fh.write(str(epoch) + '_epoch_loss' + epoch_loss + '\n')
    fh.write('\n')
    fh.close()


def adjust_learning_rate(optimizer, decay_rate=.1):

    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
    return optimizer


def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    fh = open(save_dir, 'a')
    fh.write('encode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('decode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('\n')
    fh.close()


def train_net(net):

    train_loader = get_loader(img_root, img_size, batch_size, gt_root, max_num=max_num, mode='train', num_thread=1,
                              pin=False)

    print('''
    Starting training:
        Train steps: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
    '''.format(train_steps, batch_size, lr, len(train_loader.dataset)))

    N_train = len(train_loader) * batch_size

    base_params = [params for name, params in net.named_parameters() if ("Encoder" in name)]
    other_params = [params for name, params in net.named_parameters() if ("Encoder" not in name)]

    optimizer = optim.SGD([{'params': base_params, 'lr': lr * 0.1},
                           {'params': other_params}], lr=lr, momentum=0.9, weight_decay=0.0005)

    net, optimizer = amp.initialize(net, optimizer, opt_level="O1")

    criterion = nn.BCEWithLogitsLoss()
    whole_iter_num = 0
    iter_num = math.ceil(len(train_loader.dataset) / batch_size)
    for epoch in range(epochs):

        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        print('epoch:{0}-------lr:{1}'.format(epoch + 1, lr))

        epoch_total_loss = 0
        epoch_loss = 0

        for i, data_batch in enumerate(train_loader):
            if (i + 1) > iter_num: break

            inputs = Variable(data_batch[0].squeeze(0).cuda())
            gts = Variable(data_batch[1].squeeze(0).cuda())
            gt_128 = Variable(data_batch[2].squeeze(0).cuda())
            gt_64 = Variable(data_batch[3].squeeze(0).cuda())
            gt_32 = Variable(data_batch[4].squeeze(0).cuda())

            for_loss6, for_loss5, for_loss4, for_loss3, for_loss2, for_loss1 = net(inputs)

            # loss
            loss6 = criterion(for_loss6, gt_32)
            loss5 = criterion(for_loss5, gt_32)
            loss4 = criterion(for_loss4, gt_32)
            loss3 = criterion(for_loss3, gt_64)
            loss2 = criterion(for_loss2, gt_128)
            loss1 = criterion(for_loss1, gts)

            total_loss = loss_weights[0] * loss1 + loss_weights[1] * loss2 + loss_weights[2] * loss3\
                         + loss_weights[3] * loss4 + loss_weights[4] * loss5 + loss_weights[5] * loss6

            epoch_total_loss += total_loss.cpu().data.item()
            epoch_loss += loss1.cpu().data.item()

            print('whole_iter_num: {0} --- {1:.4f} --- total_loss: {2:.6f} --- loss: {3:.6f}'.format((whole_iter_num + 1),
                                                     (i + 1) * batch_size / N_train, total_loss.item(), loss1.item()))

            optimizer.zero_grad()

            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()
            whole_iter_num += 1

            if whole_iter_num == train_steps:
                torch.save(net.state_dict(),
                           save_model_dir + 'iterations{}.pth'.format(train_steps))
                return

            if whole_iter_num == stepvalue1 or whole_iter_num == stepvalue2:
                optimizer = adjust_learning_rate(optimizer, decay_rate=lr_decay_gamma)
                save_dir = './loss.txt'
                save_lr(save_dir, optimizer)
                print('have updated lr!!')

        print('Epoch finished ! Loss: {}'.format(epoch_total_loss / iter_num))
        save_lossdir = './loss.txt'
        save_loss(save_lossdir, whole_iter_num, epoch_total_loss / iter_num, epoch_loss/iter_num, epoch+1)

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    cudnn.benchmark = True

    net = CoSODNet(3, mode='train')
    net.train()

    vgg_model = torch.load(load_model)
    net = net.init_parameters(vgg_model)
    print('Model loaded from {}'.format(load_model))

    net.cuda()
    train_net(net)
