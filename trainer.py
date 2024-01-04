from torch import optim
from dataset_domain import CMRDataset
from torch.utils.data.dataloader import DataLoader
from losses import *
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os


def train_net(net, options):
    trainset = CMRDataset(options.data_dir, mode='train')
    trainLoader = DataLoader(trainset, batch_size=options.batch_size, shuffle=True, drop_last=True)
    testset = CMRDataset(options.data_dir, mode='valid')
    testLoader = DataLoader(testset, batch_size=1, shuffle=False)
    writer = SummaryWriter(options.log_path + options.unique_name)
    optimizer = optim.Adam(net.parameters(), lr=options.lr, betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=options.weight_decay, amsgrad=False)
    scheduler_1 = optim.lr_scheduler.StepLR(optimizer, step_size=options.step_size, gamma=options.gamma)
    myloss = Myloss()
    best_mae = 5
    for epoch in range(options.epochs):
        print('Starting epoch {}/{}'.format(epoch + 1, options.epochs))
        epoch_loss = 0
        print('current lr:', optimizer.param_groups[0]['lr'])
        for i, (img1, label) in enumerate(trainLoader, 0):
            img1 = img1.cuda()
            label = label.cuda()
            label64 = F.interpolate(label, scale_factor=(0.25, 0.25))
            label128 = F.interpolate(label, scale_factor=(0.5, 0.5))
            end = time.time()
            net.train()
            optimizer.zero_grad()
            result, b128, b64 = net(img1)
            loss1 = myloss(label, result)
            loss2 = myloss(label128, b128)
            loss3 = myloss(label64, b64)
            loss = 0.6 * loss1 + 0.2 * loss2 + 0.2 * loss3
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_time = time.time() - end
            # print('batch loss: %.5f, batch_time:%.5f' % (loss.item(), batch_time))
        print('[epoch %d] epoch loss: %.5f' % (epoch + 1, epoch_loss / (i + 1)))
        writer.add_scalar('Train/Loss', epoch_loss / (i + 1), epoch + 1)
        scheduler_1.step()
        if os.path.isdir('%s%s/' % (options.cp_path, options.unique_name)):
            pass
        else:
            os.mkdir('%s%s/' % (options.cp_path, options.unique_name))
        if epoch % 20 == 0 or epoch > 200:
            torch.save(net.state_dict(), '%s%s/epoch%d.pth' % (options.cp_path, options.unique_name, epoch))
        if (epoch + 1) > 90 or (epoch + 1) % 10 == 0:
            mae = validation(net, testLoader)
            if mae <= best_mae:
                best_mae = mae
                torch.save(net.state_dict(), '%s%s/best.pth' % (options.cp_path, options.unique_name))
                print('save done')
            print('min_maeloss: %.5f' % best_mae)
            print('current_mean_loss: %.5f' % mae)


def validation(net, test_loader):
    net.eval()
    mae = []
    maeloss = nn.L1Loss(reduction='mean')
    with torch.no_grad():
        for i, (data1, label) in enumerate(test_loader):
            inputs1 = data1.cuda()
            labels = label.cuda()
            pred, _, _ = net(inputs1)
            val = maeloss(labels, pred).float().cpu
            mae.append(val().item())
    mae_mean = np.mean(mae)
    return mae_mean
