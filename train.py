from model.DRA_UNet import DRA_UNet
from optparse import OptionParser
from trainer import *
import sys


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=300, type='int', help='number of epochs')
    parser.add_option('-s', '--step-size', dest='step_size', default=40, type='int', help='number of decay epochs')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=25, type='int', help='batch size')
    parser.add_option('-g', '--gamma', dest='gamma', default=0.4, type='float',
                      help='decay coefficient of learning rate')
    parser.add_option('-c', '--resume', type='str', dest='load', default=False, help='load pretrained model')
    parser.add_option('-l', '--learning-rate', dest='lr', default=3e-4, type='float', help='learning rate')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='./data/', type='str', help='data address')
    parser.add_option('-p', '--checkpoint-path', type='str', dest='cp_path', default='./checkpoint/',
                      help='checkpoint path')
    parser.add_option('-o', '--log-path', type='str', dest='log_path', default='./log/', help='log path')
    parser.add_option('-m', type='str', dest='model', default='DRA_UNet', help='use which model')
    parser.add_option('-u', '--unique_name', type='str', dest='unique_name', default='exp1',
                      help='unique experiment name')
    parser.add_option('--weight_decay', type='float', dest='weight_decay', default=1e-4)
    parser.add_option('--gpu', type='str', dest='gpu', default='0', help='use which gpu')
    options, args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
    print('Using model:', options.model)
    if options.model == 'DRA_UNet':
        net = DRA_UNet()
    else:
        raise NotImplementedError(options.model + " has not been implemented")
    if options.load:
        net.load_state_dict(torch.load(options.load))
        print('Model loaded from {}'.format(options.load))
    param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(net)
    print(param_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    train_net(net, options)
    print('done')
    sys.exit(0)
