import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
from collections import OrderedDict
from captcha.image import ImageCaptcha


class Arguments:

    def __init__(self):
        # train
        self.resume = True
        self.ckpt = "best.pth"
        if self.resume and not self.ckpt:
            raise ValueError("if training for resume, the ckpt path must be set")
        self.epoch = 10
        self.batch_size = 128
        self.lr = 1e-3
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # dataset
        self.characters = '-' + '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.width = 192
        self.height = 64
        self.num_classes = len(self.characters)
        self.txt_length = 4  # the length of captcha
        self.seq_length = 12  # seq_length >= 2 * txt_length + 1


class CaptchaDataset(Dataset):

    def __init__(self, characters, length,
                 width, height, input_length, label_length):
        super(CaptchaDataset, self).__init__()
        self.characters = characters
        self.length = length
        self.width = width
        self.height = height
        self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        random_str = ''.join([random.choice(self.characters[1:]) for j in range(self.label_length)])
        image = to_tensor(self.generator.generate_image(random_str))
        target = torch.tensor([self.characters.find(x) for x in random_str], dtype=torch.long)
        input_length = torch.full(size=(1, ), fill_value=self.input_length, dtype=torch.long)  # tensor([12])
        target_length = torch.full(size=(1, ), fill_value=self.label_length, dtype=torch.long)  # tensor([4])
        return image, target, input_length, target_length


class Model(nn.Module):
    def __init__(self, n_classes, input_shape=(3, 64, 128)):
        super().__init__()
        self.input_shape = input_shape
        channels = [32, 64, 128, 256, 256]
        layers = [2, 2, 2, 2, 2]
        kernels = [3, 3, 3, 3, 3]
        pools = [2, 2, 2, 2, (2, 1)]
        modules = OrderedDict()

        def cba(name, in_channels, out_channels, kernel_size):
            modules[f'conv{name}'] = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(1, 1) if kernel_size == 3 else 0)
            modules[f'bn{name}'] = nn.BatchNorm2d(out_channels)
            modules[f'relu{name}'] = nn.ReLU(inplace=True)

        last_channel = 3
        for block, (n_channel, n_layer, n_kernel, k_pool) in enumerate(zip(channels, layers, kernels, pools)):
            for layer in range(1, n_layer + 1):
                cba(f'{block + 1}{layer}', last_channel, n_channel, n_kernel)
                last_channel = n_channel
            modules[f'pool{block + 1}'] = nn.MaxPool2d(k_pool)
        modules[f'dropout'] = nn.Dropout(0.25, inplace=True)

        self.cnn = nn.Sequential(modules)
        self.lstm = nn.LSTM(input_size=self.infer_features(), hidden_size=128, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(in_features=256, out_features=n_classes)

    def infer_features(self):
        x = torch.zeros((1,) + self.input_shape)
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x.shape[1]

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


class Trainer:

    def __init__(self,
                 args: Arguments = None,
                 model=None,
                 optimizer=None,
                 scheduler=None
                 ):
        self.args = args
        self.model = model

        if args.resume and args.ckpt:
            self.load(args.ckpt)

        self.model = model.to(self.args.device)

        if optimizer is None:
            raise ValueError("optimizer is None, please set a optimizer!")

        self.scheduler = scheduler

    def train(self, train_data_loader, test_data_loader):
        self.model.train()
        for epoch in range(1, self.args.epoch + 1):
            with tqdm(train_data_loader) as pbar:
                loss_mean = 0
                acc_mean = 0
                for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
                    data, target = data.to(self.args.device), target.to(self.args.device)

                    # zero grad
                    optimizer.zero_grad()
                    output = self.model(data)

                    output_log_softmax = F.log_softmax(output, dim=-1)
                    loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)

                    # calc gradient
                    loss.backward()

                    # update gradient
                    optimizer.step()

                    if self.scheduler:
                        self.scheduler.step()

                    loss = loss.item()
                    acc = calc_acc(self.args.characters, target, output)

                    if batch_index == 0:
                        loss_mean = loss
                        acc_mean = acc

                    loss_mean = 0.1 * loss + 0.9 * loss_mean
                    acc_mean = 0.1 * acc + 0.9 * acc_mean

                    pbar.set_description(f'Epoch: {epoch}/{self.args.epoch}, Train Loss: {loss_mean:.4f} Train Acc: {acc_mean:.4f} ')
            # test
            self.valid(valid_data_loader=test_data_loader)

    def valid(self, valid_data_loader):
        self.model.eval()
        for epoch in range(1, self.args.epoch + 1):
            with tqdm(valid_data_loader) as pbar, torch.no_grad():
                loss_sum = 0
                acc_sum = 0
                for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
                    data, target = data.to(args.device), target.to(args.device)

                    output = self.model(data)
                    output_log_softmax = F.log_softmax(output, dim=-1)
                    loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)

                    loss = loss.item()
                    acc = calc_acc(self.args.characters, target, output)

                    loss_sum += loss
                    acc_sum += acc

                    loss_mean = loss_sum / (batch_index + 1)
                    acc_mean = acc_sum / (batch_index + 1)

                    pbar.set_description(f'Test : {epoch}/{self.args.epoch}, Test Loss: {loss_mean:.4f} Test Acc: {acc_mean:.4f} ')

    def save(self, ckpt="best.pth"):
        torch.save(self.model.state_dict(), ckpt)

    def load(self, ckpt_path=None):
        self.model.load_state_dict(torch.load(ckpt_path, map_location=args.device))


def calc_acc(vocab, target, output):
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    target = target.cpu().numpy()
    output_argmax = output_argmax.cpu().numpy()
    a = np.array([decode_target(vocab=vocab, sequence=true) == decode(vocab=vocab, sequence=pred) for true, pred in zip(target, output_argmax)])
    return a.mean()


def decode_target(vocab, sequence):
    return ''.join([vocab[x] for x in sequence]).replace(' ', '')


def decode(vocab, sequence):
    a = ''.join([vocab[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != vocab[0] and x != a[j + 1]])
    if len(s) == 0:
        return ''
    if a[-1] != vocab[0] and s[-1] != a[-1]:
        s += a[-1]
    return s


if __name__ == "__main__":
    args = Arguments()

    train_set = CaptchaDataset(args.characters, 1000 * args.batch_size, args.width, args.height, args.seq_length,
                               args.txt_length)
    valid_set = CaptchaDataset(args.characters, 100 * args.batch_size, args.width, args.height, args.seq_length,
                               args.txt_length)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=12)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=12)

    model = Model(args.num_classes, input_shape=(3, args.height, args.width))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
    trainer = Trainer(args=args,
                      model=model,
                      optimizer=optimizer,
                      scheduler=None)

    trainer.train(train_data_loader=train_loader, test_data_loader=valid_loader)
    trainer.save("best.pth")

    dataset = CaptchaDataset(args.characters, 1, args.width, args.height, args.seq_length, args.txt_length)
    image, target, input_length, label_length = dataset[0]

    model.eval()
    output_argmax = None
    do = True
    while do or decode_target(args.characters, target) == decode(args.characters, output_argmax[0]):
        do = False
        image, target, input_length, label_length = dataset[0]
        print('true:', decode_target(args.characters, target))

        output = model(image.unsqueeze(0).cuda())
        output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
        print('pred:', decode(args.characters, output_argmax[0]))
    to_pil_image(image)

