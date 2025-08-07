import argparse
import numpy as np
import random
import time
import torch
from dataloader import IEMOCAPRobertaDataset
from model import ECERC
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from loss import Loss
import warnings
warnings.filterwarnings("ignore")


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    # np.random.shuffle(idx)  # shuffle for training data
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_bert_loaders(path=None, batch_size=32, num_workers=0, pin_memory=False, valid_rate=0.1):
    trainset = IEMOCAPRobertaDataset(train=True)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid_rate)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPRobertaDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_f, dataloader, train_flag=False, optimizer=None, cuda_flag=False, feature_type='text',
                        target_names=None):
    assert not train_flag or optimizer != None
    losses, preds, labels = [], [], []
    if train_flag:
        model.train()
    else:
        model.eval()

    for step, data in enumerate(dataloader):
        if train_flag:
            optimizer.zero_grad()

        emo_roberta, sem_roberta, audio, vision, qmask, umask, label2 = [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]
        seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        if args.feature_type == "multi":
            emo_roberta = torch.cat([emo_roberta,audio, vision], dim=-1)
            sem_roberta = torch.cat([sem_roberta], dim=-1)

        log_prob = model(emo_roberta,sem_roberta, qmask, umask, seq_lengths)

        label = torch.cat([label2[j][:seq_lengths[j]] for j in range(len(label2))])
        loss = loss_f(log_prob, label)

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train_flag:
            loss.backward()
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), float('nan'), [], []

    labels = np.array(labels)
    preds = np.array(preds)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    all_matrix = []
    all_matrix.append(metrics.classification_report(labels, preds, target_names=target_names, digits=4))
    all_matrix.append(metrics.confusion_matrix(labels, preds))

    cr = classification_report(labels, preds, target_names=target_names, digits=4)

    return avg_loss, avg_accuracy, avg_fscore, all_matrix, [labels, preds], cr


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--feature_type', type=str, default='multi', help='feature type multi/text/acouf')

    parser.add_argument('--data_dir', type=str, default='../data/iemocap/iemocap_features_roberta.pkl', help='dataset dir: iemocap_features_roberta.pkl')

    parser.add_argument('--load_model_state_dir', type=str, default='ECERC_MODEL.pkl', help='load model state dir')

    parser.add_argument('--base_layer', type=int, default=1, help='the number of base model layers,1/2')

    parser.add_argument('--batch_size', type=int, default=64, metavar='BS', help='batch size')

    parser.add_argument('--valid_rate', type=float, default=0.1, metavar='valid_rate', help='valid rate: 0.1')

    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--class_weight', action='store_true', default=True, help='use class weights')

    parser.add_argument('--wp', type=float, default=1, help='wp')

    parser.add_argument('--Dataset', type=str, default="IEMOCAP", help='DATASET')

    args = parser.parse_args()

    batch_size, data_path, base_layer, feature_type = \
        args.batch_size, args.data_dir, args.base_layer, args.feature_type
    cuda_flag = torch.cuda.is_available() and not args.no_cuda

    # IEMOCAP dataset
    n_classes, n_speakers, hidden_size, input_size = 6, 2, 128, None
    target_names = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
    # perform 1/weight according to the weight of each label in training data
    class_weights = torch.FloatTensor([1 / 0.087178797, 1 / 0.145836136, 1 / 0.229786089, 1 / 0.148392305, 1 / 0.140051123, 1 / 0.24875555])

    feat2dim = {'IS10': 1582, '3DCNN': 512, 'textCNN': 100, 'bert': 768, 'denseface': 342, 'MELD_text': 600,
                'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.Dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024

    if feature_type == 'text':
        input_size = D_text
    elif feature_type == "multi":
        input_size = D_text + D_audio + D_visual
    else:
        print('Error: feature_type not set.')
        exit(0)

    model = ECERC(args,d_t=D_text, d_a=D_audio, d_v=D_visual,
                        base_layer=base_layer,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        n_speakers=n_speakers,
                        n_classes=n_classes,
                        cuda_flag=cuda_flag)
    if cuda_flag:
        print('Running on GPU')
        class_weights = class_weights.cuda()
        model.cuda()
    else:
        print('Running on CPU')

    name = 'ECERC'
    print("The model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    print('Running on the {} features........'.format(feature_type))
    loss_f = Loss(alpha=class_weights if args.class_weight else None)
    train_loader, valid_loader, test_loader = get_IEMOCAP_bert_loaders(path=data_path, batch_size=batch_size, num_workers=0, valid_rate=args.valid_rate)

    start_time = time.time()
    model.load_state_dict(torch.load(args.load_model_state_dir))
    test_loss, test_acc, test_fscore, test_metrics, _, test_cr = train_or_eval_model(model=model, loss_f=loss_f,
                                                                                     dataloader=test_loader,
                                                                                     cuda_flag=cuda_flag,
                                                                                     feature_type=feature_type,
                                                                                     target_names=target_names)

    print('test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.format(test_loss, test_acc, test_fscore, round(time.time() - start_time, 2)))
    print(test_metrics[0])
    print(test_metrics[1])

