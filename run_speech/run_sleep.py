import os
import argparse
import utils_sleep
import torch
import numpy as np
import time
from model_sleep import SleepBase, SleepDev
log_path='C:\\Users\\dhc40\\manyDG\\run_speech\\Result'
from torch.utils.tensorboard import SummaryWriter
writer2_acc = SummaryWriter(log_dir=log_path)


def accuracy_score(y_true, y_pred):
    return np.sum(y_pred == y_true) / len(y_true)

def confusion_matrix(actual, pred):
    actual = np.array(actual)
    pred = np.array(pred)
    n_classes = int(max(np.max(actual), np.max(pred)) + 1)
    confusion = np.zeros([n_classes, n_classes])
    for i in range(n_classes):
        for j in range(n_classes):
            confusion[i, j] = np.sum((actual == i) & (pred == j))
    return confusion.astype('int')

def weighted_f1(gt, pre):
    confusion = confusion_matrix(gt, pre)
    f1_ls = []
    for i in range(confusion.shape[0]):
        if confusion[i, i] == 0:
            f1_ls.append(0)
        else:
            precision_tmp = confusion[i, i] / confusion[i].sum()
            recall_tmp = confusion[i, i] / confusion[:, i].sum()
            f1_ls.append(2 * precision_tmp * recall_tmp / (precision_tmp + recall_tmp))
    return np.mean(f1_ls)

def cohen_kappa_score(y1, y2):
    confusion = confusion_matrix(y1, y2)
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    w_mat = np.ones([n_classes, n_classes], dtype=int)
    w_mat.flat[:: n_classes + 1] = 0

    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    return 1 - k


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="choose from base, dev")
    parser.add_argument('--cuda', type=int, default=0, help="which cuda")
    parser.add_argument('--N_pat', type=int, default=100, help="number of patients")
    parser.add_argument('--dataset', type=str, default="sleep", help="dataset name")
    parser.add_argument('--MLDG_threshold', type=int, default=1024, help="threshold for MLDG")
    parser.add_argument('--epochs', type=int, default=1, help="N of epochs")
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    print ('device:', device)
    
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    path = "C:\\Users\\dhc40\\manyDG\\data\\sleep\\test_pat_map_sleep.pkl"
    if os.path.exists(path):
        train_pat_map, test_pat_map, val_pat_map = utils_sleep.load()
    else:   
        train_pat_map, test_pat_map, val_pat_map = utils_sleep.load_and_dump()

    def trainloader_for_other():
        train_X = []

        for i, (_, X) in enumerate(train_pat_map.items()):
            if i == args.N_pat: break
            train_X += X
        train_loader = torch.utils.data.DataLoader(utils_sleep.SleepLoader(train_X),
                    batch_size=256, shuffle=True, num_workers=16)
        return train_loader

    def trainloader_for_adv():
        train_X, train_ID = [], []

        for i, (_, X) in enumerate(train_pat_map.items()):
            if i == args.N_pat: break
            train_X += X
            train_ID += [i for _ in X]
        train_loader = torch.utils.data.DataLoader(utils_sleep.SleepIDLoader(train_X, train_ID),
                    batch_size=256, shuffle=True, num_workers=16)
        return train_loader

    def trainloader_for_dev():
        train_X = []
        train_X_aux = []
        for i, (_, X) in enumerate(train_pat_map.items()):
            if i == args.N_pat: break
            np.random.shuffle(X)
            train_X += X[:len(X)//2 + 1]
            train_X_aux += X[-len(X)//2 - 1:]
        
        train_loader = torch.utils.data.DataLoader(utils_sleep.SleepDoubleLoader(train_X, train_X_aux),
                batch_size=256, shuffle=True, num_workers=16)
        return train_loader


    def valloader_for_all():
        val_X = []
        for _, X in val_pat_map.items():
            val_X += X
        val_loader = torch.utils.data.DataLoader(utils_sleep.SleepLoader(val_X),
                batch_size=256, shuffle=False, num_workers=16)
        return val_loader

    def testloader_for_all():
        test_X = []
        for _, X in test_pat_map.items():
            test_X += X
        test_loader = torch.utils.data.DataLoader(utils_sleep.SleepLoader(test_X),
                batch_size=256, shuffle=False, num_workers=16)
        return test_loader

    # load model
    if args.model == "base":
        train_loader = trainloader_for_other()
        model = SleepBase(device, args.dataset).to(device)
    elif args.model == "dev":
        train_loader = trainloader_for_other()
        model = SleepDev(device, args.dataset).to(device)

    test_loader = testloader_for_all()
    val_loader = valloader_for_all()

    model_name = (args.dataset + '_' + args.model + '_' + str(args.N_pat) + '_{}').format(time.time())
    print (model_name)

    test_array, val_array = [], []
    test_kappa_array, val_kappa_array = [], []
    test_f1_array, val_f1_array = [], []
    for i in range (args.epochs):
        tic = time.time()
        if args.model == "dev":
            train_loader_dev = trainloader_for_dev()
            model.train(train_loader_dev, device,i)
        else:
            model.train(train_loader, device,i)
        
        result, gt = model.test(test_loader, device)
        print ('{}-th test accuracy: {:.4}, kappa: {:.4}, weighted_f1: {:.4}, time: {}s'.format(
            i, accuracy_score(gt, result), cohen_kappa_score(gt, result), weighted_f1(gt, result), time.time() - tic))
        test_array.append(accuracy_score(gt, result))
        test_kappa_array.append(cohen_kappa_score(gt, result))
        test_f1_array.append(weighted_f1(gt, result))
        with open('C:\\Users\\dhc40\\manyDG\\log_new\\speech\\{}.log'.format(model_name), 'a') as outfile:
            print ('{}-th test accuracy: {:.4}, kappa: {:.4}, weighted_f1: {:.4}'.format(
                i, accuracy_score(gt, result), cohen_kappa_score(gt, result), weighted_f1(gt, result)), file=outfile)
        
        result_val, gt_val = model.test(val_loader, device)
        print ('{}-th val accuracy: {:.4}, kappa: {:.4}, weighted_f1: {:.4}, time: {}s'.format(
            i, accuracy_score(gt_val, result_val), cohen_kappa_score(gt_val, result_val), weighted_f1(gt_val, result_val), time.time() - tic))
        val_array.append(accuracy_score(gt_val, result_val))
        val_kappa_array.append(cohen_kappa_score(gt_val, result_val))
        val_f1_array.append(weighted_f1(gt_val, result_val))
        
        with open('C:\\Users\\dhc40\\manyDG\\log_new\\speech\\{}.log'.format(model_name), 'a') as outfile:
             print ('{}-th val accuracy: {:.4}, kappa: {:.4}, weighted_f1: {:.4}'.format(
                i, accuracy_score(gt_val, result_val), cohen_kappa_score(gt_val, result_val), weighted_f1(gt_val, result_val)), file=outfile)
        writer2_acc.add_scalars("Accuracy", {"test":accuracy_score(gt, result),
                                        "validation": accuracy_score(gt_val, result_val)}, i)    

        # save model
        torch.save(model.state_dict(), 'C:\\Users\\dhc40\\manyDG\\pre-trained\\speech{}-{}.pt'.format(i, model_name))
        print ()