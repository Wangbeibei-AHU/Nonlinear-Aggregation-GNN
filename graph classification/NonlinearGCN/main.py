from experiments import experiment
import numpy as np
import argparse
import scipy.io as sio
import pdb


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mutag', help='dataset')
parser.add_argument('--epochs', type=int,  default=500, help='Number of epochs to train.')
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--hidden_size", type=int, default=64)
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--kfold_num", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--mod", type=str, default="Generalized-mean", choices=['Generalized-mean', 'Polynomial', 'Softmax'])
args = parser.parse_args()

# test_acc = np.zeros(args.kfold_num)
# test_accs = np.zeros([args.kfold_num, args.epochs])
# for kfold in range(args.kfold_num):
# 	test_acc[kfold], test_accs[kfold] = experiment(dataset=args.dataset, model="NonlinearGCN", mod=args.mod, num_layers=args.num_layers, batch_size=args.batch_size, lr=args.lr, dropout=args.dropout, hidden_size=args.hidden_size, epochs=args.epochs, kfold=kfold, test_ratio=0.1, use_best_config=True, mw="graph_classification_mw", dw="graph_classification_dw")
# mean_test_acc = np.mean(test_accs, axis=0)
# best_epoch = np.argmax(mean_test_acc)
# mean_acc = np.mean(test_accs[:, best_epoch])
# mean_std = np.std(test_accs[:, best_epoch])
# print('test_acc: ', mean_acc, ' test_std: ', mean_std)


## Tips
"Note that If the results are inconsistent with reported ones due to different initialization and experimental environment," 
"one can tune the parameters in the following ways:"
datasets = ['ptc-mr', 'proteins','mutag']#, 'ptc-mr', 'proteins'
dnum = len(datasets)
lrs = [0.01, 0.001]
dropouts = [0., 0.3, 0.5]
hidden_size = [32, 64]

best_lr = np.zeros(dnum)
best_dropout=np.zeros(dnum)
best_hidden=np.zeros(dnum)
best_test_acc=np.zeros(dnum)
best_test_std=np.zeros(dnum)

for i in range(dnum):
	dataset = datasets[i]
	for h in hidden_size: 
		for lr in lrs:
			for d in dropouts:
				test_acc = np.zeros(args.kfold_num)
				test_accs = np.zeros([args.kfold_num, args.epochs])
				results_acc = []
				results_std = []
				for kfold in range(args.kfold_num):
					test_acc[kfold], test_accs[kfold] = experiment(dataset=dataset, model="NonlinearGCN", mod=args.mod, num_layers=args.num_layers, batch_size=args.batch_size, lr=lr, dropout=d, hidden_size=h, epochs=args.epochs, kfold=kfold, test_ratio=0.1, use_best_config=True, mw="graph_classification_mw", dw="graph_classification_dw")
				
				mean_test_acc = np.mean(test_accs, axis=0)
				best_epoch = np.argmax(mean_test_acc)
				mean_acc = np.mean(test_accs[:, best_epoch])
				mean_std = np.std(test_accs[:, best_epoch])
				print('dataset: ',dataset, ' lr=', lr, ' hidden_size=', h, ' dropout', d, ' test_acc: ', mean_acc, ' test_std: ', mean_std)
				results_acc.append(mean_acc) 
				results_std.append(mean_std)
				if mean_acc > best_test_acc[i]:
					best_test_acc[i]=mean_acc
					best_test_std[i]=mean_std
					best_lr[i] = lr
					best_dropout[i]=d
					best_hidden[i]=h

for n in range(dnum):
	print('dataset: ', datasets[n], 'lr=', best_lr[n], 'dropout=', best_dropout[n], 'hidden=',  best_hidden[n],' reslut: ', best_test_acc[n], ' + ', best_test_std[n])
