from experiments import experiment
import numpy as np
import scipy.io as sio
import pdb


# ## single dataset
# epochs = 500
# kfold_num = 1
# test_acc = np.zeros(kfold_num)
# test_accs = np.zeros([kfold_num, epochs])
# dataset = "mutag" # mutag, imdb-b, imdb-m, ptc-mr, reddit-b, proteins
# for kfold in range(kfold_num):
# 	test_acc[kfold], test_accs[kfold] = experiment(dataset=dataset,model="nlgingcn", epochs=epochs, dropout=0.3, batch_size=128, lr=0.01, kfold=kfold, test_ratio=0.1, use_best_config=True, mw="graph_classification_mw", dw="graph_classification_dw")
# mean_test_acc = np.mean(test_accs, axis=0)
# best_epoch = np.argmax(mean_test_acc)
# mean_acc = np.mean(test_accs[:, best_epoch])
# mean_std = np.std(test_accs[:, best_epoch])
# print('test_acc: ', mean_acc, ' best_std: ', mean_std)



##multi-datasets
epochs = 500
kfold_num = 10
batch_size = 32
num_layer = 3

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
				test_acc = np.zeros(kfold_num)
				test_accs = np.zeros([kfold_num, epochs])
				results_acc = []
				results_std = []
				for kfold in range(kfold_num):
					test_acc[kfold], test_accs[kfold] = experiment(dataset=dataset, model="NonlinearGCN", mod='Generalized-mean', num_layers=num_layer, batch_size=batch_size, lr=lr, dropout=d, hidden_size=h, epochs=epochs, kfold=kfold, test_ratio=0.1, use_best_config=True, mw="graph_classification_mw", dw="graph_classification_dw")
				
				mean_test_acc = np.mean(test_accs, axis=0)
				best_epoch = np.argmax(mean_test_acc)
				mean_acc = np.mean(test_accs[:, best_epoch])
				mean_std = np.std(test_accs[:, best_epoch])
				print('dataset: ',dataset, ' lr=', lr, ' batch_size=', b, ' hidden_size=', h, ' dropout', d, ' test_acc: ', mean_acc, ' best_std: ', mean_std)
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