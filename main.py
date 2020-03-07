import os
import time
import argparse
import tensorflow as tf
from model import Model
from tqdm import tqdm
from util import *
import pickle


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--time_span', default=256, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.00005, type=float)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum, timenum] = dataset

model = Model(usernum, itemnum, timenum,args)
print('User: %d, Item: %d:, Timenum: %d'%(usernum, itemnum, timenum))
num_batch = int(len(user_train) / args.batch_size)
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: %.2f' % (cc / len(user_train)))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)
sess.run(tf.initialize_all_variables())

try:
    relation_matrix = pickle.load(open('data/relation_matrix_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.time_span),'rb'))
except:
    relation_matrix = Relation(user_train, usernum, args.maxlen, args.time_span)
    pickle.dump(relation_matrix, open('data/relation_matrix_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.time_span),'wb'))

sampler = WarpSampler(user_train, usernum, itemnum, relation_matrix, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
T = 0.0
t0 = time.time()
try:
    for epoch in range(1, args.num_epochs + 1):
    
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            
            u, seq, time_seq, time_matrix, pos, neg = sampler.next_batch()
            auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.u: u, model.input_seq: seq, model.time_matrix: time_matrix ,model.pos: pos, model.neg: neg,
                                        model.is_training: True})

        if epoch % 20 == 0:
            t1 = time.time() - t0
            T += t1
            print('Evaluating')
            t_test = evaluate(model, dataset, args, sess)
            t_valid = evaluate_valid(model, dataset, args, sess)
            print('')
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
            epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            t0 = time.time()
except:
    sampler.close()
    exit(1)
sampler.close()
print("Done")
