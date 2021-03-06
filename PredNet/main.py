# Simple version 10 (bugfix: period option and pridiction loss)

import argparse
import os
import numpy as np
#import cupy as np
from PIL import Image

import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from chainer.functions.loss.mean_squared_error import mean_squared_error
import net

parser = argparse.ArgumentParser(
description='PredNet')
parser.add_argument('--images', '-i', default='', help='Path to image list file')
parser.add_argument('--sequences', '-seq', default='', help='Path to sequence list file')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--root', '-r', default='.',
                    help='Root directory path of sequence and image files')
parser.add_argument('--initmodel', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--size', '-s', default='160,120',
                    help='Size of target images. width,height (pixels)')
parser.add_argument('--channels', '-c', default='3,48,96,192',
                    help='Number of channels on each layers')
parser.add_argument('--offset', '-o', default='0,0',
                    help='Center offset of clipping input image (pixels)')
parser.add_argument('--input_len', '-l', default=50, type=int,
                    help='Input frame length fo extended prediction on test (frames)')
parser.add_argument('--ext', '-e', default=10, type=int,
                    help='Extended prediction on test (frames)')
parser.add_argument('--bprop', default=20, type=int,
                    help='Back propagation length (frames)')
parser.add_argument('--save', default=10000, type=int,
                    help='Period of save model and state (frames)')
parser.add_argument('--period', default=1000000, type=int,
                    help='Period of training (frames)')
parser.add_argument('--xyz', default=0, type=int,
                    help='Save xyz_images, --xyz 1')
parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)
args = parser.parse_args()


if (not args.images) and (not args.sequences):
    print('Please specify images or sequences')
    exit()
    
args.size = args.size.split(',')
for i in range(len(args.size)):
    args.size[i] = int(args.size[i])
args.channels = args.channels.split(',')
for i in range(len(args.channels)):
    args.channels[i] = int(args.channels[i])
args.offset = args.offset.split(',')
for i in range(len(args.offset)):
    args.offset[i] = int(args.offset[i])

if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

#Create Model
prednet = net.PredNet(args.size[0], args.size[1], args.channels)
model = L.Classifier(prednet, lossfun=mean_squared_error)
model.compute_accuracy = False
optimizer = optimizers.Adam()
optimizer.setup(model)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    print('Running on a GPU')
else:
    print('Running on a CPU')

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('result'):
    os.makedirs('result')
if args.xyz == 1:
    if not os.path.exists('xyz_images'):
        os.makedirs('xyz_images')

def load_list(path, root):
    tuples = []
    for line in open(path):
        pair = line.strip().split()
        tuples.append(os.path.join(root, pair[0]))
    return tuples

def read_image(path):
    image = np.asarray(Image.open(path)).transpose(2, 0, 1)
    top = args.offset[1] + (image.shape[1]  - args.size[1]) // 2
    left = args.offset[0] + (image.shape[2]  - args.size[0]) // 2
    bottom = args.size[1] + top
    right = args.size[0] + left
    image = image[:, top:bottom, left:right].astype(np.float32)
    image /= 255
    return image

def write_image(image, path):
    image *= 255
    image = image.transpose(1, 2, 0)
    image = image.astype(np.uint8)
    result = Image.fromarray(image)
    result.save(path)

if args.images:
    sequencelist = [args.images]
else:
    sequencelist = load_list(args.sequences, args.root)

print('save the initial model')
serializers.save_npz('models/initial.model', model)
if args.test == True:
    logf = open('loss_prediction.txt', 'w')
    for seq in range(len(sequencelist)):
        imagelist = load_list(sequencelist[seq], args.root)
        prednet.reset_state()
        loss = 0
        batchSize = 1
        x_batch = np.ndarray((batchSize, args.channels[0], args.size[1], args.size[0]), dtype=np.float32)
        y_batch = np.ndarray((batchSize, args.channels[0], args.size[1], args.size[0]), dtype=np.float32)
        
        for i in range(0, len(imagelist) - 1):
            print('frameNo:' + str(i))
            x_batch[0] = read_image(imagelist[i])
            y_batch[0] = read_image(imagelist[i + 1])
            loss += model(chainer.Variable(xp.asarray(x_batch)),
                          chainer.Variable(xp.asarray(y_batch)))
            loss.unchain_backward()
            loss = 0
            if args.gpu >= 0:model.to_cpu()
            write_image(x_batch[0].copy(), 'result/test_' + str(i) + 'x.jpg')
            write_image(model.y.data[0].copy(), 'result/test_' + str(i) + 'y_0.jpg')
            eval_index = model.predictor.cal_eval_index(y_batch)
            logf.write(str(i))
            for e in eval_index:
                logf.write(', ' + str(float(e.data)))
            logf.write('\n')
            logf.flush()
            if args.gpu >= 0:model.to_gpu()

            if i == 0 or (args.input_len > 0 and i % args.input_len != 0):
                continue
            if args.gpu >= 0:model.to_cpu()
            x_batch[0] = model.y.data[0].copy()
            if args.gpu >= 0:model.to_gpu()
            
            for j in range(args.ext):
                print('extended frameNo:' + str(j + 1))
                loss += model(chainer.Variable(xp.asarray(x_batch)),
                              chainer.Variable(xp.asarray(y_batch)))
                loss.unchain_backward()
                loss = 0
                if args.gpu >= 0:model.to_cpu()
                write_image(model.y.data[0].copy(), 'result/test_' + str(i) + 'y_' + str(j + 1) + '.jpg')
                x_batch[0] = model.y.data[0].copy()
                if args.gpu >= 0:model.to_gpu()
            prednet.reset_state()
            
else:
    logf = open('log.txt', 'w')
    count = 0
    seq = 0
    while count < args.period:
        imagelist = load_list(sequencelist[seq], args.root)
        prednet.reset_state()
        loss = 0

        batchSize = 1
        x_batch = np.ndarray((batchSize, args.channels[0], args.size[1], args.size[0]), dtype=np.float32)
        y_batch = np.ndarray((batchSize, args.channels[0], args.size[1], args.size[0]), dtype=np.float32)
        
        if len(imagelist) == 0:
            print("Not found images.")
            break
            
        x_batch[0] = read_image(imagelist[0])
        
        for i in range(1, len(imagelist)):
            y_batch[0] = read_image(imagelist[i])
            loss += model(chainer.Variable(xp.asarray(x_batch)),
                          chainer.Variable(xp.asarray(y_batch)))

            print('frameNo:' + str(i))
            if (i + 1) % args.bprop == 0:
                model.zerograds()
                loss.backward()
                loss.unchain_backward()
                loss = 0
                optimizer.update()
                
                if args.gpu >= 0:model.to_cpu()
                
                if args.xyz == 1:
                    write_image(x_batch[0].copy(), 'xyz_images/' + str(count) + '_' + str(seq) + '_' + str(i) + 'x.jpg')
                    write_image(model.y.data[0].copy(), 'xyz_images/' + str(count) + '_' + str(seq) + '_' + str(i) + 'y.jpg')
                    write_image(y_batch[0].copy(), 'xyz_images/' + str(count) + '_' + str(seq) + '_' + str(i) + 'z.jpg')
                
                print('loss:' + str(float(model.loss.data)))
                logf.write(str(i) + ', ' + str(float(model.loss.data)) + '\n')
                logf.flush()
                if args.gpu >= 0:model.to_gpu()

            if (count%args.save) == 0:
                print('save the model')
                serializers.save_npz('models/' + str(count) + '.model', model)
                print('save the optimizer')
                serializers.save_npz('models/' + str(count) + '.state', optimizer)

            x_batch[0] = y_batch[0]
            if count > args.period:
                break
            count += 1
        
        seq = (seq + 1)%len(sequencelist)
