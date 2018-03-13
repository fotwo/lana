# Copyright 2016    Vincent Renkens
#           2016    Gaofeng Cheng
#           2018    Peking University (author: Tong Fu)
###### Attention please######
## tensorflow version: 1.0
## python version: python2.7
## only supporting: 
##                reading features(compress = false) made by kaldi; training neural network with tensorflow.
## 
## This version now only support DNN running on one gpu card, further functions need to be implemented.
## before running this ,you should 'source path.sh' first
## This work is on TIMIT.
## Results(without ivector)         epoch   dev   test 
## kaldi 2048 relu 4 layer          
## tensorflow + kaldi(see config)   
##

import os
import sys
import argparse
import ast
from six.moves import configparser
from pdb import set_trace

sys.path.append("./")
sys.path.append("steps")
sys.path.append('steps/libs/')
sys.path.append('steps/libs/tensorflow/kaldi_tensorflow')
sys.path.append('steps/libs/tensorflow/tensorflow_nnet')
sys.path.insert(0, os.path.realpath(os.path.dirname(sys.argv[0])) + '/')
current_dir = os.getcwd()

os.system('source %s' % (current_dir + '/path.sh'))
import libs.tensorflow.tensorflow_nnet.nnet as nnet
import libs.tensorflow.kaldi_tensorflow.ark as ark
import libs.tensorflow.kaldi_tensorflow.kaldiInterface as kaldiInterface
import libs.tensorflow.kaldi_tensorflow.batchdispenser as batchdispenser


# Get args from stdin.
def get_args():
    """
    Get args from stdin.
    """
    parser = argparse.ArgumentParser(
        description="""Training a DNN acoustic model.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')

    # configuration file.
    parser.add_argument("--basic-conf-file",type=str,required=True,dest='basic_conf_file')

    parser.add_argument("--gpu-card-id",type=int,dest='gpu_card_id',
                        default=0,help="""command to specific gpu card id you'd like to use,
                        for this stage, tenssorflow under kaldi only supports one gpu card""")

    # params for precedure.
    parser.add_argument("--train-nnet",type=ast.literal_eval,dest='train_nnet',default=True)

    parser.add_argument("--test-nnet",type=ast.literal_eval,dest='test_nnet',default=True)

    # networks hyper-params.
    parser.add_argument("--context-width",type=int,dest='context_width',
                        default=5,help="""size of the left and right context window""")

    parser.add_argument("--num-hidden-units",type=int,dest='num_hidden_units',
                        default=2048,help="""number of neurons in the hidden layers""")

    parser.add_argument("--num-hidden-layers",type=int,dest='num_hidden_layers',
                        default=4,help="""number of hidden layers""")

    parser.add_argument("--activation",type=str,dest='activation',
                        default='relu',help="""nonlinearity used currently supported: relu, tanh, sigmoid""")


    # training hyper-params.
    parser.add_argument("--initial-learning-rate",type=float,dest='initial_learning_rate',
                        default=0.001,help="""initial learning rate of the neural net""")

    parser.add_argument("--learning-rate-decay",type=int,dest="learning_rate_decay",
                        default=1,help="""exponential weight decay parameter""")

    parser.add_argument("--batch-size",type=int,dest='batch_size',
                        default=128,help="""size of the minibatch (#utterances).
                        to limit memory ussage (specifically for GPU) the batch can be devided into even smaller batches.
                        The gradient will be calculated by averaging the gradients of all these mini-batches.""")

    parser.add_argument("--num-frames-per-batch",type=int,dest='num_frames_per_batch',
                        default=4096,help="""This value is the size of these mini-batches in number of frames.
                        For optimal speed this value should be set as high as possible without exeeding the memory.
                        To use the entire batch set to -1""")

    parser.add_argument("--add-layer-period",type=int,dest='add_layer_period',
                        default=10,help="""the network is initialized layer by layer. This parameters determines the frequency of adding layers.
                        Adding will stop when the total number of layers is reached.
                        Set to 0 if no layer-wise initialisation is required.""")

    parser.add_argument("--starting-step",type=int,dest='starting_step',
                        default=0,help="""starting step, set to 'final' to skip nnet training""")

    parser.add_argument("--monophone",type=bool,dest='monophone',
                        default=False,help="""if you're using monophone alignments, set to True""")

    parser.add_argument("--num-epochs",type=int,dest='num_epochs',
                        default=10,help="""number of passes over the entire database""")

    # validation.
    parser.add_argument("--valid-batches",type=int,dest='valid_batches',
                        default=2,help="""size of the validation set,
                        set to 0 if you don't want to use one""")

    parser.add_argument("--valid-frequency",type=int,dest='valid_frequency',
                        default=10,help="""frequency of evaluating the validation set""")

    parser.add_argument("--valid-adapt",type=bool,dest='valid_adapt',
                        default=True,help="""if you want to adapt the learning rate based on the validation set, set to True""")

    parser.add_argument("--valid-retries",type=int,dest='valid_retries',
                        default=3,help="""number of times the learning will retry
                        (with half learning rate) before terminating the training""")

    # check
    parser.add_argument("--check-frequency",type=int,dest='check_frequency',
                        default=10,help="""how many steps are taken between two checkpoints""")

    # visualization.
    parser.add_argument("--visualize",type=bool,dest='visualize',
                        default=True,help="""you can visualise the progress of the neural net with tensorboard""")

    # regularization.
    parser.add_argument("--l2-norm",type=bool,dest='l2_norm',
                        default=False,help="""if you want to do l2 normalization
                        after every layer set to 'True'""")

    parser.add_argument("--dropout",type=float,dest='dropout',
                        default=1.0,help="""if you want to use dropout
                         set to a value smaller than 1""")

    parser.add_argument("--batch-norm",type=bool,dest='batch_norm',
                        default=False,help="""Flag for using batch normalisation""")

    # output directory
    parser.add_argument("--model-name",type=str,dest='model_name',
                        required=True,help="""name of the neural net""")

    print(" ".join(sys.argv))
    print(sys.argv)

    args=parser.parse_args()

    return args

args=get_args()
print(args)
# set_trace()
# read config file
config = configparser.ConfigParser()
# config.read('conf/wsj_tensorflow_kaldi.conf')
config.read(args.basic_conf_file)
current_dir = os.getcwd()

# setting setting gpu used
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_card_id

# get the feature input dim
reader = ark.ArkReader(config.get('directories','train_features') + '/' + config.get('dnn-features','train') + '/feats.scp')
# print(config.get('directories','train_features') + '/' + config.get('dnn-features','name') + '/feats.scp')
# set_trace()
(_,features,_) = reader.read_next_utt()
input_dim = features.shape[1]

# get number of output labels
graph_dir_split = config.get('graph','graph_dir').strip().split(',')
if len(graph_dir_split) is 0:
    print("Error: there should be lang dir")
    exit()
#

# here we suppose the pdf num shared by different graph_dir is the same
gmm_dir = config.get('directories','expdir') + '/' + config.get('gmm','gmm_name')
gmm_ali_dir = config.get('directories','expdir') + '/' + config.get('gmm','gmm_ali_name')
numpdfs = open(gmm_dir + '/'+graph_dir_split[0]+'/num_pdfs')
num_labels = numpdfs.read()
num_labels = int(num_labels[0:len(num_labels)-1])
numpdfs.close()
print("num_labels: "+str(num_labels))
print("input_dim: "+str(input_dim))

# For compatible with the original kaldi-tensorflow.
config.add_section('nnet')
config.set('nnet','name',args.model_name)
config.set('nnet','gmm_ali_name',config.get('gmm','gmm_ali_name'))
config.set('nnet','gmm_name',config.get('gmm','gmm_name'))
config.set('nnet','graph_dir',config.get('graph','graph_dir'))
config.set('nnet','context_width',str(args.context_width))
config.set('nnet','num_hidden_units',str(args.num_hidden_units))
config.set('nnet','num_hidden_layers',str(args.num_hidden_layers))
config.set('nnet','add_layer_period',str(args.add_layer_period))
config.set('nnet','starting_step',str(args.starting_step))
config.set('nnet','monophone',str(args.monophone))
config.set('nnet','nonlin',str(args.activation))
config.set('nnet','l2_norm',str(args.l2_norm))
config.set('nnet','dropout',str(args.dropout))
config.set('nnet','batch_norm',str(args.batch_norm))
config.set('nnet','num_epochs',str(args.num_epochs))
config.set('nnet','initial_learning_rate',str(args.initial_learning_rate))
config.set('nnet','learning_rate_decay',str(args.learning_rate_decay))
config.set('nnet','batch_size',str(args.batch_size))
config.set('nnet','numframes_per_batch',str(args.num_frames_per_batch))
config.set('nnet','valid_batches',str(args.valid_batches))
config.set('nnet','valid_frequency',str(args.valid_frequency))
config.set('nnet','valid_adapt',str(args.valid_adapt))
config.set('nnet','valid_retries',str(args.valid_retries))
config.set('nnet','check_freq',str(args.check_frequency))
config.set('nnet','visualise',str(args.visualize))
#
# set_trace()
#create the neural net
Nnet = nnet.Nnet(config, input_dim, num_labels)

if args.train_nnet:
    #only shuffle if we start with initialisation
    if config.get('nnet','starting_step') == '0':
        #shuffle the examples on disk
        print('------- shuffling examples ----------')
        kaldiInterface.shuffle_examples(config.get('directories','train_features') + '/' +  config.get('dnn-features','train'))
        print("train_data_root: "+str(config.get('directories','train_features') + '/' +  config.get('dnn-features','train')))
    #put all the alignments in one file
    os.system('rm -fr %s %s > /dev/null' % (gmm_ali_dir+'/ali.all.gz', gmm_ali_dir+'/ali.all.pdf.txt.gz'))
    gmm_dir_ali = gmm_ali_dir+'/ali.*.gz'
    os.system('ls %s | grep [0-9] | wc -l > %s' % (gmm_dir_ali,config.get('gmm','gmm_ali_number')))
    f = open(config.get('gmm','gmm_ali_number'), 'r')
    gmm_ali_jobs_num = f.read()
    f.close()
    if int(gmm_ali_jobs_num) is 0:
        print("Error: gmm ali reading seems going wrong")
        exit()
    alifiles = [gmm_ali_dir + '/ali.' + str(i+1) + '.gz' for i in range(int(gmm_ali_jobs_num))]
    print(alifiles)
    alifile = gmm_ali_dir + '/ali.all.gz'
    alifile_unzip = gmm_ali_dir + '/ali.all'
    ali_final_mdl = config.get('directories','expdir') + '/'+ config.get('gmm','gmm_ali_name') + '/final.mdl'
    alifile_in_pdftxt = gmm_ali_dir + '/ali.all.pdf.txt'
    alifile_in_pdftxt_gzipped = gmm_ali_dir + '/ali.all.pdf.txt.gz'

    print('alifile: ' + alifile)
    print('alifile_unzip: ' + alifile_unzip)
    print('ali_final_mdl: ' + ali_final_mdl)
    print('alifile_in_pdftxt: ' + alifile_in_pdftxt)
    # os.system('. ./path.sh')
    os.system('source %s' % (current_dir + '/path.sh'))
    os.system('cat %s > %s' % (' '.join(alifiles), alifile))
    os.system('gunzip -c %s > %s' % (alifile, alifile_unzip))
    os.system('ali-to-pdf %s ark:%s ark,t:%s' % (ali_final_mdl, alifile_unzip, alifile_in_pdftxt)) 
    os.system('gzip -c %s > %s' % (alifile_in_pdftxt, alifile_in_pdftxt_gzipped))
    #train the neural net
    # set_trace()
    print('------- training neural net ----------')
    Nnet.train(config.get('directories','train_features') + '/' +  config.get('dnn-features','train'), alifile_in_pdftxt_gzipped)

if args.test_nnet:
    # set_trace()
    #use the neural net to calculate posteriors for the testing set
    print('------- computing state pseudo-likelihoods ----------')
    savedir = config.get('directories','expdir') + '/' + args.model_name
    print("pseudo-likelihoods savedir: "+savedir)

    # decoding test feature with different graph_dir, of course, test feature dir always more than 1 
    test_feature_dir = config.get('dnn-features','test')
    test_feature_dir_split = test_feature_dir.strip().split(',')

    # decoding all test feature sets and lang dir
    for graph_dir_x in graph_dir_split:
        for test_feature_dir_x in test_feature_dir_split:
            decodedir = savedir + '/decode'+ graph_dir_x + test_feature_dir_x
            print("decodedir: "+decodedir)
            if not os.path.isdir(decodedir):
                os.mkdir(decodedir)

            Nnet.decode(config.get('directories','test_features') + test_feature_dir_x , decodedir)
            print('------- decoding testing sets ----------')
            #copy the gmm model and some files to speaker mapping to the decoding dir
            os.system('cp %s %s' %(gmm_ali_dir + '/final.mdl', decodedir))
            os.system('cp -r %s %s' %(gmm_dir + '/' + graph_dir_x, decodedir))
            os.system('cp %s %s' %(config.get('directories','test_features') + '/' + test_feature_dir_x + '/utt2spk', decodedir))
            os.system('cp %s %s' %(config.get('directories','test_features') + '/' + test_feature_dir_x + '/text', decodedir))

            os.system('cp %s %s' %(config.get('directories','test_features') + '/' + test_feature_dir_x + '/stm', decodedir))
            os.system('cp %s %s' %(config.get('directories','test_features') + '/' + test_feature_dir_x + '/glm', decodedir))

            num_job=8
            #decode using kaldi
            os.system('./steps/decode_tensorflow.sh --cmd %s --nj %s %s/graph* %s %s/kaldi_decode | tee %s/decode.log || exit 1;' % ( config.get('general','cmd'), num_job, decodedir, decodedir, decodedir, decodedir))
            
            #get results
            os.system('grep WER %s/kaldi_decode/wer_* | utils/best_wer.sh' % decodedir)
#