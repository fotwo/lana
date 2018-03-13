export KALDI_ROOT=/home/fut/work-kaldi_tensorflow/lana/kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/irstlm/bin/:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

# export PATH=\
# ${KALDI_ROOT}/src/bin:\
# ${KALDI_ROOT}/src/chainbin:\
# ${KALDI_ROOT}/src/featbin:\
# ${KALDI_ROOT}/src/fgmmbin:\
# ${KALDI_ROOT}/src/fstbin:\
# ${KALDI_ROOT}/src/gmmbin:\
# ${KALDI_ROOT}/src/ivectorbin:\
# ${KALDI_ROOT}/src/kwsbin:\
# ${KALDI_ROOT}/src/latbin:\
# ${KALDI_ROOT}/src/lmbin:\
# ${KALDI_ROOT}/src/nnet2bin:\
# ${KALDI_ROOT}/src/nnet3bin:\
# ${KALDI_ROOT}/src/nnetbin:\
# ${KALDI_ROOT}/src/online2bin:\
# ${KALDI_ROOT}/src/onlinebin:\
# ${KALDI_ROOT}/src/sgmm2bin:\
# ${KALDI_ROOT}/src/sgmmbin:\
# $PATH


# export PYTHONPATH=`pwd`/steps/:`pwd`/steps/libs:$PYTHONPATH
# there is a sub-dir called 'tensorflow' in this 'libs' dir, which prevents python normally uses the real tensorflow
# you may encounter the python report error: AttributeError: 'module' object has no attribute ******

# Actually, we do not need to export the PYTHONPATH specially, as my tensorflow script will append the these two dir.
# sys.path.append("steps")
# sys.path.append('steps/libs/')
# export PYTHONPATH=`pwd`/steps/:$PYTHONPATH
