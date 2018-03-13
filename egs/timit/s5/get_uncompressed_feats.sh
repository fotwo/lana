


fbankdir=fbank
mkdir -p data/fbank
for x in train dev test; do
  \cp -fr data/$x data/fbank/ && \rm -fr data/fbank/$x/split*;
  \rm -fr data/fbank/$x/{cmvn,feats}.scp;
  steps/make_fbank.sh --cmd "run.pl" --nj 10 --compress false data/fbank/$x exp/make_fbank/$x $fbankdir
  steps/compute_cmvn_stats.sh data/fbank/$x exp/make_fbank/$x $fbankdir
done


steps/align_fmllr.sh --nj 10 --cmd run.pl data/dev data/lang exp/tri3 exp/tri3_dev_ali 

steps/align_fmllr.sh --nj 10 --cmd run.pl data/test data/lang exp/tri3 exp/tri3_test_ali 


