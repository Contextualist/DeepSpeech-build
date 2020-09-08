## [DeepSpeech](https://github.com/mozilla/DeepSpeech) Dockerfiles

Adapted from [the official Dockerfile](https://github.com/mozilla/DeepSpeech/blob/master/Dockerfile.train.tmpl)

On dockerhub: `contextualist/deepspeech:train-latest`

To load the monkey patch do:

```
DS_MPATCH=PATH/TO/patch.py \
python -u DeepSpeech.py \
# other options
```
