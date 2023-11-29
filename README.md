# CCL_ACB_VIDEO_MOMENT_RETRIEVAL
Retrieve the moments(start and end timestamps) from the videos given sentence query.
We appreciate the contribution of the following [code](https://github.com/mxingzhang90/MSAT).

#### Training
Use the following commands for training:
```
# For ActivityNet Captions
python moment_localization/train.py --cfg experiments/activitynet/MSAT-32.yaml --verbose

# For TACoS
python moment_localization/train.py --cfg experiments/tacos/MSAT-128.yaml --verbose
```
#### Testing
Use the following commands for testing and replication of results:
```
# For ActivityNet Captions
python moment_localization/test.py --cfg experiments/activitynet/MSAT-32.yaml --verbose --split test

# For TACoS
python moment_localization/test.py --cfg experiments/tacos/MSAT-128.yaml --verbose --split test
```

#### Inference
Use the following commands for inference:
```
# For ActivityNet Captions
python moment_localization/inference_activitynet.py --cfg experiments/activitynet/MSAT-32.yaml --verbose

# For TACoS
python moment_localization/inference_tacos.py --cfg experiments/tacos/MSAT-128.yaml --verbose

```



