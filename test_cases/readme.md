### 测试用例集合

1、使用"alexnet+随机数据"测试pytorch可用性

```
python test_case_for_torch_alexnet.py
```

注：该示例代码主要来自于https://github.com/pytorch/examples/blob/master/mnist/main.py

预期将得到类似结果：

```
Train Epoch: 1 [0/60000 (0%)]	Loss: 2.344873
Train Epoch: 1 [640/60000 (1%)]	Loss: 2.318736
Train Epoch: 1 [1280/60000 (2%)]	Loss: 2.289368
Train Epoch: 1 [1920/60000 (3%)]	Loss: 2.308111
Train Epoch: 1 [2560/60000 (4%)]	Loss: 2.302441
Train Epoch: 1 [3200/60000 (5%)]	Loss: 2.292892
...
```



2、使用“torch.nn.DataParallel+随机数据”来测试多GPU同时训练的情形

```
python test_case_for_multi_gpu.py
```

注：该示例代码主要改造自于：https://github.com/pytorch/examples/blob/main/imagenet/main.py

预期得到类似结果：

```
=> creating model 'resnet50'
Epoch: [0][    0/13334]	Time  7.947 ( 7.947)	Data  1.341 ( 1.341)	Loss 7.0519e+00 (7.0519e+00)	Acc@1   1.04 (  1.04)	Acc@5   2.08 (  2.08)
Epoch: [0][    1/13334]	Time  0.166 ( 4.056)	Data  0.002 ( 0.671)	Loss 7.8058e+00 (7.4288e+00)	Acc@1   0.00 (  0.52)	Acc@5   0.00 (  1.04)
Epoch: [0][    2/13334]	Time  0.270 ( 2.794)	Data  0.112 ( 0.485)	Loss 8.3810e+00 (7.7462e+00)	Acc@1   0.00 (  0.35)	Acc@5   1.04 (  1.04)
Epoch: [0][    3/13334]	Time  0.266 ( 2.162)	Data  0.107 ( 0.390)	Loss 8.9607e+00 (8.0499e+00)	Acc@1   1.04 (  0.52)	Acc@5   1.04 (  1.04)
Epoch: [0][    4/13334]	Time  0.275 ( 1.785)	Data  0.108 ( 0.334)	Loss 9.6760e+00 (8.3751e+00)	Acc@1   1.04 (  0.63)	Acc@5   1.04 (  1.04)
```

该代码会自动利用多GPU进行计算。不同的`batch-size`会改变GPU的显存，默认batch_size=96，会占满9.65GB左右的显存，一般能把GPU跑满。

当使用多GPU时，可以相应地倍增batch_size，若使用4卡时，可以：

```
python test_case_for_multi_gpu.py -b 384
```

如果在docker中启动，使用下列命令：

```bash
docker run -ti --rm --ipc=host --volume /var/drivers/nvidia/current:/usr/local/nvidia:ro --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia4 --device=/dev/nvidia5 --device=/dev/nvidia6 --device=/dev/nvidia7 10.11.3.8:5000/bitahub/deepo:py38_cu113 bash -c "wget https://git.openi.org.cn/xwzheng/GPUCluster/raw/branch/master/test_cases/test_case_for_multi_gpu.py; python test_case_for_multi_gpu.py -b 384"
```

在上面的命令中，使用了第`4, 5, 6, 7`号GPU（可根据具体情形更改为其他号），镜像为`10.11.3.8:5000/bitahub/deepo:py38_cu113`，因使用了4张卡，batch_size更改为384。
