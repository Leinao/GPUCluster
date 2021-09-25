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

