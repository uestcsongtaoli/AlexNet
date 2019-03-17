### experiment 1
1. 没有batch normalization
2. 优化器
3. dropout
4. 试试全局pooling
5. 权重初始化固定以及初始化方法
6. 
### 结果
1. 训练集上看acc和loss并未收敛并未收敛
2. 测试集上波动太大
### experiment 2

5. 权重初始化 he_normal
6. patience = 20 因为感觉没收敛
1. 4 块GPU并行， 要重启 只能降到2块

### 结果

4块显卡和2块显卡是不是有明显不同，好像是，但是我这四块同时算，要重启
### 思考
因为 multi_gpus 是将单个 batch 拆分成 sub_batch
所以可以考虑将batch换成512？试试效果 
