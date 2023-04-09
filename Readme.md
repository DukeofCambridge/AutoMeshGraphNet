    --ckpt    模型参数文件存放处
    --config  配置文件与配置导入脚本
    --data    数据
        --CarModel 汽车模型数据
        --cd.npy   字典数据，记录了每个汽车模型的风阻值，格式：{'1':'0.357827734',...}
        --DOE List.xlsx 表格数据，直观浏览每个汽车模型的风阻值
    --log     日志
    --model
        --components  MGN模型用到的组件
        --MGN         直接调用的模型
    --tools   工具脚本

数据流：从data/CarModel中读取模型数据，从data/cd.npy读取风阻数据（label），组合为完整数据集，然后就变成了普通的回归问题，调用MGN()针对每个汽车模型输出一个风阻值，主要是各个流程的数据规格设置比较麻烦

    数据规格：
    before-normalization   
                           node:torch.Size([1, 15287, 3])
                           edge: torch.Size([1, 106973, 4])
    after-normalization
                        torch.Size([1, 15287, 3])
                        torch.Size([1, 106973, 4])
    before-encode
                        torch.Size([1, 15287, 3])
                        torch.Size([1, 106973, 4])
    after-encode
                torch.Size([1, 15287, 64])
                torch.Size([1, 106973, 64])

    after-process
            torch.Size([1, 15287, 64])
            torch.Size([1, 106973, 64])
    prediction-size
            torch.Size([1,])


为了让autoMGN调用，我把train.py里的train()里的log, eval, save等方法都注释了，还把num_worker改成了4->1