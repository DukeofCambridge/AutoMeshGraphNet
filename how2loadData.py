import os
import numpy as np

if __name__ == '__main__':
    # """
    # 读取npy文件
    # """
    # # Cd为风阻值 Coefficient of drag
    # Cd = np.load("data/cd.npy", allow_pickle=True).item()
    # print(Cd)  # {'1': '0.357827734', '2': '0.3951254', ... }
    #
    # """
    # 读取npz文件
    # """
    # model = np.load("data/CarModel/Model001.npz", allow_pickle=True)
    # print(model.files)  # ['connections', 'positions']，边和点
    # print(np.array(model['connections']).shape)  # (103872, 2)，边数和边的两个顶点
    # print(np.array(model['positions']).shape)  # (14844, 3)，点数和点的三维坐标
    #
    # """
    # 汽车模型数据编号到230，实际上只有200个
    # """
    # root = 'data/CarModel'
    # paths = [os.path.join(root, filename) for filename in os.listdir(root)]
    # paths = np.array(paths)
    # print(int(os.listdir(root)[100][5:8]))  # 获取某一文件实际序号
    fuck=2.5155
    fuck=np.array(fuck)
    print(fuck.shape)
