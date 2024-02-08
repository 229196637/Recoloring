from libsvm import svmutil
from libsvm.commonutil import svm_read_problem
from libsvm.svm import svm_problem, svm_parameter
from libsvm.svmutil import *
import numpy as np
train_path = 'HistogramData/train_data/train.txt'
test_path = 'HistogramData/test_data/test.txt'

# 读取训练数据
train_labels, train_features = svm_read_problem(train_path)

# 读取测试数据
test_labels, test_features = svm_read_problem(test_path)

prob = svm_problem(train_labels, train_features)
acc_list = np.zeros((13, 13))

for c in range(-6, 7):
    for g in range(-6, 7):
        num_1 = 2 ** c
        num_2 = 2 ** g
        print('c' + str(num_1) + ',g' + str(num_2))
        param = svm_parameter('-c ' + f'{num_1}' + ' -g ' + f'{num_2}' + ' -t 2')  # 根据之前的结果设置参数

        # 训练模型
        model = svm_train(prob, param)

        # 进行预测
        p_labels, p_acc, p_vals = svm_predict(test_labels, test_features, model)

        acc_list[c + 6][g + 6] = p_acc[0]

np.savetxt('models/acc.txt', acc_list, fmt='%.3f')

# 保存模型
svm_save_model('models/model_svm', model)

# 打印预测结果
print("Predicted Labels:", p_labels)
