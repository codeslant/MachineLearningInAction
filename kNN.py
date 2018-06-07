from numpy import *
from os import listdir
import operator


def file2matrix(filename):
    '''将文本内容转换为矩阵和类标签向量'''
    fr = open(filename)
    array_of_lines = fr.readlines()
    # 得到文件行数
    number_of_lines = len(array_of_lines)
    # 创建待返回矩阵
    return_mat = zeros((number_of_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_of_lines:
        # 解析文件数据到class_label_vector列表中
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index,:] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return(return_mat,class_label_vector)

def classify0(inX, dataSet, labels, k):
    '''
    分类算法：
    In:输入向量inX，训练集dataSet，标签向量labels，近邻数k
    Out:inX的标签
    '''
    dataSet_size = dataSet.shape[0]
    # 求欧式距离：差的平方跨列和，再开方
    diff_mat = tile(inX, (dataSet_size,1)) - dataSet
    distances = ((diff_mat**2).sum(axis=1))**0.5
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist_indicies[i]]
        class_count[vote_label] = class_count.get(vote_label,0) + 1
    # 从大到小排列
    sorted_class_count = \
       sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return(sorted_class_count[0][0])

def auto_norm(dataSet):
    '''归一化特征值，返回归一化后的矩阵'''
    # 此处axis=0，代表“跨行”
    min_vals = dataSet.min(0)
    max_vals = dataSet.max(0)
    ranges = max_vals - min_vals
    norm_dataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # numpy.tile(A,reps):A——待输入数组 reps——决定A重复的方式
    norm_dataSet = dataSet - tile(min_vals, (m,1))
    norm_dataSet = norm_dataSet/tile(ranges, (m,1))
    return(norm_dataSet, ranges, min_vals)

def dating_class_test():
    '''测试数据'''
    test_ratio = 0.1
    dating_data_mat, dating_labels = file2matrix('datingTestSet.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    # 用于测试的Samples
    test_vecs = int(m*test_ratio)
    error_ratio_list = []
    for k in range(1,101):
        error_count = 0.0
        for i in range(test_vecs):
            classifier_result = classify0(norm_mat[i,:],norm_mat[test_vecs:m,:],\
                                          dating_labels[test_vecs:m], k)
            if (classifier_result != dating_labels[i]):
                error_count += 1.0
        error_ratio_list.append(100*error_count/float(test_vecs))
    min_error = min(error_ratio_list)
    k = error_ratio_list.index(min_error) + 1
    print('The lowest error rate is: %.1f%% when k=%d' %(min_error,k))

def classify_person():
    '''预测给定的输入'''
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input("percentage of time spent playing video games?"))
    ff_miles = float(input("frequent flier miles earned per year?"))
    ice_cream = float(input("liters of ice cream consumed per year?"))
    dating_data_mat, dating_labels = file2matrix('datingTestSet.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr-min_vals)/ranges,norm_mat,dating_labels,4)
    print("You will probably like this person: ",result_list[classifier_result - 1])

def img2vector(filename):
    '''将32x32的图片转换为1x1024的向量'''
    return_vect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0,32*i+j] = int(line_str[j])
    return(return_vect)

def handwriting_class_test():
    hw_labels = []
    # 获取目录内容
    training_file_list = listdir('trainingDigits')
    m = len(training_file_list)
    training_mat = zeros((m,1024))
    for i in range(m):
        # 解析文件名获取分类数字
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        training_mat[i,:] = img2vector('trainingDigits/%s' %(file_name_str))
    test_file_list = listdir('testDigits')
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = img2vector('testDigits/%s' %(file_name_str))
        classifier_result = classify0(vector_under_test,training_mat,hw_labels,3)
        if (classifier_result != class_num_str):
            error_count += 1.0
    print("The total number of errors is: %d" %(error_count))
    print("\nThe total error rate is: %.2f%%" %(100*error_count/float(m_test)))
