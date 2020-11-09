from simhash import Simhash
from sys import argv
import os
import jieba
import jieba.analyse
# 求两篇文章相似度
def simhash_similarity(text1, text2):
    """
    :param tex1: 文本1
    :param text2: 文本2
    :return: 返回两篇文章的相似度
    """
    aa_simhash = Simhash(text1)
    bb_simhash = Simhash(text2)
    max_hashbit = max(len(bin(aa_simhash.value)), (len(bin(bb_simhash.value))))

    # 汉明距离
    distince = aa_simhash.distance(bb_simhash)

    similar = 1 - distince / max_hashbit
    similar = similar*1.1
    if similar >1:
        similar = 1.0
    return similar

if __name__ == '__main__':
    try:
        f1 = open(argv[1],'r',encoding = 'utf-8')
        file1 = f1.read()

        f2 = open(argv[2], 'r', encoding= 'utf-8')
        file2 = f2.read()

        f3 = open(argv[3],'r',encoding= 'utf-8')
        file3 = f3.read()
        f1.close()
        f2.close()
        f3.close()
        for i in range(2):
            size = os.path.getsize(argv[i])
            if size == 0:
                break
    except IndexError:
        print("输入错误")

