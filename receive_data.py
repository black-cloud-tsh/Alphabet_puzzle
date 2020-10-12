import requests
import urllib.request
import json
import  base64
from PIL import Image
from PIL import ImageChops
import numpy as np
from main import compare_images,cut_image,judge_solution,image_to_shape,demo_image,determine_letter

if __name__ == '__main__':
    url = "http://47.102.118.1:8089/api/problem?stuid=031802230 "
    html = requests.get(url).json()
    data = html['img']
    img = base64.b64decode(html['img'])
    slist = ['A', 'a', 'b', 'B', 'c', 'd', 'D', 'e', 'F', 'g', 'h', 'H', 'J', 'k', 'm', 'M', 'n', 'o', 'O', 'p', 'P',
             'q', 'Q', 'r', 's', 't', 'u', 'U', 'v', 'W', 'x', 'X', 'y', 'Y', 'z', 'Z']
    file_path = "./test/imgout.jpg"
    image = Image.open(file_path)
    # image.show()
    test_list = cut_image(image)
    # 36张图片，遍历到36
    demo_list = demo_image()
    alphabet,count = determine_letter(demo_list,test_list)
    print(alphabet)
    #print(count)#70
    test_shape = image_to_shape(count,demo_list,test_list)
    print(test_shape)
    judge = judge_solution(test_shape)
    print(judge)
