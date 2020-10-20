import requests
import json
import base64
from PIL import Image
from PIL import ImageChops
import numpy as np
import time
from queue import Queue

request_url = ' http://47.102.118.1:8089/api/problem?stuid=031802230 '+''#后面需要改动加上challenge-uuid
send_url = 'http://47.102.118.1:8089/api/answer'
token = ''
teamid = 15

def compare_images(image_one, image_two):
#    比较图片
    diff = ImageChops.difference(image_one, image_two)

    if diff.getbbox() is None:
            # 图片间没有任何不同则直接退出
        return 1
    else:
        #表示存在差异
        return 0

def cut_image(image):
    width, height = image.size
    item_width = int(width / 3)
    box_list = []
    # (left, upper, right, lower)
    for i in range(0, 3):  # 两重循环，生成9张图片基于原图的位置
        for j in range(0, 3):
            # print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
            box = (j * item_width, i * item_width, (j + 1) * item_width, (i + 1) * item_width)
            box_list.append(box)

    image_list = [image.crop(box) for box in box_list]
    return image_list

def image_to_shape(count,demo_list,test_list):
    demo_shape = np.arange(1,10).reshape((3,3))
    test_shape = np.zeros((3,3),dtype=int)
    start = int (count/9)*9
    end = start+9
    for i in range(start,end):
        count = 0
        judge = 0#判断这个图片是否为空白图片
        for j in test_list:
            demo = demo_list[i]
            sign = compare_images(demo,j)
            if (sign == 1):
                test_shape[int(count/3)][count%3] = i-start+1
                judge = 1
            count += 1
        if (judge == 0):#表示这张图片为空白
            empty_number = i-start+1
    test_shape = test_shape.tolist()
    return test_shape,empty_number

def demo_image():
    demo_list = []
    for i in range(1,37):
        path = "./demo/"+str(i)+".jpg"
        demo = Image.open(path)
        demo_list += cut_image(demo)#存有所有demo分割后的图片
    return demo_list

def determine_letter(demo_list,test_list):
    slist = ['A', 'a', 'b', 'B', 'c', 'd', 'D', 'e', 'F', 'g', 'h', 'H', 'J', 'k', 'm', 'M', 'n', 'o', 'O', 'p', 'P',
             'q', 'Q', 'r', 's', 't', 'u', 'U', 'v', 'W', 'x', 'X', 'y', 'Y', 'z', 'Z']
    count = 0 #记录此时在第几张图片
    similar_count = 0
    rest =9
    for i in demo_list:
        for j in test_list:
            if rest == 0:
                rest = 9
                similar_count = 0
            sign = compare_images(i,j)
            if sign == 1:
                similar_count +=1

                if(similar_count == 8):
                    break
        if(similar_count == 8):
            break
        count += 1
        rest -= 1
    return slist[int(count/9)],count





# 复制状态
def CopyState(state):
    s = []
    for i in state: s.append(i[:])
    return s
# 获取空格的位置

# 获取空格上移后的状态，不改变原状态
def w(state,space):#上
    ss = ''
    y, x = space[0],space[1]
    num1 = y*3+x
    num2 = (y-1)*3+x
    for i in range(len(state)):
        if(i == num1):  ss+=state[num2]
        elif(i == num2):  ss+=state[num1]
        else: ss += state[i]
    return ss
# 获取空格下移后的状态，不改变原状态
def s(state,space):#下
    ss = ''
    y, x = space[0],space[1]
    num1 = y*3+x
    num2 = (y+1)*3+x
    for i in range(len(state)):
        if(i == num1):  ss+=state[num2]
        elif(i == num2):  ss+=state[num1]
        else: ss += state[i]
    return ss
# 获取空格左移后的状态，不改变原状态
def a(state,space):#左
    ss = ''
    y, x = space[0],space[1]
    num1 = y*3+x
    num2 = y*3+x-1
    for i in range(len(state)):
        if(i == num1):  ss+=state[num2]
        elif(i == num2):  ss+=state[num1]
        else: ss += state[i]
    return ss
# 获取空格右移后的状态，不改变原状态
def d(state,space):#右
    ss = ''
    y, x = space[0],space[1]
    num1 = y*3+x
    num2 = y*3+x+1
    for i in range(len(state)):
        if(i == num1):  ss+=state[num2]
        elif(i == num2):  ss+=state[num1]
        else: ss += state[i]
    return ss

# 获取指定状态下的操作
def GetActions(state,space):
    acts = []
    y, x = space[0],space[1]
    if x > 0:acts.append("a")
    if y > 0:acts.append("w")
    if x < 2:acts.append("d")
    if y < 2: acts.append("s")
    return acts

# 边缘队列中的节点类
class Node:
    state = ""   # 状态
    action = ""  # 到达此节点所进行的操作
    space = ()
    # 用状态和步数构造节点对象
    def __init__(self, state, action, space):
        self.state = state
        self.action = action
        x = space[0]
        y = space[1]
        if(action == ""):   space = space
        elif(action[-1] == 'a'): y-= 1
        elif(action[-1] == 'w'): x-= 1
        elif(action[-1] == 's'): x+=1
        elif(action[-1] == 'd'): y+=1
        self.space = (x,y)
        # 计算估计距离


# 将状态转换为字符串
def toStr(state):
    s = ''
    for i in state:
        for j in i:
            s += str(j)
    return s

#next_permutation
def AStar1(init,space_location):
    # 边缘队列初始已有源状态节点
    queue = Queue()
    visit = {}  # 访问过的状态表

    node = Node(init,'',space_location)

    queue.put(node)#

    visit[node.state] = node.action#


    count = 0   # 循环次数
    # 队列没有元素则查找失败

    while not queue.empty():#3*1e5
        # 获取拥有最小估计距离的节点索引
        node = queue.get()#0.35
        count += 1
        # 扩展当前节点
        for act in GetActions(node.state,node.space):

            # 获取此操作下到达的状态节点并将其加入边缘队列中 1.31s
            if(act == 'a'):
                near = Node(a(node.state,node.space), node.action + act, node.space)
            elif(act == 'w'):
                near = Node(w(node.state,node.space), node.action + act, node.space)
            elif(act == 's'):
                near = Node(s(node.state,node.space), node.action + act, node.space)
            elif(act == 'd'):
                near = Node(d(node.state,node.space), node.action + act, node.space)


            if not visit.__contains__(near.state):#0.49s
                queue.put(near)
                visit[near.state] = near.action



    return count,visit

def newStr(s,swap):
    ss = ''
    for i in range(len(s)):
        if(i+1 == swap[0]): ss+=s[swap[1]-1]
        elif(i+1 == swap[1]): ss+=s[swap[0]-1]
        else: ss+=s[i]
    return ss



request_dic = {}
#队伍信息
request_dic['teamid'] = teamid
request_dic['token'] = token
#print(request_dic)
headers = {'content-type': "application/json"}
response = requests.post(request_url, data=json.dumps(request_dic),headers=headers)
response = json.loads(response.text)#json数据转为字典
print("start:",response)
swap_step = response['data']['step']
img_base64 = response['data']['img']
swap = response['data']['swap']
img = base64.b64decode(img_base64)
uuid = response['uuid']
fh = open("./test/imgout.jpg", "wb")
fh.write(img)
fh.close()
#print(swap_step)
#print(swap)
file_path = './test/imgout.jpg'
image = Image.open(file_path)
# image.show()
test_list = cut_image(image)
demo_list = demo_image()
alphabet,count = determine_letter(demo_list,test_list)
init_state,empty_number = image_to_shape(count,demo_list,test_list)
#print(init_state)
#print(empty_number)







x,y = swap[0],swap[1]
swap = (x,y)
slist = ['A', 'a', 'b', 'B', 'c', 'd', 'D', 'e', 'F', 'g', 'h', 'H', 'J', 'k', 'm', 'M', 'n', 'o', 'O', 'p', 'P',
        'q', 'Q', 'r', 's', 't', 'u', 'U', 'v', 'W', 'x', 'X', 'y', 'Y', 'z', 'Z']
# 目标状态
goal_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
for i in range(0,3):
    for j in range(0,3):
        if(goal_state[i][j] == empty_number):
            goal_state[i][j] = 0
            break
# 目标状态 值-位置表
goal_dic = {}
init_dic = {}
for i in range(0,3):
    for j in range(0,3):
        goal_dic[goal_state[i][j]] = (i,j)
for i in range(0,3):
    for j in range(0,3):
        init_dic[init_state[i][j]] = (i,j)
zero_location = goal_dic[0]
t1 = time.time()
goal_str = toStr(goal_state)
print(goal_str)
count1,ans_dic = AStar1(goal_str,zero_location)



init_str = toStr(init_state)
t2 = time.time()
print(t2-t1)
node = Node(goal_str,'',zero_location)
after_step = 999
after_operations = ''
after_swap = ()
before_operations = ''
after_str = ''
if init_str in ans_dic and len(ans_dic[init_str]) <= swap_step:#一开始就有解的情况,无需强制交换
    before_operations =  ans_dic[init_str]
    after_str = init_str
    if before_operations != '' :
        tmp_str = before_operations[::-1]
        s = ''
        for i in tmp_str:
            if(i == 'w'): s+='s'
            elif(i=='s'): s+='w'
            elif(i=='a'): s+='d'
            elif(i=='d'): s+='a'
    before_operations = s
else:
    zero_location = init_dic[0]
    count2,pos_dic = AStar1(init_str,zero_location)
    force_dic = {}#强制转换字典
    for i in pos_dic:
        if (len(pos_dic[i])<=swap_step):
            if (len(pos_dic[i])+swap_step)%2 == 0:
                    force_dic[i] = pos_dic[i]
    newf_dic = {}
    for i in force_dic:#强制转换
        newf_dic[newStr(i,swap)] = force_dic[i]
    count = 0
    for i in newf_dic:
        if ans_dic.__contains__(i):
            if len(ans_dic[i]) < after_step:
                after_str = i
                after_step = len(ans_dic[i])
                before_operations = newf_dic[i]
                after_operations = ans_dic[i]
                after_swap = ()
        else:#开始自由调换
            for j in range(1,10):
                if (j == empty_number): continue
                for k in range(j,10):
                    count += 1
                    if(k == empty_number): continue

                    ss = newStr(i,(j,k))
                    if ans_dic.__contains__(ss):
                        if len(ans_dic[ss]) < after_step:
                            after_str = ss
                            after_step = len(ans_dic[ss])
                            after_operations = ans_dic[ss]
                            before_operations = newf_dic[i]
                            after_swap = (j,k)
    #有进行强制转换操作
    if after_operations != '' :
        tmp_str = after_operations[::-1]
        s = ''
        for i in tmp_str:
            if(i == 'w'): s+='s'
            elif(i=='s'): s+='w'
            elif(i=='a'): s+='d'
            elif(i=='d'): s+='a'
    after_operations = s

    print("after_str:",after_str)
    print("before_operations:",before_operations)
    if after_operations != '' and len(before_operations)<swap_step:
        for i in range(0,len(after_str)):
            if(after_str[i] == '0'):
                space = i
                break
        x = int(space/3)
        y = int(space%3)
        if(x>0):
            while(len(before_operations)<swap_step):
                before_operations += 'ws'
        elif(x<2):
            while(len(before_operations)<swap_step):
                before_operations +='sw'
    if len(after_swap)>0:
        x,y = after_swap[0],after_swap[1]
        after_swap = [x,y]
    else:
        after_swap = []

operations = str(before_operations) + str (after_operations)
end_time = time.time()






send_dic = {}
send_dic['uuid'] = uuid
send_dic['teamid'] = teamid
send_dic['token'] = token
send_dic['answer'] = {}
send_dic['answer']['operations'] = operations
send_dic['answer']['swap'] = after_swap
headers = {'content-type': "application/json"}
response = requests.post(send_url, data=json.dumps(send_dic),headers=headers)
response = json.loads(response.text)#json数据转为字典
print("sumbit:",response)