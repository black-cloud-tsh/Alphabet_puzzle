import requests
import json
import base64
from PIL import Image
from PIL import ImageChops
import numpy as np
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
            rest_number = i-start+1
    for i in range(0,3):
        for j in range(0,3):
            if(test_shape[i][j] == 0):
                test_shape[i][j] = rest_number
                break
    return test_shape

def demo_image():
    demo_list = []
    for i in range(1,37):
        path = "./demo/"+str(i)+".jpg"
        demo = Image.open(path)
        demo_list += cut_image(demo)#存有所有demo分割后的图片
    return demo_list

def determine_letter(demo_list,test_list):
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
request_dic = {}
request_url = 'http://47.102.118.1:8089/api/challenge/start/'#后面需要改动加上challenge-uuid
#队伍信息
request_dic['teamid'] = 15
request_dic['token'] = '4e717b20-c22f-48e9-b943-a7ebd4b58436'
#print(request_dic)
headers = {'content-type': "application/json"}
response = requests.post(request_url, data=json.dumps(request_dic),headers=headers)
response = json.loads(response.text)#json数据转为字典
print(response)
change_step = response['data']['step']
img_base64 = response['data']['img']
img = base64.b64decode(img_base64)
fh = open("./test/imgout.jpg", "wb")
fh.write(img)
fh.close()
slist = ['A', 'a', 'b', 'B', 'c', 'd', 'D', 'e', 'F', 'g', 'h', 'H', 'J', 'k', 'm', 'M', 'n', 'o', 'O', 'p', 'P',
         'q', 'Q', 'r', 's', 't', 'u', 'U', 'v', 'W', 'x', 'X', 'y', 'Y', 'z', 'Z']
file_path = './test/imgout.jpg'
image = Image.open(file_path)
# image.show()
test_list = cut_image(image)
# 初始状态
init_state = [
    [4, 3, 5],
    [7, 2, 1],
    [8, 6, 0]
]
# 目标状态
goal_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]
# 目标状态 值-位置表
goal_dic = {}



for i in range(0,3):
    for j in range(0,3):
        goal_dic[goal_state[i][j]] = (i,j)
# 输出状态
def PrintState(state):
    for i in state: print(i)
# 复制状态
def CopyState(state):
    s = []
    for i in state: s.append(i[:])
    return s
# 获取空格的位置
def GetSpace(state):
    for y in range(len(state)):
        for x in range(len(state[y])):
            if state[y][x] == 0: return y, x
# 获取空格上移后的状态，不改变原状态
def MoveUp(state):
    s = CopyState(state)
    y, x = GetSpace(s)
    s[y][x], s[y - 1][x] = s[y - 1][x], s[y][x]
    return s
# 获取空格下移后的状态，不改变原状态
def MoveDown(state):
    s = CopyState(state)
    y, x = GetSpace(s)
    s[y][x], s[y + 1][x] = s[y + 1][x], s[y][x]
    return s
# 获取空格左移后的状态，不改变原状态
def MoveLeft(state):
    s = CopyState(state)
    y, x = GetSpace(s)
    s[y][x], s[y][x - 1] = s[y][x - 1], s[y][x]
    return s
# 获取空格右移后的状态，不改变原状态
def MoveRight(state):
    s = CopyState(state)
    y, x = GetSpace(s)
    s[y][x], s[y][x + 1] = s[y][x + 1], s[y][x]
    return s
# 获取两个状态之间的启发距离
def GetDistance(src, dest):
    dic, d = goal_dic, 0
    for i in range(len(src)):
        for j in range(len(src[i])):
            pos = dic[src[i][j]]
            y, x= pos[0], pos[1]
            d += abs(y - i) + abs(x - j)
    return d
# 获取指定状态下的操作
def GetActions(state):
    acts = []
    y, x = GetSpace(state)
    if x > 0:acts.append(MoveLeft)
    if y > 0:acts.append(MoveUp)
    if x < len(state[0]) - 1:acts.append(MoveRight)
    if y < len(state[0]) - 1: acts.append(MoveDown)
    return acts
# 用于统一操作序列的函数
def Start(state):
    return
# 边缘队列中的节点类
class Node:
    state = None   # 状态
    value = -1     # 启发值
    step = 0       # 初始状态到当前状态的距离（步数）
    action = Start  # 到达此节点所进行的操作
    parent = None,  # 父节点
    # 用状态和步数构造节点对象
    def __init__(self, state, step, action, parent):
        self.state = state
        self.step = step
        self.action = action
        self.parent = parent
        # 计算估计距离
        self.value = GetDistance(state, goal_state) + step
# 获取拥有最小启发值的元素索引
def GetMinIndex(queue):
    index = 0
    for i in range(len(queue)):
        node = queue[i]
        if node.value < queue[index].value:
            index = i
    return index
# 将状态转换为整数
def toInt(state):
    value = 0
    for i in state:
        for j in i:
            value = value * 10 + j
    return value
# A*算法寻找初始状态到目标状态的路径
def AStar(init, goal):
    # 边缘队列初始已有源状态节点
    queue = [Node(init, 0, Start, None)]
    visit = {}  # 访问过的状态表
    count = 0   # 循环次数
    # 队列没有元素则查找失败
    while queue:
        # 获取拥有最小估计距离的节点索引
        index = GetMinIndex(queue)
        node = queue[index]
        visit[toInt(node.state)] = True
        count += 1
        if node.state == goal:
            return node, count
        del queue[index]
        # 扩展当前节点
        for act in GetActions(node.state):
            # 获取此操作下到达的状态节点并将其加入边缘队列中
            near = Node(act(node.state), node.step + 1, act, node)
            if toInt(near.state) not in visit:
                queue.append(near)
    return None, count

# 将链表倒序，返回链头和链尾
def reverse(node):
    if node.parent == None:
        return node, node
    head, rear = reverse(node.parent)
    rear.parent, node.parent = node, None
    return head, node

def CmpState(state1,state2):
    x1,y1 = GetSpace(state1)
    x2,y2 = GetSpace(state2)
    if (x1-x2 == 1): return 'w'
    elif(x1-x2 == -1): return 's'
    elif(y1-y2 == 1): return 'a'
    elif(y1-y2 == -1): return 'd'



node, count = AStar(init_state, goal_state)

if node == None:
    print("无法从初始状态到达目标状态！")
else:
    print("搜索成功，循环次数：", count)
    node, rear = reverse(node)
    count = 0
    operation = ''

    while node:
        # 启发值包括从起点到此节点的距离
        print("第", count , "步：", node.action.__name__, "启发值为：", count, "+", node.value - count)
        PrintState(node.state)
        if(node.parent == None): break
        operation +=CmpState(node.state,node.parent.state)
        node = node.parent
        count += 1
    #print("操作数共有",len(operation),"个:")
    #print(operation)
    #awdwassawwdsds