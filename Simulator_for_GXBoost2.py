# -*- coding: utf-8 -*-
import random

from Simulator_fun_2207 import *
#from Simulator_v3 import run_time
#from Simulator_v3 import rider_speed
from re_A1_class import Store, Platform_pool
import numpy as np
import simpy
from A1_BasicFunc import  OrdergeneratorByCSV, counter, check_list, counter2, t_counter, GenerateStoreByCSVStressTest, OrdergeneratorByCSVForStressTest, OrdergeneratorByCSVForStressTest2, GenerateStoreByCSVStressTest2
import datetime

global gen_B_size
global instance_type
global order_dir
global test_run_time
global input_speed
#instance_type = '동작구'
#gen_B_size = 3
#order_dir = 'C:/Users/박태준/jupyter_notebook_base/data/' + instance_type + '/1206_NOR' + str(0) + '.txt'
#order_dir = 'C:/Users/박태준/jupyter_notebook_base/data/' + instance_type + '/R_972_1206_NOR' + str(0) + '.txt'

save_root_dir = 'C:/rev/'
#test_run_time = 30

##count 확인
counter.dist = 0
counter.bundle_consist = 0
counter.bundle_consist2 = 0
counter.dist1 = 0
counter.dist2 = 0
counter.dist3 = 0
counter.bundle_consist = 0
counter.bundle_consist2 = 0

check_list.b2 = []
check_list.b3 = []
check_list.b2_count = 0
check_list.b3_count = 0
check_list.suggested_bundle = []
##counter 정의
t_counter.t1 = 0
t_counter.t2 = 0
t_counter.t3 = 0
t_counter.t4 = 0
counter2.num1 = []
counter2.num2 = []
counter2.num3 = []
counter2.num4 = []
counter2.num5 = []



current_time = datetime.datetime.now()
rev_day = str(current_time.day)
if current_time.day < 10:
    rev_day = '0' + str(current_time.day)
rev_hour = str(current_time.hour)
if current_time.hour < 10:
    rev_hour = '0' + str(current_time.hour)
rev_min = str(current_time.minute)
if current_time.minute < 10:
    rev_min = '0' + str(current_time.minute)
rev_sec = str(current_time.second)
if current_time.second < 10:
    rev_sec = '0' + str(current_time.second)

save_id = rev_day + '_' + rev_hour + '_' + rev_min + '_' + rev_sec

env = simpy.Environment()
Platform_dict = Platform_pool()
Store_dict = {}
Orders = {}
test1 = []
store_dir = '송파구store_Coor.txt'
customer_dir = '송파구house_Coor.txt'
rider_speed = input_speed #todo 1206 : ->좌표계가 변했기 때문
heuristic_theta = 10
heuristic_r1 = 10
ellipse_w = 10
speed = input_speed #todo 1206 : ->좌표계가 변했기 때문
rider_p2 = 2 #1.5
platform_p2 = 2 #rider_p2*0.8
Saved_data = []
DummyB2 = []
DummyB3 = []
customer_p2 = 1 #2
unit_fee = 110
fee_type = 'linear'
stress_lamda = 20 # 분당 주문 발생 수 # (2400/60)/5 #기준은 한 구에 분당 3750/60
#1 주문 생성
#orders, stores, customers = OrderGen(store_dir, customer_dir, store_size = 100, customer_size = 1000, order_size = 1000, coor_random = True)

#todo : 실험 환경 현실화
StoreDetailPara = False
if StoreDetailPara == True:
    store_detail = []
    store_detail1 = [[1,0.232477446,'M','W',2],[2,0.164469119,'H','W',1.5],[3,0.121790423,'H','W',2],[4,0.104094379,'M','T',2],[5,0.078764747,'M','T',2],[6,0.065579459,'M','T',2],
     [7,0.06315059,'M','C',2],[8,0.042678695,'M','T',3],[9,0.027758501,'L','C',2],[10,0.023247745,'M','W',2],[11,0.019777932,'H','T',2],
     [12,0.015961138,'M','T',2],[13,0.012838307,'H','W',2],[14,0.0111034,'L','C',2],[15,0.00832755,'M','T',2.5],[16,0.007980569,'M','C',3]]
    # detail_pr = [rest_type_list, pr_list, frt_list, temperature_list, p2_list] -> array
    for index in [0,1,2,3,4]:
        tem = []
        for info in store_detail1:
            tem.append(info[index])
        store_detail.append(tem)
    if sum(store_detail[1]) != 1:
        rev_pr = 1 - sum(store_detail[:-1])
        store_detail[-1] = rev_pr
else:
    store_detail = None
#평균 조리 시간	분산(분)	비율(%)
#5	0.25	0.154639175
#10	0.5	0.463917526
#15	0.75	0.206185567
#20	1	0.103092784
#25	1.25	0.051546392
#30	1.5	0.020618557
cookdetail = False
if cookdetail == True:
    cook1 = [0.1546, 0.4639,0.2061,0.1030,0.0515, 0.02061]
    CookTimeDetail_pr = cook1[:-1] + [float(1 - sum(cook1[:-1]))]
    CookTimeDetail = [[5,10,15,20,25,30], CookTimeDetail_pr] #[5,10,15,20,25,30]
else:
    cook1 = None
    CookTimeDetail_pr = None
    CookTimeDetail = None
#실험 환경 현실화 끝


print('음식점 수',len(Store_dict))
test2 = 'E:/학교업무 동기화용/py_charm/BundleSimple/'+ instance_type +'/Instancestore_infos0'
test3 = 'E:/학교업무 동기화용/py_charm/BundleSimple/'+ instance_type +'/ct_data_merge'
#GenerateStoreByCSV(env, test2, Platform_dict, Store_dict)
#env.process(OrdergeneratorByCSV(env, test3, Orders, Store_dict, Platform_dict, p2_ratio = 1,rider_speed= 3,  service_time_diff = False, shuffle= True))
Store_dict = {}
#GenerateStoreByCSVStressTest(env, 50, Platform_dict, Store_dict, store_type=instance_type, detail_pr = store_detail)
GenerateStoreByCSVStressTest2(env, Platform_dict, Store_dict, order_dir, store_type=instance_type, detail_pr = store_detail)
print('음식점 수',len(Store_dict))
#input('확인2')
"""
env.process(
    OrdergeneratorByCSVForStressTest(env, Orders, Store_dict, stress_lamda, platform=Platform_dict, p2_ratio=customer_p2,
                                     rider_speed=rider_speed,
                                     unit_fee=unit_fee, fee_type=fee_type, cooktime_detail= CookTimeDetail))
"""
env.process(
    OrdergeneratorByCSVForStressTest2(env, Orders, Store_dict, stress_lamda, order_dir, platform=Platform_dict, p2_ratio=customer_p2,
                                     rider_speed=rider_speed, unit_fee=unit_fee, fee_type=fee_type))




#2번들을 탐색하는 과정
env.process(BundleProcess(env, Orders,Platform_dict, heuristic_theta, heuristic_r1,ellipse_w,platform_p2,bundle_permutation_option = True, bundle_size=[gen_B_size], speed = speed, Data = Saved_data, DummyB2_data = DummyB2, DummyB3_data = DummyB3))

env.run(test_run_time)
print(len(Orders))
print('Name :: dist :: p2 :: ratio')
for ct_num in Orders:
    ct = Orders[ct_num]
    print(ct_num, '::',distance(ct.location[0],ct.location[1], ct.store_loc[0],ct.store_loc[1]), '::',ct.p2,'::',distance(ct.location[0],ct.location[1], ct.store_loc[0],ct.store_loc[1]) / ct.p2)

saved_orders = []
print('이름;시간;x;y;s_x;s_y;p2;od_dist;service_time')
for ct_num in Orders:
    ct = Orders[ct_num]
    print(ct_num,';',ct.time_info[0], ';',ct.location[0], ';',ct.location[1], ';',ct.store_loc[0], ';',ct.store_loc[1], ';',ct.p2, ';',distance(ct.location[0],ct.location[1], ct.store_loc[0],ct.store_loc[1]),';',ct.time_info[7])
    tem = [ct.name,ct.time_info[0],ct.location[0],ct.location[1],ct.store, ct.store_loc[0],ct.store_loc[1],ct.p2,ct.cook_time,ct.cook_info[1][0],ct.cook_info[1][1],ct.time_info[6],ct.time_info[7], 3]
    saved_orders.append(tem)
print('저장1 종료')
try:
    instance_type_i = instance_type[9]
except:
    instance_type_i = instance_type[0]
#input('STOP')
order_np = np.array(saved_orders, dtype=np.float64)
#np.save('./GXBoost'+str(gen_B_size)+'/'+save_id+'saved_orders_'+instance_type_i+'_'+str(gen_B_size), order_np)
#np.save('./GXBoost'+str(gen_B_size)+'/'+save_id+'saved_orders_'+instance_type_i+'_'+str(gen_B_size), saved_orders)
np.save(save_root_dir + 'GXBoost'+str(gen_B_size)+'/'+save_id+'saved_orders_'+instance_type_i+'_'+str(gen_B_size), saved_orders)
#Feature saved Part
label_datas = []
count = 0
label1_names = []
label1_infos = []
print(len(Saved_data))
"""
#input('Saved_data')
for data in Saved_data:
    origin = Orders[data[0][0] - 1000].store_loc
    destination = Orders[data[0][-1]].location
    line_dist = data[5] - distance(origin, destination) / rider_speed
    data.append(line_dist)
"""
Saved_data.sort(key=operator.itemgetter(8))

for data in Saved_data:
    # ver1: [route, unsync_t[0], round(sum(ftds) / len(ftds), 2), unsync_t[1], order_names, round(route_time, 2),min(time_buffer), round(P2P_dist - route_time, 2), round(route_time, 2) - 시작점 지점과 끝점 이동 시간]
    print(data)
    tem = [count, len(data[4])]
    tem += data[4]
    tem += data[0]
    tem += [data[2]]
    tem += [data[5]]
    tem += [data[6]] #todo: 추가된 부분
    #거리 계산하기
    #print('거리 계산',Orders[data[0][0]- 1000].store_loc, Orders[data[0][-1]].location)
    #origin = Orders[data[0][0]- 1000].store_loc
    #destination = Orders[data[0][-1]].location
    #line_dist = data[5] - distance(origin,destination)/rider_speed
    tem += [data[8]]
    label_datas.append(tem)
    label1_names.append(data[4])
    label1_infos.append([data[8],data[5],data[9],data[10]])
    points = []
    vectors = []
    triangles = []
    for name in data[4]:
        points += [Orders[name].store_loc, Orders[name].location]
        vectors += [Orders[name].store_loc[0] - Orders[name].location[0], Orders[name].store_loc[1] - Orders[name].location[1]]
    #label1_infos.app
    count += 1
label_datas_np = np.array(label_datas)
#np.save('./GXBoost'+str(gen_B_size)+'/'+save_id+'c_'+instance_type_i+'_'+str(gen_B_size), label_datas_np)
np.save(save_root_dir + 'GXBoost'+str(gen_B_size)+'/'+save_id+'c_'+instance_type_i+'_'+str(gen_B_size), label_datas_np)
print('고객 수::', len(Orders), '찾아진 번들 수::' , count)
print('counter', counter.dist, counter.bundle_consist, counter.bundle_consist2)
if gen_B_size == 2:
    #DummyB2
    if len(DummyB2) == 0:
        dummy_thres = 1
    else:
        dummy_thres = min(0.3, (100000/len(DummyB2)))
    Dummy_B2_datas = []
    Dummy_B2_datas_names = []
    count = 0
    for data in DummyB2:
        if data not in Dummy_B2_datas_names:
            rv = random.random()
            if rv < dummy_thres:
                tem = [count, 2]
                tem += data
                Dummy_B2_datas.append(tem)
                Dummy_B2_datas_names.append(data)
        #print('data',data)
        #print('더미3',Dummy_B2_datas_names)
    Dummy_B2_datas_np = np.array(Dummy_B2_datas, dtype=int)
    #np.save('./GXBoost'+str(gen_B_size)+'/'+save_id+'Dummy_B2_datas_'+instance_type_i+'_'+str(gen_B_size), Dummy_B2_datas)
    np.save(save_root_dir + 'GXBoost' + str(gen_B_size) + '/' + save_id + 'Dummy_B2_datas_' + instance_type_i + '_' + str(gen_B_size), Dummy_B2_datas)
    print('입력2', len(label1_names))
    #label1_data = BundleFeaturesCalculator(saved_orders, label1_names, label=1)
    label1_data = BundleFeaturesCalculator2(Orders, label1_names, label=1, add_info=label1_infos, print_option = True)
    raw_data_np = np.array(label1_data, dtype=np.float64)
    # np.save('./GXBoost'+str(gen_B_size)+'/'+save_id+'raw_data_np_'+instance_type_i+'_'+str(gen_B_size), raw_data_np)
    np.save(save_root_dir + 'GXBoost' + str(gen_B_size) + '/' + save_id + 'label1_B2_raw_data_np_' + instance_type_i + '_' + str(
        gen_B_size), raw_data_np)
    print('입력2_중복제거1', len(label1_data))
    print('입력2',len(Dummy_B2_datas_names), Dummy_B2_datas_names[:5])
    #label0_data = BundleFeaturesCalculator(saved_orders, Dummy_B2_datas_names, label = 0)
    label0_data = BundleFeaturesCalculator2(Orders, Dummy_B2_datas_names, label=0, add_info=4)
    print('입력2_중복제거2',len(label0_data))
    #input('확인2')
if gen_B_size == 3:
    #DummyB3
    dummy_thres = min(0.3, (100000/len(DummyB3)))
    Dummy_B3_datas = []
    Dummy_B3_datas_names = []
    count = 0
    for data in DummyB3:
        rv = random.random()
        if rv < dummy_thres :
            tem = [count, 3]
            tem += data
            Dummy_B3_datas.append(tem)
            Dummy_B3_datas_names.append(data)
    Dummy_B3_datas_np = np.array(Dummy_B3_datas, dtype=int)
    #np.save('./GXBoost'+str(gen_B_size)+'/'+save_id+'Dummy_B3_datas_'+instance_type_i+'_'+str(gen_B_size), Dummy_B3_datas)
    np.save(save_root_dir + 'GXBoost' + str(gen_B_size) + '/' + save_id + 'Dummy_B3_datas_' + instance_type_i + '_' + str(gen_B_size), Dummy_B3_datas)
    print('입력3',len(label1_names))
    #label1_data = BundleFeaturesCalculator(saved_orders, label1_names, label = 1)
    label1_data = BundleFeaturesCalculator2(Orders, label1_names, label=1, add_info=label1_infos, print_option = True)
    raw_data_np = np.array(label1_data, dtype=np.float64)
    # np.save('./GXBoost'+str(gen_B_size)+'/'+save_id+'raw_data_np_'+instance_type_i+'_'+str(gen_B_size), raw_data_np)
    np.save(save_root_dir + 'GXBoost' + str(gen_B_size) + '/' + save_id + 'label1_B3_raw_data_np_' + instance_type_i + '_' + str(
        gen_B_size), raw_data_np)
    print('입력3_중복제거1',len(label1_data))
    #input('확인3')
    print('입력3',len(Dummy_B3_datas_names), Dummy_B3_datas_names[:5])
    #label0_data = BundleFeaturesCalculator(saved_orders, Dummy_B3_datas_names, label = 0)
    label0_data = BundleFeaturesCalculator2(Orders, Dummy_B3_datas_names, label=0, add_info=4)
    print('입력3_중복제거2',len(label0_data))
    #input('확인3')
#print(len(label1_data[0]), len(label0_data[0]))
#print(label1_data[0], label0_data[0])
#input('확인')
raw_data = label1_data + label0_data
raw_data_np = np.array(raw_data, dtype=np.float64)
#np.save('./GXBoost'+str(gen_B_size)+'/'+save_id+'raw_data_np_'+instance_type_i+'_'+str(gen_B_size), raw_data_np)
np.save(save_root_dir + 'GXBoost'+str(gen_B_size)+'/'+save_id+'raw_data_np_'+instance_type_i+'_'+str(gen_B_size), raw_data_np)
#결과 확인
#res= np.load('./GXBoost'+str(gen_B_size)+'/'+save_id+'raw_data_np_'+instance_type_i+'_'+str(gen_B_size)+'.npy')
res= np.load(save_root_dir + 'GXBoost'+str(gen_B_size)+'/'+save_id+'raw_data_np_'+instance_type_i+'_'+str(gen_B_size)+'.npy')
print('저장 결과',np.shape(res))
