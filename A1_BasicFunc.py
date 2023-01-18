# -*- coding: utf-8 -*-
#import A3_two_sided
import math
import random
import csv
from numba import jit
import numpy.random
import numpy
import time
import operator

import re_A1_class
import matplotlib.pyplot as plt


@jit(nopython=True)
def distance(p1_x, p1_y, p2_x,p2_y, rider_count = None):
    """
    Calculate 4 digit rounded euclidean distance between p1 and p2
    :para
    m p1:
    :param p2:
    :return: 4 digit rounded euclidean distance between p1 and p2
    """
    """
    counter('distance1')
    if rider_count == 'rider':
        counter('distance2')
    elif rider_count == 'xgboost':
        counter('distance3')
    else:
        pass
    """
    euc_dist = math.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)
    return euc_dist


def counter(func_name):
    if func_name == 'distance1':
        counter.dist1 += 1
    elif func_name == 'distance2':
        counter.dist2 += 1
    elif func_name == 'distance3':
        counter.dist3 += 1
    elif func_name == 'bundle_consist':
        counter.bundle_consist += 1
    elif func_name == 'bundle_consist2':
        counter.bundle_consist2 += 1
    else:
        pass

def counter2(func_name, num):
    if func_name == 'sess1': #XGBoost 투입 데이터 수
        counter2.num1.append(num)
    elif func_name == 'sess2': #XGBoost 종료 후 데이터 수
        counter2.num2.append(num)
    elif func_name == 'sess3':
        counter2.num3.append(num)
    elif func_name == 'old1': #enumerate 데이터 수
        counter2.num4.append(num)
    elif func_name == 'old2':
        counter2.num5.append(num)
    else:
        pass


def t_counter(func_name, t):
    if func_name == 'XGBoost': #RouteConstruct for XGBoost
        #print(t_counter.t1,'->')
        t_counter.t1 += t
        #print(t_counter.t1, t)
        #input('check1:')
    elif func_name == 'old': #BundleConsist
        #print(t_counter.t2, '->')
        t_counter.t2 += t
        #print(t_counter.t2, t)
        #input('check2')
    elif func_name == 'sess': #XGBoost 실행 sess.run()
        #print(t_counter.t3, '->')
        t_counter.t3 += t
        #print(t_counter.t3, t)
        #input('check3')
    elif func_name == 'enumerate': #RouteConstruct for enumerate
        t_counter.t4 += t
        #input('check4')
    elif func_name == 'score_cal': #Bundle 계산 이후 점수 계산 시간
        t_counter.t5 += t
    elif func_name == 'test1':
        t_counter.t6 += t
    elif func_name == 'test2':
        t_counter.t7 += t
    elif func_name == 'test3':
        t_counter.t8 += t
    elif func_name == 'test4':
        t_counter.t9 += t
    elif func_name == 'test5':
        t_counter.t10 += t
    elif func_name == 'test6':
        t_counter.t11 += t
    elif func_name == 'test7':
        t_counter.t12 += t
    elif func_name == 'test8':
        t_counter.t13 += t
    elif func_name == 'test9':
        t_counter.t14 += t
    elif func_name == 'test10':
        t_counter.t15 += t
    elif func_name == 'test11':
        t_counter.t16 += t
    elif func_name == 'test12':
        t_counter.t17 += t
    else:
        pass

def check_list(b_type, element):
    if b_type == 'b2':
        check_list.b2.append(element)
        check_list.b2_count += len(element)
    elif b_type == 'b3':
        check_list.b3.append(element)
        check_list.b3_count += len(element)
    elif b_type == 'unique':
        check_list.suggested_bundle += element
    else:
        pass




def RouteTime(orders, route, M = 10000, speed = 1, uncertainty = False, error = 1, sync_output_para= False, now_t = 0, bywho = 'Rider', time_buffer_para = False):
    """
    Time to move the route with speed
    :param orders: order in route
    :param route: seq
    :param speed: rider speed
    :return: time : float
    """
    #print('ROUTE SAMPLE', route)
    if uncertainty == False:
        exp_error = [1]
    else:
        exp_error = numpy.random.normal(1, 100)
        input('By What?')
    time = 0
    locs = {}
    names = []
    time_buffer = []
    un_sync_t = [0,0] #[차량 대기 시간, 음식 대기 시간]
    if type(orders) == dict:
        for order_name in orders:
            locs[order_name + M] = [orders[order_name].store_loc, 'store', orders[order_name].time_info[6],orders[order_name].time_info[0] +orders[order_name].time_info[5]]
            locs[order_name] = [orders[order_name].location, 'customer', orders[order_name].time_info[7],orders[order_name].time_info[0] +orders[order_name].time_info[5]]
            names += [order_name + M, order_name]
    elif type(orders) == list:
        for order in orders:
            locs[order.name + M] = [order.store_loc, 'store', order.time_info[6], order.time_info[0] + order.time_info[5]]
            locs[order.name] = [order.location, 'customer', order.time_info[7], order.time_info[0] + order.time_info[5]]
            names += [order.name + M, order.name]
    else:
        input('Error')
    #print('고려 대상들{} 경로{}'.format(list(locs.keys()), route))
    #print('locs',locs)
    for index in range(1,len(route)):
        bf = route[index-1]
        bf_loc = locs[bf][0]
        af = route[index]
        #print(1, bf,af,time)
        af_loc = locs[af][0]
        time += (distance(bf_loc[0],bf_loc[1],af_loc[0],af_loc[1])/speed) * random.choice(exp_error) + locs[af][2]
        if af > M:
            for order in orders:
                if order.name == af - M:
                    target = order
                    break
            #print(2, bf, af, time,target.cook_info,uncertainty)
            if target.cook_start_time > 0: #todo: 이미 조리 시작된 음식
                if bywho == 'Rider':
                    slack_t = (target.cook_start_time + target.dp_cook_time) - (now_t + time)
                else:
                    slack_t = (target.cook_start_time + target.actual_cook_time) - (now_t + time)
                if slack_t > 0:
                    time += slack_t
                    un_sync_t[0] += slack_t
                else:
                    un_sync_t[1] += -slack_t
                    pass#음식이 기다리는 상황
            else: #todo: 아직 조리 시작 되지 X음식.
                if bywho == 'Rider':
                    slack_t = target.dp_cook_time - time
                elif bywho == 'Test':
                    slack_t = target.actual_cook_time - time - 5
                else:
                    slack_t = target.actual_cook_time - time
                if slack_t > 0:
                    time += slack_t
                    un_sync_t[0] += slack_t
                else:
                    un_sync_t[1] += -slack_t
                    pass #음식이 기다리는 시간이 발생.
            """
            if (uncertainty == True and target.cook_info[0] == 'uncertainty'): #todo : 가게인 경우 추가 시간이 발생할 수 있음을 반영
                pool = numpy.random.normal(target.cook_info[1][0], target.cook_info[1][1]*error, 1000)
                exp_cook_time = random.choice(pool)
                if exp_cook_time > time:
                    #print('추가시간', exp_cook_time - time)
                    time += exp_cook_time - time
                #input('작동 확인1')            
            """
        else:
            #print('af_loc',af_loc, locs[af])
            #time_buffer.append(af_loc[3] - (now_t + time))
            time_buffer.append(locs[af][3] - (now_t + time))
        #input('작동 확인2')
    if sync_output_para == True:
        if time_buffer_para == True:
            return time, un_sync_t,time_buffer
        else:
            return time, un_sync_t
    else:
        return time


def FLT_Calculate(customer_in_order, customers, route, p2, except_names , M = 10000, speed = 1, now_t = 0, uncertainty = False, exp_error = 1, time_buffer_para = False):
    """
    Calculate the customer`s Food Delivery Time in route(bundle)

    :param orders: customer order in the route. type: customer class
    :param route: customer route. [int,...,]
    :param p2: allowable FLT increase
    :param speed: rider speed
    :return: Feasiblity : True/False, FLT list : [float,...,]
    """
    names = []
    for order in customer_in_order:
        if order.name not in names:
            names.append(order.name)
    ftds = []
    time_buffer = 0
    #input(''.format())
    #print('경로 고객들 {} 경로 {}'.format(names, route))
    #input('체크1 {} 체크2 {}'.format(customer_in_order,customers))
    for order_name in names:
        if order_name not in except_names:
            #rev_p2 = p2
            #rev_p2 = customers[order_name].p2*p2 + customers[order_name].time_info[6] + customers[order_name].time_info[7] #todo : 시간에 민감한 부분.
            if customers[order_name].p2_type1 == 'test1':
                rev_p2 = customers[order_name].p2 + customers[order_name].manual_p2+ customers[order_name].time_info[6]  + customers[order_name].time_info[7] #사전에 계산된 대로, 음식 메뉴마다 p2가 상이함.
            else:
                rev_p2 = customers[order_name].p2 * p2 + customers[order_name].time_info[6] + customers[order_name].time_info[7]  # todo : 시간에 민감한 부분.
                input('error1')
            #rev_p2 = customers[order_name].p2 + customers[order_name].time_info[6] + customers[order_name].time_info[7] # todo : 221101실험을 현실적으로 변경. -> 고객 마다 p2가 달라짐.
            #input('p2 확인 1 :: {}'.format(rev_p2))
            if customers[order_name].time_info[2] != None:
                #print('FLT 고려 대상 {} 시간 정보 {}'.format(order_name,customers[order_name].time_info))
                last_time = now_t - customers[order_name].time_info[2] #이미 음식이 실린 후 지난 시간
                #rev_p2 = p2 - last_time
                rev_p2 = customers[order_name].min_FLT - last_time
                #rev_p2 = rev_p2 - last_time # todo : 221101실험을 현실적으로 변경. -> 고객 마다 p2가 달라짐.
                #input('p2 확인 2 :: {}'.format(rev_p2))
            try:
                s = route.index(order_name + M)
                e = route.index(order_name)

                try:
                    ftd = RouteTime(customer_in_order, route[s: e + 1], speed=speed, M=M, uncertainty=uncertainty, error = exp_error, now_t = now_t)
                except:
                    ftd = 1000
                    print('경로 {}, s:{}, e :{}'.format(route,s,e))
                    print('경로 시간 계산 에러/ 현재고객 {}/ 경로 고객들 {}'.format(order_name,names))
                    input('중지')
            except:
                ftd = 0
                print('경로 {}'.format(route))
                print('인덱스 에러 발생 현재 고객 이름 {} 경로 고객들 {} 경로 {}'.format(order_name, names, route))
                #input('인덱스 에러 발생')
            #s = route.index(order_name + M)
            #e = route.index(order_name)
            if ftd > rev_p2:
                #print('시간 비교',customers[order_name].p2,ftd, rev_p2)
                if time_buffer_para == True:
                    return False, [], time_buffer
                else:
                    return False, []
            else:
                ftds.append(ftd)
    if time_buffer_para == True:
        return True, ftds,time_buffer
    else:
        return True, ftds


def RiderGenerator(env, Rider_dict, Platform, Store_dict, Customer_dict, capacity = 3, speed = 1, working_duration = 120, interval = 1, runtime = 1000,
                   gen_num = 10, history = None, freedom = True, score_type = 'simple', wait_para = False, uncertainty = False, exp_error = 1, exp_WagePerHr = 9000,
                   platform_recommend = False, input_order_select_type = None, bundle_construct = False,  p2 = 1.5, ite = 1, dir = None ):
    """
    Generate the rider until t <= runtime and rider_num<= gen_num
    :param env: simpy environment
    :param Rider_dict: 플랫폼에 있는 라이더들 {[KY]rider name : [Value]class rider, ...}
    :param rider_name: 라이더 이름 int+
    :param Platform: 플랫폼에 올라온 주문들 {[KY]order index : [Value]class order, ...}
    :param Store_dict: 플랫폼에 올라온 가게들 {[KY]store name : [Value]class store, ...}
    :param Customer_dict:발생한 고객들 {[KY]customer name : [Value]class customer, ...}
    :param working_duration: 운행 시작 후 운행을 하는 시간
    :param interval: 라이더 생성 간격
    :param runtime: 시뮬레이션 동작 시간
    :param gen_num: 생성 라이더 수
    """
    rider_num = 0
    if dir != None:
        locs = []
        f = open(dir, 'r')
        tem = f.readlines()
        tem = tem[ite]
        tem = tem.split(';')
        tem = tem[1:]
        print(len(tem), tem[1:4])
        for i in range(0, len(tem) - 1, 2):
            try:
                locs.append([int(tem[i]), int(tem[i + 1])])
            except:
                print(i, 'error 발생')
        f.close()
    while env.now <= runtime and rider_num <= gen_num:
        #single_rider = A1_Class.Rider(env,rider_num,Platform, Customer_dict,  Store_dict, start_time = env.now ,speed = speed, end_t = working_duration, capacity = capacity, freedom=freedom, order_select_type = score_type, wait_para =wait_para, uncertainty = uncertainty, exp_error = exp_error)
        """
        single_rider = re_A1_class.Rider(env,rider_num,Platform, Customer_dict,  Store_dict, start_time = env.now ,speed = speed,
                                         end_t = working_duration, capacity = capacity, freedom=freedom, order_select_type = score_type,
                                         wait_para =wait_para, uncertainty = uncertainty, exp_error = exp_error)
        """
        try:
            this_loc = locs[rider_num]
        except:
            this_loc = [25,25]
        single_rider = re_A1_class.Rider(env,rider_num,Platform, Customer_dict,  Store_dict, start_time = env.now ,speed = speed, end_t = working_duration, \
                                   capacity = capacity, freedom=freedom, order_select_type = input_order_select_type, wait_para =wait_para, \
                                      uncertainty = uncertainty, exp_error = exp_error, platform_recommend = platform_recommend,
                                         bundle_construct= bundle_construct, loc = this_loc)

        single_rider.exp_wage = exp_WagePerHr
        Rider_dict[rider_num] = single_rider
        #print('T {} 라이더 {} 생성'.format(int(env.now), rider_num))
        print('라이더 {} 생성. T {}'.format(rider_num, int(env.now)))
        if history != None:
            #next = history[rider_num + 1] - history[rider_num]
            next = history[rider_num]
            yield env.timeout(next)
        else:
            yield env.timeout(interval)
        rider_num += 1
        if rider_num == gen_num:
            print('라이더 수',len(Rider_dict))
            #input('라이더 생성 완료')


def RiderGeneratorByCSV(env, csv_dir, Rider_dict, Platform, Store_dict, Customer_dict, working_duration = 120, exp_WagePerHr = 9000 ,input_speed = None,
                        input_capacity = None, platform_recommend = False, input_order_select_type = None, bundle_construct = False, rider_num = 5,
                        lamda_list = None, p2 = 1.5, ite = 1, rider_select_print_fig = False, mix_ratio = None):
    """
    Generate the rider until t <= runtime and rider_num<= gen_num
    :param env: simpy environment
    :param Rider_dict: 플랫폼에 있는 라이더들 {[KY]rider name : [Value]class rider, ...}
    :param rider_name: 라이더 이름 int+
    :param Platform: 플랫폼에 올라온 주문들 {[KY]order index : [Value]class order, ...}
    :param Store_dict: 플랫폼에 올라온 가게들 {[KY]store name : [Value]class store, ...}
    :param Customer_dict:발생한 고객들 {[KY]customer name : [Value]class customer, ...}
    :param working_duration: 운행 시작 후 운행을 하는 시간
    :param interval: 라이더 생성 간격
    :param runtime: 시뮬레이션 동작 시간
    :param gen_num: 생성 라이더 수
    """
    datas = ReadCSV(csv_dir, interval_index = 1)
    interval_index = len(datas[0]) - 1
    for data in datas:
        name = data[0]
        if input_speed == None:
            speed = data[2]
        else:
            speed = input_speed
        if input_capacity == None:
            capacity = data[4]
        else:
            capacity = input_capacity
        freedom = data[5]
        if input_order_select_type == None:
            order_select_type = data[6]
        else:
            order_select_type = input_order_select_type
        #order_select_type = data[6]
        wait_para = data[7]
        uncertainty = data[8]
        exp_error = data[9]
        if lamda_list == None:
            lamda = 5
        else:
            lamda = lamda_list[name]
        #single_rider = A1_Class.Rider(env,name,Platform, Customer_dict,  Store_dict, start_time = env.now ,speed = speed, end_t = working_duration, \
        #                           capacity = capacity, freedom=freedom, order_select_type = order_select_type, wait_para =wait_para, \
        #                              uncertainty = uncertainty, exp_error = exp_error, platform_recommend = platform_recommend)
        single_rider = re_A1_class.Rider(env,name,Platform, Customer_dict,  Store_dict, start_time = env.now ,speed = speed, end_t = working_duration, \
                                   capacity = capacity, freedom=freedom, order_select_type = order_select_type, wait_para =wait_para, \
                                      uncertainty = uncertainty, exp_error = exp_error, platform_recommend = platform_recommend,
                                         bundle_construct= bundle_construct, lamda= lamda, p2 = p2, ite = ite)
        single_rider.rider_select_print_fig = rider_select_print_fig
        if mix_ratio != None and name < rider_num*mix_ratio:
            single_rider.bundle_construct = True
        #input('확인 {}'.format(single_rider.bundle_construct))
        if platform_recommend == False:
            single_rider.onhand_bundle = [-1,-1,-1]
        single_rider.exp_wage = exp_WagePerHr
        Rider_dict[name] = single_rider
        interval = data[interval_index]
        if interval > 0:
            yield env.timeout(interval)
        else:
            print('현재 T :{} / 마지막 고객 {} 생성'.format(int(env.now), name))
        if name >= rider_num :
            break


def GenerateStoreByCSVStressTest(env, num, platform,Store_dict, mus = [5,10,15], std_ratio = 0.2, store_type = 'Instance_random', ITE = 1, output_data = None,
                                 detail_pr = None, customer_pend = True, store_capacity = 100, csv_dir = None):
    # detail_pr = [rest_type_list, pr_list, frt_list, temperature_list, p2_list] -> array
    #mus = [11.5,13.5,15.5]
    if csv_dir != None:
        try:
            datas = ReadCSV(csv_dir)
        except:
            print('dir::',str(csv_dir))
            input('csv_dir이 없음')
    rest_type_check = []
    for count in range(num):
        if csv_dir != None:
            if count == len(datas):
                break #csv 파일을 모두 읽었기 때문에 삭제
            data = datas[count]
            name = data[0]
            loc = [data[1], data[2]]
            order_ready_time = data[3]
            capacity = data[4]
            slack = data[5]
        else:
            name = count
            if output_data == None:
                if store_type == 'Instance_random':
                    loc = [random.randint(10, 40), random.randint(10, 40)]
                else:
                    loc = [random.randint(20, 30), random.randint(20, 30)]
            else:
                try:
                    print('기존 가게')
                    loc = [output_data[count][1],output_data[count][2]]
                except:
                    pass
            #['name', 'start_loc_x', 'start_loc_y', 'order_ready_time', 'capacity', 'slack']
            order_ready_time = 5
            capacity = store_capacity
            slack = 2
        store = re_A1_class.Store(env, platform, name, loc=loc, order_ready_time=order_ready_time, capacity=capacity, print_para=False, slack = slack, customer_pend= customer_pend)
        rv = random.random()
        if rv <= 0.2: #
            store.FRT = numpy.random.normal(mus[0], mus[0]*std_ratio, 1000)
        elif rv <= 0.8:
            store.FRT = numpy.random.normal(mus[1], mus[1]*std_ratio, 1000)
        else:
            store.FRT = numpy.random.normal(mus[2], mus[2] * std_ratio, 1000)
        """
        if detail_pr != None: # todo : 221101실험을 현실적으로 변경.
            rest_type = numpy.random.choice(detail_pr[0],p = detail_pr[1])
            store.FRT = detail_pr[2][detail_pr[0].index(rest_type)]
            store.rest_type = rest_type
            store.temperature = detail_pr[3][detail_pr[0].index(rest_type)]
            store.p2 = detail_pr[4][detail_pr[0].index(rest_type)]
            rest_type_check.append(rest_type)        
        """
        if detail_pr != None:
            if name in detail_pr[0]:
                store.p2 = detail_pr[1]
            else:
                store.p2 = detail_pr[2]
        Store_dict[name] = store

    """
    f3 = open("가게_coord_정보" + str(ITE) + '_' + store_type + ".txt", 'a')
    for store_name in Store_dict:
        store = Store_dict[store_name]
        con = '{};{};{};\n'.format(store.name, store.location[0], store.location[1])
        f3.write(con)
    f3.write('Exp End' + '\n')
    f3.close()
    """
    print(list(set(rest_type_check)))
    print(len(Store_dict))
    #input('가게')

def GenerateStoreByCSVStressTest2(env, platform,Store_dict, order_dir, mus = [5,10,15], std_ratio = 0.2, store_type = 'Instance_random', ITE = 1, output_data = None,
                                 detail_pr = None, customer_pend = True, store_capacity = 100, csv_dir = None):
    f = open(order_dir , 'r')
    leadlines = f.readlines()
    store_names = []
    store_infos = []
    for info in leadlines[1:]:
        tem = info.split(';')
        data = [int(tem[1]), float(tem[2]), float(tem[3]), int(tem[5]) , float(tem[6]), float(tem[7])]
        if int(tem[1]) not in store_names:
            store_names.append(int(tem[1]))
            store_infos.append([int(tem[1]), float(tem[2]), float(tem[3])])
    for store_info in store_infos:
        name = store_info[0]
        loc = [store_info[1],store_info[2]]
        #['name', 'start_loc_x', 'start_loc_y', 'order_ready_time', 'capacity', 'slack']
        order_ready_time = 5
        capacity = store_capacity
        slack = 2
        store = re_A1_class.Store(env, platform, name, loc=loc, order_ready_time=order_ready_time, capacity=capacity, print_para=False, slack = slack, customer_pend= customer_pend)
        rv = random.random()
        if rv <= 0.2: #
            store.FRT = numpy.random.normal(mus[0], mus[0]*std_ratio, 1000)
        elif rv <= 0.8:
            store.FRT = numpy.random.normal(mus[1], mus[1]*std_ratio, 1000)
        else:
            store.FRT = numpy.random.normal(mus[2], mus[2] * std_ratio, 1000)
        if detail_pr != None: # todo : 221101실험을 현실적으로 변경.
            rest_type = numpy.random.choice(detail_pr[0],p = detail_pr[1])
            store.FRT = detail_pr[2][detail_pr[0].index(rest_type)]
            store.rest_type = rest_type
            store.temperature = detail_pr[3][detail_pr[0].index(rest_type)]
            store.p2 = detail_pr[4][detail_pr[0].index(rest_type)]
        Store_dict[name] = store
    """
    f3 = open("가게_coord_정보" + str(ITE) + '_' + store_type + ".txt", 'a')
    for store_name in Store_dict:
        store = Store_dict[store_name]
        con = '{};{};{};\n'.format(store.name, store.location[0], store.location[1])
        f3.write(con)
    f3.write('Exp End' + '\n')
    f3.close()
    """
    print(len(Store_dict))
    #input('가게')
def ReadCSV(csv_dir, interval_index = None):
    raw_datas = []
    datas = []
    #csv 파일 읽기
    f = open(csv_dir+'.csv','r')
    rdr = csv.reader(f)
    for line in rdr:
        raw_datas.append(line)
    f.close()
    for raw_data in raw_datas[1:]:
        tem = []
        for info in raw_data:
            try:
                num = float(info)
                if round(num) == num:
                    tem.append(int(num))
                else:
                    tem.append(num)
            except:
                tem.append(str(info))
        datas.append(tem)
    if interval_index != None:
        for index in range(1, len(datas)):
            interval = datas[index][interval_index] - datas[index - 1][interval_index]
            datas[index - 1].append(interval)
        datas[-1].append(0)
    return datas

def OrdergeneratorByCSV(env, csv_dir, orders, stores, platform = None, p2_ratio = None, rider_speed = 1, unit_fee = 110, fee_type = 'linear', service_time_diff = False, custom_data = None, shuffle = False):
    """
    Generate customer order
    :param env: Simpy Env
    :param orders: Order
    :param platform: 플랫폼에 올라온 주문들 {[KY]order index : [Value]class order, ...}
    :param stores: 플랫폼에 올라온 가게들 {[KY]store name : [Value]class store, ...}
    :param interval: 주문 생성 간격
    :param runtime: 시뮬레이션 동작 시간
    """
    #CSV 파일 읽기
    if custom_data == None:
        datas = ReadCSV(csv_dir, interval_index = 1)
    else:
        datas = custom_data
    interval_index = len(datas[0]) - 1
    if shuffle == True:
        random.shuffle(datas)
    count = 0
    for data in datas:
        #[customer.name, customer.time_info[0], customer.location[0],customer.location[1], customer.store, customer.store_loc[0],customer.store_loc[1], customer.p2, customer.cook_time, customer.cook_info[0], customer.cook_info[1][0], customer.cook_info[1][1]]
        if shuffle == True:
            name = count
        else:
            name = data[0]
        gen_t = data[1]
        input_location = [data[2],data[3]]
        store_num = data[4]
        store_loc = [data[5], data[6]]
        if p2_ratio == None:
            p2 = data[7]
        else:
            #p2 = (data[7] / rider_speed) * p2_ratio
            OD_dist = distance(input_location[0],input_location[1],store_loc[0],store_loc[1])
            if abs(distance(input_location[0],input_location[1],store_loc[0],store_loc[1]) - data[7]) > 0.5:
                p2 = (OD_dist/rider_speed)*p2_ratio
            else:
                p2 = (data[7]/rider_speed)*p2_ratio
            print('data체크',data[0],data[7],OD_dist,rider_speed,p2_ratio)
        count += 1
        #input('거리 {} / 생성 p2 {}/ 라이더 스피드{} / p2% {}'.format(distance(input_location, store_loc),p2, rider_speed, p2_ratio))
        cook_time = data[8]
        cook_time_type = data[9]
        cooking_time = [data[10], data[11]]
        #order = A1_Class.Customer(env, name, input_location, store=store_num, store_loc=store_loc, p2=p2,
        #                       cooking_time=cook_time, cook_info=[cook_time_type, cooking_time])
        order = re_A1_class.Customer(env, name, input_location, store=store_num, store_loc=store_loc, p2=p2,
                               cooking_time=cook_time, cook_info=[cook_time_type, cooking_time], platform = platform, unit_fee = unit_fee, fee_type = fee_type)
        order.actual_cook_time = random.choice(stores[store_num].FRT)
        order.dp_cook_time = 5*(1 + order.actual_cook_time//5)
        if order.dp_cook_time >= 15:
            order.cooking_process = env.process(order.CookingFirst(env, order.actual_cook_time)) #todo : 15분 이상 음식은 미리 조리 시작
        print('T {} 음식 {} 조리 확인/ 시간 {}'.format(int(env.now), order.name,order.actual_cook_time))
        orders[name] = order
        stores[store_num].received_orders.append(orders[name])
        interval = data[interval_index]
        if service_time_diff == True:
            try:
                order.time_info[7] = int(data[12]/2) #서비스 시간에 서로 다른 값을 넣기.
            except:
                interval = random.randint(1, 4)
                print('gen error',data[12])
        if shuffle == True:
            interval = random.randint(1,4)
        #todo : 0317 지연되는 조건 생각할 것.
        if interval > 0:
            yield env.timeout(interval)
        else:
            print('현재 T :{} / 마지막 고객 {} 생성'.format(int(env.now), name))
            pass



def OrdergeneratorByCSVForStressTest(env, orders, stores, lamda, platform = None, p2_ratio = 1, rider_speed = 1, unit_fee = 110, fee_type = 'linear',
                                     output_data = None, cooktime_detail = None, cook_first = False, customer_pend = False, manual_cook_time = 7):
    """
    Generate customer order
    :param env: Simpy Env
    :param orders: Order
    :param platform: 플랫폼에 올라온 주문들 {[KY]order index : [Value]class order, ...}
    :param stores: 플랫폼에 올라온 가게들 {[KY]store name : [Value]class store, ...}
    :param interval: 주문 생성 간격
    :param runtime: 시뮬레이션 동작 시간
    """
    dist_distribution = numpy.random.poisson(20,7) #todo: 0915 거리 조절
    for count in range(1000000):
        if output_data == None:
            store_name = random.choice(range(len(stores)))
            store = stores[store_name]
            store_loc = store.location
            #req_dist = random.randint(5,20)*rider_speed
            req_dist = max(random.choice(dist_distribution), 7)
            angle = math.radians(random.randrange(0, 360))
            num = 0
            while num < 1000:
                customer_loc = [store_loc[0] + round(req_dist*math.cos(angle),4),store_loc[1] + round(req_dist*math.sin(angle),4) ]
                if customer_loc[0] <= 50 and customer_loc[1] <= 50 and store_loc != customer_loc:
                    break
                num += 1
            if num == 1000:
                customer_loc = [random.randint(0,50),random.randint(0,50)]
        else:
            if count > len(output_data) - 2:
                break
            store_name = output_data[count][3]
            store_loc = [output_data[count][4], output_data[count][5]]
            customer_loc = [output_data[count][1], output_data[count][2]]
        name = count
        if cooktime_detail != None:
            cook_time = numpy.random.choice(cooktime_detail[0], p = cooktime_detail[1]) # todo : 221101실험을 현실적으로 변경.
            p2_ratio2 = store.p2
        else:
            cook_time = manual_cook_time
            p2_ratio2 = p2_ratio
        OD_dist = distance(store_loc[0],store_loc[1], customer_loc[0],customer_loc[1])
        p2 = (OD_dist / rider_speed) * p2_ratio2 # todo : 221101실험을 현실적으로 변경.
        cook_time_type = 0
        cooking_time = [7,1]
        #order = A1_Class.Customer(env, name, input_location, store=store_num, store_loc=store_loc, p2=p2,
        #                       cooking_time=cook_time, cook_info=[cook_time_type, cooking_time])
        order = re_A1_class.Customer(env, name, customer_loc, store=store_name, store_loc=store_loc, p2=p2,
                               cooking_time=cook_time, cook_info=[cook_time_type, cooking_time], platform = platform, unit_fee = unit_fee, fee_type = fee_type)
        #order.actual_cook_time = random.choice(stores[store_name].FRT)
        order.actual_cook_time = cook_time
        #order.dp_cook_time = cook_time
        order.dp_cook_time = 5*(1 + order.actual_cook_time//5)
        if cooktime_detail != None:
            order.temperature = store.temperature
            order.rest_type = store.rest_type
        else:
            order.temperature = 'T'
        if order.dp_cook_time >= 15 and cook_first == True:
            order.cooking_process = env.process(order.CookingFirst(env, order.actual_cook_time)) #todo : 15분 이상 음식은 미리 조리 시작
        order.cancel = customer_pend
        if customer_pend == False:
            if len(list(platform.platform.keys())) > 0:
                task_index = max(list(platform.platform.keys())) + 1
            else:
                task_index = 1
            platform_exp_error = 1
            pool = numpy.random.normal(order.cook_info[1][0], order.cook_info[1][1] * platform_exp_error, 1000)
            order.platform_exp_cook_time = random.choice(pool)
            route = [[order.name, 0, order.store_loc, 0], [order.name, 1, order.location, 0]]
            o = re_A1_class.Order(task_index, [order.name], route, 'single', fee=order.fee, parameter_info=None)
            platform.platform[task_index] = o
        print('T {} 음식 {} 조리 확인/ 시간 {}'.format(int(env.now), order.name,order.actual_cook_time))
        orders[name] = order
        stores[store_name].received_orders.append(orders[name])
        interval = 1.0/lamda
        if interval > 0:
            yield env.timeout(interval)
        else:
            print('현재 T :{} / 마지막 고객 {} 생성'.format(int(env.now), name))
            pass



def OrdergeneratorByCSVForStressTest2(env, orders, stores, lamda, order_dir, platform = None, p2_ratio = 1, rider_speed = 1, unit_fee = 110, fee_type = 'linear', cook_first = False, customer_pend = False, manual_cook_time = 7):
    """
    Generate customer order
    :param env: Simpy Env
    :param orders: Order
    :param platform: 플랫폼에 올라온 주문들 {[KY]order index : [Value]class order, ...}
    :param stores: 플랫폼에 올라온 가게들 {[KY]store name : [Value]class store, ...}
    :param interval: 주문 생성 간격
    :param runtime: 시뮬레이션 동작 시간
    """
    f = open(order_dir , 'r')
    leadlines = f.readlines()
    order_infos = []
    for info in leadlines[1:]:
        tem = info.split(';')
        data = [int(tem[1]), float(tem[2]), float(tem[3]), int(tem[5]) , float(tem[6]), float(tem[7])]
        order_infos.append(data)
    f.close()
    count = 0
    for info in order_infos:
        store_name = info[0]
        store = stores[store_name]
        store_loc = store.location
        customer_loc = [info[4], info[5]]
        name = count
        cook_time = manual_cook_time
        p2_ratio2 = p2_ratio
        OD_dist = distance(store_loc[0],store_loc[1], customer_loc[0],customer_loc[1])
        p2 = (OD_dist / rider_speed) * p2_ratio2 # todo : 221101실험을 현실적으로 변경.
        cook_time_type = 0
        cooking_time = [7,1]
        #order = A1_Class.Customer(env, name, input_location, store=store_num, store_loc=store_loc, p2=p2,
        #                       cooking_time=cook_time, cook_info=[cook_time_type, cooking_time])
        order = re_A1_class.Customer(env, name, customer_loc, store=store_name, store_loc=store_loc, p2=p2,
                               cooking_time=cook_time, cook_info=[cook_time_type, cooking_time], platform = platform, unit_fee = unit_fee, fee_type = fee_type)
        #order.actual_cook_time = random.choice(stores[store_name].FRT)
        order.actual_cook_time = cook_time
        #order.dp_cook_time = cook_time
        order.dp_cook_time = 5*(1 + order.actual_cook_time//5)
        order.temperature = 'T'
        if order.dp_cook_time >= 15 and cook_first == True:
            order.cooking_process = env.process(order.CookingFirst(env, order.actual_cook_time)) #todo : 15분 이상 음식은 미리 조리 시작
        order.cancel = customer_pend
        if customer_pend == False:
            if len(list(platform.platform.keys())) > 0:
                task_index = max(list(platform.platform.keys())) + 1
            else:
                task_index = 1
            platform_exp_error = 1
            pool = numpy.random.normal(order.cook_info[1][0], order.cook_info[1][1] * platform_exp_error, 1000)
            order.platform_exp_cook_time = random.choice(pool)
            route = [[order.name, 0, order.store_loc, 0], [order.name, 1, order.location, 0]]
            o = re_A1_class.Order(task_index, [order.name], route, 'single', fee=order.fee, parameter_info=None)
            platform.platform[task_index] = o
        print('T {} 음식 {} 조리 확인/ 시간 {}'.format(int(env.now), order.name,order.actual_cook_time))
        orders[name] = order
        stores[store_name].received_orders.append(orders[name])
        interval = 1.0/lamda
        count += 1
        if interval > 0:
            yield env.timeout(interval)
        else:
            print('현재 T :{} / 마지막 고객 {} 생성'.format(int(env.now), name))
            pass



def ReadRiderData(env, rider_data, Platform, Rider_dict, Customer_dict, Store_dict):
    #저장된 txt 데이터를 읽고, 그에 따라서 인스턴스 생성
    #rider_data = [name, start_loc, gen_time, ExpectWagePerHr]
    #order_data = [name, store_num, loc , gen_time]
    #stroe_data = [name, capacity, loc]
    f = open(rider_data + ".txt", 'r')
    lines = f.readlines()
    rider_num = 1
    for line in lines[1:]:
        line.split('')
        """
        single_rider = Class.Rider(env, rider_num, Platform, Customer_dict, Store_dict, start_time = env.now, speed = speed, end_t = working_duration,\
                                   capacity = capacity, freedom = freedom, order_select_type = score_type, wait_para = wait_para, uncertainty = uncertainty\
                                   exp_error = exp_error)
        single_rider.exp_wage = exp_WagePerHr      
        Rider_dict[rider_num] = single_rider          
        """
        interval = 1
        rider_num += 1
        yield env.timeout(interval)

    f.close()
    return None


def UpdatePlatformByOrderSelection(platform, order_index):
    """
    선택된 주문과 겹치는 고객을 가지는 주문이 플랫폼에 존재한다면, 해당 주문을 삭제하는 함수.
    @param platform: class platform
    @param order_index: 라이더가 선택한 주문.
    """
    delete_order_index = []
    order = platform.platform[order_index]
    for order_index1 in platform.platform:
        compare_order = platform.platform[order_index1]
        if order_index != order_index1:
            duplicate_customers = list(set(order.customers).intersection(compare_order.customers))
            if len(duplicate_customers) > 0:
                delete_order_index.append(compare_order.index)
    for order_index in delete_order_index:
        del platform.platform[order_index]


def ActiveRiderCalculator(rider, t_now = 0, option = None, interval = 5, print_option = True):
    """
    현재 라이더가 새로운 주문을 선택할 수 있는지 유/무를 계산.
    @param rider: class rider
    @return: True/ False
    """
    if t_now <= rider.end_t :
        if len(rider.resource.users) == 0:
            return True
        else:
            if rider.exp_end_time <= t_now and len(rider.picked_orders) < rider.max_order_num:
                return True
            else:
                return False
    else:
        return False


def ActiveRiderCalculator_ORG(rider, t_now = 0, option = None, interval = 5, print_option = True):
    """
    현재 라이더가 새로운 주문을 선택할 수 있는지 유/무를 계산.
    @param rider: class rider
    @return: True/ False
    """
    if t_now <= rider.end_t :
        #print('ActiveRiderCalculator check')
        #print (rider.name ,'::', len(rider.picked_orders) , rider.max_order_num, '::',t_now , rider.next_search_time2 , t_now + interval)
        if option == None:
            if len(rider.picked_orders) < rider.max_order_num:
                if print_option == True:
                    print('문구1/ 라이더 {} / 현재 OnHandOrder# {} / 최대 주문 수{} / 예상 선택 시간 {} / 다음 interval 시간 {}'.format(rider.name,len(rider.picked_orders), rider.max_order_num, round(rider.next_search_time2,2), t_now + interval))
                return True
        else:
            if len(rider.picked_orders) <= rider.max_order_num and (t_now <= rider.next_search_time2 <= t_now + interval or len(rider.resource.users) == 0):
                if print_option == True:
                    print('문구2/ 라이더 {} / 현재 OnHandOrder# {} / 최대 주문 수{} / 예상 선택 시간 {} / 다음 interval 시간 {}'.format(rider.name,len(rider.picked_orders), rider.max_order_num, round(rider.next_search_time2,2), t_now + interval))
                return True
    else:
        print('False time3')
        return False

def WillingtoWork(rider, t_now):
    """
    시간당 수익이 희망 시간당 수익보다 작은 경우의 그 양
    max(희망 시간당 수익 - 현재 라이더의 시간당 수익 ,0)
    @param rider: class rider
    @return: max(희망 시간당 수익 - 현재 라이더의 시간당 수익 ,0)
    """
    current_wagePerHr  = rider.income/((t_now - rider.gen_time)/60)
    if current_wagePerHr >= rider.exp_wage: #임금이 자신의 허용 범위보다 작다면
        return current_wagePerHr
    else:
        return 0


def ForABundleCount(route_info):
    B = []
    num_bundle_customer = 0
    bundle_start = 0
    b = 0
    for node in route_info:
        if node[1] == 0:
            store_index = route_info.index(node)
            for node2 in route_info[store_index:]:
                if node[0] == node2[0] and node2[1] == 1:
                    customer_index = route_info.index(node2)
                    if store_index + 1 < customer_index:
                        num_bundle_customer += 1
                        if store_index == bundle_start + 1:
                            #print('A', bundle_start, store_index, customer_index)
                            b += 1
                        else:
                            #print('B',bundle_start, store_index, customer_index)
                            B.append(b)
                            b = 0
                    break
            bundle_start = store_index
    return B, num_bundle_customer

def ResultSave(Riders, Customers, title = 'Test', sub_info = 'None', type_name = 'A', tm = None, add_info= 'test',ite = '0', add_dir = None):
    if tm == None:
        tm = time.localtime(time.time())
    sub = ['Day {} Hr{}Min{}Sec{}/ SUB {} /sctype {}/ITE {}'.format(tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec,sub_info,add_info,ite)]
    save_key = 'Day;{};Hr;{};Min;{};Sec;{};sctype;{};ITE;{};'.format(tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec,add_info,ite)
    rider_header = ['라이더 이름', '서비스 고객수', '주문 탐색 시간','선택한 번들 수','번들로 서비스된 고객 수','라이더 수익','음식점 대기시간','대기시간_번들','대기시간_단건주문','주문 선택 간격','대기 시간','주문이 없는 시간 수','경로']
    rider_infos = [sub,rider_header]
    for rider_name in Riders:
        rider = Riders[rider_name]
        if len(rider.bundle_store_wait) > 0:
            bundle_store_wait = round(sum(rider.bundle_store_wait) / len(rider.bundle_store_wait), 2)
        else:
            bundle_store_wait = 0
        if len(rider.single_store_wait) > 0:
            single_store_wait = round(sum(rider.single_store_wait) / len(rider.single_store_wait), 2)
        else:
            single_store_wait = None
        if type_name == 'A':
            bundle_num, num_bundle_customer = ForABundleCount(rider.visited_route)
            #input('라이더 {} : 번들 정보 {} : 번들 고객 수 {}'.format(rider.name, bundle_num,num_bundle_customer))
            rider.b_select = round(num_bundle_customer/2.5,2)
            rider.num_bundle_customer = num_bundle_customer
        decision_moment = []
        for time_index in range(1,len(rider.decision_moment)):
            decision_interval = rider.decision_moment[time_index] - rider.decision_moment[time_index - 1]
            decision_moment.append(decision_interval)
        #print('주문간격 시점 데이터 {}'.format(decision_moment))
        try:
            decision_moment = round(sum(decision_moment) / len(decision_moment), 2)
        except:
            decision_moment = 0
        #print('평균 주문간격{}'.format(decision_moment))
        info = [rider_name, len(rider.served), rider.idle_time, rider.b_select,rider.num_bundle_customer, int(rider.income), round(rider.store_wait,2) ,bundle_store_wait,single_store_wait,decision_moment,rider.check_t,rider.empty_serach_count,rider.visited_route]
        rider_infos.append(info)
    customer_header = ['고객 이름', '생성 시점', '라이더 선택 시점','가게 출발 시점','고객 도착 시점','가게 도착 시점','음식조리시간','음식 음식점 대기 시간'
        ,'라이더 가게 대기시간1','라이더 가게 대기시간2','수수료', '수행 라이더 정보', '직선 거리','p2(민감정도)','번들여부','조리시간','기사 대기 시간'
        ,'번들로 구성된 시점', '취소','LT', 'FLT', '라이더 번들 여부','라이더 번들 LT','조리 시작 시간','조리소요시간','가게에서 소요 시간','고객에게서 소요 시간','차량 대기 시간', '음식 대기 시간',
                       'who_picked','bundle_size','bundle_route','who_serve', 'bundletype','dynamic 번들 유형','번들 가게 순서','번들 고객 순서','번들크기','manual_p2']
    customer_infos = [sub, customer_header]
    for customer_name in Customers:
        customer = Customers[customer_name]
        wait_t = None
        try:
            wait_t = customer.ready_time - customer.time_info[8] #음식이 준비된 시간 - 가게에 도착한 시간.
        except:
            pass
        info = [customer_name] + customer.time_info[:4] +[customer.time_info[8]]+[customer.cook_time]+ \
               [customer.food_wait, customer.rider_wait]+[wait_t, customer.fee,customer.who_serve, customer.distance,
                                                          customer.p2, customer.inbundle,customer.cook_time, customer.rider_wait,customer.in_bundle_time]
        info += [customer.cancel]
        if customer.time_info[3] != None:
            info += [customer.time_info[3] - customer.time_info[0], customer.time_info[3] - customer.time_info[2]]
        #elif customer.time_info[2] != None:
        #    info += [None, customer.time_info[3] - customer.time_info[2]]
        else:
            info += [None, None]
        info += customer.rider_bundle
        info += [customer.cook_start_time, customer.actual_cook_time, customer.time_info[6], customer.time_info[7], customer.rider_wait3, customer.food_wait3]
        info += [customer.who_picked, customer.bundle_size, customer.bundle_route, customer.who_serve, customer.bundle_type, customer.dynamic_type, customer.inbundle_order[0],customer.inbundle_order[1], customer.bundle_len, customer.manual_p2]
        info += [customer.location[0], customer.location[1], customer.store_loc[0], customer.store_loc[1]]
        customer_infos.append(info)
    if add_dir == None:
        f = open(title  + "riders_" + save_key + ".txt", 'a')
    else:
        f = open(add_dir +'Rider_data/'+ title + "riders_" + save_key + ".txt", 'a')
    for info in rider_infos:
        count = 0
        for ele in info:
            data = ele
            if type(ele) != str:
                data = str(ele)
            f.write(data)
            f.write(';')
            count += 1
            if count == len(info):
                f.write('\n')
    f.close()
    if add_dir == None:
        f = open(title  + "customers_" + save_key + ".txt", 'a')
    else:
        f = open(add_dir +'Customer_data/'+ title + "customers_" + save_key + ".txt", 'a')
    for info in customer_infos:
        count = 0
        for ele in info:
            data = ele
            if type(ele) != str:
                data = str(ele)
            f.write(data)
            f.write(';')
            count += 1
            if count == len(info):
                f.write('\n')
    f.close()


def ExpValueCalculator(rider_names, riders, order_names, orders, rider_check_index = 15): #todo 1118: 요주 함수1
    """
    라이더가 선택할 주문의 가치(수수료/운행시간)을 계산.
    @param rider_names: 주문을 선택할 라이더 이름들
    @param riders: 라이더 집합
    @param order_names: 라이더가 선택할 수 있는 주문 이름들
    @param orders: 고객 집합
    @param rider_check_index: 라이더가 확인할 주문 수
    @return: [[라이더 이름, 주문 이름,주문 OD-거리, 단위 시간당 이익],...,]
    """
    m_r = []
    for rider_name in rider_names:
        rider = riders[rider_name]
        dists = []
        try:
            rider_end_loc = rider.route[-1][2]
        except:
            rider_end_loc = rider.visited_route[-1][2]
        for order_name in order_names:
            order = orders[order_name]
            dists.append([order.name, distance(rider_end_loc[0],rider_end_loc[1],order.store_loc[0],order.store_loc[1])])
        dists.sort(key=operator.itemgetter(1))
        dists = dists[:rider_check_index]
        rider_value = []
        for info in dists:
            dist1 = distance(order.store_loc[0],order.store_loc[1],order.location[1],order.location[1])
            move_t = (info[1] + dist1)/rider.speed
            val = order.fee / move_t
            rider_value.append([info[0],dist1, val])
        rider_value.sort(key=operator.itemgetter(2), reverse=True) #시간당 이윤이 높은 순서대로
        if len(dists) > 0:
            win_dist = rider_value[0]
            max_dist = dists[:rider_check_index][1]
            m_r.append([rider.name, win_dist[0],  win_dist[1], win_dist[2],max_dist])
        else:
            m_r.append([rider.name, 0 ,100, 0 ])
    return m_r


def BundleExpValueCalculator(bundle_infos, rider_names, riders, orders, M = 10000, m_r = None, output_type = 'matrix'): #todo 1118: 요주 함수2
    """
    가능한 번들 정보 집합이 주어지면, 해당 번들이 라이더들에게 선택될 법한지를 계산.
    @param bundle_infos: 번들 정보 집합. 자세한 사항은 아래 참고
    [route, unsync_t[0], round(sum(ftds) / len(ftds), 2), unsync_t[1], order_names, round(route_time, 2),min(time_buffer), round(P2P_dist - route_time, 2), line_dist, round(P2P_dist,4), distance(origin[0],origin[1], destination[0],destination[1])/speed]
    @param rider_names: t interval에 주문을 선택할 라이더 이름
    @param riders: 전체 라이더 집합
    @param orders: 주문 집합
    @param M: BundleConsist2에 사용되는 임의의 큰 수 M
    @param m_r: ExpValueCalculator의 계산 결과. 반드시 필요하지는 않지만, 있는 경우 연산 시간이 단축 됨.
    @return: [행:라이더, 열 : 주어진 bundle_infos index] 1:라이더i가 번들 j 선택할 가능성 높음. 0 : 라이더i가 번들 j 선택할 가능성 낮음
    """
    res_r = []
    rider_index = 0
    for rider_name in rider_names:
        rider = riders[rider_name]
        tem_r = []
        for b_info in bundle_infos:
            #print(bundle_infos)
            #print(b_info)
            #input('check1234')
            start_order_name = b_info[0][0] - M
            if start_order_name < 0:
                #print(b_info)
                #input('minus value')
                pass
            start_point = orders[start_order_name].store_loc
            try:
                rider_end_loc = rider.route[-1][2]
                #print(1, rider_end_loc)
            except:
                rider_end_loc = rider.visited_route[-1][2]
                #print(2, rider_end_loc)
            if len(rider_end_loc) != 2:
                rider_end_loc = rider.visited_route[-1][2]
            #print(rider_end_loc, start_point)
            d2start = distance(rider_end_loc[0], rider_end_loc[1], start_point[0], start_point[1]) / rider.speed
            move_t = d2start / rider.speed
            move_t += b_info[5]
            fee = 0
            for name in b_info[4]:
                fee += orders[name].fee
            # rider_end_loc = rider.route[-1][2]
            val = fee / move_t
            #print('check1',rider_name,d2start , m_r[rider_index][4][1])
            if d2start < m_r[rider_index][4][1]:
                #print('check2',rider_name,val, m_r[rider_index][3])
                if val > m_r[rider_index][3]:
                    #tem = val
                    tem = 1
                else:
                    #tem = m_r[rider_index][2]
                    tem = 0
            else:
                tem = 0
            tem_r.append(tem)
        res_r.append(tem_r)
        rider_index += 1
    #print('res_r',res_r)
    if output_type == 'matrix':
        return res_r
    else:
        rev_r = []
        for bundle_index in range(len(res_r[0])):
            val = 0
            for rider_index in range(len(res_r)):
                val += res_r[rider_index][bundle_index]
            rev_r.append(val)
    return rev_r

def Y_b_r(m_r, res_r): #todo 1118: 요주 함수3
    """
    계산한 값으로 부터 번들 계시 문제의 indicator "y"를 계산
    @param m_r: ExpValueCalculator 결과 값
    @param res_r: BundleExpValueCalculator 결과 값
    @return: "y" *주의 y는 [[b1을 선택할 법한 라이더 이름,b1을 선택할 법한 라이더 이름,...,],...,[]]
    """
    res = []
    num_b = len(res_r[0])
    for b_index in range(num_b):
        tem = []
        for r_index in range(len(m_r)):
            if m_r[3] < res_r[r_index][b_index]:
                tem.append(m_r[0])
        res.append(tem)
    return res

def SaveInstanceAsCSV(Rider_dict, Orders,Store_dict, instance_name = '' ):
    #시나리오 저장
    rider_header = ['name', 'start time', 'speed', 'end', 'capacity', 'freedom', 'order_select_type', 'wait_para', 'uncertainty', 'exp_error']
    saved_rider_infos = []
    saved_rider_infos.append(rider_header)
    for rider_name in Rider_dict:
        rider = Rider_dict[rider_name]
        tem = [rider.name, rider.start_time, rider.speed, rider.end_t, rider.capacity, rider.freedom, rider.order_select_type, rider.wait_para, rider.uncertainty, rider.exp_error]
        tem1 = []
        for info in tem:
            #tem1.append(';')
            tem1.append(info)
            #tem1.append(';')
        saved_rider_infos.append(tem)
    customer_header = ['name','gen t', 'loc_x', 'loc_y', 'store', 'store_x', 'store_y', 'p2', 'cook_time', 'cook_time_type','cook_time_s','cook_time_e',]
    saved_customer_infos = []
    saved_customer_infos.append(customer_header)
    for customer_name in Orders:
        customer = Orders[customer_name]
        tem = [customer.name, customer.time_info[0], customer.location[0],customer.location[1], customer.store, customer.store_loc[0],customer.store_loc[1], customer.p2, customer.cook_time, customer.cook_info[0], customer.cook_info[1][0], customer.cook_info[1][1]]
        saved_customer_infos.append(tem)
    store_header = ['name', 'start_loc_x', 'start_loc_y', 'order_ready_time', 'capacity', 'slack']
    saved_store_infos = []
    saved_store_infos.append(store_header)
    for store_name in Store_dict:
        store = Store_dict[store_name]
        tem = [store.name,store.location[0],store.location[1], store.order_ready_time, store.capacity, store.slack]
        saved_store_infos.append(tem)
    #txt로 저장
    file_name = ['rider_infos', 'customer_infos', 'store_infos']
    name_index = 0
    for infos in [saved_rider_infos, saved_customer_infos, saved_store_infos]:
        #f = open("TEST" + file_name[name_index] + ".txt", 'w')
        f = open("Instance" + file_name[name_index] + instance_name +".csv", 'w', newline = '')
        wr = csv.writer(f)
        for info in infos:
            #f.write(str(info) + '\n')
            wr.writerow(info)
        f.close()
        name_index += 1


def PrintSearchCandidate(target_customer, res_C_T, now_t = 0, titleinfo = 'None'):
    x1 = [] #주문 가게 x
    y1 = [] #주문 가게 y
    x2 = [] #주문 고객 x
    y2 = [] #주문 고객 y
    for customer_name in res_C_T:
        customer = res_C_T[customer_name]
        x1.append(customer.store_loc[0])
        y1.append(customer.store_loc[1])
        x2.append(customer.location[0])
        y2.append(customer.location[1])
    if len(x1) > 1:
        plt.scatter(x1, y1, color='k', label = 'Store')
        plt.scatter(x2, y2, marker = 'x' ,color='m',label = 'Customer')
        plt.scatter(target_customer.store_loc[0], target_customer.store_loc[1], color = 'r', label = 'BaseStore')
        plt.scatter(target_customer.location[0], target_customer.location[1], marker = 'x' , color='c', label = 'BaseCustomer')
        plt.legend()
        plt.axis([0, 50, 0, 50])
        title = '{} T;{};Base;{};CtSize{}'.format(titleinfo, now_t, target_customer.name ,len(res_C_T)-1)
        plt.title(title)
        #plt.savefig('test.png')
        #print(title)
        #print(type(title))
        #plt.savefig('save_fig/'+title+'.png', dpi = 300)
        plt.show()
        plt.close()
        #input('그림 확인')


def SaveScenario(scenario, rider_num, instance_type, ite, ellipse_w = 'None', heuristic_theta= 'None', heuristic_r1 = 'None',count = 'None',considered_customer_type = 'all',
                  search_range_index = 15, pr_para = False, add_info = [None,None] , dynamic = False):
    print('"요약 정리/ 라이더 수 {}'.format(rider_num))
    print_count = 0
    f3 = open("결과저장_sc_저장.txt", 'a')
    con = 'ITE;{};instance_type;{};search_type;{};플랫폼번들;{};라이더번들;{};'.format(ite, instance_type, scenario.search_type, scenario.platform_recommend,scenario.rider_bundle_construct)
    con1 = 'considered_customer_type;{};search_range_index;{};pr_para;{};'.format(considered_customer_type, search_range_index, pr_para)
    f3.write(con + con1 + '\n')
    sc = scenario
    res_info = []
    #input(sc.res)

    #index = -1

    for index in list(range(len(sc.res[0]))):
        tem = []
        for info in sc.res:
            if type(info) == list:
                tem.append(info[index])
            else:
                #print(info)
                pass
        if None in tem:
            res_info.append(None)
        else:
            res_info.append(tem[-1])
    tem = []
    for info in sc.res:
        if type(info) == list:
            tem.append(info[index])
        else:
            # print(info)
            pass
    if None in tem:
        res_info.append(None)
    else:
        res_info.append(sum(tem) / len(tem))
    try:
        res_info.append(sum(sc.bundle_snapshots['size'])/len(sc.bundle_snapshots['size']))
        res_info.append(sum(sc.bundle_snapshots['length']) / len(sc.bundle_snapshots['length']))
        res_info.append(sum(sc.bundle_snapshots['od']) / len(sc.bundle_snapshots['od']))
    except:
        res_info += [None,None,None]
    res_info += ['dummy','dummy','dummy']
    offered_bundle_num = len(sc.bundle_snapshots['size'])
    #print(len(res_info))
    #input(res_info)
    if print_count == 0:
        head = 'dynamic;인스턴스종류;SC;번들탐색방식;연산시간(sec);플랫폼;라이더;라이더수;obj;전체 고객;서비스된 고객;서비스율;평균LT;평균FLT;직선거리 대비 증가분;원래 O-D길이;라이더 수익 분산;LT분산;' \
               'OD증가수;OD증가 분산;OD평균;수행된 번들 수;수행된번들크기평균;b1;b2;b3;b4;b5;b수;p1;p2;p3;p4;p수;r1;r2;r3;r4;r5;r수;평균서비스시간;(테스트)음식 대기 시간;(테스트)버려진 음식 수;(테)음식대기;' \
               '(테)라이더대기;(테)15분이하 음식대기분;(테)15분이상 음식대기분;(테)15분이하 음식대기 수;(테)15분이상 음식대기 수;(테)라이더 대기 수;라이더평균운행시간;제안된 번들수;라이더수수료;size;length;ods;ellipse_w; ' \
               'heuristic_theta; heuristic_r1;rider_ratio;#dist;#bc1;#bc2;#dist(xgb);#t1;#t2;#t3;#취소된 고객'
        #print('인스턴스종류;SC;번들탐색방식;연산시간(sec);플랫폼;라이더;obj;전체 고객;서비스된 고객;서비스율;평균LT;평균FLT;직선거리 대비 증가분;원래 O-D길이;라이더 수익 분산;LT분산;'
        #     'OD증가수;OD증가 분산;OD평균;수행된 번들 수;수행된번들크기평균;제안된 번들수;size;length;ods')
        print(head)
        f3.write(head + '\n')
    ave_duration = sc.durations[-1]
    try:
        tem_data = '{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};'.format(
                dynamic, instance_type , str(sc.name[0]),sc.search_type, ave_duration,sc.platform_recommend,sc.rider_bundle_construct,rider_num,sc.obj_type, res_info[0],res_info[1],
                res_info[2], res_info[3], res_info[4], res_info[5], res_info[6], res_info[7], res_info[8],res_info[9],res_info[10],res_info[11],res_info[12],res_info[13],
            res_info[14], res_info[15], res_info[16],res_info[17], res_info[18], res_info[19],res_info[20],res_info[21],res_info[22],res_info[23], res_info[24],res_info[25],
            res_info[26],res_info[27],res_info[28],res_info[29],res_info[30],res_info[31],res_info[32], res_info[33],res_info[34], res_info[35],res_info[36], res_info[37],res_info[38],res_info[39], res_info[40], res_info[41],
            offered_bundle_num,res_info[42], res_info[44], res_info[45],res_info[46],ellipse_w, heuristic_theta, heuristic_r1, sc.mix_ratio, sc.countf[0], sc.countf[1], sc.countf[2], sc.countf[3],
        sc.countt[0], sc.countt[1],sc.countt[2],add_info[0])

        #print(tem_data)
        f3.write(tem_data + '\n')
    except:
        tem_data = '시나리오 {} ITE {} 결과 없음'.format(sc.name, count)
        #print('시나리오 {} ITE {} 결과 없음'.format(sc.name, count))
        print(tem_data)
        f3.write(tem_data + '\n')
    print_count += 1
    f3.write('Exp End' + '\n')
    f3.close()

def store_p2_reader(dir, ite = 0):
    f = open(dir, 'r')
    lines = f.readlines()
    con = lines[ite].split(',')
    stores = []
    for i in con[1:-1]:
        stores.append(int(i))
    f.close()
    return stores