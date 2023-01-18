# -*- coding: utf-8 -*-
##'Instance_random','Instance_cluster'
import time
import smtplib
from email.message import EmailMessage

"""
def email_fun(title, con):
    msg=EmailMessage()
    # 제목 입력
    msg['Subject']=title
    # 내용 입력
    msg.set_content(con)
    # 보내는 사람
    msg['From']= 'xoxoqkr@gmail.com'
    # 받는 사람
    msg['To']= 'xoxoqkr@gmail.com'
    from email.message import EmailMessage
    smtp_gmail = smtplib.SMTP('smtp.gmail.com', 587)
    # 서버 연결을 설정하는 단계
    smtp_gmail.ehlo()
    # 연결을 암호화
    smtp_gmail.starttls()
    #로그인
    smtp_gmail.login('xoxoqkr@gmail.com','mrwtwdcyssjydlbb')
    smtp_gmail.send_message(msg)
"""
#고객 주문이 발생하자 마자 바로 서비스 받을 수 있는 상태가 되는 경우 <- 이번 실험 상태
#고객들의 주문이 발생 후 cancel == True인 상태에서 다음 interval이 되면 canceal = False가 되는 상황
run_time = 20
ct_gen_time = 120
customer_pend_options = [False]
#dir = "C:/users/박태준/PycharmProjects/BundleSimple/"
#data_dir = "C:/Users/박태준/jupyter_notebook_base"
dir = "C:/Users/xoxoq/OneDrive/pythonProject/winter_vacation2/"
data_dir = "C:/Users/xoxoq/OneDrive/Ipython"
basic_infos = [[False,False,'simple_max_s','P2P']]
infos = []
c_weights = [1,0.5,0]
for customer_pend in customer_pend_options:
    for info in basic_infos:
        for c_weight in c_weights:
            infos.append(info + [customer_pend] + [c_weight])
#email_fun('test', 'test1')
#input('test')
print(infos)
#input('info 확인')
#exp_range = range(5,15) # [0]
for instance_type in ['Instance_random','Instance_cluster']: #Instance_random , Instance_cluster
    for ite in range(1):
        for info in infos:
            s_t = time.time()
            if info[2] == 'Static':
                exp_range = range(5, 10)
            elif info[2] == 'Dynamic':
                exp_range = range(5, 15)
            else:
                exp_range = range(5, 6)
            exec(open(dir + 'Simulator_v3_2.py', encoding='UTF8').read(),
                 globals().update(run_time=run_time, platform_recommend_input=info[0], dynamic_env=info[1],
                                  customer_pend=info[4], obj_type = info[2], ct_gen_time = ct_gen_time, instance_type = instance_type,
                                  exp_range = exp_range, sc_type = info[3], fee_weight = info[5], rev_dir = dir, data_dir = data_dir))
            f = open(dir + 'report_test.txt', 'a')
            e_t = time.time()
            # f.write('success; duration;{};t_now;{} \n'.format(e_t - s_t, time.strftime('%Y-%m-%d %I:%M:%S %p', e_t)))
            f.write('success ;{};{}; \n'.format(info[0], info[1]))
            f.close()
            print('success')
            """
            try:
                exec(open(dir+'Simulator_v3.py', encoding='UTF8').read(),globals().update(run_time= run_time, platform_recommend_input= info[0],dynamic_env= info[1], customer_pend = info[2]))
                f = open(dir +'report_test.txt','a')
                e_t = time.time()
                #f.write('success; duration;{};t_now;{} \n'.format(e_t - s_t, time.strftime('%Y-%m-%d %I:%M:%S %p', e_t)))
                f.write('success ;{};{}; \n'.format(info[0], info[1]))
                f.close()
                print('success')
            except:
                f = open(dir +'report_test.txt','a')
                e_t = time.time()
                #f.write('Fail; duration;{};t_now;{} \n'.format(e_t - s_t, time.strftime('%Y-%m-%d %I:%M:%S %p', e_t)))
                f.write('Fail \n')
                f.close()
                print('error')
            """
            f = open(dir + 'report_test.txt', 'a')
            e_t = time.time()
            f.write('RunTime;{}; \n'.format(e_t - s_t))
            f.close()
            input('확인')