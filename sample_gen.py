import random


#f = open('C:/Users/박태준/jupyter_notebook_base/data/store_p2.txt','a')
f = open('C:/Users/박태준/jupyter_notebook_base/data/rider_start_loc_gen.txt','a')
for _ in range(40):
    store_name_list = [' ']
    tem = 'start;'
    coords = list(range(50))
    for i in range(1000):
        tem += str(int(random.choice(coords)))+';'
    f.write(str(tem) + '\n')
f.close()
""""
f = open('C:/Users/박태준/jupyter_notebook_base/data/송파구/store_p2_송파.txt','r')
lines = f.readlines()
for i in lines:
    con = i.split(',')
    print(con[1:5])
    print(con[-5:-1])
    break

f.close()
"""