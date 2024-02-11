# แก้ demand 3 state สุดท้าย เป็น self.state[76] = self.sum_reward, self.state[77] = self.sum_real_reward, self.state[78] = self.step_count  # so all
# ไฟล์นี้จะเป็นการ ลด obd ลง เพราะเดาว่า เอา demand มา ทำให้เทรนแล้วไม่คอนเวิจ
# ไว้สำหรับเทรน ฤดูกาล รูปแบบเดียว
# เพิ่ม future demand ลงไปใน state
# ตัดพวกที่ print และ comment ออกไปใ เพื่อเอาไปใส่ใน gym
# คือ ต่อจากไฟล์ 25_16 act แต่ลองปรับ env ตาม moutain car เพื่อให้ appecnd ใน colab ถูก
# ล่าสุด 14-8-65 ตรงกับไฟล์ InvEnv38_16act.py
# มีการ normalize input ต่างๆทั้งค่า demand และ inventory ของเสตท
# ถ้ามีการเปลี่ยนแปลงในไฟล์ InvEnv ใน github ให้ pip install Env ใหม่ ดังคำสั่งด้านล่าง
# pip install -e git+https://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@github.com/thachanon27/InvEnv4#egg=inv_env

from typing import Optional

import gym

from gym import spaces
from gym.spaces import Box, Discrete
import random
import numpy as np
import itertools

from random import randint, choice

# file 18  is demand set1  but file22 will demand set2 ต่างกันแค่นี้
# print("new env @18-2-66")
print_result = True


############################################################

# class ProductionTable:
#
#     def __init__(
#             self, no_machines, no_products, lottbl_onpeak=None, lottbl_offpeak=None
#     ):
#         if lottbl_onpeak is None: lottbl_onpeak = {}
#         if lottbl_offpeak is None: lottbl_offpeak = {}
#
#         self.no_machines = no_machines
#         self.no_products = no_products
#         self.lottbl_onpeak = lottbl_onpeak
#         self.lottbl_offpeak = lottbl_offpeak
#
#         self.lotsize_tbl = {}
#         self.switch_tbl = {}
#         self.action_counter = 0
#
#         if len(self.lottbl_onpeak) > 0 and len(self.lottbl_offpeak) > 0:
#             self.init_tables()
#
#     def add_prod_lotsize(self, machine_id, prod_id, onpeak, offpeak):
#         if machine_id not in self.lottbl_onpeak:
#             self.lottbl_onpeak[machine_id] = {}
#         if machine_id not in self.lottbl_offpeak:
#             self.lottbl_offpeak[machine_id] = {}
#         self.lottbl_onpeak[machine_id][prod_id] = onpeak
#         self.lottbl_offpeak[machine_id][prod_id] = offpeak
#
#     def init_tables(self):
#         args = []
#         for machine_id in range(self.no_machines):
#             entry = [-1] + list(range(self.no_products))  # Product IDs for each machine, including halt (-1)
#             args.append(entry)
#         # args.append([True, False])  # On-peak and off-peak
#
#         action_id = 0
#         for plan in itertools.product(*args):
#             switches = np.zeros((2, self.no_machines, self.no_products))
#             lotsizes = np.zeros((2, self.no_machines, self.no_products))
#             # is_onpeak = plan[self.no_machines]
#
#             for is_onpeak in [False, True]:
#                 peak_idx = 1 if is_onpeak else 0
#                 for machine_id in range(self.no_machines):
#                     prod_id = plan[machine_id]
#                     if plan[machine_id] >= 0:
#                         switches[peak_idx, machine_id, prod_id] = 1.0
#                         if is_onpeak:
#                             lotsizes[peak_idx, machine_id, prod_id] = self.lottbl_onpeak[machine_id][prod_id]
#                         else:
#                             lotsizes[peak_idx, machine_id, prod_id] = self.lottbl_offpeak[machine_id][prod_id]
#
#             self.switch_tbl[action_id] = switches
#             self.lotsize_tbl[action_id] = lotsizes
#
#             action_id += 1
#
#         self.action_counter = action_id
#
#     def action_ids(self):
#         return self.switch_tbl.keys()
#
#     def __iter__(self):
#         return iter(self.switch_tbl)
#
#     def get_lotsize(self, action_id, is_onpeak):
#         peak_idx = 1 if is_onpeak else 0
#         return self.lotsize_tbl[action_id][peak_idx]
#
#     def get_switches(self, action_id, is_onpeak):
#         peak_idx = 1 if is_onpeak else 0
#         return self.switch_tbl[action_id][peak_idx]
#
#     def display(self):
#         print('>>> Production Table\n')
#         for action_id in self:
#             switches = self.switch_tbl[action_id]
#             lotsize_onpeak = self.lotsize_tbl[action_id][1]
#             lotsize_offpeak = self.lotsize_tbl[action_id][0]
#             print(f'Action ID: {action_id}')
#             print(f'Switches:\n{switches}')
#             print(f'Off-peak lot-size:\n{lotsize_offpeak}')
#             print(f'On-peak lot-size:\n{lotsize_onpeak}')
#             print()


############################################################

''' Production Plan

On-peak Lotsizes
Machine 1 
    p1 =  3211
    p2 =  2223
    p3 =  1668
Machine 2  
    p1 =  2717
    p2 =  1853
    p3 =  1359

Off-peak Lotsizes 
Machine 1 
    p1 =  2717
    p2 =  1881
    p3 =  1411
Machine 2  
    p1 =  2299
    p2 =  1568
    p3 =  1150
'''
# prodtbl = ProductionTable(no_machines=2, no_products=3)
# prodtbl.add_prod_lotsize(machine_id=0, prod_id=0, onpeak=3211, offpeak=2717)
# prodtbl.add_prod_lotsize(machine_id=0, prod_id=1, onpeak=2223, offpeak=1881)
# prodtbl.add_prod_lotsize(machine_id=0, prod_id=2, onpeak=1668, offpeak=1411)
# prodtbl.add_prod_lotsize(machine_id=1, prod_id=0, onpeak=2717, offpeak=2299)
# prodtbl.add_prod_lotsize(machine_id=1, prod_id=1, onpeak=1853, offpeak=1568)
# prodtbl.add_prod_lotsize(machine_id=1, prod_id=2, onpeak=1359, offpeak=1150)
# prodtbl.init_tables()


# prodtbl.display()
# input()

############################################################
############################################################

class InvEnv5_60T_MA2(gym.Env):
    def __init__(self):
        self.step_count = 0
        self.overall_time_trained = 0
        # initial inventory
        self.on_hand1 = (np.random.randint(3500, 6500) - 12000) / (12000 - 0)  # 5659
        self.on_hand2 = (np.random.randint(2500, 4500) - 12000) / (12000 - 0)  # 3051
        self.on_hand3 = (np.random.randint(2000, 3500) - 12000) / (12000 - 0)  # 2084
        # np.random.randint(3500, 6500), np.random.randint(2500, 4500), np.random.randint(2000, 3500),
        self.action_space = spaces.Discrete(4) # 0 1 2 3
        # self.observation_space = spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32)
        self.statelow = np.array([
            0, # Demand pattern #0
            0, #Demand r1 at 4 rd period  #1
            0, # Demand r1 at 28 th period  #2
            0, # Demand r1 at 56 th period  # 3
            0, 0, 0  # # reward,  real reward, period no.   # 4, 5, 6  --> so there are total 7 observations
        ])
        self.statehigh = np.array([
            np.inf,  # Demand pattern #0
            np.inf,  # Demand r1 at 4 rd period  #1
            np.inf,  # Demand r1 at 28 th period  #2
            np.inf,  # Demand r1 at 56 th period  # 3
            np.inf, np.inf, np.inf  # # reward,  real reward, period no.   # 4, 5, 6  --> so there are total 7 observations
        ])

        # self.state[49] = 0
        self.observation_space = Box(self.statelow, self.statehigh,
                                     dtype=np.float32)

        self.sum_reward = 0
        self.sum_real_reward = 0

        self.CO_var_set1 = [0, 0, 0, 0, 0, 0, 0, 0]
        self.CO_var_set2 = [0, 0, 0, 0, 0, 0, 0, 0]

        self.sum_ex_penalty_array = []
        self.sum_ex_penalty_array_2 = []
        self.sum_ex_penalty_array_3 = []

        self.demand_all = [0, 0, 0]
        self.reset()
        self.M1P1_set = []
        self.M1P2_set = []
        self.M1P3_set = []
        self.M2P1_set = []
        self.M2P2_set = []
        self.M2P3_set = []
        self.changeover_cost_of_m1 = 0
        self.switch_on_cost = 0
        self.changeover_cost_of_m2 = 0
        self.variable_cost_m1 = 0
        self.variable_cost_m2 = 0
        self.demand_real = []

        self.demand_all = []
        self.aaa = 1

        self.state = ([0,  # Demand pattern #0
                      0,  # Demand r1 at 4 rd period  #1
                      0,  # Demand r1 at 28 th period  #2
                      0,  # Demand r1 at 56 th period  # 3
                      0, 0, 0  # # reward,  real reward, period no.   # 4, 5, 6  --> so there are total 7 observations
                      ])
        #self.prodtbl = prodtbl

        self.weekend_stepcount = [2, 3, 4, 5, 16, 17, 18, 19, 30, 31, 32, 33, 44, 45, 46, 47, 59, 60]
        self.on_peak_stepcount = [0, 6, 8, 10, 12, 14, 20, 22, 24, 26, 28, 34, 36, 38, 40, 42, 48, 50, 52, 54, 56]
        self.off_peak_stepcount = [1, 7, 9, 11, 13, 15, 21, 23, 25, 27, 29, 35, 37, 39, 41, 43, 49, 51, 53, 55, 57]

    def is_weekend(self):
        return self.step_count in self.weekend_stepcount

    def is_onpeak(self):
        return self.step_count in self.on_peak_stepcount

    def is_offpeak(self):
        return self.step_count in self.off_peak_stepcount

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.step_count = 0
        # state 14 dimension =onhand ,demand ,production status of machines
        self.state = np.array([
            0,  # Demand pattern #0
            0,  # Demand r1 at 4 rd period  #1
            0,  # Demand r1 at 28 th period  #2
            0,  # Demand r1 at 56 th period  # 3
            0, 0, 0  # # reward,  real reward, period no.   # 4, 5, 6  --> so there are total 7 observations
        ])
        self.sum_reward = 0
        self.sum_real_reward = 0

        self.demand_all = [0, 0, 0]
        self.M1P1_set = []
        self.M1P2_set = []
        self.M1P3_set = []
        self.M2P1_set = []
        self.M2P2_set = []
        self.M2P3_set = []
        self.changeover_cost_of_m1 = 0
        self.switch_on_cost = 0
        self.changeover_cost_of_m2 = 0
        self.variable_cost_m1 = 0
        self.variable_cost_m2 = 0
        self.demand_real = []
        self.demand_all = []
        self.aaa = 1

        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}



    def step(self, action, demand_arr_inf):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert isinstance(demand_arr_inf, np.ndarray), f"{demand_arr_inf!r} ({type(demand_arr_inf)}) invalid"
        info = {}

        if print_result == True:
            print("=================================================self.step_count =", self.step_count)
        # all model parameters



        aaa3, \
        dr1_4, \
        dr1_28, \
        dr1_56, reward,  real_reward, period = self.state

        aaa3 = self.aaa
        self.demand_all = demand_arr_inf
        if print_result == True:
            print("===dr1_4, dr2_4, dr3_4 //dr1_16, dr2_16, dr3_16", dr1_4,"//", dr1_28)
            print("===self.demand_all =", self.demand_all)
        dr1_4 = self.demand_all[9]
        dr1_28 = self.demand_all[81]
        dr1_56 = self.demand_all[165]

        # parameter for normalize
        mind1 = 0  # min demand1
        mind2 = 0
        mind3 = 0
        maxd1 = 4500
        maxd2 = 4500
        maxd3 = 4500
        minr1 = 0
        minr2 = 0
        minr3 = 0
        maxr1 = 12000
        maxr2 = 12000
        maxr3 = 12000

        # แปลงค่า Normalize value จาก 0-1 range  กลับเป็นค่าปกติ
        dr1_4 = dr1_4 * (maxd1 - mind1) + mind1
        dr1_28 = dr1_28 * (maxd1 - mind1) + mind1
        dr1_56 = dr1_56 * (maxd1 - mind1) + mind1


        if print_result == True:
            print("===demand in this period =", dr1_4, dr1_28, dr1_56)



        # Compute Reward

        # Assign - on-and off-peak p cost to each period
        #         weekend_stepcount = [3, 4, 5, 6, 17, 18, 19, 20]
        #         on_peak_stepcount = [1, 7, 9, 11, 13, 15, 21, 23, 25, 27, 29]
        #         off_peak_stepcount = [2, 8, 10, 12, 14, 22, 24, 26, 28, 30]
        #         stp = 0
        #         stp = self.step_count + 1

        # weekend_stepcount = [2, 3, 4, 5, 16, 17, 18, 19]
        # on_peak_stepcount = [0, 6, 8, 10, 12, 14, 20, 22, 24, 26, 28]
        # off_peak_stepcount = [1, 7, 9, 11, 13, 15, 21, 23, 25, 27, 29]
        weekend_stepcount = [2, 3, 4, 5, 16, 17, 18, 19, 30, 31, 32, 33, 44, 45, 46, 47, 59, 60]
        on_peak_stepcount = [0, 6, 8, 10, 12, 14, 20, 22, 24, 26, 28, 34, 36, 38, 40, 42, 48, 50, 52, 54, 56]
        off_peak_stepcount = [1, 7, 9, 11, 13, 15, 21, 23, 25, 27, 29, 35, 37, 39, 41, 43, 49, 51, 53, 55, 57]


        reward = 9

        real_reward = 9

        self.step_count += 1
        done = bool(self.step_count >= 60)  # planning time frame period = 15
        self.overall_time_trained += 1

        raw_reward = reward
        reward = reward / 25

        real_reward = real_reward / 1000000

        self.sum_reward += reward
        self.sum_real_reward += real_reward
        last_sum_reward = 0
        last_sum_real_reward = 0

        last_sum_real_reward = 0
        last_sum_reward = 0
        if self.step_count == 59:
            last_sum_reward = self.sum_reward
            last_sum_real_reward = self.sum_real_reward

        info = [real_reward, self.overall_time_trained, self.demand_real]

        dr1_4 = (dr1_4 - mind1) / (maxd1 - mind1)
        dr1_28 = (dr1_28 - mind1) / (maxd1 - mind1)
        dr1_56 = (dr1_56 - mind1) / (maxd1 - mind1)


        if print_result == True:
            print("===dr1_4, dr1_28, dr1_56", dr1_4, "//", dr1_28, dr1_56)

        # inv data
        self.state[0] = 0
        self.state[1] = dr1_4
        self.state[2] = dr1_28
        self.state[3] = dr1_56
        self.state[4] = self.sum_reward / 100  # /100 เพื่อ normalize แบบง่ายๆ
        self.state[5] = self.sum_real_reward / 100
        self.state[6] = self.step_count / 100  # // หารแบบปัดเศษลง# so all number state variables are 79 variables

        #         print("value หลัง normalize")
        #         print("demand1 =", demand1,"=state[3]=",self.state[3])
        #         print("overage1 =", overage1,"=state[0]=",self.state[0])
        #         print("state norm", self.state)

        # Clears the variables used to store data.
        N1P_ = 0
        N1P1_ = 0
        N1P2_ = 0
        N1P3_ = 0
        N2P_ = 0
        N2P1_ = 0
        N2P2_ = 0
        N2P3_ = 0

        return np.array(self.state, dtype=np.float32), reward, done, info


############################################################


# def main():
#     env = InvEnv5_60T_MA()
#     state = env.reset()

#     done = False
#     N = 2
#     runs = int(N)  # รัน 30 peroids จำนวน N รอบ
#     for i in range(runs):
#         maxrun = 0
#         done = False
#         rand = randint(0, 999)
#         # print("rand", rand)
#         env.reset()
#         # env.seed(rand)
#         period = 0
#         demand_all = []
#         # print("============Round====", i)
#         while (done == False):
#             action = env.action_space.sample()
#             # print("period", period)
#             # print("state", state)
#             print("action = ", action)
#             state, reward, done, info = env.step(action)
#             period += 1
#             # print("####################################################period##############",period)
#             # print("state",state)
#             # print("action =",action)
#             # print("action =",action)

#             demand1 = state[1]
#             demand2 = state[2]
#             demand3 = state[3]
#             # print("d1-d3 input in next state =", state[3], state[4], state[5])
#             # print("d4-d9 =", state[21], state[22], state[23], state[24], state[25], state[26])
#             # print("extra_p_on ###### =", state[26])
#             if print_result == True:
#                 print("=== aaa3 =", state[0])
#                 print("self.sum_reward, self.sum_real_reward, self.step_count", state[4] * 100, state[5] * 100,
#                       state[6] * 100)
#             # demand_all.append(demand1)
#             # demand_all.append(demand2)
#             # demand_all.append(demand3)

#             # sum_rw_ += reward
#             # print("sum_rw",sum_rw_)
#             # maxrun += 1
#             reward2 = reward
#             print("reward", reward)
#             # print("demand =", demand_all)
#             # print("next_state",state)
#             # print("====================================================================")
#             # print(info[24])

# ############################################################


# if __name__ == '__main__':
#     main()
#     # test_production_table()
