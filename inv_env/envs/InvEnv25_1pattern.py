#ไฟล์นี้จะเป็นการ ลด obd ลง เพราะเดาว่า เอา demand มา ทำให้เทรนแล้วไม่คอนเวิจ
#ไว้สำหรับเทรน ฤดูกาล รูปแบบเดียว
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

from random import randint, choice


# file 18  is demand set1  but file22 will demand set2 ต่างกันแค่นี้
# print("new env @18-2-66")


class InvEnv4_m(gym.Env):
    def __init__(self):
        self.step_count = 0
        self.overall_time_trained = 0
        # initial inventory
        self.on_hand1 = (np.random.randint(3500, 6500) - 12000) / (12000 - 0)  # 5659
        self.on_hand2 = (np.random.randint(2500, 4500) - 12000) / (12000 - 0)  # 3051
        self.on_hand3 = (np.random.randint(2000, 3500) - 12000) / (12000 - 0)  # 2084
        # np.random.randint(3500, 6500), np.random.randint(2500, 4500), np.random.randint(2000, 3500),
        self.action_space = spaces.Discrete(16)
        # self.observation_space = spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32)
        self.statelow = np.array([
            0, 0, 0,  # initial inventory #0 1 2            ##ตอนนี้  state จะมีค่าพารามอเตอร์ทั้งหมด = 27
            0, 0, 0,  # initial demand    #3 4 5
            0, 0, 0, 0,  # initial machine status (0 = idle)   #6 7 8 9
            0, 0, 0, 0,                                        #10 11 12 13
            0, 0, 0,  # future inventory i4 i5 i6 = overage1_2, overage2_2, overage3_2    #14 15 16
            0, 0, 0,  # future inventory i7 i8 i9 = overage1_3, overage2_3, overage3_3    #17 18 19
            0, 0, 0, # overage1_4 overage2_4 overage3_4
            0, 0, 0,  # d4, d5, d6     #20 21 22
            0, 0, 0,  # d7, d8, d9     #23 24 25
            0, 0, 0,  # d10, d11, d12     #23 24 25
            0,  # extra_p_on    ---->  State 26
            0,  #Demand pattern #27
            0, 0, 0,  # Demand r1-3 at 4 rd period  # 29 30 31
            0, 0, 0,  # Demand r1-3 at 8 th period  # 32 33 34
            0, 0, 0,  # Demand r1-3 at 12 th period  # 35 36 37
            0, 0, 0,  # Demand r1-3 at 16 th period  # 38 39 40
            0, 0, 0,  # Demand r1-3 at 20 th period  # 41 42 43
            0, 0, 0,  # Demand r1-3 at 24 th period  # 44 45 46
            0, 0, 0  # Demand r1-3 at 28 th period  # 47 48 49
        ])
        self.statehigh = np.array([
            np.inf, np.inf, np.inf,  # initial inventory
            np.inf, np.inf, np.inf,  # initial demand
            1, 1, 1, 1,  # initial machine status (0 = idle)
            1, 1, 1, 1,
            np.inf, np.inf, np.inf,  # future inventory i4 i5 i6 = overage1_2, overage2_2, overage3_2
            np.inf, np.inf, np.inf,  # future inventory i7 i8 i9 = overage1_3, overage2_3, overage3_3
            np.inf, np.inf, np.inf,  # overage1_4 overage2_4 overage3_4
            np.inf, np.inf, np.inf,  # future demand d4, d5, d6
            np.inf, np.inf, np.inf,  # initial demand d7, d8, d9
            np.inf, np.inf, np.inf,  # initial demand d10, d11, d12
            1,  # extra_p_on
            np.inf,  # Demand pattern
            np.inf, np.inf, np.inf,  # Demand r1-3 at 4 rd period
            np.inf, np.inf, np.inf,  # Demand r1-3 at 8 th period
            np.inf, np.inf, np.inf,  # Demand r1-3 at 12 th period
            np.inf, np.inf, np.inf,  # Demand r1-3 at 16 th period
            np.inf, np.inf, np.inf,  # Demand r1-3 at 20 th period
            np.inf, np.inf, np.inf,  # Demand r1-3 at 24 th period
            np.inf, np.inf, np.inf  # Demand r1-3 at 28 th period
        ])

        #self.state[49] = 0
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

        (self.demand_all, self.aaa) = self.create_demand_all()

        self.state = [self.on_hand1, self.on_hand2, self.on_hand3,  # initial inventory #0 1 2
                      0, 0, 0,    # initial demand    #3 4 5
                      0, 0, 0, 0,    # initial machine status (0 = idle)   #6 7 8 9
                      0, 0, 0, 0,          # 10, 11, 12, 13
                      0, 0, 0,    # future inventory i4 i5 i6 = overage1_2, overage2_2, overage3_2    #14 15 16
                      0, 0, 0,     # future inventory i7 i8 i9 = overage1_3, overage2_3, overage3_3    #17 18 19
                      0, 0, 0,     # overage1_4 overage2_4 overage3_4
                      0, 0, 0,   # future demand # d4, d5, d6     #20 21 22
                      0, 0, 0,  # future demand # d7, d8, d9     #23 24 25
                      0, 0, 0,  # future demand # d10, d11, d12
                      1, #extra_p_on   ---->  State 26
                      0, # Demand pattern   #27
                      0, 0, 0,  # Demand r1-3 at 4 rd period
                      0, 0, 0,  # Demand r1-3 at 8 th period
                      0, 0, 0,  # Demand r1-3 at 12 th period
                      0, 0, 0,  # Demand r1-3 at 16 th period
                      0, 0, 0,  # Demand r1-3 at 20 th period
                      0, 0, 0,  # Demand r1-3 at 24 th period
                      0, 0, 0  # Demand r1-3 at 28 th period
                      ]


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
            (np.random.randint(3500, 6500) - 0) / (12000 - 0), (np.random.randint(2500, 4500) - 0) / (12000 - 0),
            (np.random.randint(2000, 3500) - 0) / (12000 - 0),
            # initial inventory  #ก่อนป้อนเข้า env ต้องทำเป็นค่า normalize แบบค่าอื่นก่อน
            # np.random.randint(3500, 6500), np.random.randint(2500, 4500), np.random.randint(2000, 3500),
            # initial inventory แบบ random
            0, 0, 0,  # initial demand
            0, 0, 0, 0,  # initial machine status (0 = idle)
            0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,  # future inventory
            0, 0, 0,  # overage1_4 overage2_4 overage3_4
            0, 0, 0, 0, 0, 0, 0, 0, 0,  # future demand
            1,  # เริ่มต้น step แรกคือ เป็นช่วง onpeak
            1,  # Demand pattern
            0, 0, 0,  # Demand r1-3 at 4 rd period
            0, 0, 0,  # Demand r1-3 at 8 th period
            0, 0, 0,  # Demand r1-3 at 12 th period
            0, 0, 0,  # Demand r1-3 at 16 th period
            0, 0, 0,  # Demand r1-3 at 20 th period
            0, 0, 0,  # Demand r1-3 at 24 th period
            0, 0, 0  # Demand r1-3 at 28 th period
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
        (self.demand_all, self.aaa) = self.create_demand_all()

        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}



    def create_index2(self):
        rng3 = randint(0, 10000000)  # 24-01-66=10000 # train with 500 set of demand data  #5000
        np.random.seed(rng3)
        aaa = np.random.randint(3, 6)  # อันนี้สุ่ม 3-5

        idrv_set = []
        for j in range(1, 91):
            idrv = round(random.uniform(0.40, 1.00), 2)
            idrv_set.append(idrv)

        # if aaa == 1:
        #     index2 = idrv_set
        # if aaa == 2:
        #     index2 = idrv_set
        if aaa == 3:
            # index = [0.713,0.744,0.83,0.96,1.09,1.179,1.253,1.311,1.261,1.174,1.1,1,0.88,0.78,0.72]  # season 1
            index2 = [0.0, 0.0, 0.0, 0.713, 0.713, 0.713, 0.0, 0.0, 0.0, 0.744, 0.744, 0.744, 0.0, 0.0, 0.0, 0.83, 0.83,
                      0.83, 0.0, 0.0, 0.0, 0.96, 0.96, 0.96, 0.0, 0.0, 0.0, 1.09, 1.09, 1.09, 0.0, 0.0, 0.0, 1.179,
                      1.179, 1.179, 0.0, 0.0, 0.0, 1.253, 1.253, 1.253, 0.0, 0.0, 0.0, 1.311, 1.311, 1.311, 0.0, 0.0,
                      0.0, 1.261, 1.261, 1.261, 0.0, 0.0, 0.0, 1.174, 1.174, 1.174, 0.0, 0.0, 0.0, 1.1, 1.1, 1.1, 0.0,
                      0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.88, 0.88, 0.88, 0.0, 0.0, 0.0, 0.78, 0.78, 0.78, 0.0,
                      0.0, 0.0, 0.72, 0.72, 0.72]
        if aaa == 4:
            # index = [1.179,1.253,1.311,1.261,1.174,1.092,1.015,0.913,0.805,0.741,0.713,0.744,0.83,0.988,1.116]   # season 2
            index2 = [0.0, 0.0, 0.0, 0.5704, 0.5704, 0.5704, 0.0, 0.0, 0.0, 0.6324, 0.6324, 0.6324, 0.0, 0.0, 0.0,
                      0.7885, 0.7885, 0.7885, 0.0, 0.0, 0.0, 0.912, 0.912, 0.912, 0.0, 0.0, 0.0, 1.1445, 1.1445, 1.1445,
                      0.0, 0.0, 0.0, 1.23795, 1.23795, 1.23795, 0.0, 0.0, 0.0, 1.3783, 1.3783, 1.3783, 0.0, 0.0, 0.0,
                      1.4421, 1.4421, 1.4421, 0.0, 0.0, 0.0, 1.3871, 1.3871, 1.3871, 0.0, 0.0, 0.0, 1.2327, 1.2327, 1.2327,
                      0.0, 0.0, 0.0, 1.155, 1.155, 1.155, 0.0, 0.0, 0.0, 1.05, 1.05, 1.05, 0.0, 0.0, 0.0, 0.836, 0.836, 0.836,
                      0.0, 0.0, 0.0, 0.663, 0.663, 0.663, 0.0, 0.0, 0.0, 0.576, 0.576, 0.576]
        if aaa == 5:
            # index = [1.1,1,0.88,0.78,0.72,0.713,0.744,0.83,0.96,1.09,1.179,1.253,1.311,1.261,1.174]  # season 3
            index2 = [0.0, 0.0, 0.0, 0.53475, 0.53475, 0.53475, 0.0, 0.0, 0.0, 0.5952, 0.5952, 0.5952, 0.0, 0.0, 0.0,
                      0.7055, 0.7055, 0.7055, 0.0, 0.0, 0.0, 0.864, 0.864, 0.864, 0.0, 0.0, 0.0, 1.1445, 1.1445, 1.1445,
                      0.0, 0.0, 0.0, 1.28511, 1.28511, 1.28511, 0.0, 0.0, 0.0, 1.44095, 1.44095, 1.44095, 0.0, 0.0, 0.0,
                      1.63875, 1.63875, 1.63875, 0.0, 0.0, 0.0, 1.45015, 1.45015, 1.45015, 0.0, 0.0, 0.0, 1.27966, 1.27966, 1.27966,
                      0.0, 0.0, 0.0, 1.155, 1.155, 1.155, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.748, 0.748, 0.748,
                      0.0, 0.0, 0.0, 0.624, 0.624, 0.624, 0.0, 0.0, 0.0, 0.54, 0.54, 0.54]
        # if aaa == 6:
        #     # index = [0.741,0.913,1.179,1.56,1.65,0.913,0.805,0.713,0.69,0.55,0.35,0.55,0.744,0.83,0.988]  # extreme1
        #     index2 = [0.0, 0.0, 0.0, 0.39215, 0.39215, 0.39215, 0.0, 0.0, 0.0, 0.5208, 0.5208, 0.5208, 0.0, 0.0, 0.0,
        #               0.664, 0.664, 0.664, 0.0, 0.0, 0.0, 0.816, 0.816, 0.816, 0.0, 0.0, 0.0, 1.09, 1.09, 1.09,
        #               0.0, 0.0, 0.0, 1.4148, 1.4148, 1.4148, 0.0, 0.0, 0.0, 1.6289, 1.6289, 1.6289, 0.0, 0.0, 0.0,
        #               1.76985, 1.76985, 1.76985, 0.0, 0.0, 0.0, 1.6393, 1.6393, 1.6393, 0.0, 0.0, 0.0, 1.4088, 1.4088, 1.4088,
        #               0.0, 0.0, 0.0, 1.1, 1.1, 1.1, 0.0, 0.0, 0.0, 0.85, 0.85, 0.85, 0.0, 0.0, 0.0, 0.704, 0.704, 0.704,
        #               0.0, 0.0, 0.0, 0.546, 0.546, 0.546, 0.0, 0.0, 0.0, 0.396, 0.396, 0.396]
        # if aaa == 7:
        #     # index = [0.69,0.55,0.35,0.55,0.744,0.83,0.988,1.215,1.355,1.56,1.65,1.121,0.805,0.713,0.69]  # extreme2
        #     index2 = [0.0, 0.0, 0.0, 0.69, 0.69, 0.69, 0.0, 0.0, 0.0, 0.55, 0.55, 0.55, 0.0, 0.0, 0.0, 0.35, 0.35, 0.35,
        #               0.0, 0.0, 0.0, 0.55, 0.55, 0.55, 0.0, 0.0, 0.0, 0.744, 0.744, 0.744, 0.0, 0.0, 0.0, 0.83, 0.83,
        #               0.83, 0.0, 0.0, 0.0, 0.988, 0.988, 0.988, 0.0, 0.0, 0.0, 1.215, 1.215, 1.215, 0.0, 0.0, 0.0,
        #               1.355, 1.355, 1.355, 0.0, 0.0, 0.0, 1.56, 1.56, 1.56, 0.0, 0.0, 0.0, 1.65, 1.65, 1.65, 0.0, 0.0,
        #               0.0, 1.121, 1.121, 1.121, 0.0, 0.0, 0.0, 0.805, 0.805, 0.805, 0.0, 0.0, 0.0, 0.713, 0.713, 0.713,
        #               0.0, 0.0, 0.0, 0.69, 0.69, 0.69]

        if aaa >= 3:
            demand_r1 =  np.random.randint(2700,3200)
            demand_r2 =  np.random.randint(2300,2600)
            demand_r3 =  np.random.randint(1500, 1700)
        if aaa == 1:
            demand_r1 =  np.random.randint(2975, 4025)  # (2500, 4500) #avg + - 15%
            demand_r2 =  np.random.randint(2338, 3163)  # 2000, 3500
            demand_r3 =  np.random.randint(1488, 2013)  # 1000, 2500
        if aaa == 2:
            demand_r1 =  np.random.randint(2975, 4025)  # (2500, 4500) #avg + - 15%
            demand_r2 =  np.random.randint(2338, 3163)  # 2000, 3500
            demand_r3 =  np.random.randint(1488, 2013)  # 1000, 2500

        return index2, demand_r1, demand_r2, demand_r3, aaa

    def create_demand_all(self):
        demand_array2 = list(range(1, 91))
        demand_all = [0, 0, 0]

        set_stepcount1 = [0, 4, 8, 12, 16, 20]
        set_stepcount3 = [24]
        #print("self.step_count## ==", self.step_count)
        #if self.step_count == 1:  #ให้ create แค่ period ที่ 1
        (index2, demand_r1, demand_r2, demand_r3, aaa) = self.create_index2()


        for step_count in range(0, 30):
            # print("step count =", step_count)
            y = step_count + 1
            # print("y = ", y)
            if step_count < 24:  # 25
                if step_count in set_stepcount1:  # ถ้าปล่อยให้ถึง 29 ค่าindex y จะหลุดนอกสมาชิก array
                    demand_array2[y * 3] = demand_r1 * index2[y * 3]  #######
                    demand_all.append(demand_array2[y * 3])
                    demand_array2[y * 3 + 1] = demand_r2 * index2[y * 3 + 1]  #######
                    demand_all.append(demand_array2[y * 3 + 1])
                    demand_array2[y * 3 + 2] = demand_r3 * index2[y * 3 + 2]  #######
                    demand_all.append(demand_array2[y * 3 + 2])
                    demand_array2[y * 3 + 3] = 0
                    demand_all.append(demand_array2[y * 3 + 3])
                    demand_array2[y * 3 + 4] = 0
                    demand_all.append(demand_array2[y * 3 + 4])
                    demand_array2[y * 3 + 5] = 0
                    demand_all.append(demand_array2[y * 3 + 5])
                    demand_array2[y * 3 + 6] = demand_r1 * index2[y * 3 + 6]  #######
                    demand_all.append(demand_array2[y * 3 + 6])
                    demand_array2[y * 3 + 7] = demand_r2 * index2[y * 3 + 7]  #######
                    demand_all.append(demand_array2[y * 3 + 7])
                    demand_array2[y * 3 + 8] = demand_r3 * index2[y * 3 + 8]  #######
                    demand_all.append(demand_array2[y * 3 + 8])
                    demand_array2[y * 3 + 9] = 0
                    demand_all.append(demand_array2[y * 3 + 9])
                    demand_array2[y * 3 + 10] = 0
                    demand_all.append(demand_array2[y * 3 + 10])
                    demand_array2[y * 3 + 11] = 0
                    demand_all.append(demand_array2[y * 3 + 11])
                    d1 = demand_array2[y * 3]
                    d2 = demand_array2[y * 3 + 1]
                    d3 = demand_array2[y * 3 + 2]
                    d4 = demand_array2[y * 3 + 3]
                    d5 = demand_array2[y * 3 + 4]
                    d6 = demand_array2[y * 3 + 5]
                    d7 = demand_array2[y * 3 + 6]
                    d8 = demand_array2[y * 3 + 7]
                    d9 = demand_array2[y * 3 + 8]


            if step_count == 24:
                demand_array2[y * 3] = demand_r1 * index2[y * 3]  #######
                demand_all.append(demand_array2[y * 3])
                demand_array2[y * 3 + 1] = demand_r2 * index2[y * 3 + 1]  #######
                demand_all.append(demand_array2[y * 3 + 1])
                demand_array2[y * 3 + 2] = demand_r3 * index2[y * 3 + 2]  #######
                demand_all.append(demand_array2[y * 3 + 2])
                demand_array2[y * 3 + 3] = 0
                demand_all.append(demand_array2[y * 3 + 3])
                demand_array2[y * 3 + 4] = 0
                demand_all.append(demand_array2[y * 3 + 4])
                demand_array2[y * 3 + 5] = 0
                demand_all.append(demand_array2[y * 3 + 5])
                demand_array2[y * 3 + 6] = demand_r1 * index2[y * 3 + 6]  #######
                demand_all.append(demand_array2[y * 3 + 6])
                demand_array2[y * 3 + 7] = demand_r2 * index2[y * 3 + 7]  #######
                demand_all.append(demand_array2[y * 3 + 7])
                demand_array2[y * 3 + 8] = demand_r3 * index2[y * 3 + 8]  #######
                demand_all.append(demand_array2[y * 3 + 8])
                demand_array2[y * 3 + 9] = 0
                demand_all.append(demand_array2[y * 3 + 9])
                demand_array2[y * 3 + 10] = 0
                demand_all.append(demand_array2[y * 3 + 10])
                demand_array2[y * 3 + 11] = 0
                demand_all.append(demand_array2[y * 3 + 11])
                d1 = demand_array2[y * 3]
                d2 = demand_array2[y * 3 + 1]
                d3 = demand_array2[y * 3 + 2]
                d4 = demand_array2[y * 3 + 3]
                d5 = demand_array2[y * 3 + 4]
                d6 = demand_array2[y * 3 + 5]
                d7 = demand_array2[y * 3 + 6]
                d8 = demand_array2[y * 3 + 7]
                d9 = demand_array2[y * 3 + 8]
                # print("demand_all2= ", demand_all)

        demand_all.extend([
            demand_r1 * index2[87],
            demand_r2 * index2[88],
            demand_r3 * index2[89],
            0, 0, 0,
        ])

        assert (len(demand_all) == 93)
        return (demand_all, aaa)

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        info = {}


        # all model parameters
        # holding cost
        h1 = 49.12  # 50.96
        h2 = 29.86  # 34.53
        h3 = 47.77  # 48.66
        # Lost of good Will
        k1 = 10200 * 25
        k2 = 6200 * 25
        k3 = 9920 * 25
        # Sell price
        p1 = 5100
        p2 = 3100
        p3 = 4960
        # unit cost
        c1 = 1434.375  # 3060
        c2 = 871.875  # 1860
        c3 = 1395.000  # 2976
        fc_m1 = 1426.140
        fc_m2 = 1326.140
        # variable_production_cost
        vcm1 = 0
        vcm2 = 0
        vc_m1_on = 1796.380  # ของเดินคือ 1796.380 แต่ลองเพิ่มให้เยอะๆขึ้น เพื่อให้เอเจนต์ฉลาดในการเลี่ยงผลิตช่วง On-peak
        vc_m1_off = 1027.450
        vc_m2_on = 1719.050  # ของเดิมคือ 1719.050
        vc_m2_off = 986.240
        # clean value
        d1 = 0
        d2 = 0
        d3 = 0
        demand1 = 0
        demand2 = 0
        demand3 = 0

        # binary variable
        CO11 = 0  # change over var of machine1:  contiguous period type
        CO21 = 0  # change over var of machine2:  contiguous period type
        CO12 = 0  # change over var of machine1: skipping 1 time interval
        CO22 = 0
        CO13 = 0  # change over var of machine1: skipping 2 time interval
        CO23 = 0
        SW1 = 0  # switch on
        SW2 = 0
        FC_M1 = 0  # fix production (N)
        FC_M2 = 0
        stepcount = 0

        self.changeover_cost_of_m1 = 0
        self.changeover_cost_of_m2 = 0
        self.switch_on_cost = 0

        extra_penalty1 = 0
        extra_penalty2 = 0
        extra_penalty3 = 0
        extra_penalty1_2 = 0
        extra_penalty2_2 = 0
        extra_penalty3_2 = 0
        extra_penalty1_3 = 0
        extra_penalty2_3 = 0
        extra_penalty3_3 = 0


        #         print("=========================================================================================")
        #         print("step :", self.step_count)
        # print("state =", state)

        # variable use to remember production data from period
        N1P_ = self.state[6]
        N1P1_ = self.state[7]
        # print("N1P1_ = ",N1P1_)
        N1P2_ = self.state[8]
        N1P3_ = self.state[9]
        N2P_ = self.state[10]
        N2P1_ = self.state[11]
        N2P2_ = self.state[12]
        N2P3_ = self.state[13]

        # Retrieves the value from the previous period to calculate the changeOver
        # with skipping 1 time and 2 time interval type before the variable takes the new value.
        # print("CO_var_set1 = ", self.CO_var_set1)
        N1P_2 = self.CO_var_set1[0]
        N1P1_2 = self.CO_var_set1[1]
        # print("N1P1_2 = ", N1P1_2)
        N1P2_2 = self.CO_var_set1[2]
        N1P3_2 = self.CO_var_set1[3]
        N2P_2 = self.CO_var_set1[4]
        N2P1_2 = self.CO_var_set1[5]
        N2P2_2 = self.CO_var_set1[6]
        N2P3_2 = self.CO_var_set1[7]

        # print("CO_var_set2 = ", self.CO_var_set2)
        N1P_3 = self.CO_var_set2[0]
        N1P1_3 = self.CO_var_set2[1]
        # print("N1P1_2 = ", N1P1_3)
        N1P2_3 = self.CO_var_set2[2]
        N1P3_3 = self.CO_var_set2[3]
        N2P_3 = self.CO_var_set2[4]
        N2P1_3 = self.CO_var_set2[5]
        N2P2_3 = self.CO_var_set2[6]
        N2P3_3 = self.CO_var_set2[7]

        # demand of this period
        demand1 = self.state[3]
        demand2 = self.state[4]
        demand3 = self.state[5]
        # print("demand this period =", demand1, demand2, demand3)

        # This clears production data from the previous period.
        self.state[6] = 0
        self.state[7] = 0
        self.state[8] = 0
        self.state[9] = 0
        self.state[10] = 0
        self.state[11] = 0
        self.state[12] = 0
        self.state[13] = 0

        # print("state[7] =", self.state[7])
        N1P = 0
        N1P1 = 0
        N1P2 = 0
        N1P3 = 0
        N2P = 0
        N2P1 = 0
        N2P2 = 0
        N2P3 = 0
        M1P = 0
        M1P1 = 0
        M1P2 = 0
        M1P3 = 0
        M2P = 0
        M2P1 = 0
        M2P2 = 0
        M2P3 = 0
        extra_p_on = 0



        on_hand1, on_hand2, on_hand3, demand1, demand2, demand3, N1P, N1P1, \
        N1P2, N1P3, N2P, N2P1, N2P2, N2P3, overage1_2, overage2_2, overage3_2, \
        overage1_3, overage2_3, overage3_3, \
        overage1_4, overage2_4, overage3_4, \
        demand4, demand5, demand6, \
        demand7, demand8, demand9, \
        demand10, demand11, demand12, \
        extra_p_on, \
        aaa3, \
        dr1_4, dr2_4, dr3_4, \
        dr1_8, dr2_8, dr3_8, \
        dr1_12, dr2_12, dr3_12, \
        dr1_16, dr2_16, dr3_16, \
        dr1_20, dr2_20, dr3_20, \
        dr1_24, dr2_24, dr3_24, \
        dr1_28, dr2_28, dr3_28 = self.state

        aaa3 = self.aaa

        #print("self.demand_all =", self.demand_all)
        dr1_4 = self.demand_all[9]
        dr2_4 = self.demand_all[10]
        dr3_4 = self.demand_all[11]
        dr1_8 = self.demand_all[21]
        dr2_8 = self.demand_all[22]
        dr3_8 = self.demand_all[23]
        dr1_12 = self.demand_all[33]
        dr2_12 = self.demand_all[34]
        dr3_12 = self.demand_all[35]
        dr1_16 = self.demand_all[45]
        dr2_16 = self.demand_all[46]
        dr3_16 = self.demand_all[47]
        dr1_20 = self.demand_all[57]
        dr2_20 = self.demand_all[58]
        dr3_20 = self.demand_all[59]
        dr1_24 = self.demand_all[69]
        dr2_24 = self.demand_all[70]
        dr3_24 = self.demand_all[71]
        dr1_28 = self.demand_all[81]
        dr2_28 = self.demand_all[82]
        dr3_28 = self.demand_all[83]

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
        demand1 = demand1 * (maxd1 - mind1) + mind1
        demand2 = demand2 * (maxd2 - mind2) + mind2
        demand3 = demand3 * (maxd3 - mind3) + mind3

        demand4 = demand4 * (maxd1 - mind1) + mind1
        demand5 = demand5 * (maxd2 - mind2) + mind2
        demand6 = demand6 * (maxd3 - mind3) + mind3
        demand7 = demand7 * (maxd1 - mind1) + mind1
        demand8 = demand8 * (maxd2 - mind2) + mind2
        demand9 = demand9 * (maxd3 - mind3) + mind3
        demand10 = demand10 * (maxd1 - mind1) + mind1
        demand11 = demand11 * (maxd2 - mind2) + mind2
        demand12 = demand12 * (maxd3 - mind3) + mind3


        # print("demand in this period =", demand1, demand2, demand3)

        # self.demand_real = info[24]   #เรียก info มาปริ้นข้างใน env ไม่ได้ จะ error เพราะ info ส่งผ่านไปข้างนอกอย่างเดียว ไม่ได้รับกลับเข้าในแต่ละ step ของ env
        self.demand_real.append(demand1)
        self.demand_real.append(demand2)
        self.demand_real.append(demand3)
        # print("real demand = ",self.demand_real)

        on_hand1 = on_hand1 * (maxr1 - 0) + 0
        on_hand2 = on_hand2 * (maxr2 - 0) + 0
        on_hand3 = on_hand3 * (maxr3 - 0) + 0
        overage1_2 = overage1_2 * (maxr1 - 0) + 0
        overage2_2 = overage2_2 * (maxr2 - 0) + 0
        overage3_2 = overage3_2 * (maxr3 - 0) + 0
        overage1_3 = overage1_3 * (maxr1 - 0) + 0
        overage2_3 = overage2_3 * (maxr2 - 0) + 0
        overage3_3 = overage3_3 * (maxr3 - 0) + 0
        overage1_4 = overage1_4 * (maxr1 - 0) + 0
        overage2_4 = overage2_4 * (maxr2 - 0) + 0
        overage3_4 = overage3_4 * (maxr3 - 0) + 0

        #         print("value หลังแปลงกลับ")
        #         print("demand1 =", demand1)
        #         print("on_hand1 =", on_hand1)

        # print("Step :", self.step_count)
        # print("onhand from last period =", on_hand1, on_hand2, on_hand3)
        # print("onhand1 from last period =", on_hand1)
        # print("onhand2 from last period =", on_hand2)
        # print("onhand3 from last period =", on_hand3)

        if action == 0:
            case = [[1, 0], [2, 3]]
            N1P = 0
            M1P = 0
            N2P3 = 1
            M2P3 = 1359
        if action == 1:
            case = [[1, 2], [2, 2]]
            N1P2 = 1
            M1P2 = 2223
            N2P2 = 1
            M2P2 = 1853
        if action == 2:
            case = [[1, 0], [2, 2]]
            N1P = 0
            M1P = 0
            N2P2 = 1
            M2P2 = 1853
        if action == 3:
            case = [[1, 2], [2, 1]]
            N1P2 = 1
            M1P2 = 2223
            N2P1 = 1
            M2P1 = 2717
        if action == 4:
            case = [[1, 1], [2, 1]]
            N1P1 = 1
            M1P1 = 3211
            N2P1 = 1
            M2P1 = 2717
        if action == 5:
            case = [[1, 1], [2, 3]]
            N1P1 = 1
            M1P1 = 3211
            N2P3 = 1
            M2P3 = 1359
        if action == 6:
            case = [[1, 3], [2, 2]]
            N1P3 = 1
            M1P3 = 1668
            N2P2 = 1
            M2P2 = 1853
        if action == 7:
            case = [[1, 2], [2, 0]]
            N1P2 = 1
            M1P2 = 2223
            N2P = 0
            M2P = 0
        if action == 8:
            case = [[1, 1], [2, 2]]
            N1P1 = 1
            M1P1 = 3211
            N2P2 = 1
            M2P2 = 1853
        if action == 9:
            case = [[1, 0], [2, 0]]
            N1P = 0
            M1P = 0
            N2P = 0
            M2P = 0
        if action == 10:
            case = [[1, 3], [2, 3]]
            N1P3 = 1
            M1P3 = 1668
            N2P3 = 1
            M2P3 = 1359
        if action == 11:
            case = [[1, 3], [2, 1]]
            N1P3 = 1
            M1P3 = 1668  # 1668   #ลองแกล้งเปลี่ยน action 11 ให้ไม่ผลิตเลยดู ว่ามันจะเลือก action ไหนแทน
            N2P1 = 1
            M2P1 = 2717  # 2717
        if action == 12:
            case = [[1, 0], [2, 1]]
            N1P = 0
            M1P = 0
            N2P1 = 1
            M2P1 = 2717
        if action == 13:
            case = [[1, 3], [2, 0]]
            N1P3 = 1
            M1P3 = 1668
            N2P = 0
            M2P = 0
        if action == 14:
            case = [[1, 1], [2, 0]]
            N1P1 = 1
            M1P1 = 3211
            N2P = 0
            M2P = 0
        if action == 15:
            case = [[1, 2], [2, 3]]
            N1P2 = 1
            M1P2 = 2223
            N2P3 = 1
            M2P3 = 1359

            # print("action =", action)
            # print("N1P1= ",N1P1 ," ,M1P1 =", M1P1)
            # print("N1P2= ", N1P2, " ,M1P2 =", M1P2)
            # print("N1P3= ", N1P3, " ,M1P3 =", M1P3)
            # print("N2P1= ",N2P1 ," ,M1P1 =", M2P1)
            # print("N2P2= ", N2P2, " ,M1P2 =", M2P2)
            # print("N2P3= ", N2P3, " ,M1P3 =", M2P3)

        ##if there a production --> NP=1
        if N1P1 == 1 or N1P2 == 1 or N1P3 == 1:
            N1P = 1
        if N2P1 == 1 or N2P2 == 1 or N2P3 == 1:
            N2P = 1

        R1 = M1P1 + M2P1  # R1 = recieve product1
        R2 = M1P2 + M2P2
        R3 = M1P3 + M2P3

        on_hand1 += R1  # onhand1
        on_hand2 += R2
        on_hand3 += R3

        # Compute Reward
        sales1 = min(on_hand1, demand1)  # ถ้า on_hand น้อยกว่า demand ก็ขายแค่ = on hand
        sales2 = min(on_hand2, demand2)
        sales3 = min(on_hand3, demand3)
        # print("sales1 =", sales1)
        sales_revenue = p1 * sales1 + p2 * sales2 + p3 * sales3
        overage1 = max(0, on_hand1 - sales1)
        overage2 = max(0, on_hand2 - sales2)
        overage3 = max(0, on_hand3 - sales3)
        # print("overage1 =", overage1)
        underage1 = max(0, demand1 - on_hand1)
        underage2 = max(0, demand2 - on_hand2)
        underage3 = max(0, demand3 - on_hand3)
        purchase_cost = c1 * (M1P1 + M2P1) + c2 * (M1P2 + M2P2) + c3 * (M1P3 + M2P3)
        holding = (
                          overage1 * h1 + overage2 * h2 + overage3 * h3) / 2  # holding devide by 2 cause holding cost rate is per day not per shift
        penalty_lost_sale = k1 * underage1 + k2 * underage2 + k3 * underage3

        ###change over of M1 แบบเปลี่ยน ช่วงติดกัน
        if N1P1_ == 1 and N1P2 == 1:  # N1P1_(last period) = 1 and N1P2 (this period) = 1 --> then there change over from P1 to P2 in M1
            CO11 = 1
        if N1P1_ == 1 and N1P3 == 1:  # N1P1_(last period) = 1 and N1P3 (this period) = 1 --> then there change over from P1 to P3 in M1
            CO11 = 1
        if N1P2_ == 1 and N1P1 == 1:
            CO11 = 1
        if N1P2_ == 1 and N1P3 == 1:
            CO11 = 1
        if N1P3_ == 1 and N1P1 == 1:
            CO11 = 1
        if N1P3_ == 1 and N1P2 == 1:
            CO11 = 1
        # print("CO11 = ", CO11)
        ###change over of M2
        if N2P1_ == 1 and N2P2 == 1:  # N2P1_(last period) = 1 and N2P2 (this period) = 1 --> then there change over from P1 to P2 in M2
            CO21 = 1
        if N2P1_ == 1 and N2P3 == 1:
            CO21 = 1
        if N2P2_ == 1 and N2P1 == 1:
            CO21 = 1
        if N2P2_ == 1 and N2P3 == 1:
            CO21 = 1
        if N2P3_ == 1 and N2P1 == 1:
            CO21 = 1
        if N2P3_ == 1 and N2P2 == 1:
            CO21 = 1
        # print("CO21 = ", CO21)

        # คิด change over แบบเว้นข้าม 1period
        if N1P1_2 == 1 and N1P_ == 0 and N1P2 == 1:  # N1P1_2(last two period) = 1  but there no production last period (N1P_ = 0),and N1P2 (this period) = 1 --> then there change over from P1 to P2 in M1
            CO12 = 1
        if N1P1_2 == 1 and N1P_ == 0 and N1P3 == 1:  # N1P1_2(last two period) = 1 but there no production last period (N1P_ = 0),and N1P3 (this period) = 1 --> then there change over from P1 to P3 in M1
            CO12 = 1
        if N1P2_2 == 1 and N1P_ == 0 and N1P1 == 1:
            CO12 = 1
        if N1P2_2 == 1 and N1P_ == 0 and N1P3 == 1:
            CO12 = 1
        if N1P3_2 == 1 and N1P_ == 0 and N1P1 == 1:
            CO12 = 1
        if N1P3_2 == 1 and N1P_ == 0 and N1P2 == 1:
            CO12 = 1
        # print("CO12 = ", CO12)
        ###change over of M2 แบบข้าม 1 ช่วง
        if N2P1_2 == 1 and N2P_ == 0 and N2P2 == 1:
            CO22 = 1
        if N2P1_2 == 1 and N2P_ == 0 and N2P3 == 1:
            CO22 = 1
        if N2P2_2 == 1 and N2P_ == 0 and N2P1 == 1:
            CO22 = 1
        if N2P2_2 == 1 and N2P_ == 0 and N2P3 == 1:
            CO22 = 1
        if N2P3_2 == 1 and N2P_ == 0 and N2P1 == 1:
            CO22 = 1
        if N2P3_2 == 1 and N2P_ == 0 and N2P2 == 1:
            CO22 = 1
        # print("CO22 = ", CO22)

        # คิด change over แบบเว้นข้าม 2period
        if N1P1_3 == 1 and N1P_2 == 0 and N1P_ == 0 and N1P2 == 1:  # N1P1_2(last two period) = 1  but there no production last period (N1P_ = 0),and N1P2 (this period) = 1 --> then there change over from P1 to P2 in M1
            CO13 = 1
        if N1P1_3 == 1 and N1P_2 == 0 and N1P_ == 0 and N1P3 == 1:  # N1P1_2(last two period) = 1 but there no production last period (N1P_ = 0),and N1P3 (this period) = 1 --> then there change over from P1 to P3 in M1
            CO13 = 1
        if N1P2_3 == 1 and N1P_2 == 0 and N1P_ == 0 and N1P1 == 1:
            CO13 = 1
        if N1P2_3 == 1 and N1P_2 == 0 and N1P_ == 0 and N1P3 == 1:
            CO13 = 1
        if N1P3_3 == 1 and N1P_2 == 0 and N1P_ == 0 and N1P1 == 1:
            CO13 = 1
        if N1P3_3 == 1 and N1P_2 == 0 and N1P_ == 0 and N1P2 == 1:
            CO13 = 1
        # print("CO13 = ", CO13)
        ###change over of M2 แบบข้าม 1 ช่วง
        if N2P1_3 == 1 and N2P_2 == 0 and N2P_ == 0 and N2P2 == 1:
            CO23 = 1
        if N2P1_3 == 1 and N2P_2 == 0 and N2P_ == 0 and N2P3 == 1:
            CO23 = 1
        if N2P2_3 == 1 and N2P_2 == 0 and N2P_ == 0 and N2P1 == 1:
            CO23 = 1
        if N2P2_3 == 1 and N2P_2 == 0 and N2P_ == 0 and N2P3 == 1:
            CO23 = 1
        if N2P3_3 == 1 and N2P_2 == 0 and N2P_ == 0 and N2P1 == 1:
            CO23 = 1
        if N2P3_3 == 1 and N2P_2 == 0 and N2P_ == 0 and N2P2 == 1:
            CO23 = 1
        # print("CO23 = ", CO23)

        ##เก็บค่าการผลิต เพื่อนำไปคิด change over แบบเว้นข้าม 2period ก่อนหน้าเพื่อมาคิดCO13 CO23
        N1P_3 = N1P_2
        N1P1_3 = N1P1_2
        # print("N1P1_3 = ", N1P1_3)
        N1P2_3 = N1P2_2
        N1P3_3 = N1P3_2
        N2P_3 = N2P_2
        N2P1_3 = N2P1_2
        N2P2_3 = N2P2_2
        N2P3_3 = N2P3_2
        self.CO_var_set2 = [N1P_3, N1P1_3, N1P2_3, N1P3_3, N2P_3, N2P1_3, N2P2_3, N2P3_3]
        # print("CO_var_set2 = ", self.CO_var_set2)

        ##Collect production values that used to calculate
        # the previous 'change over :skipping 1 period type' to calculate CO12 CO22
        N1P_2 = N1P_
        N1P1_2 = N1P1_
        # print("N1P1_2 = ", N1P1_2)
        N1P2_2 = N1P2_
        N1P3_2 = N1P3_
        N2P_2 = N2P_
        N2P1_2 = N2P1_
        N2P2_2 = N2P2_
        N2P3_2 = N2P3_
        self.CO_var_set1 = [N1P_2, N1P1_2, N1P2_2, N1P3_2, N2P_2, N2P1_2, N2P2_2, N2P3_2]
        # print("CO_var_set1 = ", self.CO_var_set1)

        ##Switch on-off variable
        if N1P_ == 0 and N1P == 1:
            SW1 = 1
        if N2P_ == 0 and N2P == 1:
            SW2 = 1

        ##Fix cost ,If producing, this cost will be incurred.
        if N1P1 + N1P2 + N1P3 == 1:
            FC_M1 = 1
        if N2P1 + N2P2 + N2P3 == 1:
            FC_M2 = 1
        # Assign - on-and off-peak p cost to each period
        #         weekend_stepcount = [3, 4, 5, 6, 17, 18, 19, 20]
        #         on_peak_stepcount = [1, 7, 9, 11, 13, 15, 21, 23, 25, 27, 29]
        #         off_peak_stepcount = [2, 8, 10, 12, 14, 22, 24, 26, 28, 30]
        #         stp = 0
        #         stp = self.step_count + 1

        weekend_stepcount = [2, 3, 4, 5, 16, 17, 18, 19]
        on_peak_stepcount = [0, 6, 8, 10, 12, 14, 20, 22, 24, 26, 28]
        off_peak_stepcount = [1, 7, 9, 11, 13, 15, 21, 23, 25, 27, 29]
        stp = 0
        #         stp = self.step_count + 1
        stp = self.step_count
        #print("stp =",stp)
        if stp in on_peak_stepcount:
           #print("On-peak")
           extra_p_on = 1
        extra_p_on1_1 = 0  # extra penalty กรณีผลิตช่วง onpeak เพื่อให้ agent ฉลาดขึ้น
        extra_p_on1_2 = 0
        extra_p_on1_3 = 0
        extra_p_on2_1 = 0
        extra_p_on2_2 = 0
        extra_p_on2_3 = 0
        extra_p_on_set = []
        extra_r_weekend1_1 = 0  # extra reward กรณีผลิตช่วง weekend เพื่อให้ agent ฉลาดขึ้น
        extra_r_weekend1_2 = 0
        extra_r_weekend1_3 = 0
        extra_r_weekend2_1 = 0
        extra_r_weekend2_2 = 0
        extra_r_weekend2_3 = 0

        ######################################################################
        penalty_onpeak = 3000
        reward_weekend = 0

        if stp in on_peak_stepcount:
            # print("yes")
            vcm1 = vc_m1_on
            # print("vcm1 = ", vcm1)
            vcm2 = vc_m2_on
            co11 = 33896
            co21 = 33896
            sw1 = 401.78
            self.changeover_cost_of_m1 = co11 * (
                    CO11 + CO12 + CO13)  # period นึงจะเกิด CO11, CO12, CO13 ได้แค่ 1 กรณี จึงจับรวมได้เลย
            self.changeover_cost_of_m2 = co21 * (CO21 + CO22 + CO23)
            self.switch_on_cost = sw1 * (SW1 + SW2)
            self.variable_cost_m1 = vcm1 * (M1P1 + M1P2 + M1P3)
            self.variable_cost_m2 = vcm2 * (M2P1 + M2P2 + M2P3)
            if M1P1 > 0:
                extra_p_on1_1 = penalty_onpeak * M1P1
                extra_p_on_set.append(extra_p_on1_1)
            if M1P2 > 0:
                extra_p_on1_2 = penalty_onpeak * M1P2
                extra_p_on_set.append(extra_p_on1_2)
            if M1P3 > 0:
                extra_p_on1_3 = penalty_onpeak * M1P3
                extra_p_on_set.append(extra_p_on1_3)
            if M2P1 > 0:
                extra_p_on2_1 = penalty_onpeak * M2P1
                extra_p_on_set.append(extra_p_on2_1)
            if M2P2 > 0:
                extra_p_on2_2 = penalty_onpeak * M2P2
                extra_p_on_set.append(extra_p_on2_2)
            if M2P3 > 0:
                extra_p_on2_3 = penalty_onpeak * M2P3
                extra_p_on_set.append(extra_p_on2_3)
        #if stp + 1 in on_peak_stepcount:  # check if next state in onpeak? to pass extra_p_on in the state[27]
        #    extra_p_on = 1  # = next step will be on-peak
        if stp == 0:
            penalty_onpeak = 50000
            if M1P1 > 0:
                extra_p_on1_1 = penalty_onpeak * M1P1
                extra_p_on_set.append(extra_p_on1_1)
            if M1P2 > 0:
                extra_p_on1_2 = penalty_onpeak * M1P2
                extra_p_on_set.append(extra_p_on1_2)
            if M1P3 > 0:
                extra_p_on1_3 = penalty_onpeak * M1P3
                extra_p_on_set.append(extra_p_on1_3)
            if M2P1 > 0:
                extra_p_on2_1 = penalty_onpeak * M2P1
                extra_p_on_set.append(extra_p_on2_1)
            if M2P2 > 0:
                extra_p_on2_2 = penalty_onpeak * M2P2
                extra_p_on_set.append(extra_p_on2_2)
            if M2P3 > 0:
                extra_p_on2_3 = penalty_onpeak * M2P3
                extra_p_on_set.append(extra_p_on2_3)

        if stp in off_peak_stepcount:
            #print("Off-peak")
            extra_p_on = 0
            penalty_onpeak = 0
            vcm1 = vc_m1_off
            vcm2 = vc_m2_off
            co11 = 32139
            co21 = 32139
            sw1 = 331.9
            self.changeover_cost_of_m1 = co11 * (
                    CO11 + CO12 + CO13)  # period นึงจะเกิด CO11, CO12, CO13 ได้แค่ 1 กรณี จึงจับรวมได้เลย
            self.changeover_cost_of_m2 = co21 * (CO21 + CO22 + CO23)
            self.switch_on_cost = sw1 * (SW1 + SW2)
            self.variable_cost_m1 = vcm1 * (M1P1 + M1P2 + M1P3)
            self.variable_cost_m2 = vcm2 * (M2P1 + M2P2 + M2P3)
        if stp in weekend_stepcount:
            penalty_onpeak = 0
            reward_weekend = 50000
            vcm1 = vc_m1_off
            vcm2 = vc_m2_off
            extra_p_on = 0
            co11 = 32139
            co21 = 32139
            sw1 = 331.9
            self.changeover_cost_of_m1 = co11 * (
                    CO11 + CO12 + CO13)  # period นึงจะเกิด CO11, CO12, CO13 ได้แค่ 1 กรณี จึงจับรวมได้เลย
            self.changeover_cost_of_m2 = co21 * (CO21 + CO22 + CO23)
            self.switch_on_cost = sw1 * (SW1 + SW2)
            self.variable_cost_m1 = vcm1 * (M1P1 + M1P2 + M1P3)
            self.variable_cost_m2 = vcm2 * (M2P1 + M2P2 + M2P3)
            if M1P1 > 0:
                extra_r_weekend1_1 = reward_weekend * M1P1
                # extra_p_on_set.append(extra_p_on1_1)
            if M1P2 > 0:
                extra_r_weekend1_2 = reward_weekend * M1P2
                # extra_p_on_set.append(extra_p_on1_2)
            if M1P3 > 0:
                extra_r_weekend1_3 = reward_weekend * M1P3
                # extra_p_on_set.append(extra_p_on1_3)
            if M2P1 > 0:
                extra_r_weekend2_1 = reward_weekend * M2P1
                # extra_p_on_set.append(extra_p_on2_1)
            if M2P2 > 0:
                extra_r_weekend2_2 = reward_weekend * M2P2
                # extra_p_on_set.append(extra_p_on2_2)
            if M2P3 > 0:
                extra_r_weekend2_3 = reward_weekend * M2P3
                # extra_p_on_set.append(extra_p_on2_3)

        #         print("penalty_onpeak =", penalty_onpeak)
        #         print(extra_p_on1_1,extra_p_on1_2,extra_p_on1_3,extra_p_on2_1,extra_p_on2_2,extra_p_on2_3)
        #         print("reward_weekend =", reward_weekend)
        #         print(extra_r_weekend1_1,extra_r_weekend1_2,extra_r_weekend1_3,extra_r_weekend2_1,extra_r_weekend2_2,extra_r_weekend2_3)

        # print("vcm1_cost = ", vcm1)
        # print("vcm2_cost = ", vcm2)
        # variable_cost_m1 = vcm1 * (M1P1 + M1P2 + M1P3)
        # variable_cost_m2 = vcm2 * (M2P1 + M2P2 + M2P3)
        # print("variable_cost_m1 = ", variable_cost_m1)
        # changeover_cost_of_m1 = co11 * (
        #        CO11 + CO12 + CO13)  # period นึงจะเกิด CO11, CO12, CO13 ได้แค่ 1 กรณี จึงจับรวมได้เลย
        # changeover_cost_of_m2 = co21 * (CO21 + CO22 + CO23)
        # print("CO11 =",CO11)
        # print("CO21 =",CO21)
        # switch_on_cost = sw1*(SW1 + SW2)
        # print("SW1 =", SW1)
        # print("SW2 =", SW2)

        fix_production_cost = fc_m1 * FC_M1 + fc_m2 * FC_M2
        # print("FC_M1 =", FC_M1)
        # print("FC_M2 =", FC_M2)

        # force buffer on_hand > safety stock
        extra_penalty1 = 0
        extra_penalty2 = 0
        extra_penalty3 = 0
        sum_extra_penalty = 0
        sum_extra_reward = 0
        extra_reward1 = 0
        extra_reward2 = 0
        extra_reward3 = 0

        s_penal = 8
        if overage1 < 500:
            extra_penalty1 = s_penal * 1000000 * 25
        if overage2 < 500:
            extra_penalty2 = s_penal * 1000000 * 25
        if overage3 < 500:
            extra_penalty3 = s_penal * 1000000 * 25

        penal = 15
        if overage1 <= 0:
            extra_penalty1 = penal * 1000000 * 40  # ถ้า < 4500 แต่ ไม่ < 0 ตรงนี้จะข้ามไป ไม่โดน penalty แต่ < 0 ด้วย 5 ล้านจะถูกแทนด้วยค่า 9 ล้าน
        if overage2 <= 0:
            extra_penalty2 = penal * 1000000 * 40
        if overage3 <= 0:
            extra_penalty3 = penal * 1000000 * 40

        if overage1 > 9000:
            extra_penalty1 = penal * 1000000 * 20
        if overage2 > 8500:
            extra_penalty2 = penal * 1000000 * 20
        if overage3 > 8000:
            extra_penalty3 = penal * 1000000 * 20

        if overage1 > 12000:  # 10000
            extra_penalty1 = penal * 1000000 * 50
        if overage2 > 12000:
            extra_penalty2 = penal * 1000000 * 50
        if overage3 > 12000:
            extra_penalty3 = penal * 1000000 * 50

        # ending penalty
        if self.step_count == 29:
            if overage1 <= 3500:
                extra_penalty1 = penal * 1000000 * 40  # ถ้า < 4500 แต่ ไม่ < 0 ตรงนี้จะข้ามไป ไม่โดน penalty แต่ < 0 ด้วย 5 ล้านจะถูกแทนด้วยค่า 9 ล้าน
            if overage2 <= 2750:
                extra_penalty2 = penal * 1000000 * 40
            if overage3 <= 1750:
                extra_penalty3 = penal * 1000000 * 40

        if overage1 in range(300, 9000) and overage2 in range(300, 9000) and overage3 in range(300,
                                                                                               9000):  # min value ไม่ควร = 0 เพราะจะหมายถึง ของหมด ก็ยังได้รางวัล
            extra_reward1 = 1000 * 1000000  # 300
        # if overage1_2 in range(300, 9000) and overage2_2 in range(300, 9000) and overage3_2 in range(300, 9000):
        #    extra_reward2 = 400 * 1000000  # 300

        sum_extra_penalty = extra_penalty1 + extra_penalty2 + extra_penalty3

        # sum_extra_reward = extra_reward1 + extra_reward2 + extra_reward3
        # print("extra penalty =", extra_penalty1, extra_penalty2, extra_penalty3)
        # print("Buffer extra penalty =", sum_extra_penalty)
        # sum_extra_penalty2 = sum_extra_penalty
        sum_ex_pen = sum_extra_penalty / 1000000
        self.sum_ex_penalty_array.append(sum_ex_pen)
        # print("penalty array =", self.sum_ex_penalty_array)

        # Genarate Damand random #กำหนดก่อนว่าให้ demand array2 จะมีค่า 90 ตำแหน่ง
        demand_array2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                         27, 28, 29, 30,
                         31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                         55, 56, 57, 58, 59, 60,
                         61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
                         85, 86, 87, 88, 89, 90]
        period_zero = [1, 2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21, 25, 26, 27,
                       31, 32, 33, 37, 38, 39, 43, 44, 45, 49, 50, 51, 55, 56, 57,
                       61, 62, 63, 67, 68, 69, 73, 74, 75, 79, 80, 81, 85, 86, 87]
        # period_value = [4, 5, 6, 10, 11, 12, 16, 17, 18, 22, 23, 24, 28,29,30]
        period_P1 = [4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70, 76, 82, 88]
        period_P2 = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89]
        period_P3 = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90]
        for t in period_zero:
            # if t == stepcount:
            # print("t =", t)
            demand_array2[t - 1] = 0

        d1 = 0
        d2 = 0
        d3 = 0
        d4 = 0
        d5 = 0
        d6 = 0
        d7 = 0
        d8 = 0
        d9 = 0
        # เป็น step_count+1  เพราะ ค่า demand เริ่มต้นกำหนดใน def reset ให้ = 0 ไว้แล้ว จึงข้าม 0 0 0 สามตัวแรกใน demand array2 ได้
        set_stepcount1 = [0, 4, 8, 12, 16, 20]
        set_stepcount2 = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
        set_stepcount3 = [24]
        # set_stepcount2 = [28]
        rng_indexset = np.random.randint(1,
                                         8)  # เลขไว้สุ่มชุด index1-7 #1,2=randomV, 3 =season1, 4=season2, 5=season3, 6=extreme1, 7=extreme2
        idrv_set = []
        index_season1 = [0.713, 0.744, 0.83, 0.96, 1.09, 1.179, 1.253, 1.311, 1.261, 1.174, 1.1, 1, 0.88, 0.78, 0.72]
        index_season2 = [1.179, 1.253, 1.311, 1.261, 1.174, 1.092, 1.015, 0.913, 0.805, 0.741, 0.713, 0.744, 0.83,
                         0.988, 1.116]
        index_season3 = [1.1, 1, 0.88, 0.78, 0.72, 0.713, 0.744, 0.83, 0.96, 1.09, 1.179, 1.253, 1.311, 1.261, 1.174]
        index_extreme1 = [0.741, 0.913, 1.179, 1.56, 1.65, 0.913, 0.805, 0.713, 0.69, 0.55, 0.35, 0.55, 0.744, 0.83,
                          0.988]
        index_extreme2 = [0.69, 0.55, 0.35, 0.55, 0.744, 0.83, 0.988, 1.215, 1.355, 1.56, 1.65, 1.121, 0.805, 0.713,
                          0.69]
        i = 0
        demand_r1 = 0
        demand_r2 = 0
        demand_r3 = 0
        #index2 = []
        # if self.step_count < 24:  # 25
        #     # print("step count =", self.step_count)
        #     if self.step_count in set_stepcount1:  # ถ้าปล่อยให้ถึง 29 ค่าindex y จะหลุดนอกสมาชิก array
        #         y = self.step_count + 1
        #
        #         d1 = demand_array2[y * 3]
        #         d2 = demand_array2[y * 3 + 1]
        #         d3 = demand_array2[y * 3 + 2]
        #         d4 = demand_array2[y * 3 + 3]
        #         d5 = demand_array2[y * 3 + 4]
        #         d6 = demand_array2[y * 3 + 5]
        #         d7 = demand_array2[y * 3 + 6]
        #         d8 = demand_array2[y * 3 + 7]
        #         d9 = demand_array2[y * 3 + 8]

                # print("self.demand_all= ", self.demand_all)
        # if self.step_count == 24:
        #     if self.step_count in set_stepcount3:  # ถ้าปล่อยให้ถึง 29 ค่าindex y จะหลุดนอกสมาชิก array
        #         y = self.step_count + 1
        #
        #         d1 = demand_array2[y * 3]
        #         d2 = demand_array2[y * 3 + 1]
        #         d3 = demand_array2[y * 3 + 2]
        #         d4 = demand_array2[y * 3 + 3]
        #         d5 = demand_array2[y * 3 + 4]
        #         d6 = demand_array2[y * 3 + 5]
        #         d7 = demand_array2[y * 3 + 6]
        #         d8 = demand_array2[y * 3 + 7]
        #         d9 = demand_array2[y * 3 + 8]
                # print("demand_all2= ", self.demand_all)

        # assign demand of next period from array of demand data

        # demand9 = 0
        #         ppp = (self.step_count + 1) * 3
        #         ppp8 = (self.step_count + 1) * 3 + 8
        #         print("================ppp============", ppp)
        #         print("================ppp + 8============", ppp8)
        #         demand1 = self.demand_all[(self.step_count + 1) * 3]
        #         demand2 = self.demand_all[(self.step_count + 1) * 3 + 1]
        #         demand3 = self.demand_all[(self.step_count + 1) * 3 + 2]

        y = self.step_count + 1
        # y = self.step_count

        if self.step_count < 24:
            # print("len(demand_all)  =", len(self.demand_all))
            # print(f'y * 3 = {y * 3}')
            # print(f'y * 3 + 8 = {y * 3 + 8}')
            if y * 3 + 8 >= len(self.demand_all):
                input('>>> Error here!')
            demand1 = self.demand_all[y * 3]
            demand2 = self.demand_all[y * 3 + 1]
            demand3 = self.demand_all[y * 3 + 2]
            demand4 = self.demand_all[y * 3 + 3]
            demand5 = self.demand_all[y * 3 + 4]
            demand6 = self.demand_all[y * 3 + 5]
            demand7 = self.demand_all[y * 3 + 6]
            demand8 = self.demand_all[y * 3 + 7]
            demand9 = self.demand_all[y * 3 + 8]
            demand10 = self.demand_all[y * 3 + 9]
            demand11 = self.demand_all[y * 3 + 10]
            demand12 = self.demand_all[y * 3 + 11]

        if self.step_count < 27:  # =27
            # y = self.step_count + 1
            # print("len(demand_all) at step27 =",len(self.demand_all))
            demand1 = self.demand_all[y * 3]
            demand2 = self.demand_all[y * 3 + 1]
            demand3 = self.demand_all[y * 3 + 2]
            demand4 = self.demand_all[y * 3 + 3]
            demand5 = self.demand_all[y * 3 + 4]
            demand6 = self.demand_all[y * 3 + 5]
            demand7 = self.demand_all[y * 3 + 6]
            demand8 = self.demand_all[y * 3 + 7]
            demand9 = self.demand_all[y * 3 + 8]
            demand10 = self.demand_all[y * 3 + 9]
            demand11 = self.demand_all[y * 3 + 10]
            demand12 = self.demand_all[y * 3 + 11]

        if self.step_count == 27:  # =27
            # y = self.step_count + 1
            # print("len(demand_all) at step27 =",len(self.demand_all))
            demand1 = self.demand_all[y * 3]
            demand2 = self.demand_all[y * 3 + 1]
            demand3 = self.demand_all[y * 3 + 2]
            demand4 = self.demand_all[y * 3 + 3]
            demand5 = self.demand_all[y * 3 + 4]
            demand6 = self.demand_all[y * 3 + 5]
            demand7 = self.demand_all[y * 3 + 6]
            demand8 = self.demand_all[y * 3 + 7]
            demand9 = self.demand_all[y * 3 + 8]
            demand10 = 0
            demand11 = 0
            demand12 = 0

        if self.step_count == 28:
            # y = self.step_count + 1
            demand1 = self.demand_all[y * 3]
            demand2 = self.demand_all[y * 3 + 1]
            demand3 = self.demand_all[y * 3 + 2]
            demand4 = self.demand_all[y * 3 + 3]
            demand5 = self.demand_all[y * 3 + 4]
            demand6 = self.demand_all[y * 3 + 5]
            demand7 = 0
            demand8 = 0
            demand9 = 0
            demand10 = 0
            demand11 = 0
            demand12 = 0
        if self.step_count == 29:
            # y = self.step_count + 1
            demand1 = self.demand_all[y * 3]
            demand2 = self.demand_all[y * 3 + 1]
            demand3 = self.demand_all[y * 3 + 2]
            demand4 = 0
            demand5 = 0
            demand6 = 0
            demand7 = 0
            demand8 = 0
            demand9 = 0
            demand10 = 0
            demand11 = 0
            demand12 = 0


        # print("self.step_count =", self.step_count)
        # print("Action =", action)
        # print("d1-d3, demand of r1 r2 r3 in next periods =",demand1,demand2,demand3)
        # print("d4-d9 =", demand4, demand5, demand6, demand7, demand8, demand9)
        # print("d10-d12 =", demand10, demand11, demand12)
        # print("self.demand_all", self.demand_all)

        # demand1 from random
        # print("demand1",demand1)

        # Penalty for future shortage
        extra_penalty1_2 = 0
        extra_penalty2_2 = 0
        extra_penalty3_2 = 0
        extra_penalty1_3 = 0
        extra_penalty2_3 = 0
        extra_penalty3_3 = 0
        extra_penalty1_4 = 0
        extra_penalty2_4 = 0
        extra_penalty3_4 = 0
        sum_extra_penalty_2 = 0
        sum_extra_penalty_3 = 0
        #         overage1_2 = overage1 - d4 + R1
        #         overage2_2 = overage2 - d5 + R2
        #         overage3_2 = overage3 - d6 + R3
        #         overage1_3 = overage1 - d7 + R1
        #         overage2_3 = overage2 - d8 + R2
        #         overage3_3 = overage3 - d9 + R3
        #Inv in next period
        overage1_2 = overage1 - demand4
        overage2_2 = overage2 - demand5
        overage3_2 = overage3 - demand6
        #Inv in next 2 period
        overage1_3 = overage1_2 - demand7
        overage2_3 = overage2_2 - demand8
        overage3_3 = overage3_2 - demand9
        # Inv in next 3 period
        overage1_4 = overage1_3 - demand10
        overage2_4 = overage2_3 - demand11
        overage3_4 = overage3_3 - demand12
        # print("overage1_2 =", overage1_2)
        #print("overage1 = ",overage1)
        #print("overage1_2,  overage1_3, overage1_4  = ",overage1_2,  overage1_3, overage1_4 )

        #         if overage1_2 < 1500:
        #             extra_penalty1_2 = s_penal*2000000
        #         if overage2_2 < 1000:
        #             extra_penalty2_2 = s_penal*2000000
        #         if overage3_2 < 1000:
        #             extra_penalty3_2 = s_penal*2000000
        #         if overage1_3 < 1000:
        #             extra_penalty1_3 = s_penal*1000000
        #         if overage2_3 < 1000:
        #             extra_penalty2_3 = s_penal*1000000
        #         if overage3_3 < 1000:
        #             extra_penalty3_3 = s_penal*1000000

        if overage1_2 <= 0:  # ลองแก้จาก 0 เป็นติด - ดู เพราะเหมือนมันจะ overstock มากไป
            extra_penalty1_2 = penal * 1000000 * 15
        if overage2_2 <= 0:
            extra_penalty2_2 = penal * 1000000 * 15
        if overage3_2 <= 0:
            extra_penalty3_2 = penal * 1000000 * 15
        if overage1_3 <= 0:
            extra_penalty1_3 = penal * 1000000 * 5
        if overage2_3 <= 0:
            extra_penalty2_3 = penal * 1000000 * 5
        if overage3_3 <= 0:
            extra_penalty3_3 = penal * 1000000 * 5
        if overage1_4 <= -1000:
            extra_penalty1_4 = penal * 1000000 * 5
        if overage2_4 <= -1000:
            extra_penalty2_4 = penal * 1000000 * 5
        if overage3_4 <= -1000:
            extra_penalty3_4 = penal * 1000000 * 5
        #print("extra penalty1_2,3,4 =", extra_penalty1_2, extra_penalty1_3, extra_penalty1_4)


        if overage1_2 > 10000:
            extra_penalty1_2 = penal * 1000000 * 30  # ยื่งตุนนาน ยิ่งโดนปรับเยอะ
        if overage2_2 > 10000:
            extra_penalty2_2 = penal * 1000000 * 30
        if overage3_2 > 10000:
            extra_penalty3_2 = penal * 1000000 * 30
        if overage1_3 > 10000:
            extra_penalty1_3 = penal * 1000000 * 50
        if overage2_3 > 10000:
            extra_penalty2_3 = penal * 1000000 * 50
        if overage3_3 > 10000:
            extra_penalty3_3 = penal * 1000000 * 50

        # ending penalty
        if self.step_count == 28:  # periodสุดท้ายคือ29อันนี้มองล่วงหน้าไป1periodก่อนจบ
            if overage1_2 <= 3500:
                extra_penalty1_2 = penal * 1000000 * 40  # ถ้า < 4500 แต่ ไม่ < 0 ตรงนี้จะข้ามไป ไม่โดน penalty แต่ < 0 ด้วย 5 ล้านจะถูกแทนด้วยค่า 9 ล้าน
            if overage2_2 <= 2750:
                extra_penalty2_2 = penal * 1000000 * 40
            if overage3_2 <= 1750:
                extra_penalty3_2 = penal * 1000000 * 40
        if self.step_count == 27:
            if overage1_3 <= 3500:
                extra_penalty1_3 = penal * 1000000 * 40  # ถ้า < 4500 แต่ ไม่ < 0 ตรงนี้จะข้ามไป ไม่โดน penalty แต่ < 0 ด้วย 5 ล้านจะถูกแทนด้วยค่า 9 ล้าน
            if overage2_3 <= 2750:
                extra_penalty2_3 = penal * 1000000 * 40
            if overage3_3 <= 1750:
                extra_penalty3_3 = penal * 1000000 * 40

        if overage1_2 in range(100, 6000) and overage2_2 in range(100, 6000) and overage3_2 in range(100, 6000):
            extra_reward2 = 300 * 1000000
        if overage1_3 in range(100, 6000) and overage2_3 in range(100, 6000) and overage3_3 in range(100, 6000):
            extra_reward3 = 300 * 1000000
        # if overage2_2 in range(1500,8000):
        #    extra_reward2 = 200*1000000
        # if overage3_2 in range(1500,7500):
        #    extra_reward3 = 200*1000000

        # sum_extra_reward =  extra_reward1 + extra_reward2
        sum_extra_reward = extra_reward2
        sum_extra_penalty_2 = extra_penalty1_2 + extra_penalty2_2 + extra_penalty3_2
        sum_extra_penalty_3 = extra_penalty1_3 + extra_penalty2_3 + extra_penalty3_3
        sum_extra_penalty_3_2 = extra_penalty1_4 + extra_penalty2_4 + extra_penalty3_4
        # print("extra penalty3 =", extra_penalty1_3, extra_penalty2_3, extra_penalty3_3)
        # print("Buffer extra penalty3 =", sum_extra_penalty_3)
        sum_extra_penalty4 = (sum_extra_penalty_2)
        sum_extra_penalty5 = (sum_extra_penalty_3)
        self.sum_ex_penalty_array_2.append(sum_extra_penalty4)
        self.sum_ex_penalty_array_3.append(sum_extra_penalty5)
        # print("penalty array2 =", self.sum_ex_penalty_array_3)

        # reward that use to train agent
        #         reward_ = (((-purchase_cost + holding + penalty_lost_sale
        #                      - (changeover_cost_of_m1 + changeover_cost_of_m2) * 100
        #                      - switch_on_cost + fix_production_cost + (variable_cost_m1 + variable_cost_m2)
        #                      - sum_extra_penalty + sum_extra_penalty_2 + sum_extra_penalty_3
        #                      - (
        #                                  extra_p_on1_1 + extra_p_on1_2 + extra_p_on1_3 + extra_p_on2_1 + extra_p_on2_2 + extra_p_on2_3)) / 1000000))

        # for Tanh activation fn
        # ใส่ _ = ยังไม่เอามาคิด ถ้าจะคิดก็เอา _ ออก    #450
        #         reward_tan = ((sum_extra_reward/1000000 - (purchase_cost + holding + penalty_lost_sale
        #                             + (changeover_cost_of_m1 + changeover_cost_of_m2) * 10
        #                             + switch_on_cost + fix_production_cost + (variable_cost_m1 + variable_cost_m2)
        #                             + sum_extra_penalty + sum_extra_penalty_2 + sum_extra_penalty_3
        #                             + (
        #                                        extra_p_on1_1 + extra_p_on1_2 + extra_p_on1_3 + extra_p_on2_1 + extra_p_on2_2 + extra_p_on2_3)) / 1000000)) / 100
        # print("reward",reward)
        #         reward_tanh = (0 - ((purchase_cost + holding + penalty_lost_sale
        #                             + (changeover_cost_of_m1 + changeover_cost_of_m2) * 10
        #                             + switch_on_cost + fix_production_cost + (variable_cost_m1 + variable_cost_m2)
        #                             + sum_extra_penalty + sum_extra_penalty_2 + sum_extra_penalty_3
        #                             + (
        #                                        extra_p_on1_1 + extra_p_on1_2 + extra_p_on1_3 + extra_p_on2_1 + extra_p_on2_2 + extra_p_on2_3)) / 1000000)) / 500

        # for Gelu activation fn
        # sum_extra_reward/1000000
        # ใส่ _ = ยังไม่เอามาคิด ถ้าจะคิดก็เอา _ ออก    #450
        reward = (2100 + (
            sales_revenue) / 1000000 + extra_reward1 / 1000000 + extra_reward2 / 1000000 + extra_reward3 / 1000000 - (
                          (purchase_cost + holding + penalty_lost_sale
                           + (self.changeover_cost_of_m1 + self.changeover_cost_of_m2) * 10
                           + self.switch_on_cost + fix_production_cost + (
                                   self.variable_cost_m1 + self.variable_cost_m2)
                           + sum_extra_penalty + sum_extra_penalty_2 + sum_extra_penalty_3 + sum_extra_penalty_3_2
                           - (
                                   extra_r_weekend1_1 + extra_r_weekend1_2 + extra_r_weekend1_3 + extra_r_weekend2_1 + extra_r_weekend2_2 + extra_r_weekend2_3)
                           + (
                                   extra_p_on1_1 + extra_p_on1_2 + extra_p_on1_3 + extra_p_on2_1 + extra_p_on2_2 + extra_p_on2_3)) / 1000000)) / 2100  # 650

        #         pure_reward = (purchase_cost + holding + penalty_lost_sale
        #                             + (self.changeover_cost_of_m1 + self.changeover_cost_of_m2) * 10
        #                             + self.switch_on_cost + fix_production_cost + (self.variable_cost_m1 + self.variable_cost_m2)
        #                             + sum_extra_penalty + sum_extra_penalty_2 + sum_extra_penalty_3
        #                             + (
        #                                        extra_p_on1_1 + extra_p_on1_2 + extra_p_on1_3 + extra_p_on2_1 + extra_p_on2_2 + extra_p_on2_3))

        # normalize reward อีกที
        # reward = reward/25

        # ใส่ _ = ยังไม่เอามาคิด ถ้าจะคิดก็เอา _ ออกก
        #         reward___ = (415 + (sales_revenue) / 1000000 - (purchase_cost + holding * 3 + penalty_lost_sale
        #                                                      + (changeover_cost_of_m1 + changeover_cost_of_m2) * 100
        #                                                      + switch_on_cost + fix_production_cost + (
        #                                                                  variable_cost_m1 + variable_cost_m2)
        #                                                      + sum_extra_penalty + sum_extra_penalty_2 + sum_extra_penalty_3
        #                                                      + (
        #                                                                  extra_p_on1_1 + extra_p_on1_2 + extra_p_on1_3 + extra_p_on2_1 + extra_p_on2_2 + extra_p_on2_3)) / 1000000) / 415

        # real reward that equal to real revenue
        #         reward_____ = (sales_revenue
        #                        - purchase_cost
        #                        - holding
        #                        - penalty_lost_sale
        #                        - (changeover_cost_of_m1 + changeover_cost_of_m2)
        #                        - switch_on_cost
        #                        - fix_production_cost
        #                        - (variable_cost_m1 + variable_cost_m2))

        # real reward that equal to real revenue
        real_reward = (sales_revenue
                       - purchase_cost
                       - holding
                       - penalty_lost_sale
                       - (self.changeover_cost_of_m1 + self.changeover_cost_of_m2)
                       - self.switch_on_cost
                       - fix_production_cost
                       - (self.variable_cost_m1 + self.variable_cost_m2)) / 34.84  # แปลงจากบาท to dollar

        #         print("############################################################# ")
        #         print("stepcount ", self.step_count )
        #         print("self.changeover_cost_of_m1 ", self.changeover_cost_of_m1 )
        #         print("self.changeover_cost_of_m2 ", self.changeover_cost_of_m2 )
        #         print("self.switch_on_cost ", self.switch_on_cost )
        #         print("self.variable_cost_m1 ", self.variable_cost_m1 )
        #         print("self.variable_cost_m2 ", self.variable_cost_m2 )
        #         print("fix_production_cost ", fix_production_cost )
        #         print("holding ", holding )
        #         print("purchase_cost ", purchase_cost )
        #         print("sum_extra_penalty ", sum_extra_penalty )
        #         print("sum_extra_penalty2 ", sum_extra_penalty_2 )
        #         print("sum_extra_penalty3 ", sum_extra_penalty_3 )
        #         print("pure_reward ", pure_reward )

        # print("Step", self.step_count)
        self.step_count += 1
        done = bool(self.step_count >= 30)  # planning time frame period = 15
        self.overall_time_trained += 1

        # if done == True:
        #   self.rn = np.random.random_integers(1, high=100000, size=None)

        # Normalize the reward
        raw_reward = reward
        reward = reward / 25

        # reward = (315 + reward)    #ไว้แก้ไม่ให้ค่า reward เป็นลบ  คำนวณได้ว่าค่า max penalty per period จะประมาณ 121000000 หรือ 121 M
        # reward = (500 + reward)     #เหมือนใช้ 315 แล้วไม่ค่อยโอเค อาจจะยังมีติดลบอยู่ เลยเพิ่มเห็น 500
        # reward = (200 + reward)
        real_reward = real_reward / 1000000

        self.sum_reward += reward
        self.sum_real_reward += real_reward
        last_sum_reward = 0
        last_sum_real_reward = 0

        last_sum_real_reward = 0
        last_sum_reward = 0
        if self.step_count == 29:
            last_sum_reward = self.sum_reward
            last_sum_real_reward = self.sum_real_reward

        self.M1P1_set.append(M1P1)
        self.M1P2_set.append(M1P2)
        self.M1P3_set.append(M1P3)
        self.M2P1_set.append(M2P1)
        self.M2P2_set.append(M2P2)
        self.M2P3_set.append(M2P3)

        # pass CO data value to use in next state by save it in 'info' part
        # the real reward is at info[13]
        info = [self.CO_var_set1, self.CO_var_set2, self.sum_reward, last_sum_reward, last_sum_real_reward,
                self.sum_ex_penalty_array,
                d4, d5, d6, d7,
                d8, d9, self.sum_ex_penalty_array_3, real_reward,
                self.overall_time_trained,
                self.demand_all, extra_p_on_set,  # last one is info[16]
                self.M1P1_set, self.M1P2_set, self.M1P3_set,  # info[17-19]
                self.M2P1_set, self.M2P2_set, self.M2P3_set,  # info[20-22]
                raw_reward, self.demand_real,  # info[23-24]
                self.aaa] #info25

        #         print("value ก่อน normalize")
        #         print("demand1 =", demand1)
        #         print("onhand at end of this period =", overage1, overage2, overage3)
        #         print("overage1 =", overage1)
        #         print("overage2 =", overage2)
        #         print("overage3 =", overage3)

        # Normalize value to 0-1 range  =before sendout to neural network
        demand1 = (demand1 - mind1) / (maxd1 - mind1)
        demand2 = (demand2 - mind2) / (maxd2 - mind2)
        demand3 = (demand3 - mind3) / (maxd3 - mind3)

        demand4 = (demand4 - mind1) / (maxd1 - mind1)
        demand5 = (demand5 - mind2) / (maxd2 - mind2)
        demand6 = (demand6 - mind3) / (maxd3 - mind3)
        demand7 = (demand7 - mind1) / (maxd1 - mind1)
        demand8 = (demand8 - mind2) / (maxd2 - mind2)
        demand9 = (demand9 - mind3) / (maxd3 - mind3)
        demand10 = (demand10 - mind1) / (maxd1 - mind1)
        demand11 = (demand11 - mind2) / (maxd2 - mind2)
        demand12 = (demand12 - mind3) / (maxd3 - mind3)


        overage1 = (overage1 - minr1) / (maxr1 - minr1)
        overage2 = (overage2 - minr2) / (maxr2 - minr2)
        overage3 = (overage3 - minr3) / (maxr3 - minr3)
        overage1_2 = (overage1_2 - minr1) / (maxr1 - minr1)
        overage2_2 = (overage2_2 - minr2) / (maxr2 - minr2)
        overage3_2 = (overage3_2 - minr3) / (maxr3 - minr3)
        overage1_3 = (overage1_3 - minr1) / (maxr1 - minr1)
        overage2_3 = (overage2_3 - minr2) / (maxr2 - minr2)
        overage3_3 = (overage3_3 - minr3) / (maxr3 - minr3)
        overage1_4 = (overage1_4 - minr1) / (maxr1 - minr1)
        overage2_4 = (overage2_4 - minr2) / (maxr2 - minr2)
        overage3_4 = (overage3_4 - minr3) / (maxr3 - minr3)

        dr1_4 = (dr1_4 - mind1) / (maxd1 - mind1)
        dr2_4 = (dr2_4 - mind2) / (maxd2 - mind2)
        dr3_4 = (dr3_4 - mind3) / (maxd3 - mind3)
        dr1_8 = (dr1_8 - mind1) / (maxd1 - mind1)
        dr2_8 = (dr2_8 - mind2) / (maxd2 - mind2)
        dr3_8 = (dr3_8 - mind3) / (maxd3 - mind3)
        dr1_12 = (dr1_12 - mind1) / (maxd1 - mind1)
        dr2_12 = (dr2_12 - mind2) / (maxd2 - mind2)
        dr3_12 = (dr3_12 - mind3) / (maxd3 - mind3)
        dr1_16 = (dr1_16 - mind1) / (maxd1 - mind1)
        dr2_16 = (dr2_16 - mind2) / (maxd2 - mind2)
        dr3_16 = (dr3_16 - mind3) / (maxd3 - mind3)
        dr1_20 = (dr1_20 - mind1) / (maxd1 - mind1)
        dr2_20 = (dr2_20 - mind2) / (maxd2 - mind2)
        dr3_20 = (dr3_20 - mind3) / (maxd3 - mind3)
        dr1_24 = (dr1_24 - mind1) / (maxd1 - mind1)
        dr2_24 = (dr2_24 - mind2) / (maxd2 - mind2)
        dr3_24 = (dr3_24 - mind3) / (maxd3 - mind3)
        dr1_28 = (dr1_28 - mind1) / (maxd1 - mind1)
        dr2_28 = (dr2_28 - mind1) / (maxd1 - mind1)
        dr3_28 = (dr3_28 - mind3) / (maxd3 - mind3)

        # inv data
        self.state[0] = 0
        self.state[1] = 0
        self.state[2] = 0
        self.state[26] = 0
        self.state[27] = 0
        # update demand data this period
        self.state[3] = demand1
        self.state[4] = demand2
        self.state[5] = demand3

        self.state[0] += overage1  # Inventory that has already subtracted demand
        self.state[1] += overage2
        self.state[2] += overage3
        # collect Production data to use in next state
        self.state[6] = N1P
        self.state[7] = N1P1
        self.state[8] = N1P2
        self.state[9] = N1P3
        self.state[10] = N2P
        self.state[11] = N2P1
        self.state[12] = N2P2
        self.state[13] = N2P3
        self.state[14] = overage1_2
        self.state[15] = overage2_2
        self.state[16] = overage3_2
        self.state[17] = overage1_3
        self.state[18] = overage2_3
        self.state[19] = overage3_3
        self.state[20] = overage1_4
        self.state[21] = overage2_4
        self.state[22] = overage3_4
        self.state[23] = demand4
        self.state[24] = demand5
        self.state[25] = demand6
        self.state[26] = demand7
        self.state[27] = demand8
        self.state[28] = demand9
        self.state[29] = demand10
        self.state[30] = demand11
        self.state[31] = demand12
        self.state[32] = extra_p_on

        self.state[33] = self.aaa
        self.state[34] = dr1_4
        self.state[35] = dr2_4
        self.state[36] = dr3_4
        self.state[37] = dr1_8
        self.state[38] = dr2_8
        self.state[39] = dr3_8
        self.state[40] = dr1_12
        self.state[41] = dr2_12
        self.state[42] = dr3_12
        self.state[43] = dr1_16
        self.state[44] = dr2_16
        self.state[45] = dr3_16
        self.state[46] = dr1_20
        self.state[47] = dr2_20
        self.state[48] = dr3_20
        self.state[49] = dr1_24
        self.state[50] = dr2_24
        self.state[51] = dr3_24
        self.state[52] = dr1_28
        self.state[53] = dr2_28
        self.state[54] = dr3_28    # so all number state variables are 55 variables

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


        # print("state[7]_2 =", self.state[7])
        # print("state_2 =", self.state)
        # print("End_On_hand_P1 this period :", self.state[0])
        # print("End_On_hand_P2 this period :", self.state[1])
        # print("End_On_hand_P3 this period :", self.state[2])
        # print("period buy :", self.state[1:lt])
        # print("demand of next period =", demand1, demand2, demand3)
        # print("Reward : ", reward)
        # print("Sum_reward : ", self.sum_reward)
        # print("---------------------------------------------")
        # print("obs : ", self.state)
        # return self._normalize_obs(), reward, done, info
        # demand_array2 = []
        # เนื่องจาก reward ตอนที่ A3C คิดน่าจะ เป็น sum_reward ในแต่ละ episode อยู่แล้ว ดังนั้น reward ที่ return ควรเป็น reward
        return np.array(self.state, dtype=np.float32), reward, done, info

env = InvEnv4_m()
state = env.reset()

done = False
N = 2
runs = int(N)   #รัน 30 peroids จำนวน N รอบ
for i in range(runs):
    maxrun = 0
    done = False
    rand = randint(0,999)
    #print("rand", rand)
    env.reset()
    #env.seed(rand)
    period = 0
    demand_all = []
    #print("============Round====", i)
    while(done==False):
        action = env.action_space.sample()
        #print("period", period)
        #print("state", state)
        #print("action = ", action)
        state, reward, done, info = env.step(action)
        period += 1
        #print("################period##############",period)
        #print("state",state)
        #print("action =",action)
        #print("action =",action)

        demand1 = state[3]
        demand2 = state[4]
        demand3 = state[5]
        #print("d1-d3 input in next state =", state[3], state[4], state[5])
        # print("d4-d9 =", state[21], state[22], state[23], state[24], state[25], state[26])
        # print("extra_p_on ###### =", state[26])
        #print("aaa3 =",state[27], info[25])
        #demand_all.append(demand1)
        #demand_all.append(demand2)
        #demand_all.append(demand3)

        #sum_rw_ += reward
        #print("sum_rw",sum_rw_)
        #maxrun += 1
        reward2 = reward
        # print("reward", reward)
        # print("demand =", demand_all)
        # print("next_state",state)
        #print("====================================================================")
        # print(info[24])
