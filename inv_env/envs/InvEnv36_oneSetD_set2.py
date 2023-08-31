# อันนี้คือ env3 (deamnd set3)
# this code is same as invEnv36_oneSetDemandGithub.py
# แก้ env ให้เป็น 24 dimension
# ตัดพวกที่ print และ comment ออกไปใ เพื่อเอาไปใส่ใน gym
# คือ ต่อจากไฟล์ 25_16 act แต่ลองปรับ env ตาม moutain car เพื่อให้ appecnd ใน colab ถูก
import math
from typing import Optional

import pygame
from pygame import gfxdraw

import gym
from gym import spaces
from gym.utils import seeding

from gym import Env
from gym import spaces
from gym.spaces import Box, Discrete
import random
import numpy as np

from gym import Env
from scipy.stats import poisson
from random import randint, choice


# file 18  is demand set1  but file22 will demand set2 ต่างกันแค่นี้


class InvEnv7(gym.Env):
    def __init__(self):
        self.step_count = 0
        self.overall_time_trained = 0
        # initial inventory
        self.on_hand1 = (5659-0)/(12000-0)  #5659
        self.on_hand2 = (3051-0)/(12000-0)
        self.on_hand3 = (2084-0)/(12000-0)
        self.action_space = spaces.Discrete(16)
        # self.observation_space = spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32)
        self.statelow = np.array([
            0, 0, 0,  # initial inventory
            0, 0, 0,  # initial demand
            0, 0, 0, 0,  # initial machine status (0 = idle)
            0, 0, 0, 0,
            0, 0, 0,  # future inventory i4 i5 i6 = overage1_2, overage2_2, overage3_2
            0, 0, 0,  # future inventory i7 i8 i9 = overage1_3, overage2_3, overage3_3
            0, 0, 0,  # d4, d5, d6
            0, 0, 0,  # d7, d8, d9
            0
        ])
        self.statehigh = np.array([
            np.inf, np.inf, np.inf,  # initial inventory
            np.inf, np.inf, np.inf,  # initial demand
            1, 1, 1, 1,  # initial machine status (0 = idle)
            1, 1, 1, 1,
            np.inf, np.inf, np.inf,  # future inventory i4 i5 i6 = overage1_2, overage2_2, overage3_2
            np.inf, np.inf, np.inf,  # future inventory i7 i8 i9 = overage1_3, overage2_3, overage3_3
            np.inf, np.inf, np.inf,  # future demand d4, d5, d6
            np.inf, np.inf, np.inf,  # initial demand d7, d8, d9
            1
        ])
        self.observation_space = Box(self.statelow, self.statehigh,
                                     dtype=np.float32)

        self.state = [self.on_hand1, self.on_hand2, self.on_hand3,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0,  # future inventory position state[14] - state[19]
                      0, 0, 0, 0, 0, 0, 0]  # future demand position state[20] - state[25]

        self.sum_reward = 0
        self.sum_real_reward = 0

        self.CO_var_set1 = [0, 0, 0, 0, 0, 0, 0, 0]
        self.CO_var_set2 = [0, 0, 0, 0, 0, 0, 0, 0]

        self.sum_ex_penalty_array = []
        self.sum_ex_penalty_array_2 = []
        self.sum_ex_penalty_array_3 = []
        self.demand_all = [10, 10, 10]
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
            (5659-0)/(12000-0), (3051-0)/(12000-0), (2084-0)/(12000-0),  # initial inventory  #2084
            #8659, 3051, 2084,  # initial inventory               #5659
            0, 0, 0,  # initial demand
            0, 0, 0, 0,  # initial machine status (0 = idle)
            0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,  # future inventory
            0, 0, 0, 0, 0, 0,  # future demand
            1
        ])
        self.sum_reward = 0
        self.sum_real_reward = 0
        self.demand_all = [10, 10, 10]
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
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}

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
        k1 = 51000
        k2 = 31000
        k3 = 49600
        # Sell price
        p1 = 5100
        p2 = 3100
        p3 = 4960
        # unit cost
        c1 = 1434.375  # 3060
        c2 = 871.875  # 1860
        c3 = 1395.000  # 2976
        # chage over
        co11 = 4631.490
        co21 = 4631.490
        # switch_on_cost
        sw1 = 401.790
        sw2 = 401.790
        # fix_production_cost
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

        extra_penalty1 = 0
        extra_penalty2 = 0
        extra_penalty3 = 0
        extra_penalty1_2 = 0
        extra_penalty2_2 = 0
        extra_penalty3_2 = 0
        extra_penalty1_3 = 0
        extra_penalty2_3 = 0
        extra_penalty3_3 = 0

        extra_p_on = 0
        
        self.changeover_cost_of_m1 = 0
        self.switch_on_cost = 0
        self.changeover_cost_of_m2 = 0
        self.variable_cost_m1 = 0
        self.variable_cost_m2 = 0

        print("=======================================================================================================")
        print("step :", self.step_count)
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

        on_hand1, on_hand2, on_hand3, demand1, demand2, demand3, N1P, N1P1, \
        N1P2, N1P3, N2P, N2P1, N2P2, N2P3, overage1_2, overage2_2, overage3_2, \
        overage1_3, overage2_3, overage3_3, demand4, demand5, \
        demand6, demand7, demand8, demand9, extra_p_on = self.state

        # print("Step :", self.step_count)
        # print("onhand1 from last period =", on_hand1)
        # print("onhand2 from last period =", on_hand2)
        # print("onhand3 from last period =", on_hand3)
        
        #parameter for normalize
        mind1 = 0  #min demand1
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
        
         
        
#         print("value ก่อนแปลงกลับ") 
#         print("demand1 =", demand1, "=state[3]=",self.state[3])
#         print("on_hand1 =", on_hand1, "=state[0]=",self.state[0])
        # แปลงค่า Normalize value จาก 0-1 range  กลับเป็นค่าปกติ
        demand1 = demand1*(maxd1-mind1)+mind1
        demand2 = demand2*(maxd2-mind2)+mind2
        demand3 = demand3*(maxd3-mind3)+mind3
        
        demand4 = demand4*(maxd1-mind1)+mind1
        demand5 = demand5*(maxd2-mind2)+mind2
        demand6 = demand6*(maxd3-mind3)+mind3
        demand7 = demand7*(maxd1-mind1)+mind1
        demand8 = demand8*(maxd2-mind2)+mind2
        demand9 = demand9*(maxd3-mind3)+mind3
        
        on_hand1 = on_hand1*(maxr1-0)+0
        on_hand2 = on_hand2*(maxr2-0)+0
        on_hand3 = on_hand3*(maxr3-0)+0
        overage1_2 = overage1_2*(maxr1-0)+0
        overage2_2 = overage2_2*(maxr2-0)+0
        overage3_2 = overage3_2*(maxr3-0)+0
        overage1_3 = overage1_3*(maxr1-0)+0
        overage2_3 = overage2_3*(maxr2-0)+0
        overage3_3 = overage3_3*(maxr3-0)+0
        
#         if self.step_count == 0:
#             print("---------------------on_hand1=", on_hand1)
#             on_hand1 = 5659
#             print("on_hand1=", on_hand1)
#             action == 9 #บังคับให้เริ่มต้น ยังไม่ต้องผลิตอะไร เพราะไม่จำเป็น
            
        
#         print("value หลังแปลงกลับ") 
        print("demand1", demand1)
#         print("on_hand1", on_hand1)

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

        print("action =", action)
        print("N1P1= ",N1P1 ," ,M1P1 =", M1P1)
        print("N1P2= ", N1P2, " ,M1P2 =", M1P2)
        print("N1P3= ", N1P3, " ,M1P3 =", M1P3)
        print("N2P1= ",N2P1 ," ,M2P1 =", M2P1)
        print("N2P2= ", N2P2, " ,M2P2 =", M2P2)
        print("N2P3= ", N2P3, " ,M2P3 =", M2P3)

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
        overage1 = on_hand1 - sales1
        overage2 = on_hand2 - sales2
        overage3 = on_hand3 - sales3
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
        # Assign - on-peak and off-peak p cost to each period
#         weekend_stepcount = [3, 4, 5, 6, 17, 18, 19, 20]
#         on_peak_stepcount = [1, 7, 9, 11, 13, 15, 21, 23, 25, 27, 29]
#         off_peak_stepcount = [2, 8, 10, 12, 14, 22, 24, 26, 28, 30]
        weekend_stepcount = [2, 3, 4, 5, 16, 17, 18, 19]
        on_peak_stepcount = [0, 6, 8, 10, 12, 14, 20, 22, 24, 26, 28]
        off_peak_stepcount = [1, 7, 9, 11, 13, 15, 21, 23, 25, 27, 29]
        stp = 0
        #         stp = self.step_count + 1
        stp = self.step_count
        # print("stp =",stp)
        # if stp in on_peak_stepcount:
        #    print("yes")
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
        if stp + 1 in on_peak_stepcount:  # check if next state in onpeak? to pass extra_p_on in the state[27]
            extra_p_on = 1  # = next step will be on-peak
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
            penalty_onpeak = 0
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
            extra_penalty1 = s_penal * 1000000 * 15
        if overage2 < 500:
            extra_penalty2 = s_penal * 1000000 * 15
        if overage3 < 500:
            extra_penalty3 = s_penal * 1000000 * 15

        penal = 15
        if overage1 <= 0:
            extra_penalty1 = penal * 1000000 * 30  # ถ้า < 4500 แต่ ไม่ < 0 ตรงนี้จะข้ามไป ไม่โดน penalty แต่ < 0 ด้วย 5 ล้านจะถูกแทนด้วยค่า 9 ล้าน
        if overage2 <= 0:
            extra_penalty2 = penal * 1000000 * 30
        if overage3 <= 0:
            extra_penalty3 = penal * 1000000 * 30

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

        if overage1 in range(300, 9000) and overage2 in range(300, 9000) and overage3 in range(300, 9000):  #min value ไม่ควร = 0 เพราะจะหมายถึง ของหมด ก็ยังได้รางวัล
            extra_reward1 = 1000 * 1000000  # 300
        if overage1_2 in range(300, 9000) and overage2_2 in range(300, 9000) and overage3_2 in range(300, 9000):
            extra_reward2 = 400 * 1000000  # 300
        

        sum_extra_penalty = extra_penalty1 + extra_penalty2 + extra_penalty3

        # sum_extra_reward = extra_reward1 + extra_reward2 + extra_reward3
        # print("extra penalty =", extra_penalty1, extra_penalty2, extra_penalty3)
        # print("Buffer extra penalty =", sum_extra_penalty)
        # sum_extra_penalty2 = sum_extra_penalty
        sum_ex_pen = sum_extra_penalty / 1000000
        self.sum_ex_penalty_array.append(sum_ex_pen)
        # print("penalty array =", self.sum_ex_penalty_array)

        # assign demand of next period from array of demand data @
        period = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        demand_array2 =  [2514, 2486, 1219, 0, 0, 0, 4362, 2456, 1436, 0, 0, 0, 2559, 2468, 2132, 0, 0, 0,
                          2528, 2910, 1090, 0, 0, 0, 2550, 2342, 1077, 0, 0, 0, 3193, 2194, 1127, 0, 0, 0,
                          4031, 2424, 1964, 0, 0, 0, 2557, 2354, 1094, 0, 0, 0, 3865, 2839, 1806, 0, 0, 0, 
                          3615, 2705, 1729, 0, 0, 0, 3444, 2551, 1993, 0, 0, 0, 4136, 2147, 1579, 0, 0, 0, 
                          2521, 2166, 1182, 0, 0, 0, 3095, 2106, 1837, 0, 0, 0, 2762, 2258, 1797,0,0,0]
        # step_count+1 because the initial demand value is set in 'def reset' to = 0,
        # so the first three 0 0 0 in demand array2 can be skipped.
        # if self.step_count == 0:
        #    d1 = 0
        #    d2 = 0
        #    d3 = 0
        # if self.step_count > 0 & self.step_count <= 28:  # If 29, index y is dropped outside the array members.
        if self.step_count <= 29:  # If 29, index y is dropped outside the array members.
            y = 3 * (self.step_count)
            # print("y =", y)
            d1 = demand_array2[y]
            d2 = demand_array2[y + 1]
            d3 = demand_array2[y + 2]
        else:
            d1 = 0
            d2 = 0
            d3 = 0

        ##เพิ่ม ตัวแปร future demand เพื่อให้กำหนด penalty ให้โปรแกรมผลิตล่วงหน้า
        if self.step_count <= 25:  # ถ้าปล่อยให้ถึง 29 ค่าindex y จะหลุดนอกสมาชิก array
            y = 3 * (self.step_count + 1)
            # print("y =", y)
            d4 = demand_array2[y + 3]
            d5 = demand_array2[y + 4]
            d6 = demand_array2[y + 5]
        else:
            d4 = 0
            d5 = 0
            d6 = 0
        if self.step_count <= 22:  # ถ้าปล่อยให้ถึง 29 ค่าindex y จะหลุดนอกสมาชิก array
            y = 3 * (self.step_count + 1)
            # print("y =", y)
            d7 = demand_array2[y + 6]
            d8 = demand_array2[y + 7]
            d9 = demand_array2[y + 8]
        else:
            d7 = 0
            d8 = 0
            d9 = 0
        # print("d1 d2 d3 =", d1, d2, d3)
        # print("d =", d1, d2, d3)
        demand1 = d1
        demand2 = d2
        demand3 = d3
        demand4 = d4
        demand5 = d5
        demand6 = d6
        demand7 = d7
        demand8 = d8
        demand9 = d9

        # Penalty for future shortage
        extra_penalty1_2 = 0
        extra_penalty2_2 = 0
        extra_penalty3_2 = 0
        extra_penalty1_3 = 0
        extra_penalty2_3 = 0
        extra_penalty3_3 = 0
        overage1_2 = overage1 - d4 + R1
        overage2_2 = overage2 - d5 + R2
        overage3_2 = overage3 - d6 + R3
        overage1_3 = overage1 - d7 + R1
        overage2_3 = overage2 - d8 + R2
        overage3_3 = overage3 - d9 + R3
        # print("overage1_2 =", overage1_2)

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

        if overage1_3 in range(100, 6000) and overage2_3 in range(100, 6000) and overage3_3 in range(100, 6000):
            extra_reward2 = 500 * 1000000
        # if overage2_2 in range(1500,8000):
        #    extra_reward2 = 200*1000000
        # if overage3_2 in range(1500,7500):
        #    extra_reward3 = 200*1000000

        # sum_extra_reward =  extra_reward1 + extra_reward2
        sum_extra_reward = extra_reward2
        sum_extra_penalty_2 = extra_penalty1_2 + extra_penalty2_2 + extra_penalty3_2
        sum_extra_penalty_3 = extra_penalty1_3 + extra_penalty2_3 + extra_penalty3_3
        # print("extra penalty3 =", extra_penalty1_3, extra_penalty2_3, extra_penalty3_3)
        # print("Buffer extra penalty3 =", sum_extra_penalty_3)
        sum_extra_penalty4 = (sum_extra_penalty_2)
        sum_extra_penalty5 = (sum_extra_penalty_3)
        self.sum_ex_penalty_array_2.append(sum_extra_penalty4)
        self.sum_ex_penalty_array_3.append(sum_extra_penalty5)
        # print("penalty array2 =", self.sum_ex_penalty_array_3)


        # reward that use to train agent
#         reward = (sales_revenue - (purchase_cost + holding + penalty_lost_sale
#                                    + (changeover_cost_of_m1 + changeover_cost_of_m2) * 100
#                                    + switch_on_cost + fix_production_cost + (variable_cost_m1 + variable_cost_m2)
#                                    + sum_extra_penalty + sum_extra_penalty_2 + sum_extra_penalty_3
#                                    + (
#                                                extra_p_on1_1 + extra_p_on1_2 + extra_p_on1_3 + extra_p_on2_1 + extra_p_on2_2 + extra_p_on2_3)) / 1000000)

        reward = (2100 + (
            sales_revenue) / 1000000 + extra_reward1 / 1000000 + extra_reward2 / 1000000 + extra_reward3 / 1000000 - (
                              (purchase_cost + holding + penalty_lost_sale
                               + (self.changeover_cost_of_m1 + self.changeover_cost_of_m2) * 10
                               + self.switch_on_cost + fix_production_cost + (
                                           self.variable_cost_m1 + self.variable_cost_m2)
                               + sum_extra_penalty + sum_extra_penalty_2 + sum_extra_penalty_3
                               - (
                                           extra_r_weekend1_1 + extra_r_weekend1_2 + extra_r_weekend1_3 + extra_r_weekend2_1 + extra_r_weekend2_2 + extra_r_weekend2_3)
                               + (
                                       extra_p_on1_1 + extra_p_on1_2 + extra_p_on1_3 + extra_p_on2_1 + extra_p_on2_2 + extra_p_on2_3)) / 1000000)) / 2100  # 650
        
        # real reward that equal to real revenue
        real_reward = (sales_revenue
                       - purchase_cost
                       - holding
                       - penalty_lost_sale
                       - (self.changeover_cost_of_m1 + self.changeover_cost_of_m2)
                       - self.switch_on_cost
                       - fix_production_cost
                       - (self.variable_cost_m1 + self.variable_cost_m2))/34.84   #แปลงจากบาท to dollar

       

        #if self.step_count == 0:
        #print("=======================================")
        #print("self.step_count", self.step_count)
        print("overage1 =" , overage1)
        print("overage2 =" , overage2)
        print("overage3 =" , overage3)


        # print("Step",self.step_count )
        self.step_count += 1
        done = bool(self.step_count >= 30)  # planning time frame period = 15
        self.overall_time_trained += 1

        # if done == True:
        #   self.rn = np.random.random_integers(1, high=100000, size=None)

        # Normalize the reward
        reward = reward / 10000000
        # reward = (1000 + reward)/1000    #ไว้แก้ไม่ให้ค่า reward เป็นลบ

        real_reward = real_reward / 1000000

        self.sum_reward += reward
        self.sum_real_reward += real_reward
        last_sum_reward = 0
        last_sum_real_reward = 0

        if self.step_count == 30:
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
                self.M2P1_set, self.M2P2_set, self.M2P3_set]  # info[20-22]
        
#         print("value ก่อน Normalize") 
#         print("demand1", demand1)
#         print("overage1", overage1)
        #Normalize value to 0-1 range  =before sendout to neural network
        demand1 = (demand1-mind1)/(maxd1-mind1)
        demand2 = (demand2-mind2)/(maxd2-mind2)
        demand3 = (demand3-mind3)/(maxd3-mind3)
        
        demand4 = (demand4-mind1)/(maxd1-mind1)
        demand5 = (demand5-mind2)/(maxd2-mind2)
        demand6 = (demand6-mind3)/(maxd3-mind3)
        demand7 = (demand7-mind1)/(maxd1-mind1)
        demand8 = (demand8-mind2)/(maxd2-mind2)
        demand9 = (demand9-mind3)/(maxd3-mind3)
        
        overage1 = (overage1-minr1)/(maxr1-minr1)
        overage2 = (overage2-minr2)/(maxr2-minr2)
        overage3 = (overage3-minr3)/(maxr3-minr3)
        overage1_2 = (overage1_2-minr1)/(maxr1-minr1)
        overage2_2 = (overage2_2-minr2)/(maxr2-minr2)
        overage3_2 = (overage3_2-minr3)/(maxr3-minr3)
        overage1_3 = (overage1_3-minr1)/(maxr1-minr1)
        overage2_3 = (overage2_3-minr2)/(maxr2-minr2)
        overage3_3 = (overage3_3-minr3)/(maxr3-minr3)
        
        
        
         # inv data
        self.state[0] = 0
        self.state[1] = 0
        self.state[2] = 0
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
        self.state[20] = demand4
        self.state[21] = demand5
        self.state[22] = demand6
        self.state[23] = demand7
        self.state[24] = demand8
        self.state[25] = demand9
        self.state[26] = extra_p_on
        
#         print("value หลัง Normalize") 
#         print("demand1", demand1)
#         print("overage1", overage1)
        #print("#######################State####################", self.state)
        
        # Clears the variables used to store data.
        N1P_ = 0
        N1P1_ = 0
        N1P2_ = 0
        N1P3_ = 0
        N2P_ = 0
        N2P1_ = 0
        N2P2_ = 0
        N2P3_ = 0

        '''
        if R1 > 0:
            print("Recieve product 1 =", R1)
            if M1P1 > 0:
                print("Machine 1 produce P1 =", M1P1)
            if M2P1 > 0:
                print("Machine 2 produce P1 =", M2P1)
        if R2 > 0:
            print("Recieve product 2 =", R2)
            if M1P2 > 0:
                print("Machine 1 produce P2 =", M1P2)
            if M2P2 > 0:
                print("Machine 2 produce P2 =", M2P2)
        if R3 > 0:
            print("Recieve product 3 =", R3)
            if M1P3 > 0:
                print("Machine 1 produce P3 =", M1P3)
            if M2P3 > 0:
                print("Machine 2 produce P3 =", M2P3)
        if sales1 > 0:
            print("sale P1 : ", sales1)
        if sales2 > 0:
            print("sale P2 : ", sales2)
        if sales3 > 0:
            print("sale P3 : ", sales3)
        '''

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

        # เนื่องจาก reward ตอนที่ A3C คิดน่าจะ เป็น sum_reward ในแต่ละ episode อยู่แล้ว ดังนั้น reward ที่ return ควรเป็น reward
        return np.array(self.state, dtype=np.float32), reward, done, info
    
