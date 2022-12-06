import gym
from gym import Env
import numpy as np
from gym import spaces
from gym.spaces import Discrete, Box
from scipy.stats import poisson
from random import randint, choice

#file 18  is demand set1  but file22 will demand set2 ต่างกันแค่นี้

class InventoryEnv(Env):
    def __init__(self):
        self.step_count = 0
        # initial inventory
        self.on_hand1 = np.random.random_integers(2000, high=7000, size=None)  
        self.on_hand2 = np.random.random_integers(1500, high=5000, size=None)  
        self.on_hand3 = np.random.random_integers(1000, high=3500, size=None)  
        self.action_space = spaces.Discrete(16)
        #self.observation_space = spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32)
        self.statelow = np.array([
             0, 0, 0, # initial inventory
             0, 0, 0, #initial demand
             0, 0, 0, 0,  #initial machine status (0 = idle)
             0, 0, 0, 0,
             ])
        self.statehigh = np.array([
             np.inf, np.inf ,np.inf, # initial inventory
             np.inf, np.inf, np.inf, #initial demand
             0, 0, 0, 0,  #initial machine status (0 = idle)
             0, 0, 0, 0,
             ])
        self.observation_space = Box(self.statelow,  self.statehigh ,
                                     dtype=np.float32)

        self.state = [self.on_hand1,self.on_hand2,self.on_hand3,
                      0, 0, 0, 0 , 0, 0,0,0,0,0,0]
        self.reset()
        self.sum_reward = 0

        self.CO_var_set1 = [0, 0, 0, 0, 0, 0, 0, 0]
        self.CO_var_set2 = [0, 0, 0, 0, 0, 0, 0, 0]

        self.sum_ex_penalty_array   = []
        self.sum_ex_penalty_array_2 = []
        self.sum_ex_penalty_array_3 = []
        self.demand_all = [0,0,0]

    def reset(self):
        self.step_count = 0
        #state 14 dimension =onhand ,demand ,production status of machines
        self.state = np.array([
             np.random.random_integers(2000, high=7000, size=None),  # initial inventory
             np.random.random_integers(1500, high=5000, size=None),
             np.random.random_integers(1000, high=3500, size=None),  
             0, 0, 0, #initial demand
             0, 0, 0, 0,  #initial machine status (0 = idle)
             0, 0, 0, 0,
             ])
        self.sum_reward = 0
        return self.state

    def step(self, action):
        #all model parameters
        # holding cost
        h1 = 50.96
        h2 = 34.53
        h3 = 48.66
        # Lost of good Will
        k1 = 6200
        k2 = 4200
        k3 = 5920
        # Sell price
        p1 = 3100
        p2 = 2100
        p3 = 2960
        # unit cost
        c1 = 1860.000
        c2 = 1260.000
        c3 = 1776.000
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
        vc_m1_on = 5000  # ของเดินคือ 1796.380 แต่ลองเพิ่มให้เยอะๆขึ้น เพื่อให้เอเจนต์ฉลาดในการเลี่ยงผลิตช่วง On-peak
        vc_m1_off = 1027.450
        vc_m2_on = 4500  # ของเดิมคือ 1719.050
        vc_m2_off = 986.240
        #clean value
        d1 = 0
        d2 = 0
        d3 = 0
        demand1 = 0
        demand2 = 0
        demand3 = 0

    #binary variable
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

        #print("=========================================================================================")
        #print("step :", self.step_count)
        #print("state =", state)

        #variable use to remember production data from period
        N1P_ = self.state[6]
        N1P1_ = self.state[7]
        #print("N1P1_ = ",N1P1_)
        N1P2_ = self.state[8]
        N1P3_ = self.state[9]
        N2P_ = self.state[10]
        N2P1_ = self.state[11]
        N2P2_ = self.state[12]
        N2P3_ = self.state[13]

        #Retrieves the value from the previous period to calculate the changeOver
        # with skipping 1 time and 2 time interval type before the variable takes the new value.
        #print("CO_var_set1 = ", self.CO_var_set1)
        N1P_2 = self.CO_var_set1[0]
        N1P1_2 = self.CO_var_set1[1]
        #print("N1P1_2 = ", N1P1_2)
        N1P2_2 = self.CO_var_set1[2]
        N1P3_2 = self.CO_var_set1[3]
        N2P_2 = self.CO_var_set1[4]
        N2P1_2 = self.CO_var_set1[5]
        N2P2_2 = self.CO_var_set1[6]
        N2P3_2 = self.CO_var_set1[7]

        #print("CO_var_set2 = ", self.CO_var_set2)
        N1P_3 = self.CO_var_set2[0]
        N1P1_3 = self.CO_var_set2[1]
        #print("N1P1_2 = ", N1P1_3)
        N1P2_3 = self.CO_var_set2[2]
        N1P3_3 = self.CO_var_set2[3]
        N2P_3 = self.CO_var_set2[4]
        N2P1_3 = self.CO_var_set2[5]
        N2P2_3 = self.CO_var_set2[6]
        N2P3_3 = self.CO_var_set2[7]

        #demand of this period
        demand1 = self.state[3]
        demand2 = self.state[4]
        demand3 = self.state[5]
        #print("demand this period =", demand1, demand2, demand3)

        #This clears production data from the previous period.
        self.state[6] = 0
        self.state[7] = 0
        self.state[8] = 0
        self.state[9] = 0
        self.state[10] = 0
        self.state[11] = 0
        self.state[12] = 0
        self.state[13] = 0

        #print("state[7] =", self.state[7])
        N1P = 0
        N1P1 = 0
        N1P2 = 0
        N1P3 = 0
        N2P =  0
        N2P1 = 0
        N2P2 = 0
        N2P3 = 0
        M1P = 0
        M1P1 = 0
        M1P2 = 0
        M1P3 = 0
        M2P = 0
        M2P1=0
        M2P2=0
        M2P3 = 0

        on_hand1,on_hand2,on_hand3,demand1,demand2,demand3, N1P, N1P1, N1P2, N1P3, N2P, N2P1, N2P2, N2P3  = self.state

        #print("Step :", self.step_count)
        #print("onhand1 from last period =", on_hand1)
        #print("onhand2 from last period =", on_hand2)
        #print("onhand3 from last period =", on_hand3)

        ##generate all action possible##########################################
        # declaring a list
        machine = [1, 2]
        product = [0, 1, 2, 3]
        # output
        output = []
        all_case = []
        set_m1_produce = []
        set_m2_produce = []
        # looping over list
        for i in range(0, len(machine)):
            for j in range(0, len(product)):
                output.append((machine[i], product[j]))
        for k in range(0, len(output)):
            if output[k][0] == 1:
                set_m1_produce.append(output[k])
            if output[k][0] == 2:
                set_m2_produce.append(output[k])

        for i in range(0, len(set_m1_produce)):
            for j in range(0, len(set_m2_produce)):
                all_case.append((set_m1_produce[i], set_m2_produce[j]))

        LM1P1 = 3211  # lotsize of Machine 1 for produce product 1
        LM1P2 = 2223
        LM1P3 = 1668
        LM2P1 = 2717
        LM2P2 = 1853
        LM2P3 = 1359
        action_set = []

        num_case = -1  # เริ่มที่ลบ1 เพราะต้องการให้ action อันแรก เริ่มที่ 0
        for i in range(len(all_case)):  # 0    1     2    3    4    5    6   7     8   9    10   11
            action_subset = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0]  # action = [N1P1,M1P1,N1P2,M1P2,N1P3,M1P3,N2P1,M2P1,N2P2,M2P2,N2P3,M2P3]
            num_case += 1
            case = all_case[i]
            M1 = case[0]
            M2 = case[1]
            if M1[1] == 1:
                N1P1 = 1
                M1P1 = LM1P1
                action_subset[0] = 1
                action_subset[1] = LM1P1
            if M1[1] == 2:
                N1P2 = 1
                M1P2 = LM1P2
                action_subset[2] = 1
                action_subset[3] = LM1P2
            if M1[1] == 3:
                N1P3 = 1
                M1P3 = LM1P3
                action_subset[4] = 1
                action_subset[5] = LM1P3
            # =====================================
            if M2[1] == 1:
                N2P1 = 1
                M2P1 = LM2P1
                action_subset[6] = 1
                action_subset[7] = LM2P1
            if M2[1] == 2:
                N2P2 = 1
                M2P2 = LM2P2
                action_subset[8] = 1
                action_subset[9] = LM2P2
            if M2[1] == 3:
                N2P3 = 1
                M2P3 = LM2P3
                action_subset[10] = 1
                action_subset[11] = LM2P3
            action_set.append(action_subset)

        # print("action_set", action_set)
        # action = 2
        action_2 = []
        a = action
        for i in range(0, len(action_set)):
            if i == a:
                action_2 = action_set[i]

        #print("action ", action, " =", action_2)
        N1P1 = action_2[0]
        M1P1 = action_2[1]
        N1P2 = action_2[2]
        M1P2 = action_2[3]
        N1P3 = action_2[4]
        M1P3 = action_2[5]
        N2P1 = action_2[6]
        M2P1 = action_2[7]
        N2P2 = action_2[8]
        M2P2 = action_2[9]
        N2P3 = action_2[10]
        M2P3 = action_2[11]
        # print("M1M1",M1P1,"N2P2",N2P2,"M2P2",M2P2,"M2P3",M2P3)

        # over inventory cap -> no production
        if self.state[0] >= 15789: #if R1 >= 19000 - max(3211,2717) > forbid M1,M2 to produce P1
            N1P1 = 0 #over inventory cap -> no production
            M1P1 = 0 #assume if you can't produce = lot size, then => don't produce at all
            N2P1 = 0
            M2P1 = 0
        if self.state[1] >= 16777: #if R2 >= 19000 - max(2223,1853) -> forbid M1,M2 to produce P2
            N1P2 = 0 #over inventory cap -> no production
            M1P2 = 0
            N2P2 = 0
            M2P2 = 0
        if self.state[2] >= 17332:
            N1P3 = 0  # over inventory cap -> no production
            M1P3 = 0
            N2P3 = 0
            M2P3 = 0

        #print("N1P1= ",N1P1 ," ,M1P1 =", M1P1)
        #print("N1P2= ", N1P2, " ,M1P2 =", M1P2)
        #print("N1P3= ", N1P3, " ,M1P3 =", M1P3)
        #print("N2P1= ",N2P1 ," ,M1P1 =", M2P1)
        #print("N2P2= ", N2P2, " ,M1P2 =", M2P2)
        #print("N2P3= ", N2P3, " ,M1P3 =", M2P3)

        ##if there a production --> NP=1
        if N1P1 == 1 or N1P2 == 1 or N1P3 == 1:
            N1P = 1
        if N2P1 == 1 or N2P2 == 1 or N2P3 == 1:
            N2P = 1

        R1 = M1P1 + M2P1  # R1 = recieve product1
        R2 = M1P2 + M2P2
        R3 = M1P3 + M2P3

        on_hand1 += R1    # onhand1
        on_hand2 += R2
        on_hand3 += R3


        # Compute Reward
        sales1 = min(on_hand1, demand1)  # ถ้า on_hand น้อยกว่า demand ก็ขายแค่ = on hand
        sales2 = min(on_hand2, demand2)
        sales3 = min(on_hand3, demand3)
        #print("sales1 =", sales1)
        sales_revenue = p1 * sales1 + p2 * sales2 + p3 * sales3
        overage1 = on_hand1 - sales1
        overage2 = on_hand2 - sales2
        overage3 = on_hand3 - sales3
        #print("overage1 =", overage1)
        underage1 = max(0, demand1 - on_hand1)
        underage2 = max(0, demand2 - on_hand2)
        underage3 = max(0, demand3 - on_hand3)
        purchase_cost = c1 * R1 + c2*R2 + c3*R3
        holding = overage1 * h1 + overage2 * h2 + overage3 * h3
        penalty_lost_sale = k1 * underage1 + k2 * underage2 + k3 * underage3

        ###change over of M1 แบบเปลี่ยน ช่วงติดกัน
        if N1P1_ == 1 and  N1P2 == 1:  # N1P1_(last period) = 1 and N1P2 (this period) = 1 --> then there change over from P1 to P2 in M1
            CO11 = 1
        if N1P1_ == 1 and  N1P3 == 1:  # N1P1_(last period) = 1 and N1P3 (this period) = 1 --> then there change over from P1 to P3 in M1
            CO11 = 1
        if N1P2_ == 1 and  N1P1 == 1:
            CO11 = 1
        if N1P2_ == 1 and  N1P3 == 1:
            CO11 = 1
        if N1P3_ == 1 and  N1P1 == 1:
            CO11 = 1
        if N1P3_ == 1 and  N1P2 == 1:
            CO11 = 1
        #print("CO11 = ", CO11)
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
        #print("CO21 = ", CO21)

        #คิด change over แบบเว้นข้าม 1period
        if N1P1_2 == 1 and N1P_ == 0 and  N1P2 == 1:  # N1P1_2(last two period) = 1  but there no production last period (N1P_ = 0),and N1P2 (this period) = 1 --> then there change over from P1 to P2 in M1
            CO12 = 1
        if N1P1_2 == 1 and N1P_ == 0 and  N1P3 == 1:  # N1P1_2(last two period) = 1 but there no production last period (N1P_ = 0),and N1P3 (this period) = 1 --> then there change over from P1 to P3 in M1
            CO12 = 1
        if N1P2_2 == 1 and N1P_ == 0 and  N1P1 == 1:
            CO12 = 1
        if N1P2_2 == 1 and N1P_ == 0 and  N1P3 == 1:
            CO12 = 1
        if N1P3_2 == 1 and N1P_ == 0 and  N1P1 == 1:
            CO12 = 1
        if N1P3_2 == 1 and N1P_ == 0 and  N1P2 == 1:
            CO12 = 1
        #print("CO12 = ", CO12)
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
        #print("CO22 = ", CO22)

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
        #print("CO13 = ", CO13)
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
        #print("CO23 = ", CO23)

        ##เก็บค่าการผลิต เพื่อนำไปคิด change over แบบเว้นข้าม 2period ก่อนหน้าเพื่อมาคิดCO13 CO23
        N1P_3 = N1P_2
        N1P1_3 = N1P1_2
        #print("N1P1_3 = ", N1P1_3)
        N1P2_3 = N1P2_2
        N1P3_3 = N1P3_2
        N2P_3 = N2P_2
        N2P1_3 = N2P1_2
        N2P2_3 = N2P2_2
        N2P3_3 = N2P3_2
        self.CO_var_set2 = [N1P_3, N1P1_3, N1P2_3, N1P3_3, N2P_3, N2P1_3, N2P2_3, N2P3_3]
        #print("CO_var_set2 = ", self.CO_var_set2)

        ##Collect production values that used to calculate
        # the previous 'change over :skipping 1 period type' to calculate CO12 CO22
        N1P_2 = N1P_
        N1P1_2 = N1P1_
        #print("N1P1_2 = ", N1P1_2)
        N1P2_2 = N1P2_
        N1P3_2 = N1P3_
        N2P_2 = N2P_
        N2P1_2 = N2P1_
        N2P2_2 = N2P2_
        N2P3_2 = N2P3_
        self.CO_var_set1 = [N1P_2, N1P1_2, N1P2_2, N1P3_2, N2P_2, N2P1_2, N2P2_2, N2P3_2]
        #print("CO_var_set1 = ", self.CO_var_set1)

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
        #Assign - on-peak and off-peak p cost to each period
        weekend_stepcount = [3,4,5,6,17,18,19,20]
        on_peak_stepcount = [1,7,9,11,13,15,21,23,25,27,29]
        off_peak_stepcount = [2,8,10,12,14,22,24,26,28,30]
        stp = 0
        stp = self.step_count+1
        #print("stp =",stp)
        #if stp in on_peak_stepcount:
        #    print("yes")
        if stp in on_peak_stepcount:
            #print("yes")
            vcm1 = vc_m1_on
            #print("vcm1 = ", vcm1)
            vcm2 = vc_m2_on
        if stp in off_peak_stepcount:
            vcm1 = vc_m1_off
            vcm2 = vc_m2_off
        if stp in weekend_stepcount:
            vcm1 = vc_m1_off
            vcm2 = vc_m2_off

        #print("vcm1_cost = ", vcm1)
        #print("vcm2_cost = ", vcm2)
        variable_cost_m1 = vcm1*(M1P1 + M1P2 + M1P3)
        variable_cost_m2 = vcm2*(M2P1 + M2P2 + M2P3)
        #print("variable_cost_m1 = ", variable_cost_m1)
        changeover_cost_of_m1 = co11*(CO11 + CO12 + CO13)  # period นึงจะเกิด CO11, CO12, CO13 ได้แค่ 1 กรณี จึงจับรวมได้เลย
        changeover_cost_of_m2 = co21*(CO21 + CO22 + CO23)
        #print("CO11 =",CO11)
        #print("CO21 =",CO21)
        switch_on_cost = sw1*SW1 + sw2*SW2
        #print("SW1 =", SW1)
        #print("SW2 =", SW2)

        fix_production_cost = fc_m1*FC_M1 + fc_m2*FC_M2
        #print("FC_M1 =", FC_M1)
        #print("FC_M2 =", FC_M2)

        #force buffer on_hand > safety stock
        extra_penalty1 = 0
        extra_penalty2 = 0
        extra_penalty3 = 0
        sum_extra_penalty = 0
        if overage1 < 3000:
           extra_penalty1 = 9000000
        if overage2 < 2000:
            extra_penalty2 = 9000000
        if overage3 < 1500:
            extra_penalty3 = 9000000
        sum_extra_penalty = extra_penalty1 + extra_penalty2 + extra_penalty3
        #print("extra penalty =", extra_penalty1, extra_penalty2, extra_penalty3)
        #print("Buffer extra penalty =", sum_extra_penalty)
        sum_extra_penalty2 = sum_extra_penalty / 100000
        self.sum_ex_penalty_array.append(sum_extra_penalty2)
        #print("penalty array =", self.sum_ex_penalty_array)

        #Genarate Damand random
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
        if self.step_count in set_stepcount1:  # ถ้าปล่อยให้ถึง 29 ค่าindex y จะหลุดนอกสมาชิก array
            y = self.step_count + 1
            demand_array2[y * 3] = np.random.random_integers(2000, high=4500, size=None)
            self.demand_all.append(demand_array2[y * 3])
            demand_array2[y * 3 + 1] = np.random.random_integers(1200, high=3000, size=None)
            self.demand_all.append(demand_array2[y * 3 + 1])
            demand_array2[y * 3 + 2] = np.random.random_integers(700, high=2500, size=None)
            self.demand_all.append(demand_array2[y * 3 + 2])
            demand_array2[y * 3 + 3] = 0
            self.demand_all.append(demand_array2[y * 3 + 3])
            demand_array2[y * 3 + 4] = 0
            self.demand_all.append(demand_array2[y * 3 + 4])
            demand_array2[y * 3 + 5] = 0
            self.demand_all.append(demand_array2[y * 3 + 5])
            demand_array2[y * 3 + 6] = np.random.random_integers(2000, high=4500, size=None)
            self.demand_all.append(demand_array2[y * 3 + 6])
            demand_array2[y * 3 + 7] = np.random.random_integers(1200, high=3000, size=None)
            self.demand_all.append(demand_array2[y * 3 + 7])
            demand_array2[y * 3 + 8] = np.random.random_integers(700, high=2500, size=None)
            self.demand_all.append(demand_array2[y * 3 + 8])
            demand_array2[y * 3 + 9] = 0
            self.demand_all.append(demand_array2[y * 3 + 9])
            demand_array2[y * 3 + 10] = 0
            self.demand_all.append(demand_array2[y * 3 + 10])
            demand_array2[y * 3 + 11] = 0
            self.demand_all.append(demand_array2[y * 3 + 11])
            d1 = demand_array2[y * 3]
            d2 = demand_array2[y * 3 + 1]
            d3 = demand_array2[y * 3 + 2]
            d4 = demand_array2[y * 3 + 3]
            d5 = demand_array2[y * 3 + 4]
            d6 = demand_array2[y * 3 + 5]
            d7 = demand_array2[y * 3 + 6]
            d8 = demand_array2[y * 3 + 7]
            d9 = demand_array2[y * 3 + 8]

        if self.step_count in set_stepcount3:  # ถ้าปล่อยให้ถึง 29 ค่าindex y จะหลุดนอกสมาชิก array
            y = self.step_count + 1
            demand_array2[y * 3] = np.random.random_integers(2000, high=4500, size=None)
            self.demand_all.append(demand_array2[y * 3])
            demand_array2[y * 3 + 1] = np.random.random_integers(1200, high=3000, size=None)
            self.demand_all.append(demand_array2[y * 3 + 1])
            demand_array2[y * 3 + 2] = np.random.random_integers(700, high=2500, size=None)
            self.demand_all.append(demand_array2[y * 3 + 2])
            demand_array2[y * 3 + 3] = 0
            self.demand_all.append(demand_array2[y * 3 + 3])
            demand_array2[y * 3 + 4] = 0
            self.demand_all.append(demand_array2[y * 3 + 4])
            demand_array2[y * 3 + 5] = 0
            self.demand_all.append(demand_array2[y * 3 + 5])
            demand_array2[y * 3 + 6] = np.random.random_integers(2000, high=4500, size=None)
            self.demand_all.append(demand_array2[y * 3 + 6])
            demand_array2[y * 3 + 7] = np.random.random_integers(1200, high=3000, size=None)
            self.demand_all.append(demand_array2[y * 3 + 7])
            demand_array2[y * 3 + 8] = np.random.random_integers(700, high=2500, size=None)
            self.demand_all.append(demand_array2[y * 3 + 8])
            demand_array2[y * 3 + 9] = 0
            self.demand_all.append(demand_array2[y * 3 + 9])
            demand_array2[y * 3 + 10] = 0
            self.demand_all.append(demand_array2[y * 3 + 10])
            demand_array2[y * 3 + 11] = 0
            self.demand_all.append(demand_array2[y * 3 + 11])
            ##เสริมให้เต็ม 30
            demand_array2[y * 3 + 12] = np.random.random_integers(2000, high=4500, size=None)
            self.demand_all.append(demand_array2[y * 3 + 12])
            demand_array2[y * 3 + 13] = np.random.random_integers(1200, high=3000, size=None)
            self.demand_all.append(demand_array2[y * 3 + 13])
            demand_array2[y * 3 + 14] = np.random.random_integers(700, high=2500, size=None)
            self.demand_all.append(demand_array2[y * 3 + 14])
            d1 = demand_array2[y * 3]
            d2 = demand_array2[y * 3 + 1]
            d3 = demand_array2[y * 3 + 2]
            d4 = demand_array2[y * 3 + 3]
            d5 = demand_array2[y * 3 + 4]
            d6 = demand_array2[y * 3 + 5]
            d7 = demand_array2[y * 3 + 6]
            d8 = demand_array2[y * 3 + 7]
            d9 = demand_array2[y * 3 + 8]


        #assign demand of next period from array of demand data

        if self.step_count <= 28:
            # demand of product 1  next period
            demand1 = self.demand_all[(self.step_count + 1) * 3]
            # demand of product 2  next period
            demand2 = self.demand_all[(self.step_count + 1) * 3 + 1]
            # demand of product 3  next period
            demand3 = self.demand_all[(self.step_count + 1) * 3 + 2]
        if self.step_count >= 29:
            demand1 = 0
            demand2 = 0
            demand3 = 0

        # Penalty for future shortage
        extra_penalty1_2 = 0
        extra_penalty2_2 = 0
        extra_penalty3_2 = 0
        extra_penalty1_3 = 0
        extra_penalty2_3 = 0
        extra_penalty3_3 = 0
        overage1_2 = overage1 - d4
        overage2_2 = overage2 - d5
        overage3_2 = overage3 - d6
        overage1_3 = overage1 - d7
        overage2_3 = overage2 - d8
        overage3_3 = overage3 - d9
        #print("overage1_2 =", overage1_2)
        if overage1_2 < 0:
            extra_penalty1_2 = 7000000
        if overage2_2 < 0:
            extra_penalty2_2 = 7000000
        if overage3_2 < 0:
            extra_penalty3_2 = 7000000
        if overage1_3 < 0:
            extra_penalty1_3 = 7000000
        if overage2_3 < 0:
            extra_penalty2_3 = 7000000
        if overage3_3 < 0:
            extra_penalty3_3 = 7000000
        sum_extra_penalty_2 = extra_penalty1_2 + extra_penalty2_2 + extra_penalty3_2
        sum_extra_penalty_3 = extra_penalty1_3 + extra_penalty2_3 + extra_penalty3_3
        #print("extra penalty3 =", extra_penalty1_3, extra_penalty2_3, extra_penalty3_3)
        #print("Buffer extra penalty3 =", sum_extra_penalty_3)
        sum_extra_penalty4 = (sum_extra_penalty_2) / 100000
        sum_extra_penalty5 = (sum_extra_penalty_3) / 100000
        self.sum_ex_penalty_array_2.append(sum_extra_penalty4)
        self.sum_ex_penalty_array_3.append(sum_extra_penalty5)
        #print("penalty array2 =", self.sum_ex_penalty_array_3)

        # reward that use to train agent
        reward = sales_revenue \
                 - purchase_cost \
                 - holding \
                 - penalty_lost_sale \
                 - (changeover_cost_of_m1 + changeover_cost_of_m2) \
                 - switch_on_cost \
                 - fix_production_cost \
                 - (variable_cost_m1 + variable_cost_m2) \
                 - sum_extra_penalty \
                 - sum_extra_penalty_2 \
                 - sum_extra_penalty_3
        # real reward that equal to real revenue
        real_reward = sales_revenue \
                      - purchase_cost \
                      - holding \
                      - penalty_lost_sale \
                      - (changeover_cost_of_m1 + changeover_cost_of_m2) \
                      - switch_on_cost \
                      - fix_production_cost \
                      - (variable_cost_m1 + variable_cost_m2)

        #inv data
        self.state[0] = 0
        self.state[1] = 0
        self.state[2] = 0
        #update demand data this period
        self.state[3] = demand1
        self.state[4] = demand2
        self.state[5] = demand3

        self.state[0] += overage1  # Inventory that has already subtracted demand
        self.state[1] += overage2
        self.state[2] += overage3
        #collect Production data to use in next state
        self.state[6] = N1P
        self.state[7] = N1P1
        self.state[8] = N1P2
        self.state[9] = N1P3
        self.state[10] = N2P
        self.state[11] = N2P1
        self.state[12] = N2P2
        self.state[13] = N2P3

        #Clears the variables used to store data.
        N1P_ = 0
        N1P1_ = 0
        N1P2_ = 0
        N1P3_ = 0
        N2P_ = 0
        N2P1_ = 0
        N2P2_ = 0
        N2P3_ = 0

        self.step_count += 1
        done = bool(self.step_count >= 29)  # planning time frame period = 15


        #if done == True:
        #   self.rn = np.random.random_integers(1, high=100000, size=None)

        # Normalize the reward
        reward = reward / 10000000
        #reward = (reward + 100)/(110)  #r_max = 10 , r_min = -100
        #reward = (reward + 300)/(350)  #r_max = 50 , r_min = -300
        #reward = (reward + 100)/(200)  #r_max = 100 , r_min = -100

        
        self.sum_reward += reward

        #pass CO data value to use in next state by save it in 'info' part
        #the real reward is at info[11]
        info = [self.CO_var_set1, self.CO_var_set2, self.sum_reward, self.sum_ex_penalty_array, d4, d5, d6, d7, d8, d9,
                self.sum_ex_penalty_array_3, real_reward]

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

        #print("state[7]_2 =", self.state[7])
        #print("state_2 =", self.state)
        #print("End_On_hand_P1 this period :", self.state[0])
        #print("End_On_hand_P2 this period :", self.state[1])
        #print("End_On_hand_P3 this period :", self.state[2])
        # print("period buy :", self.state[1:lt])
        #print("demand of next period =", demand1, demand2, demand3)
        #print("Reward : ", reward)
        #print("Sum_reward : ", self.sum_reward)
        #print("---------------------------------------------")
        # print("obs : ", self.state)
        # return self._normalize_obs(), reward, done, info

        #เนื่องจาก reward ตอนที่ A3C คิดน่าจะ เป็น sum_reward ในแต่ละ episode อยู่แล้ว ดังนั้น reward ที่ return ควรเป็น reward
        return self.state, reward, done, info
