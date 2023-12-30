from inv_env.envs.InventoryEnv18_gym import InventoryEnv
from inv_env.envs.InventoryEnv22_gym_Dset2 import InventoryEnv2
from inv_env.envs.InvEnv25_3act import InvEnv3

from inv_env.envs.InvEnv25_16act import InvEnv4   #id='inv-v3'
from inv_env.envs.InvEnv25_1pattern import InvEnv4_a2   #id='inv-v3_a2'  #อันนี้คือ กรณีเปลี่ยนรูปแบบ demand ให้มีแต่ season ป่องกลางอย่างเดียว

#from inv_env.envs.InvEnv26_A1_RandomV_noP import InvEnv5_a1   #id='inv-v3_a-1'
# from inv_env.envs.InvEnv26_A2_season5_noP import InvEnv5_a2   #id='inv-v3_a-2'

#Train 60T
#from inv_env.envs.InvEnv26_A1_60T_RandomV2_noP2 import InvEnv5_60T_a1   #id='inv-v3_60t_a-1'
from inv_env.envs.InvEnv26_A1_randomV_60T_add_action2_noP import InvEnv5_60T_a1   #id='inv-v3_60t_a-1'
from inv_env.envs.InvEnv26_A2_season6_60T_add_action4_noP import InvEnv5_60T_a2   #id='inv-v3_60t_a-2'
from inv_env.envs.InvEnv26_A3_midseason_60T_add_action6_noP import InvEnv5_60T_a3   #id='inv-v3_60t_a-3'
from inv_env.envs.InvEnv26_A4_highseason_60T_add_action_noP import InvEnv5_60T_a4   #id='inv-v3_60t_a-4'


#test 60T 
from inv_env.envs.InvEnv27_A2_season_test_set1_60T_m6 import InvEnv5_60T_a2_set1  #id= 'inv_a2_testset1'
from inv_env.envs.InvEnv27_A2_season_test_set2_60T import InvEnv5_60T_a2_set2  #id= 'inv_a2_testset2'
from inv_env.envs.InvEnv27_A2_season_test_set3_60T import InvEnv5_60T_a2_set3  #id= 'inv_a2_testset3'

from inv_env.envs.InvEnv27_A1_randomV_test_set1_60T_m2 import InvEnv5_60T_a1_set1  #id= 'inv_a1_testset1'
from inv_env.envs.InvEnv27_A1_randomV_test_set2_60T_m2 import InvEnv5_60T_a1_set2  #id= 'inv_a1_testset1'
from inv_env.envs.InvEnv27_A1_randomV_test_set3_60T_m import InvEnv5_60T_a1_set3  #id= 'inv_a1_testset1'

from inv_env.envs.InvEnv27_A3_season_test_set1_60T import InvEnv5_60T_a3_set1  #id= 'inv_a3_testset1'
from inv_env.envs.InvEnv27_A3_season_test_set2_60T import InvEnv5_60T_a3_set2  #id= 'inv_a3_testset2'
from inv_env.envs.InvEnv27_A3_season_test_set3_60T import InvEnv5_60T_a3_set3  #id= 'inv_a3_testset3'

from inv_env.envs.InvEnv27_A4_season_test_set1_60T import InvEnv5_60T_a4_set1  #id= 'inv_a4_testset1'
from inv_env.envs.InvEnv27_A4_season_test_set2_60T import InvEnv5_60T_a4_set2  #id= 'inv_a4_testset2'
from inv_env.envs.InvEnv27_A4_season_test_set3_60T import InvEnv5_60T_a4_set3  #id= 'inv_a4_testset3'






from inv_env.envs.InventoryEnv31_30T import InvEnv5
from inv_env.envs.Robot_in_room import RobotInRoom
from inv_env.envs.mountain_car_env import MountainCarEnv2

from inv_env.envs.InvEnv36_oneSetD_set1 import InvEnv6   #id='inv-v5'
from inv_env.envs.InvEnv36_oneSetD_set2 import InvEnv7  #id= 'inv-v5-2'
from inv_env.envs.InvEnv36_oneSetD_set3 import InvEnv8  #id= 'inv-v5-3'
from inv_env.envs.InvEnv36_oneSetD_set4 import InvEnv9  #id= 'inv-v5-4'
from inv_env.envs.InvEnv36_oneSetD_set5 import InvEnv10  #id= 'inv-v5-5'

from inv_env.envs.InvEnv36_oneSetD_set6 import InvEnv11  #id= 'inv-v5-6'
from inv_env.envs.InvEnv36_oneSetD_set7 import InvEnv12  #id= 'inv-v5-7'
from inv_env.envs.InvEnv36_oneSetD_set8 import InvEnv13  #id= 'inv-v5-8'
from inv_env.envs.InvEnv36_oneSetD_set9 import InvEnv14  #id= 'inv-v5-9'
from inv_env.envs.InvEnv36_oneSetD_set10 import InvEnv15  #id= 'inv-v5-10'

from inv_env.envs.InvEnv36_oneSetD_set11 import InvEnv16  #id= 'inv-v5-11'
from inv_env.envs.InvEnv36_oneSetD_set12 import InvEnv17  #id= 'inv-v5-12'
from inv_env.envs.InvEnv36_oneSetD_set13 import InvEnv18  #id= 'inv-v5-13'
from inv_env.envs.InvEnv36_oneSetD_set14 import InvEnv19  #id= 'inv-v5-14'
from inv_env.envs.InvEnv36_oneSetD_set15 import InvEnv20  #id= 'inv-v5-15'

from inv_env.envs.Inv_env_sen_set1 import InvSen1  #id= 'inv-sen1'
from inv_env.envs.Inv_env_sen_set2 import InvSen2  #id= 'inv-sen2'
from inv_env.envs.Inv_env_sen_set3 import InvSen3  #id= 'inv-sen3'
from inv_env.envs.Inv_env_sen_set4 import InvSen4  #id= 'inv-sen4'
from inv_env.envs.Inv_env_sen_set5 import InvSen5  #id= 'inv-sen5'
from inv_env.envs.Inv_env_sen_set6 import InvSen6  #id= 'inv-sen6'
from inv_env.envs.Inv_env_sen_set7 import InvSen7  #id= 'inv-sen7'
from inv_env.envs.Inv_env_sen_set8 import InvSen8  #id= 'inv-sen8'
from inv_env.envs.Inv_env_sen_set9 import InvSen9  #id= 'inv-sen9'
from inv_env.envs.Inv_env_sen_set10 import InvSen10  #id= 'inv-sen10'

#======================================================
## Original 30Periods & 27Observation 
from inv_env.envs.Inv_z_Env36_oneSetD_set1_27obs import Inv_z_Env6   #id='inv-v5-z1'




