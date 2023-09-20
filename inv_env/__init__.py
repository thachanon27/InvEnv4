from gym.envs.registration import register

register(
    id='inventory-v0',
    entry_point='inv_env.envs:InventoryEnv',
)
register(
    id='inventory-v1',
    entry_point='inv_env.envs:InventoryEnv2',
)
register(
    id='inv-v2',
    entry_point='inv_env.envs:InvEnv3',
)
#========================================================
register(
    id='inv-v3',
    entry_point='inv_env.envs:InvEnv4',
)
register(
    id='inv-v3_a2',
    entry_point='inv_env.envs:InvEnv4_a2',
)     #ใช้อันนี้กรณีต้องการเปลี่ยนมาเทรนแบบ season เต็มๆ
#========================================================
#Multi_agents
register(
    id='inv-v3_a-1',
    entry_point='inv_env.envs:InvEnv5_a1',
)     #agent1 random variation
register(
    id='inv-v3_60T_a-2',
    entry_point='inv_env.envs:InvEnv5_60T_a2',
)     #agent2 seasonal

#========================================================


register(
    id='inv-v4',
    entry_point='inv_env.envs:InvEnv5',
)
register(
    id='inv-v5',
    entry_point='inv_env.envs:InvEnv6',
)
register(
    id='robot-v0',
    entry_point='inv_env.envs:RobotInRoom',
)
register(
    id='mtcar-v0',
    entry_point='inv_env.envs:MountainCarEnv2',
)
register(
    id='inv-v5-2',
    entry_point='inv_env.envs:InvEnv7',
)
register(
    id='inv-v5-3',
    entry_point='inv_env.envs:InvEnv8',
)
register(
    id='inv-v5-4',
    entry_point='inv_env.envs:InvEnv9',
)
register(
    id='inv-v5-5',
    entry_point='inv_env.envs:InvEnv10',
)

register(
    id='inv-v5-6',
    entry_point='inv_env.envs:InvEnv11',
)
register(
    id='inv-v5-7',
    entry_point='inv_env.envs:InvEnv12',
)
register(
    id='inv-v5-8',
    entry_point='inv_env.envs:InvEnv13',
)
register(
    id='inv-v5-9',
    entry_point='inv_env.envs:InvEnv14',
)
register(
    id='inv-v5-10',
    entry_point='inv_env.envs:InvEnv15',
)

register(
    id='inv-v5-11',
    entry_point='inv_env.envs:InvEnv16',
)
register(
    id='inv-v5-12',
    entry_point='inv_env.envs:InvEnv17',
)
register(
    id='inv-v5-13',
    entry_point='inv_env.envs:InvEnv18',
)
register(
    id='inv-v5-14',
    entry_point='inv_env.envs:InvEnv19',
)
register(
    id='inv-v5-15',
    entry_point='inv_env.envs:InvEnv20',
)


register(
    id='inv-sen1',
    entry_point='inv_env.envs:InvSen1',
)
register(
    id='inv-sen2',
    entry_point='inv_env.envs:InvSen2',
)
register(
    id='inv-sen3',
    entry_point='inv_env.envs:InvSen3',
)
register(
    id='inv-sen4',
    entry_point='inv_env.envs:InvSen4',
)
register(
    id='inv-sen5',
    entry_point='inv_env.envs:InvSen5',
)
register(
    id='inv-sen6',
    entry_point='inv_env.envs:InvSen6',
)
register(
    id='inv-sen7',
    entry_point='inv_env.envs:InvSen7',
)
register(
    id='inv-sen8',
    entry_point='inv_env.envs:InvSen8',
)
register(
    id='inv-sen9',
    entry_point='inv_env.envs:InvSen9',
)
register(
    id='inv-sen10',
    entry_point='inv_env.envs:InvSen10',
)












#inventory-v1=demand set2
#inv-v2=3actions
#inv-v3=16actions
#inv-v4=16actions with new efficient way แต่รันใน ppo แล้ว error 
#inv-v5 = 16actions แบบ มี demand ชุดเดียว เอาไว้ เอา demand ไปกรอกตอน test หลังสุด
