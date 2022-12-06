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
register(
    id='inv-v3',
    entry_point='inv_env.envs:InvEnv4',
)
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




#inventory-v1=demand set2
#inv-v2=3actions
#inv-v3=16actions
#inv-v4=16actions with new efficient way แต่รันใน ppo แล้ว error 
#inv-v5 = 16actions แบบ มี demand ชุดเดียว เอาไว้ เอา demand ไปกรอกตอน test หลังสุด
