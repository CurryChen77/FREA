# validation-based-on-carla-testing
## 静态场景改变
carla_runner.py中的self._init_world(m_i)用于生成静态场景，诸如天气，自车需要行驶的global route等
都由scenario/scenario_data/route下的.xml文件或.json文件生成  

### 静态地图载入流程

1. 一类生成场景算法(adv_behavior_single)包含一系列的data_id，即不同的scenario template id, route id的组合
2. sceanrio_type中的yaml文件去除一部分的data_id，例如，采用SAC算法进行adv_behavior_single生成，指定scenario_id=8，route id=null，即只进行第八类scenario template的场景(NoSignalJunctionCrossingRoute)，但选择第八类全部的route进行生成。其中不同的route包含不同地图下的不同route
3. 针对挑选scenario id和route id后的data id进行具体的scenario_05_route_03.xml文件中解析其route上的waypoints以及额外信息，并对每一个data_id依据town进行分类
4. 针对根据town分类后的data id生成地图，每一个具体的地图中，可能包含不同scenario template，以及route。例如，可能为Town5中，所有可能发生NoSignalJunctionCrossingRoute模板场景的route，如果不仅仅包含一个scenario template，可能还会包含多种scenario template的多种route，在一个地图内。
5. 当carla_runner每run一次，会针对一个地图，挑出该地图中，所有的data id包含的config (每一个config包含一个具体的scenario template的route，即发生NoSignalJunctionCrossingRoute的route 1)，并且将其打乱，包装为一个data_loader训练测试时，对data_loader进行sample，挑出某些包含场景的route。并且在训练和测试时，同时sample到的多条route进行仿真。



### 各类文件作用
1. [Scenario_type](safebench/scenario/config/scenario_type)中的.json文件包含某一生成场景算法(adv_behavior_single, adv_init_state等)，所有的**Scenario_id, route_id**组合，使用时通过.yaml文件确定在哪些**scenario模板**以及**哪些地图的哪些route**进行部署, 
    其中针对同一scenario以及route可能重复多次，但其data_id不同
2. **确定Scenario_id (Scenario模板): **[scenario_01.json](safebench/scenario/scenario_data/route/scenarios)文件表示，某一scenario template (8种template)在所有的7张地图(Town_Safebench_Light,Town1~Town6)中，可能在哪些地图中出现，且周围的环境车辆的初始位置信息，如果available_event_configurations为空，则代表在该地图下，不可能发生该scenario。例如scenario_01.json表示DynamicObjectCrossing这一场景只会在Town_Safebench_Light地图和Town5地图下可能出现
3. **确定route_id (某一地图某一Route): **[Scenario_01_routes](safebench/scenario/scenario_data/route/scenario_01_routes)文件夹中的Scenario_01_route_xx.xml文件表示在可能的地图中（scenario_01例子中，为Town_Safebench_Light地图和Town5地图）发生DynamicObjectCrossing这一场景的route (包含起始点(保留位姿)，终止点，以及路径中离散的关键点)
4. 天气情况由[scenario_01_routes](./safebench/scenario/scenario_data/route/scenario_01_routes)中的.xml文件指定，也可由[carla_runner.py](safebench/carla_runner.py)中的self._init_world 函数设置
## 动态场景改变
1. initial action: 改变初始状态的通过sceanrio difination中具体scenario的create_behavior跳转
可改，通过改变某一scenario的create_behavior以及initialize_actor
2. adversarial behavior改变行进过程中的状态(调整target speed，目前不支持变道，只能直行)的通过scenario policy update_behavior跳转 
可改，通过改变接收到scenario action后的使用方式，扩展到变道，不仅仅改变target speed  
### Example
1. adversarial behavior:  
[adv_behavior_single](./safebench/scenario/scenario_definition/adv_behavior_single/junction_crossing_route.py) 
update_behavior function, 当触发场景后，改变周车行驶过程中的target speed (仅限于直行),且在实现控制时，加速度即throttle保持一定  
*行车过程中，周车的target speed通过算法控制，初始状态提前设定(pre-defined)*
2. initial action:  
[adv_init_state](./safebench/scenario/scenario_definition/adv_init_state/junction_crossing_route.py) 
create_behavior function改变周车初始行为中的target speed (周车目标速度), trigger distance threshold (提前多少距离触发新场景)
改变他车forward vector的尺度 (改变他车的初始位置), 当自车接近到周车提前设定的位置时，激活场景，通过update_behavior更新动作 
*行车过程中，初始状态由算法提前设定，激活后update behavior提前设定(rule-based)无法通过算法动态改变*

## Train

### 流程

1. get static_obs(静态场景的obs)
2. get scenario init action (**scenario_policy.get_init_action**(static_obs)) (根据静态场景生成Scenario vehicle的初始状态)
3. get obs (env.reset(scenario_init_action)) (根据Scenario vehicle的初始状态获得真正的Obs)
4. store init information in replay buffer
5. get ego action (**agent_policy.get_action**(**obs**, infos)) #用obs
6. get Scenario vehicle action (**scenario_policy.get_action**(obs, **infos**)) #用infos
7. env.step(ego_action, Scenario vehicle action)
8. replay buffer store **[ego_actions, scenario_actions, obs, next_obs, rewards, dones]**
9. train **agent_policy or scenario_policy using replay buffer**

### observation

#### Ego agent原始

Ego agent obs包括

1. camera,

2. lidar

3. birdeye

4. state**默认obs仅包含state**

   lateral_dis(自车与前方最近waypoint距离),

    -delta_yaw(与waypoint偏航角差？),

    speed,

    self.vehicle_front(自车前方的车辆是否是障碍物，即在自车车道且距在一定距离内)) 其中agent.yaml文件中定义的obs_type遵循以下原则 (planning默认obs_type=0, perception obs_type=3)

from [env_wrapper.py](safebench/gym_carla/env_wrapper.py)

```python
# only use the 4-dimensional state space
if self.obs_type == 0:
    return obs['state'][:4].astype(np.float64)
# concat the 4-dimensional state space and lane info
elif self.obs_type == 1:
    new_obs = np.array([
        obs['state'][0], obs['state'][1], obs['state'][2], obs['state'][3],
        obs['command'], 
        obs['forward_vector'][0], obs['forward_vector'][1],
        obs['node_forward'][0], obs['node_forward'][1],
        obs['target_forward'][0], obs['target_forward'][1]
    ])
    return new_obs
# return a dictionary with bird-eye view image and state
elif self.obs_type == 2:
    return {"img": obs['birdeye'], "states": obs['state'][:4].astype(np.float64)}
# return a dictionary with front-view image and state
elif self.obs_type == 3:
    return {"img": obs['camera'], "states": obs['state'][:4].astype(np.float64)}
else:
    raise NotImplementedError
```

#### Ego state 改进后

包含10维state，包括自车信息，global route 信息，周车信息，以及路灯，交叉口等语义信息

```python
state = np.array([ego_location.x, ego_location.y, ego_yaw, ego_speed,  # ego
                  waypoint_dis, -waypoint_delta_yaw,  # pre waypoint dis
                  controlled_bv_dis,  # controlled bv dis
                  self.vehicle_front,  # whether exist front vehicle
                  pre_waypoint_is_junction,  # whether the pre waypoint is in the junction
                  self.red_light_state,  # whether the ego encountered red light
                  ])
```

#### Scenario agent

##### 原始周车state

周车的state来源于info而不是Ego的obs

info包括

```python
info = {
    'waypoints': self.waypoints,
    'route_waypoints': self.route_waypoints,
    'vehicle_front': self.vehicle_front,
    'cost': self._get_cost()
    'actor_info': [ego_state, other_actors_state]
}
```

使用时只采用了actor_info部分，即包含ego state, other actors states

其中每一个state包括

```python
def _get_actor_state(actor):
    actor_trans = actor.get_transform()
    actor_x = actor_trans.location.x
    actor_y = actor_trans.location.y
    actor_yaw = actor_trans.rotation.yaw / 180 * np.pi
    yaw = np.array([np.cos(actor_yaw), np.sin(actor_yaw)])
    velocity = actor.get_velocity()
    acc = actor.get_acceleration()
    return [actor_x, actor_y, actor_yaw, yaw[0], yaw[1], velocity.x, velocity.y, acc.x, acc.y]  # 9 dimension
```

other actors的顺序遵循scenario_01.json文件中关于other_actor的顺序

##### 改进后周车state

根据离自车的远近更新的info中的9个维度的state（来源于_get_actor_state）

在env.reset和env.update时都会更新一次

TODO需要将以自车为中心改为，以被控舟车为中心

### Action

* 以SAC Agent为例

​	ego action dim=2, ego state dim=4, ego action limit=1

```python
if self.discrete:
    acc = self.discrete_act[0][ego_action // self.n_steer]  # 'discrete_acc': [-3.0, 0.0, 3.0]
    steer = self.discrete_act[1][ego_action % self.n_steer]  # 'discrete_steer': [-0.2, 0.0, 0.2]
else:
    acc = ego_action[0]  # continuous action: acc
    steer = ego_action[1]  # continuous action: steering
```

* 以SAC scenario policy为例

  scenario action dim=1, scenario state dim=18 (2*9  actor number * actor info dim)

  由scenario_poliy.get_action得到的scenario action直接传入各个scenario的update_behavior()中。

  **目前仅为一维（target speed scale）,可改为两维，类似ego，在update_behavior上修改acc以及steering。通过控制self.other_actors[]列表控制具体的车辆**

## TODO

1. Agent 的state如何设置，原先的state中只包含前车是否存在的bool值，以及追寻next waypoint的state，不包含周车的state。直接类似info加入actor的9维值会导致在计算distribution时mu 和log std时出现None，**实质为通过linear层后**输入的数值不稳定或梯度爆炸导致，actor state无法与原先的4维state兼容，仅考虑preview point也会出现Nan，但Scenario policy使用SAC时，同样模型，只包含actor state不会出现Nan
1. ego agent 中的behavior模块会出现self.incoming_waypoint没有is_junction的命令，即无self.incoming_waypoint，可能由于global route过短，导致过早停止，即出现无incoming_waypoints





