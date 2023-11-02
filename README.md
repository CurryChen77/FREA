# RESA: Reasonable Evaluation via Safety Assurance

# Content

* [Train Agent](#Train Agent)
* [Train Scenario](#Train Scenario)
* [Train Safety network](#Train Safety network)
* [Eval](#Eval)
* [Visualization](#Visualization)

## Train Agent

### Usage

``````bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Launch in another terminal
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard.yaml --mode train_agent
``````

### Policy:

* **behavior:** Carla default agent (no state)

* **expert:** Carla leaderboard default rule-based agent (ego state)
* **plant:**  transformer based planning agent, output ego's future waypoints (ego state)
* **sac:** RL-based agent (from Safebench, default 4dim simple state)

### state/observation:

For learnable agent (SAC PPO......)

* **Safebench default state for learning**

  1. **lateral dis** from the target point

  2. **-delta yaw** from the target point

  3. **ego speed**

  4.  **front got vehicle or not**  

     shape: 4

* **PlanT style encoded state**

  1. cls token after transformer block

     shape: [1, 1, 512]

For Carla Leaderboard agent (Expert or PlanT)

* **Ego state for PlanT and Expert**:
  1. ego position (x, y)
  2. ego speed (m/s)
  3. ego yaw (radius)

## Train Scenario

### Usage

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Launch in another terminal
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard.yaml --mode train_scenario
```

### Policy

1. **sac**
2. **standard (autopilot)**
3. **ppo**
4. **td3**

### optional 

* Select the controlled bv method

  1. **attention-based** (default)
  2. **rule-based**

```bash
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard.yaml --mode train_scenario  --cbv_selection 'attention-based'  # different method of selecting controlled bv
```

* Input state

  **Actor info** (ego and surrounding 3 bv's 9-dim state)

* Output action

  2-dim continues action

  acc ~ [-3.0, 3.0]

  steer ~ [-0.3, 0.3]

## Train Safety network

### Usage

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Launch in another terminal
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard.yaml --safety_network_cfg HJR.yaml --mode train_safety_network
```

### Policy

* HJ-Reachability

### Input state

[yaml file](safebench/safety_network/config/HJR.yaml)  ```obs_type: "plant" (default)```

* **PlanT:** (encoded state) default 

* **Actor info:** (ego and surrounding 3 bv's 9-dim state) *no route map information*

## Eval

### Usage

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Launch in another terminal
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard.yaml --safety_network_cfg HJR.yaml --mode eval
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard.yaml --safety_network_cfg HJR.yaml --mode eval --safety_eval  # use the trained safety network to help evaluation
```

## Visualization

### Ego route

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Launch in another terminal
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard.yaml --mode eval --viz_route  # visualization the global route
```

### State encoder attn map

[state encoder yaml](safebench/agent/config/state_encoder.yaml)   ```viz_attn_map=True (default)```

### render the whole pygame window

 ```bash
 # Launch CARLA
 ./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000
 
 # Launch in another terminal
 python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard.yaml --mode eval --render  # show the pygame window
 ```

### enable the semantic segmentation camera in the pygame window

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Launch in another terminal
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard.yaml --mode eval --render --envable_sem # show the pygame window and enable the 3rd-person view using the semantic segmentation camera
```

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
