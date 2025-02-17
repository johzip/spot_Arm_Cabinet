# Spot Robot Setup Tutorials in Isaac Sim
 
This repository contains tutorials for setting up the Spot robot in NVIDIA Isaac Sim. The tutorials guide users through installation, configuration, and running basic simulations with the Spot robot.

# Prerequisites

Before starting, ensure you have the following installed:

please check the requirements and follow the instruction 
[Isaac Sim Official Website](https://docs.isaacsim.omniverse.nvidia.com/latest/installation)

NVIDIA Isaac Sim (The code is tested on 4.2.0 version)

Isaac Lab framework(For the small demo)

# Tutorials
## Basic Setup:
1. How to import and configure Spot in Isaac Sim.
You can download the urdf and usd from the repo [spor_urdf](https://github.com/carolzyy/urdf_spot)

You can import the spot from urdf file or just add the spot USD as a reference on the stage.
[How To Import URDF](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/import_urdf.html)

2. Try the existing spot demo[Spot Locomotion](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_simulation/ext_isaacsim_robot_policy_example.html):

   a.The Boston Dynamics Spot quadruped example can be accessed by creating a empty stage.
   
   b.Open the example menu using Robotics Examples > POLICY > Quadruped.
   
   c.Press LOAD to open the scene.
   
   d.Control with the keyboard

4. Camera setup
Add camera by GUI and code[Camera sensor](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/features/sensors_simulation/isaac_sim_sensors_camera.html)

* Tips:

  * For the camera included in USD, just set the prim.
   
  * For the camera move with the robot, the camera need to be placed under the root, in spot is the body.
   
5. Particle objects simulation
[Official Demo](https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/physics-particles.html)

* Tips:

    * For elastic object,e.g foam,rubber and toys, simulate with the finite element model.
   
    * For deformable objects,e.g. soil, sand and cloth, use the particle model.
   
    * Setup with the GUI would be easier but many parameters only can be set in the code.
   

## Small Demo:
The demo is for control the spot body and arm with the kayboard, and the observation is the rgb data from the front camera we mounted additionally.
