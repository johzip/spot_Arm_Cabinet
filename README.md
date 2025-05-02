# Spot Robot Setup Tutorials in Isaac Sim

This repository contains tutorials for setting up the Spot robot in NVIDIA Isaac Sim. The tutorials guide users through installation, configuration, and running basic simulations with the Spot robot.

## Prerequisites

Before starting, ensure you have the following installed:

Please check the requirements and follow the instructions provided in the official documentation:\
[Isaac Sim Official Website](https://docs.isaacsim.omniverse.nvidia.com/latest/installation)

- **NVIDIA Isaac Sim** (Tested on version 4.2.0)
- **Isaac Lab framework** (Required for the small demo,Tested on version 1.2.0)

## Tutorials

### Basic Setup

1. **Importing and Configuring Spot in Isaac Sim**\
   You can download the URDF and USD files from this repository:\
   [Spot URDF Repository](https://github.com/carolzyy/urdf_spot)

   You can import the Spot robot from a URDF file or add the Spot USD as a reference on the stage.\
   [How to Import URDF](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/import_urdf.html)

2. **Trying the Existing Spot Demo**\
   [Spot Locomotion Demo](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_simulation/ext_isaacsim_robot_policy_example.html):

   - Create an empty stage.
   - Open the example menu: `Robotics Examples > POLICY > Quadruped`.
   - Press **LOAD** to open the scene.
   - Control the robot using the keyboard.

3. **Camera Setup**\
   Add a camera using the GUI or through code.\
   [Camera Sensor Documentation](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/features/sensors_simulation/isaac_sim_sensors_camera.html)

   **Tips:**

   - For cameras included in USD, simply set the prim.
   - To move the camera with the robot, it should be placed under the root; for Spot, this means under the body.

4. **Particle Object Simulation**\
   [Official Particle Simulation Demo](https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/physics-particles.html)

   **Tips:**

   - For elastic objects (e.g., foam, rubber, and toys), use the finite element model.
   - For deformable objects (e.g., soil, sand, and cloth), use the particle model.
   - GUI setup is easier, but many parameters can only be adjusted via code.

## Small Demo

This demo allows control of Spot's body and arm using the keyboard, with observations provided by an additional front-mounted RGB camera.

### Arm Movement

- **Move along x-axis**: `W` / `S`
- **Move along y-axis**: `A` / `D`
- **Move along z-axis**: `Z` / `X`

### Base Movement

- **Move forward/backward**: `↑` / `↓` (Up/Down Arrow)
- **Move left/right**: `←` / `→` (Left/Right Arrow)
- **Rotate along z-axis**: `M` / `N`

### Running the Teleoperation Script

Isaac Lab installation is required to run the teleoperation script for the Spot robot. Use the following command:

```
/path-to-isaac/isaac-sim-4.2.0/python.sh /path-to-code/Isaac_spot_tutorials/script/teleop_se3_agent.py
```

### Other Interesting tutorials maybe help
- [Articulation Controller](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_simulation/articulation_controller.html)
- [Kinematics Solver](https://docs.isaacsim.omniverse.nvidia.com/latest/manipulators/manipulators_lula_kinematics.html)


