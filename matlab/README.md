# Quadrotor Simulator for WPI-RBE502 (Robot Control)

## Overview

This repository hosts the simulator for the course project of WPI-RBE502 (Robot Control). The simulator is designed to facilitate the development and testing of control algorithms for a quadrotor tasked with monitoring a restricted airspace. The primary objective is to design a controller that enables the quadrotor to autonomously detect, follow, and capture an unidentified Unmanned Aerial Vehicle (UAV) that enters the airspace, and then return to its nest.

## Project Objective

The project focuses on creating a robust control system for a quadrotor, leveraging the simulator provided in this repository. Students will engage in designing and implementing a controller that meets specific mission specifications, including UAV detection, pursuit, capture, and return to base operations within a defined airspace.

## Simulator Features

- Quadrotor dynamics simulation (currently based on Euler angles).
- Scenario setup for UAV intrusion detection and interception.
- Customizable parameters for control algorithm testing and optimization.

## Getting Started

To use this simulator, clone the repository and follow the setup instructions in the documentation.

### Examples

- **example_01.m**: Demonstrates how to use the Quadrotor class independently.
- **example_02.m**: Shows a simple usage of the UAV class.
- **example_03.m**: Integrates everything using the Simulator class. This example demonstrates the full capabilities of the simulator, including the zero-order-hold approximation of the control function, which simulates the effect of implementing a control function via a microprocessor based on samples.

## Contributing

Contributions to enhance the simulator or fix issues are welcome. Please follow the standard pull request process for contributions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
