#!/usr/bin/env python3

from math import floor
from threading import Lock, Thread
from time import sleep

import rclpy

from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import BatteryState
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator

BATTERY_HIGH = 0.95
BATTERY_LOW = 0.2  
BATTERY_CRITICAL = 0.1  


class BatteryMonitor(Node):

    def __init__(self, lock):
        super().__init__('battery_monitor')

        self.lock = lock

        
        self.battery_state_subscriber = self.create_subscription(
            BatteryState,
            'battery_state',
            self.battery_state_callback,
            qos_profile_sensor_data)

  
    def battery_state_callback(self, batt_msg: BatteryState):
        with self.lock:
            self.battery_percent = batt_msg.percentage

    def thread_function(self):
        executor = SingleThreadedExecutor()
        executor.add_node(self)
        executor.spin()


def main(args=None):
    rclpy.init(args=args)

    lock = Lock()
    battery_monitor = BatteryMonitor(lock)

    navigator = TurtleBot4Navigator()
    battery_percent = None
    position_index = 0

    thread = Thread(target=battery_monitor.thread_function, daemon=True)
    thread.start()

    # Start on dock
    if not navigator.getDockedStatus():
        navigator.info('Docking before intialising pose')
        navigator.dock()

    # Set initial pose
    initial_pose = navigator.getPoseStamped([0.0, 0.0], TurtleBot4Directions.NORTH)
    navigator.setInitialPose(initial_pose)

    # Wait for Nav2
    navigator.waitUntilNav2Active()

    # Undock
    navigator.undock()

    
    goal_pose = []
    goal_pose.append(navigator.getPoseStamped([-1.33696, -7.4321], TurtleBot4Directions.NORTH))
    goal_pose.append(navigator.getPoseStamped([23.5573, -5.1291], TurtleBot4Directions.WEST))
    goal_pose.append(navigator.getPoseStamped([23.38397, -3.361244], TurtleBot4Directions.SOUTH))
    goal_pose.append(navigator.getPoseStamped([-0.08597, -6.97], TurtleBot4Directions.WEST))
    goal_pose.append(navigator.getPoseStamped([-1.72589, 26.4496], TurtleBot4Directions.NORTH))
    goal_pose.append(navigator.getPoseStamped([13.5875, 27.0957], TurtleBot4Directions.SOUTH))
    goal_pose.append(navigator.getPoseStamped([-1.72589, 26.4496], TurtleBot4Directions.NORTH))
    goal_pose.append(navigator.getPoseStamped([-1.0, 0.0], TurtleBot4Directions.NORTH))




    while True:
        with lock:
            battery_percent = battery_monitor.battery_percent

        if (battery_percent is not None):
            navigator.info(f'Battery is at {(battery_percent*100):.2f}% charge')

            # Check battery charge level
            if (battery_percent < BATTERY_CRITICAL):
                navigator.error('Battery critically low. Charge or power down')
                break
            elif (battery_percent < BATTERY_LOW):
                # Go near the dock
                navigator.info('Docking for charge')
                navigator.startToPose(navigator.getPoseStamped([-1.0, 1.0],
                                      TurtleBot4Directions.EAST))
                navigator.dock()

                if not navigator.getDockedStatus():
                    navigator.error('Robot failed to dock')
                    break


                navigator.info('Charging...')
                battery_percent_prev = 0
                while (battery_percent < BATTERY_HIGH):
                    sleep(15)
                    battery_percent_prev = floor(battery_percent*100)/100
                    with lock:
                        battery_percent = battery_monitor.battery_percent

                    # Print charge level every time it increases a percent
                    if battery_percent > (battery_percent_prev + 0.01):
                        navigator.info(f'Battery is at {(battery_percent*100):.2f}% charge')

                # Undock
                navigator.undock()
                position_index = 0

            else:
                # Navigate to next position
                navigator.startToPose(goal_pose[position_index])

                position_index = position_index + 1
                if position_index >= len(goal_pose):
                    position_index = 0

    battery_monitor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
