#! /usr/bin/env python3
import os
from contextlib import suppress

import cv2
import edge_tpu_lane_detection.msg
import numpy as np
import rospkg
from edge_ai_Lanedetection.edge_tpu_inference import EdgeTPUInference

import cv_bridge
import geometry_msgs.msg
import rospy
import sensor_msgs.msg
from ros_base import node_base
from timer.timer import Timer
from transformation.birds_eyed_view import BirdseyedviewTransformation
from transformation.coordinate_transform import CoordinateTransform


class LaneDetectionAiNode(node_base.NodeBase):
    """ROS lane detection ai node."""

    def __init__(self):
        super().__init__(
            name="ros_lane_detection_ai_node", log_level=rospy.INFO
        )  # Name can be overwritten in launch file

        self.debug_image_publisher = None
        self.camera_image_subscriber = None
        self.result_publisher = None

        # Initialise param object to itself so that IDE will suggest it. Keep in mind, that the parameters
        # itself are not known to IDE and therefore won't show up in the autocompletion.
        self.param = self.param
        self.package_path = rospkg.RosPack().get_path('edge_tpu_lane_detection')

        self.cv_bridge = cv_bridge.CvBridge()
        self.coord_trans = CoordinateTransform()
        self.birds_eyed = BirdseyedviewTransformation()
        self.model = EdgeTPUInference(self.package_path,
                                          os.path.join(self.package_path, self.param.model_config_path))

        # Don't pass any parameter if no function should be called periodically.
        # This way it is in a spinning mode.
        self.run()

    def start(self):
        """Gets called automatically by node_base.NodeBase when the node is started."""

        self.result_publisher = rospy.Publisher(
            self.param.result_publisher,
            edge_tpu_lane_detection.msg.LaneDetectionResult,
            queue_size=1
        )

        self.camera_image_subscriber = rospy.Subscriber(
            self.param.camera_image_subscriber,
            sensor_msgs.msg.Image,
            callback=self.camera_image_cb,
        )

        # Only register publisher and subscriber if the debug is turned on
        if self.param.debug:
            self.debug_image_publisher = rospy.Publisher(
                self.param.debug_image_publisher,
                sensor_msgs.msg.Image,
                queue_size=1
            )

        super().start()

    def stop(self):
        """Gets automatically called by node_base.NodeBase when the node is shut down."""

        # Attribute errors can occur if the node has not been completely started
        # before shutting down.
        with suppress(AttributeError):
            self.result_publisher.unregister()
            self.camera_image_subscriber.unregister()

            # Only unregister publisher if the debug was turned on
            if self.param.debug:
                self.debug_image_publisher.unregister()

        super().stop()

    def camera_image_cb(self, camera_image_msg: sensor_msgs.msg.Image):
        """Receives new camera image as a callback function from ROS."""

        # Do not proceed if the node is not set to be active.
        if not self.param.active:
            return

        timer = Timer(name="total", filter_strength=40)
        timer.start()

        # Get image from the example message
        camera_image = self.cv_bridge.imgmsg_to_cv2(camera_image_msg, desired_encoding='8UC1')
        camera_image = cv2.cvtColor(camera_image, cv2.COLOR_GRAY2RGB)
        camera_image = cv2.convertScaleAbs(camera_image, alpha=4, beta=0.5)

        # # Get result
        with Timer(name="prediction", filter_strength=40):
            result = self.model.predict(camera_image)
        #     left_lane = self.coord_trans.camera_to_world(result[0]) if result[0] is not None else np.asarray([])
        #     center_lane = self.coord_trans.camera_to_world(result[1]) if result[1] is not None else np.asarray([])
        #     right_lane = self.coord_trans.camera_to_world(result[2]) if result[2] is not None else np.asarray([])

        # with Timer(name="polyfit", filter_strength=40):
            # # Fit lane to a polynomial function of 2nd degree. In our representation of the polynomial function,
            # # the x and y are swapped: x = a * y² + b * y + c
            # left_lane_coeff = None
            # center_lane_coeff = None
            # right_lane_coeff = None
            # if len(left_lane) > 0:
            #     left_lane_coeff = np.asarray(np.polyfit(left_lane.T[1], left_lane.T[0], 2))
            # if len(center_lane) > 0:
            #     center_lane_coeff = np.asarray(np.polyfit(center_lane.T[1], center_lane.T[0], 2))
            # if len(right_lane) > 0:
            #     right_lane_coeff = np.asarray(np.polyfit(right_lane.T[1], right_lane.T[0], 2))
            #
            # # Build the two driving lanes (left and right) from the center lane and the left/right lane
            # driving_lane_left_coeff = [0, 0, 0]
            # driving_lane_right_coeff = [0, 0, 0]

            # if right_lane_coeff is not None and center_lane_coeff is not None:
            #     driving_lane_right_coeff = np.average([right_lane_coeff, center_lane_coeff], axis=0)
            # if center_lane_coeff is not None and left_lane_coeff is not None:
            #     driving_lane_left_coeff = np.average([center_lane_coeff, left_lane_coeff], axis=0)

        # # Check if this node is still active before publishing results
        # if self.param.active:
        #     # Generate vec3 arrays for the output message
        #     left_lane_vec3s = [geometry_msgs.msg.Vector3(x=coord[0], y=coord[1], z=coord[2]) for coord in left_lane]
        #     center_lane_vec3s = [geometry_msgs.msg.Vector3(x=coord[0], y=coord[1], z=coord[2]) for coord in center_lane]
        #     right_lane_vec3s = [geometry_msgs.msg.Vector3(x=coord[0], y=coord[1], z=coord[2]) for coord in right_lane]
        #
        #     # Publish result
        #     rospy.logdebug(f"Driving lane left: {driving_lane_left_coeff}")
        #     rospy.logdebug(f"Driving lane right: {driving_lane_right_coeff}")
        #     self.result_publisher.publish(
        #         lane_detection_ai.msg.LaneDetectionResult(
        #             left=lane_detection_ai.msg.Lane(left_lane_vec3s, len(left_lane_vec3s) > 0),
        #             center=lane_detection_ai.msg.Lane(center_lane_vec3s, len(center_lane_vec3s) > 0),
        #             right=lane_detection_ai.msg.Lane(right_lane_vec3s, len(right_lane_vec3s) > 0),
        #             trajectory_left=geometry_msgs.msg.Vector3(x=driving_lane_left_coeff[0],
        #                                                       y=driving_lane_left_coeff[1],
        #                                                       z=driving_lane_left_coeff[2]),
        #             trajectory_right=geometry_msgs.msg.Vector3(x=driving_lane_right_coeff[0],
        #                                                        y=driving_lane_right_coeff[1],
        #                                                        z=driving_lane_right_coeff[2]),
        #         )
        #     )

            with Timer(name="debug_image", filter_strength=40):
                if self.param.debug:
                    debug_image = camera_image
                    if len(result[0]) > 0:
                        for coord in result[0].astype(int):
                            cv2.circle(debug_image, coord, 5, (255, 0, 0), -1)
                    if len(result[1]) > 0:
                        for coord in result[1].astype(int):
                            cv2.circle(debug_image, coord, 5, (0, 255, 0), -1)
                    if len(result[2]) > 0:
                        for coord in result[2].astype(int):
                            cv2.circle(debug_image, coord, 5, (0, 0, 255), -1)

                    # # Draw the trajectories
                    # self._draw_trajectory(debug_image, driving_lane_left_coeff)
                    # self._draw_trajectory(debug_image, driving_lane_right_coeff)

                    self.debug_image_publisher.publish(
                        self.cv_bridge.cv2_to_imgmsg(debug_image, encoding='rgb8')
                    )

        timer.stop()
        timer.print()

    # def _draw_trajectory(self, img: np.ndarray, driving_lane_coeff: np.ndarray) -> np.ndarray:
    #     """Draws the trajectory on the given image.
    #
    #     Args:
    #         img (np.ndarray): Image to draw the trajectory on.
    #         driving_lane_coeff (np.ndarray): Coefficients of the polynomial function of the driving lane.
    #
    #     Returns:
    #         np.ndarray: Image with the trajectory drawn on it.
    #     """
    #     # create lane points for fitted parabola
    #     row_points = np.asarray(range(-500, 2500, 10))
    #     col_points = driving_lane_coeff[0] * row_points ** 2 + driving_lane_coeff[1] * row_points + driving_lane_coeff[2]
    #     fitted_lane_points_world = np.vstack([col_points, row_points, np.zeros(300)]).T
    #
    #     # transform into bev
    #     fitted_lane_points_bev = self.coord_trans.world_to_bird(fitted_lane_points_world)
    #     fitted_lane_points_bev = fitted_lane_points_bev[(fitted_lane_points_bev >= 750).all(axis=1)]
    #
    #     for coord in fitted_lane_points_bev.astype(int):
    #         cv2.circle(img, coord, 2, (255, 255, 0), -1)
    #
    #     return img


if __name__ == "__main__":
    # Start the node
    with suppress(rospy.ROSInterruptException):
        LaneDetectionAiNode()
