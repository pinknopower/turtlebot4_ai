import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from vision_msgs.msg import Detection3DArray
import cv2
import numpy as np


class Detection3DViewer(Node):
    labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
		    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    def __init__(self):
        super().__init__('detection3d_viewer')
        self.image = None
        self.detections = []

        self.image_sub = self.create_subscription(
            CompressedImage,
            '/oak/rgb/image_rect/compressed',
            self.image_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection3DArray,
            '/oak/nn/spatial_detections',
            self.detection_callback,
            10
        )

        self.get_logger().info("Detection3D Viewer 已啟動")

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            raw_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            self.image = cv2.resize(raw_image, (300, 300), interpolation=cv2.INTER_AREA)
            self.draw_and_show()
        except Exception as e:
            self.get_logger().error(f" 解壓縮影像失敗: {e}")

    def detection_callback(self, msg):
        self.detections = msg.detections

    def draw_and_show(self):
        if self.image is None:
            return

        img = self.image.copy()

        for det in self.detections:
            if not det.results:
                continue

            hypothesis = det.results[0].hypothesis
            class_id_str = hypothesis.class_id
            score = hypothesis.score
            try:
                class_id_int = int(class_id_str)
                class_name = self.labelMap[class_id_int] if 0 <= class_id_int < len(self.labelMap) else f"id:{class_id_str}"
            except:
                class_name = f"id:{class_id_str}"

            label = f"{class_name} ({score:.2f})"

            cx = int(det.bbox.center.position.x)
            cy = int(det.bbox.center.position.y)
            w = int(det.bbox.size.x)
            h = int(det.bbox.size.y)

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            
            z = det.results[0].pose.pose.position.z
            font_scale = max(0.1, min(0.5, h / 100))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(img, class_name, (x1 + 3, y1 + int(25 * font_scale)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
            
            info_text = f"Score:{score:.2f} Z:{z:.2f}m"
            cv2.putText(img, info_text, (x1 + 3, y1 + int(50 * font_scale)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), 2)
            
        zoom=2.0
        img = cv2.resize(img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Detection3D (Zoom x2)", img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = Detection3DViewer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
