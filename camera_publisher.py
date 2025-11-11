import gi
import numpy as np
import cv2 as cv
import queue
import signal
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import socket
from queue import Queue
import threading
import time

gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GLib


class GStreamerPipeline(Node):
    def __init__(self):
        super().__init__("FUCKFUCKFUCK")

def main(args = None):
    try:
        Gst.init(None)
        rclpy.init(args=args)
        print(rclpy.ok())
        cam = GStreamerPipeline()
        
        while True:
            print("WHAT")
    except Exception as error:
        print(error)

if __name__ == "__main__":
    main()