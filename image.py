#!/usr/bin/env python

#publish:sending subscribe:receive

#packageのimport
import rospy #from ROS.org
from opencv_apps.msg import RotatedRectStamped #from ROS.org
from image_view2.msg import ImageMarker2 #from ROS.org
from geometry_msgs.msg import Point #from ROS.org

#関数の定義
def cb(msg):
print msg.rect #矩形の表示
marker = ImageMarker2() #markerの定義
marker.type = 0 #markerの形を円形に設定 
marker.position = Point(msg.rect.center.x, msg.rect.center.y, 0) #markerの中心座標を矩形の中心に設定
pub.publish(marker) #markerを送信
rospy.init_node('client') #ノード名の宣言(ROS Masterとの通信の開始)
rospy.Subscriber('/camshift/track_box', RotatedRectStamped, cb) #(topic名, 型,callback関数)
pub = rospy.Publisher('/image_marker', ImageMarker2) #(topic名, 型, size)
rospy.spin() #node処理が終了するまで繰り返す
