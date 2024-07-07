#! /usr/bin/env python3.10
import rospy
import actionlib
import frida_vision_interfaces.msg
import torch
import cv2
from PIL import Image as PILImage
from moondream import Moondream, detect_device, LATEST_REVISION
from cv_bridge import CvBridge
from transformers import TextIteratorStreamer, AutoTokenizer

# adding Folder_2 to the system path
class MoondreamAction(object):
     
# create messages that are used to publish feedback/result
    _result = frida_vision_interfaces.msg.Moondream2Result()

    def __init__(self, name):
        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, frida_vision_interfaces.msg.Moondream2Action, execute_cb=self.execute_cb, auto_start = False)
        self._as.start()
        device, dtype = detect_device()
        if device != torch.device("cpu"):
            rospy.loginfo("Using device: %s" %(device))
            rospy.loginfo("Setting Up Model")
        model_id = "vikhyatk/moondream2"
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=LATEST_REVISION)
        self._moondream = Moondream.from_pretrained(
            model_id,
            revision=LATEST_REVISION,
            torch_dtype=dtype,
        ).to(device=device)
        self._moondream.eval()
        rospy.loginfo("Setup finished")
    

    def execute_cb(self, goal):
        success = True
        rospy.loginfo('%s: Executing, resolving frame with prompt "%s" ' % (self._action_name, goal.prompt))
        
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(goal.frame, desired_encoding='passthrough')
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        image_embeds = self._moondream.encode_image(pil_image)
        
        if goal.prompt != "":
            answer = self._moondream.answer_question(image_embeds, goal.prompt, self._tokenizer)
        else:
            answer = "";
            
        if success:
            self._result.response = answer
            rospy.loginfo('%s: Completed' % self._action_name)
            self._as.set_succeeded(self._result)

if __name__ == '__main__':
    rospy.init_node('moondream')
    server = MoondreamAction(rospy.get_name())
    rospy.spin()

