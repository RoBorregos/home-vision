#! /usr/bin/env python3.10

import rospy
import actionlib
import moondream_msgs.msg

class MoondreamAction(object):
    # create messages that are used to publish feedback/result
    _feedback = moondream_msgs.msg.moondreamFeedback()
    _result = moondream_msgs.msg.moondreamResult()

    def __init__(self, name):
        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, moondream_msgs.msg.moondreamAction, execute_cb=self.execute_cb, auto_start = False)
        self._as.start()
      
    def execute_cb(self, goal):
        # helper variables
        success = True
        
        # append the seeds for the fibonacci sequence
        self._feedback.percentage = 5.0
        
        # publish info to the console for the user
        rospy.loginfo('%s: Executing, creating fibonacci sequence of order %i with seeds %i' % (self._action_name, goal.order, self._feedback.percentage))
        
        # start executing the action
        #for i in range(1, goal.order):
            # check that preempt has not been requested by the client
         #   if self._as.is_preempt_requested():
          #      rospy.loginfo('%s: Preempted' % self._action_name)
           #     self._as.set_preempted()
            #    success = False
             #   break
           # self._feedback.sequence.append(self._feedback.sequence[i] + self._feedback.sequence[i-1])
            # publish the feedback
        self._as.publish_feedback(self._feedback)
            # this step is not necessary, the sequence is computed at 1 Hz for demonstration purposes
            #r.sleep()
          
        if success:
            self._result.count = 6
            rospy.loginfo('%s: Succeeded' % self._action_name)
            self._as.set_succeeded(self._result)
        
if __name__ == '__main__':
    rospy.init_node('counter')
    server = MoondreamAction(rospy.get_name())
    rospy.spin()
