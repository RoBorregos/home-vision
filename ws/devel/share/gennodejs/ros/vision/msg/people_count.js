// Auto-generated. Do not edit!

// (in-package vision.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class people_count {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.detected_people = null;
      this.people_standing = null;
      this.people_sitting = null;
      this.people_raising_hand = null;
      this.people_pointing = null;
    }
    else {
      if (initObj.hasOwnProperty('detected_people')) {
        this.detected_people = initObj.detected_people
      }
      else {
        this.detected_people = 0;
      }
      if (initObj.hasOwnProperty('people_standing')) {
        this.people_standing = initObj.people_standing
      }
      else {
        this.people_standing = 0;
      }
      if (initObj.hasOwnProperty('people_sitting')) {
        this.people_sitting = initObj.people_sitting
      }
      else {
        this.people_sitting = 0;
      }
      if (initObj.hasOwnProperty('people_raising_hand')) {
        this.people_raising_hand = initObj.people_raising_hand
      }
      else {
        this.people_raising_hand = 0;
      }
      if (initObj.hasOwnProperty('people_pointing')) {
        this.people_pointing = initObj.people_pointing
      }
      else {
        this.people_pointing = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type people_count
    // Serialize message field [detected_people]
    bufferOffset = _serializer.int64(obj.detected_people, buffer, bufferOffset);
    // Serialize message field [people_standing]
    bufferOffset = _serializer.int64(obj.people_standing, buffer, bufferOffset);
    // Serialize message field [people_sitting]
    bufferOffset = _serializer.int64(obj.people_sitting, buffer, bufferOffset);
    // Serialize message field [people_raising_hand]
    bufferOffset = _serializer.int64(obj.people_raising_hand, buffer, bufferOffset);
    // Serialize message field [people_pointing]
    bufferOffset = _serializer.int64(obj.people_pointing, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type people_count
    let len;
    let data = new people_count(null);
    // Deserialize message field [detected_people]
    data.detected_people = _deserializer.int64(buffer, bufferOffset);
    // Deserialize message field [people_standing]
    data.people_standing = _deserializer.int64(buffer, bufferOffset);
    // Deserialize message field [people_sitting]
    data.people_sitting = _deserializer.int64(buffer, bufferOffset);
    // Deserialize message field [people_raising_hand]
    data.people_raising_hand = _deserializer.int64(buffer, bufferOffset);
    // Deserialize message field [people_pointing]
    data.people_pointing = _deserializer.int64(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 40;
  }

  static datatype() {
    // Returns string type for a message object
    return 'vision/people_count';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'dd5d4b83e54fd0abe744ecf17478c695';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int64 detected_people
    int64 people_standing
    int64 people_sitting
    int64 people_raising_hand
    int64 people_pointing
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new people_count(null);
    if (msg.detected_people !== undefined) {
      resolved.detected_people = msg.detected_people;
    }
    else {
      resolved.detected_people = 0
    }

    if (msg.people_standing !== undefined) {
      resolved.people_standing = msg.people_standing;
    }
    else {
      resolved.people_standing = 0
    }

    if (msg.people_sitting !== undefined) {
      resolved.people_sitting = msg.people_sitting;
    }
    else {
      resolved.people_sitting = 0
    }

    if (msg.people_raising_hand !== undefined) {
      resolved.people_raising_hand = msg.people_raising_hand;
    }
    else {
      resolved.people_raising_hand = 0
    }

    if (msg.people_pointing !== undefined) {
      resolved.people_pointing = msg.people_pointing;
    }
    else {
      resolved.people_pointing = 0
    }

    return resolved;
    }
};

module.exports = people_count;
