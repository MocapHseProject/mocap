/*
package main.java.Animation.Animator;

import java.util.HashMap;
import java.util.Map;

public class KeyFrame {
    private final float delta;
    private final Map<String, SimpleJoint> positions;

    public KeyFrame(float timeStamp, Map<String, SimpleJoint> jointKeyFrames) {
        this.delta = timeStamp;
        this.positions = jointKeyFrames;
    }

    public SimpleJoint getJoint(String name) {
        return positions.get(name);
    }

    public KeyFrame(float timeStamp) {
        this.delta = timeStamp;
        this.positions = new HashMap<>();
    }

    public float getTimeStamp() {
        return delta;
    }

    public Map<String, SimpleJoint> getJointKeyFrames() {
        return positions;
    }

    public boolean putJoint(String name, SimpleJoint joint) {
        if (positions.containsKey(name)) {
            return false;
        }
        positions.put(name, joint);
        return true;
    }
}
*/