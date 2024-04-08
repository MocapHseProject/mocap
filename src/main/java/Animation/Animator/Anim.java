package main.java.Animation.Animator;

public class Anim {

    private final float length;

    private final KeyFrame[] keyFrames;

    public Anim(float length, KeyFrame[] frames) {
        this.keyFrames = frames;
        this.length = length;
    }

    public float getLength() {
        return length;
    }

    protected KeyFrame getKeyFrame(int index) {
        if (index >= keyFrames.length) {
            return null;
        }
        return keyFrames[index];
    }

    protected boolean putKeyFrame(int index, KeyFrame keyFrame) {
        if (index >= keyFrames.length) {
            return false;
        }
        keyFrames[index] = keyFrame;
        return true;
    }

    public KeyFrame[] getKeyFrames() {
        return keyFrames;
    }
}
