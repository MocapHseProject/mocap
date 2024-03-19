package main.java.Animation.Animator;

import main.java.MyMath.Matrix;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.util.HashMap;
import java.util.Map;

public class Animator {

    private float animationTime = 0;

    private float deltaTime = 0;
    private final AnimatorModel model;

    private Anim currentAnimation;

    public Animator(AnimatorModel entity) {
        this(entity, 0.1F);
    }

    public Animator(AnimatorModel model, float deltaTime) {
        this.model = model;
        this.deltaTime = deltaTime;
    }

    public void addTime(float deltaTime) {
        this.deltaTime += deltaTime;
    }

    public float getTime() {
        return deltaTime;
    }

    public void doAnim(Anim animation) {
        this.animationTime = 0;
        this.currentAnimation = animation;
    }

    @Contract(" -> new")
    private KeyFrame @NotNull [] getPrNeFrames() {
        KeyFrame[] allFrames = currentAnimation.getKeyFrames();
        KeyFrame previousFrame = allFrames[0];
        KeyFrame nextFrame = allFrames[0];
        for (int i = 1; i < allFrames.length; i++) {
            nextFrame = allFrames[i];
            if (nextFrame.getTimeStamp() > animationTime) {
                break;
            }
            previousFrame = allFrames[i];
        }
        return new KeyFrame[] { previousFrame, nextFrame };
    }

    private void applyPosToJoints(@NotNull Map<String, Matrix> currentPose, @NotNull Joint joint, Matrix parentTransform) {
        Matrix currentLocalTransform = currentPose.get(joint.name);
        Matrix currentTransform = Matrix.multiplicate(parentTransform, currentLocalTransform);
        for (Joint childJoint : joint.jointList) {
            applyPosToJoints(currentPose, childJoint, currentTransform);
        }
        currentTransform = Matrix.multiplicate(currentTransform, joint.getInvBindTrans());
        joint.setAnimTrans(currentTransform);
    }
    private @NotNull Map<String, Matrix> calcCurrentAnimPos() {
        KeyFrame[] frames = getPrNeFrames();
        float progression = calcProgress(frames[0], frames[1]);
        return interpolatePos(frames[0], frames[1], progression);
    }


    private float calcProgress(@NotNull KeyFrame previousFrame, @NotNull KeyFrame nextFrame) {
        float totalTime = nextFrame.getTimeStamp() - previousFrame.getTimeStamp();
        float currentTime = animationTime - previousFrame.getTimeStamp();
        return currentTime / totalTime;
    }

    @NotNull
    private Map<String, Matrix> interpolatePos(@NotNull KeyFrame previousFrame, KeyFrame nextFrame, float progression) {
        Map<String, Matrix> currentPose = new HashMap<>();
        for (String jointName : previousFrame.getJointKeyFrames().keySet()) {
            SimpleJoint previousTransform = previousFrame.getJointKeyFrames().get(jointName);
            SimpleJoint nextTransform = nextFrame.getJointKeyFrames().get(jointName);
            SimpleJoint currentTransform = SimpleJoint.interpolate(previousTransform, nextTransform, progression);
            currentPose.put(jointName, currentTransform.getLocalTrans());
        }
        return currentPose;
    }

    public void update() {
        if (currentAnimation == null) {
            return;
        }
        increaseAnimTime();
        Map<String, Matrix> currentPose = calcCurrentAnimPos();
        applyPosToJoints(currentPose, model.getRootJoint(), new Matrix());
    }

    private void increaseAnimTime() {
        animationTime += deltaTime;
        //animationTime += DisplayManager.getFrameTime();
        if (animationTime > currentAnimation.getLength()) {
            this.animationTime %= currentAnimation.getLength();
        }
    }
}
