package main.java.Animation.Animator;

import com.jogamp.opengl.GL3;
import com.jogamp.opengl.util.texture.Texture;
import main.java.Buffer.VertexArray;
import main.java.MyMath.Matrix;
import com.jogamp.opengl.util.GLBuffers;
import org.jetbrains.annotations.NotNull;

public class AnimatorModel {

    private final Joint rootJoint;

    private final int jointCount;
    private final VertexArray model;

    private final Texture texture;

    private final Animator animator;

    public AnimatorModel(VertexArray model, Texture texture, @NotNull Joint joint, int count) {
        this.model = model;
        this.texture = texture;
        this.rootJoint = joint;
        this.jointCount = count;
        this.animator = new Animator(this);
        joint.calcInvBindTrans(new Matrix());
    }

    public void doAnimation(Anim animation) {
        animator.doAnim(animation);
    }

    public void delete() {
        model.destroy((GL3) new GLBuffers());
        texture.destroy((GL3) new GLBuffers());
    }

    public VertexArray getModel() {
        return model;
    }


    public void update() {
        animator.update();
    }

    public Matrix[] getJointTransforms() {
        Matrix[] jointMatrices = new Matrix[jointCount];
        addJointsToArray(rootJoint, jointMatrices);
        return jointMatrices;
    }

    public Texture getTexture() {
        return texture;
    }

    public Joint getRootJoint() {
        return rootJoint;
    }

    private void addJointsToArray(@NotNull Joint headJoint, Matrix @NotNull [] jointMatrices) {
        jointMatrices[headJoint.index] = headJoint.getAnimTrans();
        for (Joint childJoint : headJoint.jointList) {
            addJointsToArray(childJoint, jointMatrices);
        }
    }

}
