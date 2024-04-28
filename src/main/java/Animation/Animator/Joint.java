/*
package main.java.Animation.Animator;

import main.java.MyMath.Matrix;

import java.util.ArrayList;
import java.util.List;

public class Joint {
    public final int index;

    private final Matrix localBindTrans;

    public final String name;

    public final List<Joint> jointList = new ArrayList<>();

    private Matrix animTrans = new Matrix();

    private final Matrix invBindTrans = new Matrix();

    public Joint(String name) {
        this(0, name, new Matrix());
    }

    public Joint(int index, String name, Matrix bindLocalTrans) {
        this.index = index;
        this.name = name;
        this.localBindTrans = bindLocalTrans;
    }

    protected Joint getJoint(int index) {
        if (index >= jointList.size()) {
            return null;
        }
        return jointList.get(index);
    }

    protected void calcInvBindTrans(Matrix parentBindTrans) {
        Matrix bindTransform = Matrix.multiplicate(parentBindTrans, localBindTrans);
        Matrix.invert(bindTransform, invBindTrans);
        for (Joint child : jointList) {
            child.calcInvBindTrans(bindTransform);
        }
    }


    public void addJoint(Joint child) {
        this.jointList.add(child);
    }

    protected List<Joint> getJoints() {
        return jointList;
    }

    public Matrix getAnimTrans() {
        return animTrans;
    }

    public void setAnimTrans(Matrix animationTransform) {
        this.animTrans = animationTransform;
    }

    public Matrix getInvBindTrans() {
        return invBindTrans;
    }
}
*/
