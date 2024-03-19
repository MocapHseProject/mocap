/*
package main.java.Animation.Animator;

import main.java.MyMath.*;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

public class SimpleJoint {

    private final Vector position;

    private final Quaternion rotation;

    public Matrix getLocalTrans() {
        Matrix matrix = new Matrix();
        matrix.translate(position);
        matrix = Matrix.multiplicate(matrix, rotation.toRotationMatrix());
        return matrix;
    }

    public SimpleJoint(Vector position, Quaternion rotation) {
        this.position = position;
        this.rotation = rotation;
    }

    @Contract(value = "_, _, _ -> new", pure = true)
    public static @NotNull Vector interpolate(@NotNull Vector start, @NotNull Vector end, float progression) {
		return new Vector(
                start.x + (end.x - start.x) * progression,
                start.y + (end.y - start.y) * progression,
                start.z + (end.z - start.z) * progression);
	}

    public static @NotNull SimpleJoint interpolate(@NotNull SimpleJoint frameA, @NotNull SimpleJoint frameB, float progression) {
        Vector pos = interpolate(frameA.position, frameB.position, progression);
        Quaternion rot = Quaternion.interpolate(frameA.rotation, frameB.rotation, progression);
        return new SimpleJoint(pos, rot);
    }

}
*/