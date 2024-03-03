package main.java.MyMath;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

public class Quaternion {

    private float x, y, z, w;

    public Quaternion(float x, float y, float z, float w) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = w;
        normalize();
    }

    public void normalize() {
        float sqrt = (float) Math.sqrt(x * x + y * y + z * z + w * w);
        x /= sqrt;
        y /= sqrt;
        z /= sqrt;
        w /= sqrt;
    }

    @Contract(pure = true)
    public static float scalarProduct(@NotNull Quaternion a, @NotNull Quaternion b) {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    public static @NotNull Quaternion interpolate(Quaternion a, Quaternion b, float blend) {
        Quaternion result;
        float blendI = 1f - blend;
        if (scalarProduct(a, b) < 0) {
            result = new Quaternion(
                    blendI * a.x + blend * -b.x,
                    blendI * a.y + blend * -b.y,
                    blendI * a.z + blend * -b.z,
                    blendI * a.w + blend * -b.w
            );
        } else {
            result = new Quaternion(
                    blendI * a.x + blend * b.x,
                    blendI * a.y + blend * b.y,
                    blendI * a.z + blend * b.z,
                    blendI * a.w + blend * b.w
            );
        }
        result.normalize();
        return result;
    }

    public Matrix toRotationMatrix() {
        final float xy = x * y;
        final float xz = x * z;
        final float xw = x * w;
        final float yz = y * z;
        final float yw = y * w;
        final float zw = z * w;
        final float xSquared = x * x;
        final float ySquared = y * y;
        final float zSquared = z * z;
        return new Matrix(
                1 - 2 * (ySquared + zSquared), 2 * (xy - zw), 2 * (xz + yw), 0,
                2 * (xy + zw), 1 - 2 * (xSquared + zSquared), 2 * (yz - xw), 0,
                2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xSquared + ySquared), 0,
                0, 0, 0, 1
        );
    }
}
