package main.java.MyMath;

import java.util.ArrayList;

public class Matrix {
    public float[] matrix;

    public final Vector getRight() {
        return new Vector(matrix[4 * 0 + 0], matrix[4 * 0 + 1], matrix[4 * 0 + 2]);
    }

    public final Vector getUp() {
        return new Vector(matrix[4 * 1 + 0], matrix[4 * 1 + 1], matrix[4 * 1 + 2]);
    }

    public final Vector getForward() {
        return new Vector(matrix[4 * 2 + 0], matrix[4 * 2 + 1], matrix[4 * 2 + 2]);
    }

    public final Vector getPosition() {
        return new Vector(matrix[4 * 3 + 0], matrix[4 * 3 + 1], matrix[4 * 3 + 2]);
    }

    public Matrix() {
        matrix = new float[16];
        for (int i = 0; i < 16; i++) {
            if (i % 4 == 0) {
                matrix[i] = 1F;
            } else {
                matrix[i] = 0F;
            }
        }
    }

    public Matrix(float value) {
        matrix = new float[16];
        for (int i = 0; i < 16; i++) {
            matrix[i] = value;
        }
    }

    public Matrix(ArrayList<Float> matrix_) {
        matrix = new float[16];
        if (matrix_.size() != 16) {
            return;
        }
        for (int i = 0; i < 16; i++) {
            matrix[i] = matrix_.get(i);
        }
    }

    public Matrix(float[] matrix_) {
        if (matrix_.length != 16) {
            return;
        }
        matrix = matrix_;
    }

    public Matrix(float a00, float a01, float a02, float a03,
                  float a10, float a11, float a12, float a13,
                  float a20, float a21, float a22, float a23,
                  float a30, float a31, float a32, float a33) {
        matrix = new float[16];

        matrix[0] = a00;
        matrix[1] = a01;
        matrix[2] = a02;
        matrix[3] = a03;

        matrix[4] = a10;
        matrix[5] = a11;
        matrix[6] = a12;
        matrix[7] = a13;

        matrix[8] = a20;
        matrix[9] = a21;
        matrix[10] = a22;
        matrix[11] = a23;

        matrix[12] = a30;
        matrix[13] = a31;
        matrix[14] = a32;
        matrix[15] = a33;
    }

    public Matrix(Vector right, Vector up, Vector forward, Vector origin) {
        new Matrix(right.x, right.y, right.z, 0F, up.x, up.y, up.z, 0F, forward.x, forward.y, forward.z, 0F, origin.x, origin.y, origin.z, 1F);
    }

    public float getDeterminant() {
        return matrix[4 * 0 + 0] * getMatrix3x3Determinant(matrix[4 * 1 + 1], matrix[4 * 1 + 2], matrix[4 * 1 + 3],
                matrix[4 * 2 + 1], matrix[4 * 2 + 2], matrix[4 * 2 + 3],
                matrix[4 * 3 + 1], matrix[4 * 3 + 2], matrix[4 * 3 + 3]) -
                matrix[4 * 0 + 1] * getMatrix3x3Determinant(matrix[4 * 1 + 0], matrix[4 * 1 + 2], matrix[4 * 1 + 3],
                        matrix[4 * 2 + 0], matrix[4 * 2 + 2], matrix[4 * 2 + 3],
                        matrix[4 * 3 + 0], matrix[4 * 3 + 2], matrix[4 * 3 + 3]) +
                matrix[4 * 0 + 2] * getMatrix3x3Determinant(matrix[4 * 1 + 0], matrix[4 * 1 + 1], matrix[4 * 1 + 3],
                        matrix[4 * 2 + 0], matrix[4 * 2 + 1], matrix[4 * 2 + 3],
                        matrix[4 * 3 + 0], matrix[4 * 3 + 1], matrix[4 * 3 + 3]) -
                matrix[4 * 0 + 3] * getMatrix3x3Determinant(matrix[4 * 1 + 0], matrix[4 * 1 + 1], matrix[4 * 1 + 2],
                        matrix[4 * 2 + 0], matrix[4 * 2 + 1], matrix[4 * 2 + 2],
                        matrix[4 * 3 + 0], matrix[4 * 3 + 1], matrix[4 * 3 + 2]);
    }

    public Matrix getTransposed() {
        return new Matrix(matrix[0], matrix[4], matrix[8], matrix[12],
                matrix[1], matrix[5], matrix[9], matrix[13],
                matrix[2], matrix[6], matrix[10], matrix[14],
                matrix[3], matrix[7], matrix[11], matrix[15]);
    }

    public Matrix transpose() {
        matrix[4 * 0 + 1] += matrix[4 * 1 + 0];
        matrix[4 * 1 + 0] = matrix[4 * 0 + 1] - matrix[4 * 1 + 0];
        matrix[4 * 0 + 1] -= matrix[4 * 1 + 0];

        matrix[4 * 0 + 2] += matrix[4 * 2 + 0];
        matrix[4 * 2 + 0] = matrix[4 * 0 + 2] - matrix[4 * 2 + 0];
        matrix[4 * 0 + 2] -= matrix[4 * 2 + 0];

        matrix[4 * 1 + 2] += matrix[4 * 2 + 1];
        matrix[4 * 2 + 1] = matrix[4 * 1 + 2] - matrix[4 * 2 + 1];
        matrix[4 * 12] -= matrix[4 * 2 + 1];

        matrix[4 * 0 + 3] += matrix[4 * 3 + 0];
        matrix[4 * 3 + 0] = matrix[4 * 0 + 3] - matrix[4 * 3 + 0];
        matrix[4 * 0 + 3] -= matrix[4 * 3 + 0];

        matrix[4 * 1 + 3] += matrix[4 * 3 + 1];
        matrix[4 * 3 + 1] = matrix[4 * 1 + 3] - matrix[4 * 3 + 1];
        matrix[4 * 1 + 3] -= matrix[4 * 3 + 1];

        matrix[4 * 2 + 3] += matrix[4 * 3 + 2];
        matrix[4 * 3 + 2] = matrix[4 * 2 + 3] - matrix[4 * 3 + 2];
        matrix[4 * 2 + 3] -= matrix[4 * 3 + 2];

        return this;
    }

    public static Matrix multiplicate(Matrix a, Matrix b) {
        Matrix result = new Matrix(0);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    result.matrix[4 * i + j] += a.matrix[4 * i + k] * b.matrix[4 * k + j];
                }
            }
        }
        return result;
    }

    public static float getMatrix3x3Determinant(float a00, float a01, float a02,
                                                float a10, float a11, float a12,
                                                float a20, float a21, float a22) {
        return a00 * (a11 * a22 - a12 * a21) -
                a01 * (a10 * a22 - a12 * a20) +
                a02 * (a10 * a21 - a11 * a20);
    }

    public Matrix inverse() {
        float determinant = getDeterminant();

        if (determinant != 0)
            return new Matrix(
                    getMatrix3x3Determinant(
                            matrix[4 * 1 + 1], matrix[4 * 1 + 2], matrix[4 * 1 + 3],
                            matrix[4 * 2 + 1], matrix[4 * 2 + 2], matrix[4 * 2 + 3],
                            matrix[4 * 3 + 1], matrix[4 * 3 + 2], matrix[4 * 3 + 3]) / determinant,
                    -getMatrix3x3Determinant(
                            matrix[4 * 0 + 1], matrix[4 * 0 + 2], matrix[4 * 0 + 3],
                            matrix[4 * 2 + 1], matrix[4 * 2 + 2], matrix[4 * 2 + 3],
                            matrix[4 * 3 + 1], matrix[4 * 3 + 2], matrix[4 * 3 + 3]) / determinant,
                    getMatrix3x3Determinant(
                            matrix[4 * 0 + 1], matrix[4 * 0 + 2], matrix[4 * 0 + 3],
                            matrix[4 * 1 + 1], matrix[4 * 1 + 2], matrix[4 * 1 + 3],
                            matrix[4 * 3 + 1], matrix[4 * 3 + 2], matrix[4 * 3 + 3]) / determinant,
                    -getMatrix3x3Determinant(
                            matrix[4 * 0 + 1], matrix[4 * 0 + 2], matrix[4 * 0 + 3],
                            matrix[4 * 1 + 1], matrix[4 * 1 + 2], matrix[4 * 1 + 3],
                            matrix[4 * 2 + 1], matrix[4 * 2 + 2], matrix[4 * 2 + 3]) / determinant,

                    -getMatrix3x3Determinant(
                            matrix[4 * 1 + 0], matrix[4 * 1 + 2], matrix[4 * 1 + 3],
                            matrix[4 * 2 + 0], matrix[4 * 2 + 2], matrix[4 * 2 + 3],
                            matrix[4 * 3 + 0], matrix[4 * 3 + 2], matrix[4 * 3 + 3]) / determinant,
                    getMatrix3x3Determinant(
                            matrix[4 * 0 + 0], matrix[4 * 0 + 2], matrix[4 * 0 + 3],
                            matrix[4 * 2 + 0], matrix[4 * 2 + 2], matrix[4 * 2 + 3],
                            matrix[4 * 3 + 0], matrix[4 * 3 + 2], matrix[4 * 3 + 3]) / determinant,
                    -getMatrix3x3Determinant(
                            matrix[4 * 0 + 0], matrix[4 * 0 + 2], matrix[4 * 0 + 3],
                            matrix[4 * 1 + 0], matrix[4 * 1 + 2], matrix[4 * 1 + 3],
                            matrix[4 * 3 + 0], matrix[4 * 3 + 2], matrix[4 * 3 + 3]) / determinant,
                    getMatrix3x3Determinant(
                            matrix[4 * 0 + 0], matrix[4 * 0 + 2], matrix[4 * 0 + 3],
                            matrix[4 * 1 + 0], matrix[4 * 1 + 2], matrix[4 * 1 + 3],
                            matrix[4 * 2 + 0], matrix[4 * 2 + 2], matrix[4 * 2 + 3]) / determinant,

                    getMatrix3x3Determinant(
                            matrix[4 * 1 + 0], matrix[4 * 1 + 1], matrix[4 * 1 + 3],
                            matrix[4 * 2 + 0], matrix[4 * 2 + 1], matrix[4 * 2 + 3],
                            matrix[4 * 3 + 0], matrix[4 * 3 + 1], matrix[4 * 3 + 3]) / determinant,
                    -getMatrix3x3Determinant(
                            matrix[4 * 0 + 0], matrix[4 * 0 + 1], matrix[4 * 0 + 3],
                            matrix[4 * 2 + 0], matrix[4 * 2 + 1], matrix[4 * 2 + 3],
                            matrix[4 * 3 + 0], matrix[4 * 3 + 1], matrix[4 * 3 + 3]) / determinant,
                    getMatrix3x3Determinant(
                            matrix[4 * 0 + 0], matrix[4 * 0 + 1], matrix[4 * 0 + 3],
                            matrix[4 * 1 + 0], matrix[4 * 1 + 1], matrix[4 * 1 + 3],
                            matrix[4 * 3 + 0], matrix[4 * 3 + 1], matrix[4 * 3 + 3]) / determinant,
                    -getMatrix3x3Determinant(
                            matrix[4 * 0 + 0], matrix[4 * 0 + 1], matrix[4 * 0 + 3],
                            matrix[4 * 1 + 0], matrix[4 * 1 + 1], matrix[4 * 1 + 3],
                            matrix[4 * 2 + 0], matrix[4 * 2 + 1], matrix[4 * 2 + 3]) / determinant,

                    -getMatrix3x3Determinant(
                            matrix[4 * 1 + 0], matrix[4 * 1 + 1], matrix[4 * 1 + 2],
                            matrix[4 * 2 + 0], matrix[4 * 2 + 1], matrix[4 * 2 + 2],
                            matrix[4 * 3 + 0], matrix[4 * 3 + 1], matrix[4 * 3 + 2]) / determinant,
                    getMatrix3x3Determinant(
                            matrix[4 * 0 + 0], matrix[4 * 0 + 1], matrix[4 * 0 + 2],
                            matrix[4 * 2 + 0], matrix[4 * 2 + 1], matrix[4 * 2 + 2],
                            matrix[4 * 3 + 0], matrix[4 * 3 + 1], matrix[4 * 3 + 2]) / determinant,
                    -getMatrix3x3Determinant(
                            matrix[4 * 0 + 0], matrix[4 * 0 + 1], matrix[4 * 0 + 2],
                            matrix[4 * 1 + 0], matrix[4 * 1 + 1], matrix[4 * 1 + 2],
                            matrix[4 * 3 + 0], matrix[4 * 3 + 1], matrix[4 * 3 + 2]) / determinant,
                    getMatrix3x3Determinant(
                            matrix[4 * 0 + 0], matrix[4 * 0 + 1], matrix[4 * 0 + 2],
                            matrix[4 * 1 + 0], matrix[4 * 1 + 1], matrix[4 * 1 + 2],
                            matrix[4 * 2 + 0], matrix[4 * 2 + 1], matrix[4 * 2 + 2]) / determinant);
        return new Matrix();
    }

    public static void invert(Matrix a, Matrix b) {
        b = a.inverse();
    }

    public static Matrix getViewMatrix(Vector origin, Vector direction, Vector right, Vector up) {
        return new Matrix(
                right.x, up.x, -direction.x, 0F,
                right.y, up.y, -direction.y, 0F,
                right.z, up.z, -direction.z, 0F,
                -Vector.scalarProduction(origin, right),
                -Vector.scalarProduction(origin, up),
                Vector.scalarProduction(origin, direction),
                1F
        );
    }

    public static Matrix getProjectionMatrix(
            float left,
            float right,
            float bottom,
            float top,
            float near,
            float far
    ) {
        return new Matrix(
                2 * near / (right - left), 0, 0, 0,
                0, 2 * near / (top - bottom), 0, 0,
                (right + left) / (right - left),
                (top + bottom) / (top - bottom),
                -(far + near) / (far - near),
                -1,
                0, 0, -2 * far * near / (far - near), 0
        );
    }

    public void translate(Vector vector) {
        for (int i = 0; i < 16; i++) {
            matrix[i] *= Vector.scalarProduction(vector, vector);
        }
    }
}
