package main.java.Camera;

import main.java.MyMath.Matrix;
import main.java.MyMath.Vector;

import java.rmi.MarshalException;

import static java.lang.Math.max;

// Camera class
public class Camera {
    private Vector origin, direction, right, up;                       // Camera basic vectors
    private float far, near;                                           // Projection clipping distances
    private int width, height;                                         // Width and height of projection
    private float projectionSize;                                      // Size of projection
    private Matrix viewMatrix, projectionMatrix, viewProjectionMatrix; // Camera basic matrices

    /**
     * Get camera's view*projection matrix function.
     *
     * @return camera's VP matrix
     */
    public final Matrix getViewProjectionMatrix() {
        return viewProjectionMatrix;
    } // End of 'getViewProjectionMatrix' function

    /**
     * Camera default constructor function.
     */
    public Camera() {
        origin = new Vector(0);
        direction = new Vector(0, 0, 1);
        right = new Vector(1, 0, 0);
        up = new Vector(0, 1, 0);
        far = 1000;
        near = 1;
        projectionSize = 0.1F;
        width = height = 0;
        setViewMatrix();
        setProjection(0, 0);
    } // End of 'Camera' function

    /**
     * Camera constructor function.
     *
     * @param width_  width of camera projection
     * @param height_ height of camera projection
     */
    public Camera(int width_, int height_) {
        origin = new Vector(0);
        direction = new Vector(0, 0, -1);
        right = new Vector(1, 0, 0);
        up = new Vector(0, 1, 0);
        far = 1000F;
        near = 0.1F;
        projectionSize = 0.1F;
        viewMatrix = new Matrix();
        projectionMatrix = new Matrix();
        viewProjectionMatrix = new Matrix();
        setProjection(width_, height_);
        setViewMatrix();
    } // End of 'Camera' function

    /**
     * Setting camera projection function.
     *
     * @param width_  width of camera projection
     * @param height_ height of camera projection
     */
    public void setProjection(int width_, int height_) {
        far = 1000F;
        near = 0.1F;
        projectionSize = 0.1F;
        width = width_;
        height = height_;
        setProjectionMatrix();
    } // End of 'setProjection' function

    /**
     * Setting camera projection matrix function.
     */
    public void setProjectionMatrix() {
        float projectionWidth =
                max(1.0F, (float) width / (float) height) * projectionSize,
                projectionHeight =
                        max(1.0F, (float) height / (float) width) * projectionSize;
        projectionMatrix = Matrix.getProjectionMatrix(
                -projectionWidth / 2, projectionWidth / 2, -projectionHeight / 2, projectionHeight / 2, near, far
        );
        viewProjectionMatrix = Matrix.multiplicate(viewMatrix, projectionMatrix);
    } // End of 'setProjectionMatrix' function

    /**
     * Setting camera view matrix function.
     */
    public void setViewMatrix() {
        viewMatrix = Matrix.getViewMatrix(origin, direction, right, up);
        viewProjectionMatrix = Matrix.multiplicate(viewMatrix, projectionMatrix);
    } // End of 'setViewMatrix' function
} // End of 'Camera' class
