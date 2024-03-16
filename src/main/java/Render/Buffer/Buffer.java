package main.java.Render.Buffer;

import com.jogamp.opengl.GL3;

import java.nio.IntBuffer;

// Buffer abstract class.
public abstract class Buffer {
    protected IntBuffer bufferId; // Ids of each of buffers
    protected int sizeOfBuffer;   // Size of buffer in bytes

    /**
     * Receiving buffer's id function.
     *
     * @return integer buffer of buffer's ids
     */
    public final IntBuffer getBufferId() {
        return bufferId;
    } // End of 'getBufferId' function

    /**
     * Receiving buffer's size function.
     *
     * @return buffer's size in bytes
     */
    public final int getSizeOfBuffer() {
        return sizeOfBuffer;
    } // End of 'getSizeOfBuffer' function

    /**
     * Buffer creating function.
     *
     * @param gl OpenGL interface
     */
    public void create(GL3 gl) {
    } // End of 'create' function

    /**
     * Buffer destroying function.
     *
     * @param gl OpenGL interface
     */
    public void destroy(GL3 gl) {
    } // End of 'destroy' function
} // End of 'Buffer' abstract class
