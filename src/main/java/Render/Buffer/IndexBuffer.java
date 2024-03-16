package main.java.Render.Buffer;

import com.jogamp.opengl.GL3;
import com.jogamp.opengl.util.GLBuffers;

import java.nio.IntBuffer;
import java.util.ArrayList;

// Index buffer class
public class IndexBuffer extends Buffer {
    /**
     * Reformatting index buffer data from array list to array of ints function.
     *
     * @param indexBufferData array list of index buffer data
     * @return index buffer data in array of ints
     */
    private int[] reformatIndexBufferData(ArrayList<Integer> indexBufferData) {
        int[] rawBufferData = new int[indexBufferData.size()];
        for (int i = 0; i < indexBufferData.size(); i++) {
            rawBufferData[i] = indexBufferData.get(i);
        }
        return rawBufferData;
    } // End of 'reformatIndexBufferData' function

    /**
     * Index buffer class default constructor function.
     */
    public IndexBuffer() {
        sizeOfBuffer = 0;
        bufferId = null;
    } // End of 'IndexBuffer' function

    /**
     * Index buffer constructor function.
     *
     * @param gl              OpenGL interface
     * @param indexBufferData index buffer data
     */
    public IndexBuffer(GL3 gl, int[] indexBufferData) {
        create(gl);

        IntBuffer buffer = GLBuffers.newDirectIntBuffer(indexBufferData);
        sizeOfBuffer = buffer.capacity() * Integer.BYTES;
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, sizeOfBuffer, buffer, gl.GL_STATIC_DRAW);
    } // End of 'IndexBuffer' function

    /**
     * Index buffer constructor function (not recommended for using because of low efficiency).
     *
     * @param gl              OpenGL interface
     * @param indexBufferData index buffer data as array list
     */
    public IndexBuffer(GL3 gl, ArrayList<Integer> indexBufferData) {
        create(gl);

        IntBuffer buffer = GLBuffers.newDirectIntBuffer(reformatIndexBufferData(indexBufferData));
        sizeOfBuffer = buffer.capacity() * Integer.BYTES;
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, sizeOfBuffer, buffer, gl.GL_STATIC_DRAW);
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0);
    } // End of 'IndexBuffer' function

    /**
     * Index buffer creating function.
     *
     * @param gl OpenGL interface
     */
    @Override
    public void create(GL3 gl) {
        bufferId = GLBuffers.newDirectIntBuffer(1);
        gl.glGenBuffers(1, bufferId);
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, bufferId.get(0));
    } // End of 'create' function

    /**
     * Index buffer destroying function.
     *
     * @param gl OpenGL interface
     */
    @Override
    public void destroy(GL3 gl) {
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0);
        gl.glDeleteBuffers(1, bufferId);
    } // End of 'destroy' function
} // End of 'IndexBuffer' class
