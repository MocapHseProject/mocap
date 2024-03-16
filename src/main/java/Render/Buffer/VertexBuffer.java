package main.java.Render.Buffer;

import com.jogamp.opengl.GL3;
import com.jogamp.opengl.util.GLBuffers;

import java.nio.FloatBuffer;
import java.util.ArrayList;

// Vertex buffer class
public class VertexBuffer extends Buffer {
    private int sizeOfVertex; // Size of one vertex buffer's element in bytes

    /**
     * Receive vertex size of vertex buffer function.
     *
     * @return size of one vertex buffer's element in bytes
     */
    public final int getSizeOfVertex() {
        return sizeOfVertex;
    } // End of 'getSizeOfVertex' function

    /**
     * Reformatting vertex buffer data from array list to array of floats function.
     *
     * @param vertexBufferData array list of vertex buffer data
     * @return index buffer data in array of floats
     */
    private float[] reformatVertexBufferData(ArrayList<Float> vertexBufferData) {
        float[] rawBufferData = new float[vertexBufferData.size()];
        for (int i = 0; i < vertexBufferData.size(); i++) {
            rawBufferData[i] = vertexBufferData.get(i);
        }
        return rawBufferData;
    } // End of 'reformatVertexBufferData' function

    /**
     * Setting vertex buffer's attributes function.
     *
     * @param gl                 OpenGL interface
     * @param vertexBufferFormat format of each vertex in vertex buffer (example: v3v3v3v2, each 'v' is one attribute, next number is its size in floats)
     */
    private void setVertexBufferAttributes(GL3 gl, String vertexBufferFormat) {
        sizeOfVertex = 0;
        for (int i = 0; i < vertexBufferFormat.length(); i++) {
            char sign = vertexBufferFormat.charAt(i);
            if (sign == 'v') {
                continue;
            } else {
                int attributeSize = sign - '0';
                sizeOfVertex += attributeSize * Float.BYTES;
            }
        }

        int attributeIndex = -1;
        int stride = 0;
        for (int i = 0; i < vertexBufferFormat.length(); i++) {
            char sign = vertexBufferFormat.charAt(i);
            if (sign == 'v') {
                attributeIndex++;
            } else {
                int attributeSize = sign - '0';
                gl.glVertexAttribPointer(attributeIndex, attributeSize, gl.GL_FLOAT, false, sizeOfVertex, stride * Float.BYTES);
                gl.glEnableVertexAttribArray(attributeIndex);
                stride += attributeSize;
            }
        }
    } // End of 'setVertexBufferAttributes' function

    /**
     * Vertex buffer class default constructor function.
     */
    public VertexBuffer() {
        bufferId = null;
        sizeOfBuffer = 0;
        sizeOfVertex = 0 * Float.BYTES;
    } // End of 'VertexBuffer' function

    /**
     * Vertex buffer constructor function.
     *
     * @param gl                 OpenGL interface
     * @param vertexBufferData   vertex buffer data
     * @param vertexBufferFormat format of each vertex in vertex buffer (example: v3v3v3v2, each 'v' is one attribute, next number is its size in floats)
     */
    public VertexBuffer(GL3 gl, float[] vertexBufferData, String vertexBufferFormat) {
        create(gl);

        FloatBuffer buffer = GLBuffers.newDirectFloatBuffer(vertexBufferData);
        sizeOfBuffer = buffer.capacity() * Float.BYTES;
        gl.glBufferData(gl.GL_ARRAY_BUFFER, sizeOfBuffer, buffer, gl.GL_STATIC_DRAW);
        setVertexBufferAttributes(gl, vertexBufferFormat);
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0);
    } // End of 'VertexBuffer' function

    /**
     * Vertex buffer constructor function.
     *
     * @param gl                 OpenGL interface
     * @param vertexBufferData   vertex buffer data as array list
     * @param vertexBufferFormat format of each vertex in vertex buffer (example: v3v3v3v2, each 'v' is one attribute, next number is its size in floats)
     * @apiNote not recommended for using because of low efficiency
     */
    public VertexBuffer(GL3 gl, ArrayList<Float> vertexBufferData, String vertexBufferFormat) {
        create(gl);

        FloatBuffer buffer = GLBuffers.newDirectFloatBuffer(reformatVertexBufferData(vertexBufferData));
        sizeOfBuffer = buffer.capacity() * Float.BYTES;
        sizeOfVertex = (3 + 3 + 3 + 2) * Float.BYTES;
        gl.glBufferData(gl.GL_ARRAY_BUFFER, sizeOfBuffer, buffer, gl.GL_STATIC_DRAW);

        setVertexBufferAttributes(gl, vertexBufferFormat);
    } // End of 'VertexBuffer' function

    /**
     * Vertex buffer creating function.
     *
     * @param gl OpenGL interface
     */
    @Override
    public void create(GL3 gl) {
        bufferId = GLBuffers.newDirectIntBuffer(1);
        gl.glGenBuffers(1, bufferId);
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, bufferId.get(0));
    } // End of 'create' function

    /**
     * Vertex buffer destroying function.
     *
     * @param gl OpenGL interface
     */
    @Override
    public void destroy(GL3 gl) {
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0);
        gl.glDeleteBuffers(1, bufferId);
    } // End of 'destroy' function
} // End of 'VertexBuffer' class
