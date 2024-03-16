package main.java.Render.Buffer;

import com.jogamp.opengl.*;
import com.jogamp.opengl.util.GLBuffers;

import java.util.ArrayList;

// Vertex array class
public class VertexArray extends Buffer {
    // Rendering types of vertex array enum
    public enum RenderType {
        QUADS,
        TRIANGLES,
        TRIANGLES_STRIP,
        LINES
    } // End of 'RenderType' enum

    private VertexBuffer VB; // Vertex buffer of vertex array
    private IndexBuffer IB;  // Index buffer of vertex array (can be null)

    /**
     * Vertex array class default constructor function.
     */
    public VertexArray() {
        bufferId = null;
        sizeOfBuffer = 0;
        VB = null;
        IB = null;
    } // End of 'VertexArray' function

    /**
     * Vertex array constructor function.
     *
     * @param gl                 OpenGL interface
     * @param vertexBufferData   vertex buffer data
     * @param indexBufferData    index buffer data
     * @param vertexBufferFormat format of each vertex in vertex buffer (example: v3v3v3v2, each 'v' is one attribute, next number is its size in floats)
     */
    public VertexArray(GL3 gl, float[] vertexBufferData, int[] indexBufferData, String vertexBufferFormat) {
        if (vertexBufferData == null) {
            return;
        }

        create(gl);

        VB = new VertexBuffer(gl, vertexBufferData, vertexBufferFormat);
        if (IB != null && IB.getSizeOfBuffer() != 0)
            IB = new IndexBuffer(gl, indexBufferData);
        sizeOfBuffer = VB.getSizeOfBuffer();
        gl.glBindVertexArray(0);
    } // End of 'VertexArray' function

    /**
     * Vertex array constructor function.
     *
     * @param gl                 OpenGL interface
     * @param vertexBufferData   vertex buffer data as array list
     * @param indexBufferData    index buffer data as array list
     * @param vertexBufferFormat format of each vertex in vertex buffer (example: v3v3v3v2, each 'v' is one attribute, next number is its size in floats)
     * @apiNote not recommended for using because of low efficiency
     */
    public VertexArray(GL3 gl, ArrayList<Float> vertexBufferData, ArrayList<Integer> indexBufferData, String vertexBufferFormat) {
        if (vertexBufferData == null) {
            return;
        }

        create(gl);

        VB = new VertexBuffer(gl, vertexBufferData, vertexBufferFormat);
        if (IB != null && IB.getSizeOfBuffer() != 0)
            IB = new IndexBuffer(gl, indexBufferData);
        sizeOfBuffer = VB.getSizeOfBuffer();
        gl.glBindVertexArray(0);
    } // End of 'VertexArray' function

    /**
     * Vertex array constructor function.
     *
     * @param gl                 OpenGL interface
     * @param vertexBufferData   vertex buffer data as array list
     * @param indexBufferData    index buffer data
     * @param vertexBufferFormat format of each vertex in vertex buffer (example: v3v3v3v2, each 'v' is one attribute, next number is its size in floats)
     * @apiNote not recommended for using because of low efficiency
     */
    public VertexArray(GL3 gl, ArrayList<Float> vertexBufferData, int[] indexBufferData, String vertexBufferFormat) {
        if (vertexBufferData == null) {
            return;
        }

        create(gl);

        VB = new VertexBuffer(gl, vertexBufferData, vertexBufferFormat);
        if (IB != null && IB.getSizeOfBuffer() != 0)
            IB = new IndexBuffer(gl, indexBufferData);
        sizeOfBuffer = VB.getSizeOfBuffer();
        gl.glBindVertexArray(0);
    } // End of 'VertexArray' function

    /**
     * Vertex array constructor function.
     *
     * @param gl                 OpenGL interface
     * @param vertexBufferData   vertex buffer data
     * @param indexBufferData    index buffer data as array list
     * @param vertexBufferFormat format of each vertex in vertex buffer (example: v3v3v3v2, each 'v' is one attribute, next number is its size in floats)
     * @apiNote not recommended for using because of low efficiency
     */
    public VertexArray(GL3 gl, float[] vertexBufferData, ArrayList<Integer> indexBufferData, String vertexBufferFormat) {
        if (vertexBufferData == null) {
            return;
        }

        create(gl);

        VB = new VertexBuffer(gl, vertexBufferData, vertexBufferFormat);
        if (IB.getSizeOfBuffer() == 0)
            IB = new IndexBuffer(gl, indexBufferData);
        sizeOfBuffer = VB.getSizeOfBuffer();
        gl.glBindVertexArray(0);
    } // End of 'VertexArray' function

    /**
     * Rendering vertex array function.
     *
     * @param gl   OpenGL interface
     * @param type rendering type
     */
    public void render(GL3 gl, RenderType type) {
        gl.glBindVertexArray(bufferId.get(0));
        int glDrawingType = type == RenderType.TRIANGLES ? gl.GL_TRIANGLES
                : type == RenderType.TRIANGLES_STRIP ? gl.GL_TRIANGLE_STRIP
                : type == RenderType.QUADS ? gl.GL_QUADS
                : gl.GL_LINES;
        if (IB != null) {
            gl.glDrawElements(
                    glDrawingType,
                    IB.getSizeOfBuffer() / Integer.BYTES, gl.GL_UNSIGNED_INT, 0
            );
        } else {
            gl.glDrawArrays(
                    gl.GL_TRIANGLES,
                    0, VB.getSizeOfBuffer() / VB.getSizeOfVertex()
            );
        }
        gl.glBindVertexArray(0);
    } // End of 'render' function

    /**
     * Vertex array creating function.
     *
     * @param gl OpenGL interface
     */
    @Override
    public void create(GL3 gl) {
        bufferId = GLBuffers.newDirectIntBuffer(1);
        gl.glGenVertexArrays(1, bufferId);
        gl.glBindVertexArray(bufferId.get(0));
    } // End of 'create' function

    /**
     * Vertex array destroying function.
     *
     * @param gl OpenGL interface
     */
    @Override
    public void destroy(GL3 gl) {
        gl.glBindVertexArray(bufferId.get(0));
        VB.destroy(gl);
        if (IB != null) {
            IB.destroy(gl);
        }
        gl.glDeleteVertexArrays(1, bufferId);
        gl.glBindVertexArray(0);
        bufferId = null;
    } // End of 'destroy' function
} // End of 'VertexArray' class
