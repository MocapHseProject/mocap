package main.java.Render.Primitive;

import com.jogamp.opengl.GL3;
import main.java.Render.Shader.Shader;

import java.util.ArrayList;

// Basic primitive class
public class BasicPrimitive extends Primitive {
    /**
     * Basic primitive default constructor function.
     */
    public BasicPrimitive() {
        shader = new Shader();
        VAs = new ArrayList<>();
        deletedShaders = new ArrayList<>();
    } // End of 'BasicPrimitive' function

    /**
     * Basic primitive constructor function.
     *
     * @param gl                 OpenGL interface
     * @param vertexBufferData   basic primitive's vertex buffer data
     * @param vertexBufferFormat format of each vertex in vertex buffer (example: v3v3v3v2, each 'v' is one attribute, next number is its size in floats)
     * @param indexBufferData    basic primitive's index buffer data
     * @param shaderPath         basic primitive's shader path
     */
    public BasicPrimitive(GL3 gl, float[] vertexBufferData, String vertexBufferFormat, int[] indexBufferData, String shaderPath) {
        create(gl, vertexBufferData, vertexBufferFormat, indexBufferData, shaderPath);
    } // End of 'BasicPrimitive' function
} // End of 'BasicPrimitive' class
