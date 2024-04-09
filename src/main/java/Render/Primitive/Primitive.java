package main.java.Render.Primitive;

import com.jogamp.opengl.GL3;
import main.java.Camera.Camera;
import main.java.Render.Buffer.VertexArray;
import main.java.Render.Shader.Shader;

import java.nio.FloatBuffer;
import java.util.ArrayList;

// Primitive abstract class
public abstract class Primitive {
    protected Shader shader;                    // Primitive's main shader
    protected ArrayList<VertexArray> VAs;       // Primitive's vertex arrays
    protected ArrayList<Shader> deletedShaders; // For shaders after setting some new shader

    /**
     * Getting shader primitive's instance function.
     *
     * @return shader instance
     */
    public final Shader getShader() {
        return shader;
    } // End of 'getShader' function

    /**
     * Setting new primitive's shader
     *
     * @param shader_ new primitive's shader
     */
    public void setShader(Shader shader_) {
        deletedShaders.add(shader);
        shader = shader_;
    } // End of 'setShader' function

    /**
     * Primitive creation function.
     *
     * @param gl                 OpenGL interface
     * @param vertexBufferData   primitive's vertex buffer data
     * @param vertexBufferFormat format of each vertex in vertex buffer (example: v3v3v3v2, each 'v' is one attribute, next number is its size in floats)
     * @param indexBufferData    primitive's index buffer data
     * @param shaderPath         primitive's shader path
     */
    protected void create(GL3 gl, float[] vertexBufferData, String vertexBufferFormat, int[] indexBufferData, String shaderPath) {
        VAs = new ArrayList<>();
        deletedShaders = new ArrayList<>();
        shader = new Shader(gl, shaderPath);
        VAs.add(new VertexArray(gl, vertexBufferData, indexBufferData, vertexBufferFormat));
    } // End of 'create' function

    /**
     * Rendering primitive function.
     *
     * @param gl          OpenGL interface
     * @param frameCamera main frame camera
     */
    public void render(GL3 gl, Camera frameCamera, VertexArray.RenderType renderType) {
        gl.glUseProgram(shader.getShaderId());
        int uniformLocation = gl.glGetUniformLocation(shader.getShaderId(), "viewProjection");
        if (uniformLocation != -1) {
            gl.glUniformMatrix4fv(uniformLocation, 1, false, FloatBuffer.wrap(frameCamera.getViewProjectionMatrix().matrix));
        }
        VAs.getFirst().render(gl, renderType);
        gl.glUseProgram(0);
    } // End of 'render' function

    /**
     * Destroying primitive function.
     *
     * @param gl OpenGL interface
     */
    public void destroy(GL3 gl) {
        for (Shader deleteShader : deletedShaders) {
            deleteShader.destroy(gl);
        }
        shader.destroy(gl);
        for (VertexArray VA : VAs) {
            VA.destroy(gl);
        }
    } // End of 'destroy' function
} // End of 'Primitive' abstract class
