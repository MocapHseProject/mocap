package main.java.Render.Shader;

import com.jogamp.opengl.GL3;
import com.jogamp.opengl.GLException;
import com.jogamp.opengl.util.glsl.ShaderCode;
import com.jogamp.opengl.util.glsl.ShaderProgram;

import java.io.*;
import java.nio.file.FileSystemNotFoundException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

// Shader class
public class Shader {
    private int shaderId;                          // Id of shader
    private ArrayList<ShaderCode> shadersPrograms; // Array list of shader programs

    /**
     * Receiving shader's id function.
     *
     * @return shader id
     */
    public final int getShaderId() {
        return shaderId;
    } // End of 'getShaderId' function

    /**
     * Loading from shader program file its source code function.
     *
     * @param filePath path to shader program file
     * @return shader program source
     */
    private String loadShaderProgramSourceFromFile(String filePath) throws Exception {
        try {
            Path path = Paths.get(filePath);
            ArrayList<String> source = new ArrayList<String>();

            File file = new File(path.toString());
            if (!file.exists() || file.isDirectory()) {
                return null;
            }
            return file.toString();
        } catch (
                IllegalArgumentException e) {
            System.err.println("Preconditions on the uri parameter do not hold. The format of the URI is provider specific while loading: " + filePath);
            throw new RuntimeException(e);
        } catch (
                FileSystemNotFoundException e) {
            System.err.println("The file system, identified by the URI, does not exist and cannot be created automatically, or the provider identified by the URI's scheme component is not installed while loading: " + filePath);
            throw new RuntimeException(e);
        } catch (
                SecurityException e) {
            System.err.println("Security manager is installed and it denies an unspecified permission to access the file system while loading: " + filePath);
            throw new RuntimeException(e);
        } catch (
                NullPointerException e) {
            System.err.println("Pathname argument is null while loading: " + filePath);
            throw new RuntimeException(e);
        }
    } // End of 'loadShaderProgramSourceFromFile' function

    /**
     * Shader class default constructor function.
     */
    public Shader() {
        shaderId = 0;
        shadersPrograms = null;
    } // End of 'Shader' function

    /**
     * Shader constructor function.
     *
     * @param gl         OpenGL interface
     * @param shaderPath filepath to shader's programs
     */
    public Shader(GL3 gl, String shaderPath) throws GLException {
        shadersPrograms = new ArrayList<>();
        String[] names = {"vertex", "fragment", "geometry"};

        for (String name : names) {
            ShaderCode prog = null;
            boolean isExist = true;
            try {
                prog = ShaderCode.create(gl, name.equals("vertex") ? gl.GL_VERTEX_SHADER :
                                name.equals("fragment") ? gl.GL_FRAGMENT_SHADER : gl.GL_GEOMETRY_SHADER,
                        this.getClass(), shaderPath, null, name, "glsl", null, true);
            } catch (
                    GLException e) {
                if (name.equals("geometry")) {
                    System.err.println("Geometry shader program for \'" + shaderPath + "\' does not exist or some problems in it...");
                    isExist = false;
                } else {
                    throw new RuntimeException(e);
                }
            }
            if (isExist) {
                shadersPrograms.add(prog);
            }
        }

        ShaderProgram shaderProgram = new ShaderProgram();

        for (ShaderCode program : shadersPrograms) {
            shaderProgram.add(program);
        }
        shaderProgram.init(gl);
        shaderId = shaderProgram.program();
        if (!shaderProgram.link(gl, System.err)) {
            throw new RuntimeException("Error in linking shader program: " + shaderPath);
        }
    } // End of 'Shader' function

    /**
     * Destroying shader function.
     *
     * @param gl OpenGL interface
     */
    public void destroy(GL3 gl) {
        for (ShaderCode program : shadersPrograms) {
            if (shaderId != 0) {
                gl.glDetachShader(shaderId, program.id());
            }
            gl.glDeleteShader(program.id());
        }
        if (shaderId != 0) {
            gl.glDeleteProgram(shaderId);
        }
        shaderId = 0;
    } // End of 'destroy' function
} // End of 'Shader' class
