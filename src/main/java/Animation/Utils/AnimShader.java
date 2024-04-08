package main.java.Animation.Utils;

import com.jogamp.opengl.GL3;
import main.java.Shader.Shader;

import java.io.IOException;

public class AnimShader extends Shader {
    public AnimShader(GL3 gl, String shaderPath) throws IOException {
        super(gl, shaderPath);
    }
    public void connectTextureUnits(String file) {

    }
}
