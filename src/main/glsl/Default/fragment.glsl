#version 330 core

out vec4 outColor;

in vec3 inColor;

void main() {
    outColor = vec4(inColor, 1);
}