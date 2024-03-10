#version 330 core
layout (location = 0) in vec3 Position;
layout (location = 1) in vec3 Color;
layout (location = 2) in vec3 Normal;

out vec3 inColor;
//out vec3 inNormal;

uniform mat4 viewProjection;

void main() {
    gl_Position = viewProjection * vec4(Position.xyz, 1);
    //inNormal = mat3(transpose(inverse(transform))) * normal.normalize();
    inColor = Color;
}