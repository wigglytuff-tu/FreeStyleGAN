#version 420 core

#define EPS 10e-6

layout(location = 0) in vec3 position;

uniform mat4 viewMatrix1;
uniform mat4 projectionMatrix1;
uniform vec2 lensDistortion = vec2(0);
uniform float zClamp = 1000;

out vec3 frag_position;

//------------------------------------------

vec4 applyLensDistortion(vec4 pos)
{
    pos /= pos.w;
    float r = length(pos.xy);
    float factor = 1 + dot(lensDistortion, vec2(pow(r, 2), pow(r, 4.)));
    pos.xy /= factor;
    return pos;
}

//------------------------------------------

void main()
{
    vec3 inPos = position;
    inPos.z = max(inPos.z, -(zClamp + EPS));
    gl_Position = projectionMatrix1 * viewMatrix1 * vec4(inPos, 1);
    if (lensDistortion != vec2(0)) gl_Position = applyLensDistortion(gl_Position);
    frag_position = inPos;    
}