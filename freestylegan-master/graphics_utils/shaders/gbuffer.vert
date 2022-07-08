#version 420 core

layout(location = 0) in vec3 position;

uniform mat4 modelviewMatrix;
uniform mat4 projectionMatrix;
uniform vec2 lensDistortion = vec2(0);

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
    gl_Position = projectionMatrix * modelviewMatrix * vec4(position, 1);
    if (lensDistortion != vec2(0)) gl_Position = applyLensDistortion(gl_Position);
    frag_position = position;    
}