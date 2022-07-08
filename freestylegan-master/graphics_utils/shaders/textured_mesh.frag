#version 420

uniform mat4 alignmentMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

layout (binding = 1) uniform sampler2D inputTexture2D;

in vec2 frag_uv;

out vec4 out_color;

void main()
{	
	vec2 uv = vec2(frag_uv.x, 1 - frag_uv.y);
	out_color = texture(inputTexture2D, uv);
}