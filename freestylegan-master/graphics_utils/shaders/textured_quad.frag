#version 420

layout (binding = 0) uniform sampler2D inputTexture2D;
layout (binding = 1) uniform sampler2DArray inputTexture2DArray;

uniform ivec2 outputRes;
uniform int level;
uniform int layer = -1;  // -1 means use texture sampler
uniform bool transposed = false;
uniform bool remap = false;

in vec2 uv;

out vec4 out_color;

void main()
{	
	ivec2 inputRes = (layer == -1) ? textureSize(inputTexture2D, 0) : textureSize(inputTexture2DArray, 0).xy;
	
	vec2 customUV = uv;
	if (transposed) inputRes = inputRes.yx;
	if (outputRes != inputRes)
	{
		vec2 ratios = vec2(outputRes) / inputRes;
		customUV *= ratios / min(ratios.x, ratios.y);
	}
	if (transposed) customUV = 1 - customUV.yx;

	if (layer == -1) out_color = textureLod(inputTexture2D, customUV, level);
	else out_color = textureLod(inputTexture2DArray, vec3(customUV, layer), level);

	if (remap)
		out_color.rg = 0.5 * (out_color.rg + vec2(1));

}