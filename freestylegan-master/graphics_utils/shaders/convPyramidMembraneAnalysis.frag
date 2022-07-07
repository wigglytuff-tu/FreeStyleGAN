#version 420

layout(binding=0) uniform sampler2D boundaryTexture2D;
layout(binding=3) uniform sampler2D analysisTexture2D;		

uniform int level;									

out vec4 analysis;

//===================================================================

vec4 texelFetch2DBorderZero(const in sampler2D sampler, const in ivec2 texCoord, const in int level)
{
	ivec2 size = textureSize(sampler, level).xy;
	return texCoord.x < size.x && texCoord.y < size.y && texCoord.x >= 0 && texCoord.y >= 0 ? texelFetch(sampler, texCoord, level) : vec4(0);
}

//===================================================================

// filter weights are taken from the authors:
// https://www.cs.huji.ac.il/labs/cglab/projects/convpyr/data/filters.txt
float kernelH1(const ivec2 kernelCoord)
{
	const float values[5] = float[] ( 0.1507146, 0.6835785, 1.0334191, 0.6835785, 0.1507146 );
	return values[kernelCoord.x + 2] * values[kernelCoord.y + 2];
}

//===================================================================

vec4 convolutionH1(sampler2D sampler, const ivec2 coord, const int level)
{
	vec4 ret = vec4(0);
	for (int y = -2; y <= 2; y++)
	{
		for (int x = -2; x <= 2; x++)
		{
			ivec2 offset = ivec2(x, y) + ivec2(0, 0);
			vec4 texel = texelFetch2DBorderZero(sampler, coord + offset, level);
			ret += kernelH1(offset) * texel;
		}
	}
	return ret;
}

//===================================================================

void main()
{
	if (level == 0) 
		analysis = texelFetch(boundaryTexture2D, ivec2(gl_FragCoord.xy), 0);
	else
		analysis = convolutionH1(analysisTexture2D, (ivec2(gl_FragCoord.xy) << 1), level - 1);
}