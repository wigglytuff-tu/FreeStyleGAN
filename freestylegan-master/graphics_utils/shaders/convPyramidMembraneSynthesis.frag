#version 420

// filter weights are taken from the authors:
// https://www.cs.huji.ac.il/labs/cglab/projects/convpyr/data/filters.txt

layout(binding=1) uniform sampler2D fgTexture2D;
layout(binding=2) uniform sampler2D bgTexture2D;
layout(binding=3) uniform sampler2D analysisTexture2D;
layout(binding=4) uniform sampler2D synthesisTexture2D;

uniform int level;									
uniform int maxLevel;

out vec4 synthesis;

//===================================================================

vec4 texelFetch2DBorderZero(const in sampler2D sampler, const in ivec2 texCoord, const in int level)
{
	ivec2 size = textureSize(sampler, level).xy;
	return texCoord.x < size.x && texCoord.y < size.y && texCoord.x >= 0 && texCoord.y >= 0 ? texelFetch(sampler, texCoord, level) : vec4(0);
}

//===================================================================

float kernelG(const ivec2 kernelCoord)
{	
	const float values[3] = float[] ( 0.0311849, 0.7752854, 0.0311849 );
	return values[kernelCoord.x + 1] * values[kernelCoord.y + 1];
}

//===================================================================

float kernelH2(const ivec2 kernelCoord)
{
	const float values[5] = float[] ( 0.1507146, 0.6835785, 1.0334191, 0.6835785, 0.1507146 );
	const float scaling = 0.0269546;
	return values[kernelCoord.x + 2] * values[kernelCoord.y + 2] * scaling;
}

//===================================================================

vec4 convolutionG(sampler2D sampler, const ivec2 coord, const int level)
{
	vec4 ret = vec4(0);
	for (int y = -1; y <= 1; y++)
	{
		for (int x = -1; x <= 1; x++)
		{
			ivec2 offset = ivec2(x, y);
			vec4 texel = texelFetch2DBorderZero(sampler, coord + offset, level);
			ret += kernelG(offset) * texel;
		}
	}
	return ret;
}

//===================================================================

vec4 convolutionH2(sampler2D sampler, const ivec2 coord, const int level)
{
	vec4 ret = vec4(0);
	for (int y = -2; y <= 2; y++)
	{
		for (int x = -2; x <= 2; x++)
		{
			ivec2 offset = ivec2(x, y);
			ivec2 convCoord = coord + offset;
			if (mod(convCoord, ivec2(2)) == ivec2(0))
			{
				vec4 texel = texelFetch2DBorderZero(sampler, convCoord / 2, level + 1);
				ret += kernelH2(offset) * texel;
			}
		}
	}
	return ret;
}

//===================================================================

void main()
{
	ivec2 coord2D = ivec2(gl_FragCoord.xy);

	if (level == maxLevel)
		synthesis = convolutionG(analysisTexture2D, coord2D, level);
	else
	{
		vec4 h = convolutionH2(synthesisTexture2D, coord2D, level);
		vec4 g = convolutionG(analysisTexture2D, coord2D, level);
		synthesis = g + h;

		if (level == 0)
		{
			vec4 fg = texelFetch(fgTexture2D, coord2D, 0);
			vec4 bg = texelFetch(bgTexture2D, coord2D, 0);
			vec4 membrane = synthesis / synthesis.a;
			synthesis = mix(bg, membrane + fg, fg.a);
		}
	}	
}