#version 420

#define EPS 10e-3

layout(location=0) out vec4 out_color;

layout(binding=0) uniform sampler2D inputTexture2D;

uniform int radius;
uniform ivec4 channels;


//===================================================================================

void main()
{  					
	ivec2 sampleCoord = ivec2(gl_FragCoord.xy);
	ivec2 fSize = textureSize(inputTexture2D, 0);
		
	vec4 sum = vec4(0);
	float weightSum = 0;

	float sigma = (radius / 3.) + EPS;

	vec4 centerSample = vec4(0);

	for (int x = -radius; x <= radius; x++) {
		for (int y = -radius; y <= radius; y++)
		{
			ivec2 coord = sampleCoord + ivec2(x, y);
			if (coord.x >= 0 && coord.x < fSize.x &&
				coord.y >= 0 && coord.y < fSize.y)
			{
				vec4 value = texelFetch(inputTexture2D, coord, 0);
				float weight = exp(-(x * x + y * y) / (2 * sigma * sigma));
				sum += value * vec4(weight);
				weightSum += weight;
				if (x == 0 && y == 0)
				centerSample = value;
			}
		}
	}
	if (weightSum == 0) weightSum = 1;
	out_color = vec4(sum / weightSum);

	out_color = mix(centerSample, out_color, vec4(channels));

}

