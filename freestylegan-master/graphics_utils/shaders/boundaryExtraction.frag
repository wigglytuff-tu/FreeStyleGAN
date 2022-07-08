#version 420

layout(location=0) out vec4 out_color;

layout(binding=0) uniform sampler2D fgTexture2D;
layout(binding=1) uniform sampler2D bgTexture2D;

uniform float threshold;

//===================================================================================

void main()
{  					
	ivec2 coord2D = ivec2(gl_FragCoord.xy);
	out_color = vec4(0);
	
	vec4 fg = texelFetch(fgTexture2D, coord2D, 0);
	
	if (fg.a == 0) return;
	
	float bestDst = 10000000.;

	// Am I a boundary pixel? See if one of my neighbors has zero alpha.
	for (int y=-1; y<=1; y++)
	{
		for (int x=-1; x<=1; x++)
		{
			float alphaNeigh = texelFetch(fgTexture2D, coord2D + ivec2(x, y), 0).a;
			if (alphaNeigh == 0)
			{
				vec3 bg = texelFetch(bgTexture2D, coord2D, 0).rgb;
				vec3 color_diff = bg - fg.rgb;
				float color_dist = length(color_diff);
				if (color_dist < threshold && color_dist < bestDst)
				{
					out_color = vec4(color_diff, 1);
					bestDst = color_dist;
				}
			}
		}
	}
}
