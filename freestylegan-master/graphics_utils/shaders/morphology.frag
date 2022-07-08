#version 420

#define INFTY 100000.0

layout(location=0) out vec4 out_color;

uniform int kernelSize;
uniform bool erode;
uniform bool alphaOnly;

bool circle = true; // circle as structuring element?

layout(binding=0) uniform sampler2D inputTexture2D;

//=============================================================================================

void main(void)
{  			
	ivec2 coord = ivec2(gl_FragCoord.xy);
	out_color = (erode) ? vec4(INFTY) : vec4(-INFTY);

	for (int y = -kernelSize; y <= kernelSize; y++)
	{
		for (int x = -kernelSize; x <= kernelSize; x++)
		{
			ivec2 offset = ivec2(x, y);
			if (circle && length(offset) > kernelSize) continue;
			
			if (alphaOnly)
			{
				float fetch = texelFetch(inputTexture2D, coord + offset, 0).a;
				out_color.a = (erode) ? min(out_color.a, fetch) : max(out_color.a, fetch);
			}
			else
			{
				vec4 fetch = texelFetch(inputTexture2D, coord + offset, 0);
				out_color = (erode) ? min(out_color, fetch) : max(out_color, fetch);
			}
		}
	}
	if (alphaOnly) out_color.rgb = texelFetch(inputTexture2D, coord, 0).rgb;
}
