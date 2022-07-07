#version 420

layout(location=0) out vec4 out_color;

layout(binding=0) uniform sampler2D inputTexture2D;
layout(binding=1) uniform sampler2D flowTexture2D;

//===================================================================================

void main()
{  					
	vec4 warpCoord = texelFetch(flowTexture2D, ivec2(gl_FragCoord.xy), 0);
	if (warpCoord.a == 0) 
		out_color = vec4(0);
	else
	{
		warpCoord.xy = vec2(warpCoord.y, 1 - warpCoord.x);
		out_color = texture(inputTexture2D, warpCoord.xy);
	}
}
