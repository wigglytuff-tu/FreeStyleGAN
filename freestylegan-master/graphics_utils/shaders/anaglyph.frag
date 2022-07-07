#version 420

layout(location=0) out vec4 out_color;

layout(binding=0) uniform sampler2D leftTexture2D;
layout(binding=1) uniform sampler2D rightTexture2D;

//===================================================================================

void main()
{  					
	ivec2 coord2D = ivec2(gl_FragCoord.xy);

	vec4 leftColor = texelFetch(leftTexture2D, coord2D, 0);
	vec4 rightColor = texelFetch(rightTexture2D, coord2D, 0);
	
	mat3 leftFilter = mat3(
		0,   0, 0,
		0.7, 0, 0,
		0.3, 0, 0);

	mat3 rightFilter = mat3(
		0, 0, 0,
		0, 1, 0,
		0, 0, 1);

	leftColor.rgb = leftFilter * leftColor.rgb;
	leftColor.r = pow(leftColor.r, 2.0/3.0);
	rightColor.rgb = rightFilter * rightColor.rgb;

	out_color.rgb = vec3(leftColor.rgb + rightColor.rgb);
	out_color.a = mix(leftColor.a, rightColor.a, 0.5);

}
