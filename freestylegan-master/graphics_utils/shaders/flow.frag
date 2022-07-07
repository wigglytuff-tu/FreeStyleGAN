#version 420

uniform mat4 viewMatrix0;
uniform mat4 projectionMatrix0;

in vec3 frag_position;

out vec4 frag_color;

void main()
{	
	vec4 projectedPos = projectionMatrix0 * viewMatrix0 * vec4(frag_position, 1);
	projectedPos /= projectedPos.w;
	projectedPos.xyz = 0.5 * (projectedPos.xyz + vec3(1));
	vec2 pxCoord = projectedPos.xy;

	if (any(lessThan(pxCoord, vec2(0))) ||  any(greaterThan(pxCoord, vec2(1))))
		discard;

	frag_color = vec4(1 - pxCoord.y , pxCoord.x, 0, 1);
}