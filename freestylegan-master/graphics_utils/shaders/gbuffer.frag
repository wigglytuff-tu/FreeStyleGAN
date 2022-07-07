#version 420

uniform vec3 camPosition;
uniform int mode;

in vec3 frag_position;

out vec4 frag_color;

void main()
{	
	switch(mode){
		case(0): frag_color.xyz = frag_position; break;
		case(1): frag_color.xyz = vec3(distance(frag_position, camPosition)); break;
	}
	frag_color.w = 1;
}