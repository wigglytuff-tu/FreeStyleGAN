#version 420

#define NUM_CAMS 30  // upper limit
#define INFTY_W 100000.0
#define EPS 0.0000001

in vec2 uv;
layout(location = 0) out vec4 out_color;

// Uniforms
uniform vec3 novelCamPosition;
uniform int camsCount;
uniform float epsilonOcclusion = 0.5;
uniform bool showBlendingWeights = false;
uniform bool varianceWeightToAlpha = false;
uniform bool varianceWeightToRGB = false;

// Textures.
layout(binding=0) uniform sampler2D gBufferTexture2D;
layout(binding=1) uniform sampler2DArray input_rgbs;
layout(binding=2) uniform sampler2DArray input_depths;

// Input cameras.
struct CameraInfos
{
	mat4 vp;
	vec3 pos;
	bool transposed;
};

layout(binding=3) uniform InputCameras
{
   CameraInfos cameras[NUM_CAMS];
};

//=============================================================================================

vec4 getRGBD(vec2 coords, int index, bool transposed)
{
	if (transposed) coords = 1 - coords.yx;
	vec3 lookupCoord = vec3(coords, index);
	vec4 rgba = texture(input_rgbs, lookupCoord);
	if (rgba.a == 0) rgba.rgb = vec3(0);
	float depth = texture(input_depths, lookupCoord).r;
    return vec4(rgba.rgb, depth);
}

//=============================================================================================

vec3 project(vec3 point, mat4 proj)
{
	vec4 p1 = proj * vec4(point, 1.0);
  	vec3 p2 = (p1.xyz/p1.w);
  	return (0.5 * p2.xyz + 0.5);
}

//=============================================================================================

bool frustumTest(vec2 ndc)
{
  	return !any(greaterThan(ndc, vec2(1.0)));
}

//=============================================================================================

uint baseHash(uint p)
{
	p = 1103515245U*((p >> 1U)^(p));
	uint h32 = 1103515245U*((p)^(p>>3U));
	return h32^(h32 >> 16);
}

vec3 getRandomColor(int x)
{
	x = x+1;
	uint n = baseHash(uint(x));
	uvec3 rz = uvec3(n, n*16807U, n*48271U);
	return vec3(rz & uvec3(0x7fffffffU))/float(0x7fffffff);
}

//=============================================================================================

vec3 sqr3(vec3 x)
{
	return x * x;
}

//=============================================================================================

void main(void)
{  			
	vec4 point = texture(gBufferTexture2D, uv);
	
	// discard if there was no intersection with the proxy
	if (point.w == 0) discard;
	
	vec4  color0 = vec4(0, 0, 0, INFTY_W);
	vec4  color1 = vec4(0, 0, 0, INFTY_W);
	vec4  color2 = vec4(0, 0, 0, INFTY_W);
	vec4  color3 = vec4(0, 0, 0, INFTY_W);

	bool atLeastOneValid = false;

	for (int i = 0; i < camsCount; i++)
	{	
		vec3 uvd = project(point.xyz, cameras[i].vp);
		vec2 ndc = abs(2 * uvd.xy - 1);

		if (frustumTest(ndc))
		{
	 		vec4 rgbd = getRGBD(uvd.xy, i, cameras[i].transposed);

			if(all(equal(rgbd.xyz, vec3(0)))) continue;
			if (distance(point.xyz, cameras[i].pos) - rgbd.w >= epsilonOcclusion) continue;

			if (showBlendingWeights) rgbd.xyz = getRandomColor(i);

			vec3 v1 = (point.xyz - cameras[i].pos);
			vec3 v2 = (point.xyz - novelCamPosition);
			float dist_i2p 	= length(v1);
			float dist_n2p 	= length(v2);

			rgbd.w = max(0.0001, acos(dot(v1, v2) / (dist_i2p * dist_n2p)));
			atLeastOneValid = true;

			// compare with best four candiates and insert at the appropriate rank
			if (rgbd.w < color3.w) { // better than fourth best candidate
	 			if (rgbd.w < color2.w) {    // better than third best candidate
					color3 = color2;
					if (rgbd.w < color1.w) {    // better than second best candidate
	 					color2 = color1;
	 					if (rgbd.w < color0.w) {    // better than best candidate
	 						color1 = color0;
	 						color0 = rgbd;
	 					} else {
	 						color1 = rgbd;
	 					}
	 				} else {
	 					color2 = rgbd;
	 				}
	 			} else {
	 				color3 = rgbd;
	 			}
	 		}
	 	}  
	}
	
	if(!atLeastOneValid) discard;


	float thresh = (1 + EPS) * color3.w;
	color0.w = max(0, 1.0 - color0.w/thresh);
	color1.w = max(0, 1.0 - color1.w/thresh);
	color2.w = max(0, 1.0 - color2.w/thresh);
	color3.w = 1.0 - 1.0 / (1 + EPS);

	// ignore any candidate which is uninitialized
	if (color0.w == INFTY_W) color0.w = 0;
	if (color1.w == INFTY_W) color1.w = 0;
	if (color2.w == INFTY_W) color2.w = 0;
	
	// blending
	float weightSum = color0.w + color1.w + color2.w + color3.w;
	vec3 weightedColor = (
			color0.w * color0.xyz +
			color1.w * color1.xyz +
			color2.w * color2.xyz +
			color3.w * color3.xyz) / weightSum;

	out_color = vec4(weightedColor, 1);
	
	if (varianceWeightToAlpha || varianceWeightToRGB)
	{
		// compute color variance
		vec3 colorVar =  (color0.w * sqr3(color0.rgb - weightedColor)
						+ color1.w * sqr3(color1.rgb - weightedColor)
						+ color2.w * sqr3(color2.rgb - weightedColor)
						+ color3.w * sqr3(color3.rgb - weightedColor)) / weightSum;

		// turn into weight	
		float colVarFalloff = 100.;
		float varWeight = exp(-colVarFalloff * dot(colorVar, vec3(1)) / 3.);

		if (varianceWeightToAlpha) out_color.a = varWeight;
		if (varianceWeightToRGB) out_color.rgb = vec3(varWeight);
	}
}
