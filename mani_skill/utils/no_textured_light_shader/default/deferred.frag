#version 450

layout (constant_id = 0) const int NUM_DIRECTIONAL_LIGHTS = 3;
layout (constant_id = 1) const int NUM_POINT_LIGHTS = 10;
layout (constant_id = 2) const int NUM_DIRECTIONAL_LIGHT_SHADOWS = 1;
layout (constant_id = 3) const int NUM_POINT_LIGHT_SHADOWS = 3;
layout (constant_id = 4) const int NUM_TEXTURED_LIGHT_SHADOWS = 1;
layout (constant_id = 5) const int NUM_SPOT_LIGHT_SHADOWS = 10;
layout (constant_id = 6) const int NUM_SPOT_LIGHTS = 10;

#define SET_NUM 0
#include "./scene_set.glsl"
#undef SET_NUM

#define SET_NUM 1
#include "./camera_set.glsl"
#undef SET_NUM

layout(set = 2, binding = 0) uniform sampler2D samplerAlbedo;
layout(set = 2, binding = 1) uniform sampler2D samplerPositionRaw;
layout(set = 2, binding = 2) uniform sampler2D samplerSpecular;
layout(set = 2, binding = 3) uniform sampler2D samplerNormal;
layout(set = 2, binding = 4) uniform sampler2D samplerEmission;
layout(set = 2, binding = 5) uniform sampler2D samplerGbufferDepth;
layout(set = 2, binding = 6) uniform sampler2D samplerCustom;

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outLighting;

vec4 world2camera(vec4 pos) {
  return cameraBuffer.viewMatrix * pos;
}

vec3 getBackgroundColor(vec3 texcoord) {
  texcoord = texcoord.xzy;
  return textureLod(samplerEnvironment, texcoord, 0).rgb + sceneBuffer.ambientLight.rgb;
}

vec3 diffuseIBL(vec3 albedo, vec3 N) {
  N = N.xzy;
  vec3 color = textureLod(samplerEnvironment, N, 5).rgb;
  return color * albedo;
}

vec3 specularIBL(vec3 fresnel, float roughness, vec3 N, vec3 V) {
  float dotNV = max(dot(N, V), 0);
  vec3 R = 2 * dot(N, V) * N - V;
  R = R.xzy;
  vec3 color = textureLod(samplerEnvironment, R, roughness * 5).rgb;
  vec2 envBRDF = texture(samplerBRDFLUT, vec2(roughness, dotNV)).xy;
  return color * (fresnel * envBRDF.x + envBRDF.y);
}

void main() {
  vec3 albedo = texture(samplerAlbedo, inUV).xyz;
  vec3 frm = texture(samplerSpecular, inUV).xyz;
  float specular = frm.x;
  float roughness = frm.y;
  float metallic = frm.z;

  vec3 normal = normalize(texture(samplerNormal, inUV).xyz);
  float depth = texture(samplerGbufferDepth, inUV).x;

  vec4 csPosition = cameraBuffer.projectionMatrixInverse * (vec4(inUV * 2 - 1, depth, 1));
  csPosition /= csPosition.w;

  vec3 camDir = -normalize(csPosition.xyz);

  vec3 diffuseAlbedo = albedo * (1 - metallic);
  vec3 fresnel = specular * (1 - metallic) + albedo * metallic;

  vec4 emission = texture(samplerEmission, inUV);
  vec3 color = emission.rgb * emission.a;

  // point light
  for (int i = 0; i < NUM_POINT_LIGHT_SHADOWS; ++i) {
    vec3 pos = world2camera(vec4(sceneBuffer.pointLights[i].position.xyz, 1.f)).xyz;
    mat4 shadowProj = shadowBuffer.pointLightBuffers[6 * i].projectionMatrix;

    vec3 l = pos - csPosition.xyz;
    vec3 wsl = vec3(cameraBuffer.viewMatrixInverse * vec4(l, 0));

    vec3 v = abs(wsl);
    vec4 p = shadowProj * vec4(0, 0, -max(max(v.x, v.y), v.z), 1);
    float pixelDepth = p.z / p.w;
    float shadowDepth = texture(samplerPointLightDepths[i], wsl).x;

    float visibility = step(pixelDepth - shadowDepth, 0);
    color += visibility * computePointLight(
        sceneBuffer.pointLights[i].emission.rgb,
        l, normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  for (int i = NUM_POINT_LIGHT_SHADOWS; i < NUM_POINT_LIGHTS; i++) {
    vec3 pos = world2camera(vec4(sceneBuffer.pointLights[i].position.xyz, 1.f)).xyz;
    vec3 l = pos - csPosition.xyz;
    color += computePointLight(
        sceneBuffer.pointLights[i].emission.rgb,
        l, normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  // directional light
  for (int i = 0; i < NUM_DIRECTIONAL_LIGHT_SHADOWS; ++i) {
    mat4 shadowView = shadowBuffer.directionalLightBuffers[i].viewMatrix;
    mat4 shadowProj = shadowBuffer.directionalLightBuffers[i].projectionMatrix;

    vec3 lightDir = mat3(cameraBuffer.viewMatrix) * sceneBuffer.directionalLights[i].direction.xyz;

    vec4 ssPosition = shadowView * cameraBuffer.viewMatrixInverse * vec4((csPosition.xyz), 1);
    vec4 shadowMapCoord = shadowProj * ssPosition;
    shadowMapCoord /= shadowMapCoord.w;
    shadowMapCoord.xy = shadowMapCoord.xy * 0.5 + 0.5;

    float resolution = textureSize(samplerDirectionalLightDepths[i], 0).x;
    float visibility = ShadowMapPCF(
        samplerDirectionalLightDepths[i], shadowMapCoord.xyz, resolution, 1 / resolution, 1);

    color += visibility * computeDirectionalLight(
        lightDir,
        sceneBuffer.directionalLights[i].emission.rgb,
        normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  for (int i = NUM_DIRECTIONAL_LIGHT_SHADOWS; i < NUM_DIRECTIONAL_LIGHTS; ++i) {
    color += computeDirectionalLight(
        mat3(cameraBuffer.viewMatrix) * sceneBuffer.directionalLights[i].direction.xyz,
        sceneBuffer.directionalLights[i].emission.rgb,
        normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  // spot light
  for (int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; ++i) {
    mat4 shadowView = shadowBuffer.spotLightBuffers[i].viewMatrix;
    mat4 shadowProj = shadowBuffer.spotLightBuffers[i].projectionMatrix;

    vec3 pos = world2camera(vec4(sceneBuffer.spotLights[i].position.xyz, 1.f)).xyz;
    vec3 centerDir = mat3(cameraBuffer.viewMatrix) * sceneBuffer.spotLights[i].direction.xyz;
    vec3 l = pos - csPosition.xyz;

    vec4 ssPosition = shadowView * cameraBuffer.viewMatrixInverse * vec4((csPosition.xyz), 1);
    vec4 shadowMapCoord = shadowProj * ssPosition;
    shadowMapCoord /= shadowMapCoord.w;
    shadowMapCoord.xy = shadowMapCoord.xy * 0.5 + 0.5;

    float resolution = textureSize(samplerSpotLightDepths[i], 0).x;
    float visibility = ShadowMapPCF(
        samplerSpotLightDepths[i], shadowMapCoord.xyz, resolution, 1 / resolution, 1);

    color += visibility * computeSpotLight(
        sceneBuffer.spotLights[i].emission.a,
        sceneBuffer.spotLights[i].direction.a,
        centerDir,
        sceneBuffer.spotLights[i].emission.rgb,
        l, normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  for (int i = NUM_SPOT_LIGHT_SHADOWS; i < NUM_SPOT_LIGHTS; ++i) {
    vec3 pos = world2camera(vec4(sceneBuffer.spotLights[i].position.xyz, 1.f)).xyz;
    vec3 l = pos - csPosition.xyz;
    vec3 centerDir = mat3(cameraBuffer.viewMatrix) * sceneBuffer.spotLights[i].direction.xyz;
    color += computeSpotLight(
        sceneBuffer.spotLights[i].emission.a,
        sceneBuffer.spotLights[i].direction.a,
        centerDir,
        sceneBuffer.spotLights[i].emission.rgb,
        l, normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  // environmental light
  vec3 wnormal = mat3(cameraBuffer.viewMatrixInverse) * normal;
  color += diffuseIBL(diffuseAlbedo, wnormal);
  color += specularIBL(fresnel, roughness,
                       wnormal,
                       mat3(cameraBuffer.viewMatrixInverse) * camDir);

  color += sceneBuffer.ambientLight.rgb * albedo.rgb;

  if (depth == 1) {
    outLighting = vec4(getBackgroundColor((cameraBuffer.viewMatrixInverse * csPosition).xyz), 0.f);
  } else {
    outLighting = vec4(color, 1);
  }
}
