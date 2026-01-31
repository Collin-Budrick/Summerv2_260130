// https://zhuanlan.zhihu.com/p/525500877
vec2 offset = vec2(10.0 * invViewSize.x, 0.0);
float bloomLuminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}
vec4 uvTable[9] = vec4[](
    vec4(0.0, 0.0, 1.0, 1.0),
    vec4(0.0, 0.0, 0.5, 0.5),
    vec4(0.0, 0.0, 0.25, 0.25),
    vec4(0.25, 0.0, 0.375, 0.125) + offset.xyxy,
    vec4(0.375, 0.0, 0.4375, 0.0625) + 2.0 * offset.xyxy,
    vec4(0.4375, 0.0, 0.46875, 0.03125) + 3.0 * offset.xyxy,
    vec4(0.46875, 0.0, 0.484375, 0.015625) + 4.0 * offset.xyxy,
    vec4(0.484375, 0.0, 0.4921875, 0.0078125) + 5.0 * offset.xyxy,
    vec4(0.4921875, 0.0, 0.49609375, 0.00390625) + 6.0 * offset.xyxy
);

#ifdef BLOOM_DOWNSAMPLE
    const int BLOOM_MAX_MIPS = 4;
    const float BLOOM_LUMA_THRESHOLD = 0.02;

    vec4 uvI = uvTable[BLOOM_LOD - 1];
    vec4 uvO = uvTable[BLOOM_LOD];
    vec2 uv = vec2(remap(texcoord.s, uvO.x, uvO.z, uvI.x, uvI.z), remap(texcoord.t, uvO.y, uvO.w, uvI.y, uvI.w));

    vec3 color = vec3(0.0);
    bool shouldProcess = BLOOM_LOD <= BLOOM_MAX_MIPS;
    float texLuma = 0.0;
    #if BLOOM_LOD == 1
        if(!isOutsideRange(texcoord, uvO.xy, uvO.zw)) {
            texLuma = bloomLuminance(texture(colortex0, uv).rgb);
            if (texLuma > BLOOM_LUMA_THRESHOLD && shouldProcess) {
                vec3 blurH = gaussianBlur6x1(colortex0, uv, 2.0, 0.0);
                vec3 blurV = gaussianBlur1x6(colortex0, uv, 2.0, 0.0);
                color = 0.5 * (blurH + blurV);
            }
        }
        if(any(greaterThan(texcoord.xy, uvO.zw + vec2(offset.x)))) discard;
    #elif BLOOM_LOD == 2
        if(!isOutsideRange(texcoord, uvO.xy, uvO.zw)) {
            texLuma = bloomLuminance(texture(colortex1, uv).rgb);
            if (texLuma > BLOOM_LUMA_THRESHOLD && shouldProcess) {
                vec3 blurH = gaussianBlur6x1(colortex1, uv, 2.0, 0.0);
                vec3 blurV = gaussianBlur1x6(colortex1, uv, 2.0, 0.0);
                color = 0.5 * (blurH + blurV);
            }
        }
        if(texcoord.x > uvTable[8].z + offset.x || texcoord.y > 0.25 + offset.x) discard;
    #elif BLOOM_LOD == 3
        color = texture(colortex1, texcoord).rgb;
        if(!isOutsideRange(texcoord, uvO.xy, uvO.zw)) {
            texLuma = bloomLuminance(texture(colortex1, uv).rgb);
            if (texLuma > BLOOM_LUMA_THRESHOLD && shouldProcess) {
                vec3 blurH = gaussianBlur6x1(colortex1, uv, 2.0, 0.0);
                vec3 blurV = gaussianBlur1x6(colortex1, uv, 2.0, 0.0);
                color = 0.5 * (blurH + blurV);
            }
        }
        if(texcoord.x > uvTable[8].z + offset.x || texcoord.y > 0.25 + offset.x) discard;
    #else
        color = texture(colortex1, texcoord).rgb;
        if(!isOutsideRange(texcoord, uvO.xy, uvO.zw)) {
            texLuma = bloomLuminance(texture(colortex1, uv).rgb);
            if (texLuma > BLOOM_LUMA_THRESHOLD && shouldProcess) {
                vec3 blurH = gaussianBlur6x1(colortex1, uv, 2.0, 0.0);
                vec3 blurV = gaussianBlur1x6(colortex1, uv, 2.0, 0.0);
                color = 0.5 * (blurH + blurV);
            }
        }
        if(texcoord.x > uvO.z + offset.x || texcoord.y > 0.25 + offset.x) discard;
    #endif
#endif

#ifdef BLOOM_UPSAMPLE
    const vec4 uvO = vec4(0.0, 0.0, 1.0, 1.0);
    const int BLOOM_MAX_LAYERS = 4;
    const int BLOOM_STABLE_LAYERS = 2;
    const float BLOOM_LUMA_STABLE_DELTA = 0.01;

    #if BLOOM_MODE == 0
        const float bloomWeights[7] = float[](
            1.62, 1.56, 1.49, 1.41, 1.32, 1.19, 1.0
        );
    #elif BLOOM_MODE == 1
        const float bloomWeights[7] = float[](
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        );
    #elif BLOOM_MODE == 2
        const float bloomWeights[7] = float[](
            1.0, 1.19, 1.32, 1.41, 1.49, 1.56, 1.62
        );
    #endif

    float lumDelta = abs(curLum - preLum);
    bool reusePrev = lumDelta < BLOOM_LUMA_STABLE_DELTA;
    int maxLayers = min(BLOOM_LAYERS, reusePrev ? BLOOM_STABLE_LAYERS : BLOOM_MAX_LAYERS);
    vec3 historyBloom = texture(colortex2, texcoord).rgb;

    for (int i = 0; i < maxLayers; ++i) {
        vec4 uvI = uvTable[i+2];
        vec2 uv = vec2(remap(texcoord.s, uvO.x, uvO.z, uvI.x, uvI.z), 
                    remap(texcoord.t, uvO.y, uvO.w, uvI.y, uvI.w));
        blur += textureBicubic(colortex1, uv).rgb * bloomWeights[i];
    }

    if (reusePrev) {
        blur = mix(historyBloom, blur, saturate(lumDelta / BLOOM_LUMA_STABLE_DELTA));
    }
#endif
