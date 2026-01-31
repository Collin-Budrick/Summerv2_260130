vec4 EqualWeightSeparableBlur(
    sampler2D colorTex,
    sampler2D depthTex,
    vec2 uv,
    vec2 dir,
    float radius,
    float quality,
    bool useNormal,
    bool useDepth,
    float normalThreshold,
    float depthThreshold)
{
    const int MAX_TAPS = 4;
    const float gaussianWeights[5] = float[](
        0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216
    );
    int taps = int(clamp(floor(quality), 1.0, float(MAX_TAPS)));
    float stepPx = radius / float(MAX_TAPS);

    vec4 cSum = vec4(0.0);
    float wSum = 0.0;

    vec3 centerN = vec3(0.0, 0.0, 1.0);
    float centerZ = 0.0;
    if (useNormal) {
        centerN = normalize(getNormal(uv));
    }
    if (useDepth) {
        centerZ = linearizeDepth(texture(depthTex, uv).r);
    }

    vec2 baseStep = dir * (stepPx * invViewSize);
    for (int i = 0; i <= MAX_TAPS; ++i) {
        if (i > taps) {
            continue;
        }
        float weight = gaussianWeights[i];
        vec2 offset = baseStep * float(i);
        vec2 sampleUV = uv + offset;

        if (outScreen(sampleUV)) continue;

        float w = 1.0;
        if (useNormal) {
            vec3 n = normalize(getNormal(sampleUV));
            float wN = saturate(dot(n, centerN) * normalThreshold); 
            w *= wN;
        }
        if (useDepth) {
            float z = linearizeDepth(texture(depthTex, sampleUV).r);
            float wZ = saturate(1.2 - abs(z - centerZ) * depthThreshold); 
            w *= wZ;
        }

        if (w <= 1e-5) continue;

        vec4 col = texture(colorTex, sampleUV);
        float finalWeight = weight * w;
        cSum += col * finalWeight;
        wSum += finalWeight;

        if (i > 0) {
            vec2 sampleUVNeg = uv - offset;
            if (!outScreen(sampleUVNeg)) {
                float wNeg = 1.0;
                if (useNormal) {
                    vec3 n = normalize(getNormal(sampleUVNeg));
                    float wN = saturate(dot(n, centerN) * normalThreshold); 
                    wNeg *= wN;
                }
                if (useDepth) {
                    float z = linearizeDepth(texture(depthTex, sampleUVNeg).r);
                    float wZ = saturate(1.2 - abs(z - centerZ) * depthThreshold); 
                    wNeg *= wZ;
                }

                if (wNeg > 1e-5) {
                    vec4 colNeg = texture(colorTex, sampleUVNeg);
                    float finalWeightNeg = weight * wNeg;
                    cSum += colNeg * finalWeightNeg;
                    wSum += finalWeightNeg;
                }
            }
        }
    }

    if (wSum <= 1e-5) {
        return texture(colorTex, uv);
    } else {
        return cSum / wSum;
    }
}

vec4 EqualWeightBlur_Horizontal(
    sampler2D colorTex, sampler2D depthTex,
    vec2 uv, float radius, float quality,
    bool useNormal, float normalThreshold, 
    bool useDepth, float depthThreshold)
{
    return EqualWeightSeparableBlur(colorTex, depthTex, uv, vec2(1.0, 0.0),
                                    radius, quality, useNormal, useDepth, normalThreshold, depthThreshold);
}

vec4 EqualWeightBlur_Vertical(
    sampler2D colorTex, sampler2D depthTex,
    vec2 uv, float radius, float quality,
    bool useNormal, float normalThreshold, 
    bool useDepth, float depthThreshold)
{
    return EqualWeightSeparableBlur(colorTex, depthTex, uv, vec2(0.0, 1.0),
                                    radius, quality, useNormal, useDepth, normalThreshold, depthThreshold);
}

vec4 JointBilateralFiltering_hrr_Horizontal(){
    // return texelFetch(colortex3, ivec2(gl_FragCoord.xy), 0);
    
    ivec2 pix = ivec2(gl_FragCoord.xy);

    #if defined DISTANT_HORIZONS && !defined NETHER && !defined END
        ivec2 uvC = ivec2(pix * 2.0 - vec2(0.0, 1.0) * viewSize);
        float depthHrrC = texelFetch(depthtex0, uvC, 0).r;
        float dhDepthC = texelFetch(dhDepthTex0, uvC, 0).r;
        bool isSkyC = depthHrrC == 1.0 && dhDepthC == 1.0;
    #endif

    vec4 curGD = texelFetch(colortex6, pix, 0);
    // vec3  normal0 = unpackNormal(curGD.r);
    float z0      = linearizeDepth(curGD.g);

    const float radius  = 6.0;
    const float quality = 6.0;
    float d = 2.0 * radius / quality;

    vec4 wSum = vec4(vec3(0.0), 0.0);
    vec4  cSum = vec4(0.0);

    for (float dx = -radius; dx <= radius + 0.001; dx += d) {
        ivec2 offset = ivec2(dx, 0.0);
        ivec2 p = pix + offset;

        if (outScreen((p * invViewSize) * 2.0 - vec2(0.0, 1.0))) continue;

        vec4 w = vec4(1.0);
        if(isEyeInWater == 0.0){
            vec4 gd = texelFetch(colortex6, p, 0);
            // vec3  n  = unpackNormal(gd.r);
            float z  = linearizeDepth(gd.g);
            
            float wZ = saturate(1.2 - abs(z - z0) * 1.0);      // 深度权重
            w  = vec4(wZ);

            #if defined DISTANT_HORIZONS && !defined NETHER && !defined END
                ivec2 uv = ivec2(p * 2.0 - vec2(0.0, 1.0) * viewSize);
                float depthHrr = texelFetch(depthtex0, uv, 0).r;
                float dhDepth = texelFetch(dhDepthTex0, uv, 0).r;
                bool isSky = depthHrr == 1.0 && dhDepth == 1.0;
                if(isSkyC != isSky){
                    w *= 0.01;
                }
            #endif
        }
        vec4 col = texelFetch(colortex3, p, 0);
        cSum += col * w;
        wSum += w;
    }

    return cSum / max(vec4(1e-4), wSum);
}

vec4 JointBilateralFiltering_hrr_Vertical(){
    // return texelFetch(colortex1, ivec2(gl_FragCoord.xy), 0);

    ivec2 pix = ivec2(gl_FragCoord.xy);

    #if defined DISTANT_HORIZONS && !defined NETHER && !defined END
        ivec2 uvC = ivec2(pix * 2.0 - vec2(0.0, 1.0) * viewSize);
        float depthHrrC = texelFetch(depthtex0, uvC, 0).r;
        float dhDepthC = texelFetch(dhDepthTex0, uvC, 0).r;
        bool isSkyC = depthHrrC == 1.0 && dhDepthC == 1.0;
    #endif

    vec4 curGD = texelFetch(colortex6, pix, 0);
    vec3  normal0 = unpackNormal(curGD.r);
    float z0      = linearizeDepth(curGD.g);

    const float radius  = 6.0;
    const float quality = 6.0;
    float d = 2.0 * radius / quality;

    vec4 wSum = vec4(vec3(0.0), 0.0);
    vec4  cSum = vec4(0.0);

    for (float dy = -radius; dy <= radius + 0.001; dy += d) {
        ivec2 offset = ivec2(0.0, dy);
        ivec2 p = pix + offset;

        if (outScreen((p * invViewSize) * 2.0 - vec2(0.0, 1.0))) continue;

        vec4 w = vec4(1.0);
        if(isEyeInWater == 0.0){
            vec4 gd = texelFetch(colortex6, p, 0);
            // vec3  n  = unpackNormal(gd.r);
            float z  = linearizeDepth(gd.g);
            
            float wZ = saturate(1.2 - abs(z - z0) * 1.0);
            w  = vec4(wZ);

            #if defined DISTANT_HORIZONS && !defined NETHER && !defined END
                ivec2 uv = ivec2(p * 2.0 - vec2(0.0, 1.0) * viewSize);
                float depthHrr = texelFetch(depthtex0, uv, 0).r;
                float dhDepth = texelFetch(dhDepthTex0, uv, 0).r;
                bool isSky = depthHrr == 1.0 && dhDepth == 1.0;
                if(isSkyC != isSky){
                    w *= 0.01;
                }
            #endif
        }

        vec4 col = texelFetch(colortex1, p, 0);
        cSum += col * w;
        wSum += w;
    }

    return cSum / max(vec4(1e-4), wSum);
}
