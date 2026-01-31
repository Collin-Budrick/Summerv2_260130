#ifndef AO_HALF_RES
#define AO_HALF_RES 1
#endif

#ifndef GTAO_QUALITY_HIGH
#define GTAO_QUALITY_HIGH 1
#endif

float getAoDepth(vec2 uv){
    return linearizeDepth(texture(depthtex2, uv).r);
}

vec2 getAoVelocity(vec2 uv){
#if !defined GBF && !defined SHD
    return texture(colortex9, uv).rg;
#else
    return vec2(0.0);
#endif
}

float SSAO_Core(vec2 uv, vec3 viewPos, vec3 normal, float sampleScale){
    float noise = rand2_1(uv + sin(frameTimeCounter));
    vec3 randomVec = rand2_3(uv + sin(frameTimeCounter)) * 2.0 - 1.0;

    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = normalize(cross(normal, tangent));
    mat3 TBN = mat3(tangent, bitangent, normal);

    float baseSamples = remapSaturate(length(viewPos), 0.0, 120.0, SSAO_MAX_SAMPLES, SSAO_MIN_SAMPLES);
    float N_SAMPLES = clamp(baseSamples * sampleScale, 2.0, SSAO_MAX_SAMPLES);
    int sampleCount = max(1, int(floor(N_SAMPLES + 0.5)));
    const float radius = SSAO_SEARCH_RADIUS;

    float ao = 0.0;
    for(int i = 0; i < sampleCount; ++i){
        vec3 offset = rand2_3(uv + sin(frameTimeCounter) + i + noise);
        offset.xy = offset.xy * 2.0 - 1.0;
            float scale = float(i) / float(sampleCount);
            scale = lerp(0.1, 1.0, scale * scale);
            offset *= scale;
        offset = TBN * offset;

        vec3 sampleViewPos = viewPos.xyz + offset * radius;
        vec3 sampleScreenPos = viewPosToScreenPos(vec4(sampleViewPos, 1.0)).xyz;
        float sampleDepth = texture(depthtex2, sampleScreenPos.xy).r;

        sampleDepth = linearizeDepth(sampleDepth);
        sampleScreenPos.z = linearizeDepth(sampleScreenPos.z);

        float nowAO = 0.0;
        if(sampleDepth < sampleScreenPos.z){
            float weight = 1.0;            

            float rangeCheck = smoothstep(0.0, 1.0, radius / (sampleScreenPos.z - sampleDepth));
            weight *= rangeCheck;
            
            if(outScreen(sampleScreenPos.xy))
                weight = 0.0;

            nowAO = 1.0 * weight;
        }
        ao += (1.0 - nowAO);
    }

    ao /= float(sampleCount);
    ao = pow(ao, SSAO_INTENSITY);
    return saturate(ao);
}

float HBAO(vec3 viewPos, vec3 normal){
    const int N_SAMPLES = 64;
    float dist = length(viewPos);
    float radius = 0.75;

    float ao = 0.0;
    for(int i = 0; i < N_SAMPLES; ++i){
        float rand1 = rand2_1(texcoord + sin(frameTimeCounter) + i);
        float rand2 = rand2_1(texcoord + sin(frameTimeCounter) + i + vec2(17.33333));
        float angle = rand2 * _2PI;
        vec2 offsetUV = vec2(rand1 * sin(angle), rand1 * cos(angle)) * radius / dist;

        vec2 curUV = texcoord + offsetUV;
            
        float z = texture(depthtex2, curUV).r;
        vec3 curViewPos = screenPosToViewPos(vec4(curUV, z, 1.0)).xyz;
        
        vec3 vector = curViewPos - viewPos + normalize(viewPos) * 0.1;

        float cosTheta = dot(normal, normalize(vector));
        // float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
        // if(sign(cosTheta) < 0.0)
        //     sinTheta = 0.0;

        float weight = max(0.0, 1.0 - length(vector) / radius);

        if(outScreen(curUV)) weight = 0.0;

        ao += saturate(cosTheta) * weight;
    }
    ao /= N_SAMPLES;

    return saturate(1.0 - 10.0 * ao);
}

// Practical Real-Time Strategies for Accurate Indirect Occlusion
// https://www.activision.com/cdn/research/Practical_Real_Time_Strategies_for_Accurate_Indirect_Occlusion_NEW%20VERSION_COLOR.pdf
float GTAO_Core(vec2 uv, vec3 viewPos, vec3 normal, float dhTerrain, float sampleScale){
    float rand = temporalBayer64(gl_FragCoord.xy);
    float dist = length(viewPos);
    int sliceCount = max(1, int(floor(float(GTAO_SLICE_COUNT) * sampleScale + 0.5)));
    int directionSampleCount = max(1, int(floor(float(GTAO_DIRECTION_SAMPLE_COUNT) * sampleScale + 0.5)));
    float scaling = GTAO_SEARCH_RADIUS / dist;
    
    float visibility = 0.0;
    viewPos += normal * 0.05;
    vec3 viewV = normalize(-viewPos);
    
    for (int slice = 0; slice < sliceCount; slice++) {
        float phi = (PI / float(sliceCount)) * (float(slice) + rand * 17.3333);
        vec2 omega = normalize(vec2(cos(phi), sin(phi)));
        vec3 directionV = vec3(omega.x, omega.y, 0.0);
        
        vec3 orthoDirectionV = directionV - dot(directionV, viewV) * viewV;
        vec3 axisV = cross(directionV, viewV);
        
        vec3 projNormalV = normal - axisV * dot(normal, axisV);
        
        float sgnN = sign(dot(orthoDirectionV, projNormalV));
        float cosN = saturate(dot(projNormalV, viewV) / max(length(projNormalV), 0.0001));
        float n = sgnN * acos(cosN);
        
        for (int side = 0; side <= 1; side++) {
            float cHorizonCos = -1.0;
            for (int samples = 0; samples < directionSampleCount; samples++) {
                float s = (float(samples) + 0.1 + rand) / float(directionSampleCount);
                
                vec2 offset = (2.0 * float(side) - 1.0) * s * scaling * omega;
                vec2 sampleUV = uv * 2.0 + offset;
                if(outScreen(sampleUV))
                    continue;
                
                float sampleDepth = texture(depthtex2, sampleUV).r;
                vec4 sampleScreenPos = vec4(sampleUV, sampleDepth, 1.0);
                vec3 sPosV = screenPosToViewPos(sampleScreenPos).xyz;

                #if defined DISTANT_HORIZONS && !defined NETHER && !defined END
                    if(dhTerrain > 0.5){
                        float dhSampleDepth = texture(dhDepthTex0, sampleUV).r;
                        sPosV = screenPosToViewPosDH(vec4(sampleUV, dhSampleDepth, 1.0)).xyz;
                    }
                #endif
                
                vec3 sHorizonV = normalize(sPosV - viewPos);
                float horizonCos = dot(sHorizonV, viewV);
                horizonCos = mix(-1.0, horizonCos, (smoothstep(0.0, 1.0, GTAO_SEARCH_RADIUS * 1.41 / distance(sPosV, viewPos))));
                cHorizonCos = max(cHorizonCos, horizonCos);
            } 

            float h = n + clamp((2.0 * float(side) - 1.0) * acos(cHorizonCos) - n, -PI/2.0, PI/2.0);
            visibility += length(projNormalV) * (cosN + 2.0 * h * sin(n) - cos(2.0 * h - n)) / 4.0;
        }
    }
    visibility /= float(sliceCount);
    return pow(visibility, GTAO_INTENSITY);
}

float AOHistoryBlend(vec2 uv, vec3 normal, float depth, float ao){
    vec2 velocity = getAoVelocity(uv);
    vec2 prevUV = uv - velocity;
    if(outScreen(prevUV))
        return ao;

    float prevDepth = getAoDepth(prevUV);
    vec3 prevNormal = getNormal(prevUV);

    float depthDiff = abs(prevDepth - depth) / max(depth, 0.001);
    float normalMatch = pow(saturate(dot(normal, prevNormal)), 4.0);
    float motionPixels = length(velocity * viewSize);
    float stability = exp(-depthDiff * 40.0) * normalMatch * saturate(1.0 - motionPixels * 0.2);

    float history = texture(colortex3, prevUV).a;
    float blend = saturate(stability * 0.8);
    return mix(ao, history, blend);
}

float computeAOSampleScale(vec3 viewPos, vec3 normal){
    float dist = length(viewPos);
    float depthFactor = remapSaturate(dist, 20.0, 120.0, 0.0, 1.0);
    float normalVariance = saturate(length(fwidth(normal)) * 3.0);
    return mix(1.0, 0.6, depthFactor) * mix(1.0, 0.7, normalVariance);
}

float computeAOAtUV(vec2 uv, vec3 viewPos, vec3 normal, float dhTerrain, bool useGTAO){
    float sampleScale = computeAOSampleScale(viewPos, normal);
    if(useGTAO){
        return GTAO_Core(uv, viewPos, normal, dhTerrain, sampleScale);
    }
    return SSAO_Core(uv, viewPos, normal, sampleScale * 0.8);
}

float computeAOHalfRes(vec3 viewPos, vec3 normal, float dhTerrain, bool useGTAO){
    vec2 halfPixel = invViewSize * 2.0;
    vec2 halfBase = (floor(texcoord / halfPixel) + vec2(0.5)) * halfPixel;
    float depthCenter = getAoDepth(texcoord);

    float aoSum = 0.0;
    float wSum = 0.0;
    for(int y = 0; y <= 1; ++y){
        for(int x = 0; x <= 1; ++x){
            vec2 sampleUV = halfBase + vec2(float(x), float(y)) * halfPixel;
            if(outScreen(sampleUV))
                continue;

            float sampleDepthRaw = texture(depthtex2, sampleUV).r;
            vec3 sampleViewPos = screenPosToViewPos(vec4(sampleUV, sampleDepthRaw, 1.0)).xyz;
            vec3 sampleNormal = getNormal(sampleUV);
            float sampleAO = computeAOAtUV(sampleUV, sampleViewPos, sampleNormal, dhTerrain, useGTAO);

            float sampleDepth = linearizeDepth(sampleDepthRaw);
            float depthWeight = exp(-abs(sampleDepth - depthCenter) * 4.0);
            float normalWeight = pow(saturate(dot(normal, sampleNormal)), 8.0);
            float weight = depthWeight * normalWeight;

            aoSum += sampleAO * weight;
            wSum += weight;
        }
    }

    if(wSum > 0.0)
        return aoSum / wSum;

    return computeAOAtUV(texcoord, viewPos, normal, dhTerrain, useGTAO);
}

float SSAO(vec3 viewPos, vec3 normal, float dhTerrain){
    float ao = 1.0;
#if AO_HALF_RES > 0
    ao = computeAOHalfRes(viewPos, normal, dhTerrain, false);
#else
    ao = computeAOAtUV(texcoord, viewPos, normal, dhTerrain, false);
#endif

    float depth = getAoDepth(texcoord);
    return AOHistoryBlend(texcoord, normal, depth, ao);
}

float GTAO(vec3 viewPos, vec3 normal, float dhTerrain){
    float depth = getAoDepth(texcoord);
    float normalVariance = saturate(length(fwidth(normal)) * 3.0);
    float motionPixels = length(getAoVelocity(texcoord) * viewSize);
    float discontinuity = saturate(fwidth(depth) * 50.0 + normalVariance);

    bool useGTAO = (GTAO_QUALITY_HIGH > 0)
        && (motionPixels < 2.5)
        && (discontinuity < 0.7);

    float ao = 1.0;
#if AO_HALF_RES > 0
    ao = computeAOHalfRes(viewPos, normal, dhTerrain, useGTAO);
#else
    ao = computeAOAtUV(texcoord, viewPos, normal, dhTerrain, useGTAO);
#endif

    return AOHistoryBlend(texcoord, normal, depth, ao);
}

vec3 AOMultiBounce(vec3 BaseColor, float ao){
	vec3 a =  2.0404 * BaseColor - 0.3324;
	vec3 b = -4.7951 * BaseColor + 0.6417;
	vec3 c =  2.7552 * BaseColor + 0.6903;
	return max(vec3(ao), (( ao * a + b) * ao + c) * ao);
}
